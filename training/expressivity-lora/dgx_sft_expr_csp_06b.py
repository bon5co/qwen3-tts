# CSP-FT step 2/2 — Characteristic-Specific Partial Fine-Tune of Qwen3-TTS CustomVoice (arXiv 2501.14273).
#
# Fork of dgx_sft_expr_lang.py (LEFT UNTOUCHED). The ONLY behavioural change is WHICH params are trainable:
# instead of a hand-picked band (L16-26 / L0-27) we train ONLY the emotion-specific layers chosen by the
# probe (csp_probe.py -> csp_layers.json), and we train the WHOLE block (attn + mlp + norms), not just
# gate_proj+attn. EVERYTHING ELSE is FROZEN — crucially the pronunciation/prosody layers — which is what
# keeps speech clean (paper CER 1.2% vs full-FT 3.9%) and removes the need for the disentangle τ
# subtraction (freezing the knowledge-rich layers does the protecting). See plan_emo_v2.md.
#
# Same as the lang trainer otherwise: language-tagged dataset (so one mixed FT routes per-language),
# voice-agnostic (random preset at the speaker slot), instruct-conditioned, output = a full CV checkpoint.
#
# Self-test (NO model — verifies the freeze mask selects exactly the probed blocks):
#   python3 dgx_sft_expr_csp.py --self-test --csp-layers 20,24
#
# Real run (DGX):
#   CSP=$(python3 -c "import json;print(json.load(open('/root/qwen-ft/csp_layers_italian.json'))['selected']['top_k'])" | tr -d '[] ')
#   python3 dgx_sft_expr_csp.py --csp-layers "$CSP" --train_jsonl /root/qwen-ft/data/italian_emotion.jsonl \
#       --output_model_path /root/qwen-ft/out_csp_italian --num_epochs 10
import argparse, json, os, random, shutil, sys
import torch

CSP_LAYERS = set()          # filled from --csp-layers (block indices chosen by the probe)
TRAIN_TEXT_PROJ = False     # CSP default: FREEZE text_projection too (pronunciation bridge). Flag to enable.
SCOPE = "full"              # "full" = whole block | "attn_mlp" = only self_attn + mlp.gate_proj (legacy band style)


def is_trainable(name):
    if TRAIN_TEXT_PROJ and "text_projection" in name:
        return True
    for L in CSP_LAYERS:
        prefix = f"talker.model.layers.{L}."
        if name.startswith(prefix) or f".{prefix}" in name:
            if SCOPE == "full":
                return True
            if SCOPE == "attn_mlp" and ("mlp.gate_proj" in name or "self_attn" in name):
                return True
    return False


def _parse_layers(spec):
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", default="/root/qwen-ft/models/1.7B-CustomVoice")
    ap.add_argument("--output_model_path", default="/root/qwen-ft/out_csp")
    ap.add_argument("--train_jsonl", required=False)
    ap.add_argument("--csp-layers", dest="csp_layers", required=False,
                    help="comma/range list of emotion-specific BLOCK indices from the probe, e.g. 20,24 or 18-20")
    ap.add_argument("--scope", choices=["full", "attn_mlp"], default="full",
                    help="full = fine-tune the whole selected block (CSP default) | attn_mlp = legacy band style")
    ap.add_argument("--train-text-projection", dest="train_text_proj", action="store_true",
                    help="also unfreeze text_projection (default: FROZEN, to protect pronunciation)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num_epochs", type=int, default=10)   # paper trains the selected layers for 10 epochs
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    args = ap.parse_args()

    global CSP_LAYERS, TRAIN_TEXT_PROJ, SCOPE
    SCOPE = args.scope
    TRAIN_TEXT_PROJ = args.train_text_proj
    if not args.csp_layers:
        ap.error("--csp-layers required (from csp_probe.py output)")
    CSP_LAYERS = _parse_layers(args.csp_layers)

    if args.self_test:
        _self_test(); return

    if not args.train_jsonl:
        ap.error("--train_jsonl required for a real run")

    from accelerate import Accelerator
    from dgx_dataset_expr_lang import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    PRESET_IDS = [3066, 3065, 3010, 3061, 2861, 2873, 2864, 2875, 2878]
    print(f"[csp] trainable BLOCKS: {sorted(CSP_LAYERS)}  scope={SCOPE}  text_projection={'train' if TRAIN_TEXT_PROJ else 'FROZEN'}", flush=True)

    acc = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.init_model_path)

    for p in qwen3tts.model.parameters():
        p.requires_grad = False
    ntr = 0; ntr_names = []
    for n, p in qwen3tts.model.named_parameters():
        if is_trainable(n):
            p.requires_grad = True; ntr += p.numel(); ntr_names.append(n)
    total = sum(p.numel() for p in qwen3tts.model.parameters())
    acc.print(f"[freeze] trainable {ntr/1e6:.1f}M / {total/1e6:.1f}M = {100*ntr/total:.1f}% "
              f"(CSP blocks {sorted(CSP_LAYERS)}); rest FROZEN incl. pronunciation/prosody layers")

    data = [json.loads(l) for l in open(args.train_jsonl)]
    ds = TTSDataset(data, qwen3tts.processor, config)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    opt = AdamW([p for p in qwen3tts.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    model, opt, dl = acc.prepare(qwen3tts.model, opt, dl)
    model.train()

    for epoch in range(args.num_epochs):
        for step, b in enumerate(dl):
            with acc.accumulate(model):
                input_ids = b["input_ids"]; B = input_ids.shape[0]
                tmask = b["text_embedding_mask"]; cmask = b["codec_embedding_mask"]
                spk_pos = b["spk_pos"]
                # 0.6B: bridge text_hidden 2048 -> talker hidden 1024 via text_projection before te+ce.
                te = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * tmask
                ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * cmask
                spk_ids = torch.tensor([random.choice(PRESET_IDS) for _ in range(B)], device=model.device)
                spk_emb = model.talker.model.codec_embedding(spk_ids).to(ce.dtype)
                for j in range(B):
                    ce[j, spk_pos[j], :] = spk_emb[j]
                emb = te + ce
                for i in range(1, 16):
                    emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](b["codec_ids"][:, :, i]) * b["codec_mask"].unsqueeze(-1)
                out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=b["attention_mask"][:, :-1],
                                   labels=b["codec_0_labels"][:, 1:], output_hidden_states=True)
                hs = out.hidden_states[0][-1][b["codec_mask"][:, :-1]]
                sub_ids = b["codec_ids"][b["codec_mask"]]
                _, sub_loss = model.talker.forward_sub_talker_finetune(sub_ids, hs)
                loss = out.loss + 0.3 * sub_loss
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if step % 10 == 0:
                acc.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if acc.is_main_process:
        out_dir = os.path.join(args.output_model_path, "checkpoint-final")
        shutil.copytree(args.init_model_path, out_dir, dirs_exist_ok=True)
        sd = {k: v.detach().to("cpu") for k, v in acc.unwrap_model(model).state_dict().items()}
        save_file(sd, os.path.join(out_dir, "model.safetensors"))
        acc.print(f"[save] -> {out_dir}")


def _self_test():
    """Verify the freeze mask selects EXACTLY the probed blocks (and nothing from other layers)."""
    names = []
    for L in range(28):
        for sub in ["self_attn.q_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight",
                    "input_layernorm.weight"]:
            names.append(f"talker.model.layers.{L}.{sub}")
    names += ["talker.model.text_projection.weight", "talker.model.codec_embedding.weight"]
    sel_full = sorted({n for n in names if is_trainable(n)})
    by_layer = {}
    for n in sel_full:
        if ".layers." in n:
            L = int(n.split(".layers.")[1].split(".")[0]); by_layer.setdefault(L, []).append(n)
    print(f"--csp-layers parsed -> blocks {sorted(CSP_LAYERS)}  scope={SCOPE}  text_proj={'train' if TRAIN_TEXT_PROJ else 'frozen'}")
    print("trainable layers:", sorted(by_layer))
    assert sorted(by_layer) == sorted(CSP_LAYERS), f"freeze mask trained {sorted(by_layer)}, expected {sorted(CSP_LAYERS)}"
    if SCOPE == "full":
        # a non-attn/mlp param (a layernorm) of a selected block MUST be trainable in full scope
        L = sorted(CSP_LAYERS)[0]
        assert is_trainable(f"talker.model.layers.{L}.input_layernorm.weight"), "full scope must train norms too"
    assert not is_trainable("talker.model.codec_embedding.weight"), "codec_embedding must stay frozen"
    if not TRAIN_TEXT_PROJ:
        assert not is_trainable("talker.model.text_projection.weight"), "text_projection must be frozen by default"
    print("SELF-TEST PASS — only the probed blocks are unfrozen; pronunciation layers stay frozen.")


if __name__ == "__main__":
    train()
