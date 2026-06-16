# LoRA variant of dgx_sft_expr.py — route (b): a TINY expressivity adapter instead of a full
# 4GB checkpoint. Same data / forward / loss / voice-agnostic trick as the full FT; the ONLY
# change is PEFT LoRA on the same target modules (L16-26 self_attn q/k/v/o + mlp.gate_proj)
# instead of unfreezing+training the full matrices. Output = a small PEFT adapter dir
# (adapter_model.safetensors, tens of MB).
import argparse, json, os, random, time
import torch
from accelerate import Accelerator
from dgx_dataset_expr import TTSDataset
from peft import LoraConfig, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

PRESET_IDS = [3066, 3065, 3010, 3061, 2861, 2873, 2864, 2875, 2878]  # serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan


def parse_layers(spec):
    """'16-26' / '0-27' / '0-12,16-26' -> sorted list of layer indices.
    The emotion band is 16-26; the prosody map (docs/prosody-map.md) found general/linguistic
    prosody lives EARLIER (L00-L12), so a language-prosody LoRA wants a wider/earlier band."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", default="/root/qwen-ft/models/1.7B-CustomVoice")
    ap.add_argument("--output_model_path", default="/root/qwen-ft/out_expr_lora")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)        # LoRA likes a higher LR than full FT
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--layers", default="16-26",
                    help="Talker layers to adapt: '16-26' (emotion, default) | '0-27' (all) | "
                         "'0-12,16-26' (early prosody + emotion). See docs/prosody-map.md.")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    acc = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.init_model_path)

    # LoRA on attn q/k/v/o + mlp.gate_proj, restricted to the chosen layer band (--layers).
    # Default 16-26 = emotion band (the full FT); a wider/earlier band (e.g. 0-12,16-26) also
    # adapts the general-prosody layers (docs/prosody-map.md).
    train_layers = parse_layers(args.layers)
    if acc.is_main_process:
        acc.print(f"[layers] adapting {len(train_layers)} layers: {train_layers}", flush=True)
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        layers_to_transform=train_layers, layers_pattern="layers",
    )
    model = get_peft_model(qwen3tts.model, lcfg)
    if acc.is_main_process:
        model.print_trainable_parameters()

    data = [json.loads(l) for l in open(args.train_jsonl)]
    ds = TTSDataset(data, qwen3tts.processor, config)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    model, opt, dl = acc.prepare(model, opt, dl)
    model.train()

    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * args.num_epochs
    acc.print(f"[plan] {len(data)} samples | {steps_per_epoch} steps/epoch x {args.num_epochs} epochs "
              f"= {total_steps} steps | batch {args.batch_size}", flush=True)

    def save_adapter(tag):
        if not acc.is_main_process:
            return
        out_dir = os.path.join(args.output_model_path, tag)
        acc.unwrap_model(model).save_pretrained(out_dir)
        sz = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
                 if f.endswith(".safetensors"))
        acc.print(f"[save] adapter -> {out_dir}  ({sz/1e6:.1f} MB)", flush=True)

    t0 = time.time()
    gstep = 0
    for epoch in range(args.num_epochs):
        ep_loss, ep_n = 0.0, 0
        for step, b in enumerate(dl):
            with acc.accumulate(model):
                input_ids = b["input_ids"]; B = input_ids.shape[0]
                tmask = b["text_embedding_mask"]; cmask = b["codec_embedding_mask"]
                spk_pos = b["spk_pos"]
                tk = model.talker  # PEFT forwards attribute access to the base model
                te = tk.model.text_embedding(input_ids[:, :, 0]) * tmask
                ce = tk.model.codec_embedding(input_ids[:, :, 1]) * cmask
                # voice-agnostic: random preset speaker at the (shifted) speaker slot
                spk_ids = torch.tensor([random.choice(PRESET_IDS) for _ in range(B)], device=model.device)
                spk_emb = tk.model.codec_embedding(spk_ids).to(ce.dtype)
                for j in range(B):
                    ce[j, spk_pos[j], :] = spk_emb[j]
                emb = te + ce
                for i in range(1, 16):
                    emb = emb + tk.code_predictor.get_input_embeddings()[i - 1](b["codec_ids"][:, :, i]) * b["codec_mask"].unsqueeze(-1)
                out = tk(inputs_embeds=emb[:, :-1, :], attention_mask=b["attention_mask"][:, :-1],
                         labels=b["codec_0_labels"][:, 1:], output_hidden_states=True)
                hs = out.hidden_states[0][-1][b["codec_mask"][:, :-1]]
                sub_ids = b["codec_ids"][b["codec_mask"]]
                _, sub_loss = tk.forward_sub_talker_finetune(sub_ids, hs)
                loss = out.loss + 0.3 * sub_loss
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            gstep += 1
            ep_loss += loss.item(); ep_n += 1
            if step % 10 == 0:
                el = time.time() - t0
                rate = gstep / el if el > 0 else 0.0
                eta = (total_steps - gstep) / rate / 60 if rate > 0 else 0.0
                acc.print(f"epoch {epoch}/{args.num_epochs - 1} step {step}/{steps_per_epoch} "
                          f"gstep {gstep}/{total_steps} loss {loss.item():.4f} "
                          f"| {rate:.2f} it/s | elapsed {el / 60:.1f}m ETA {eta:.1f}m", flush=True)
        acc.print(f"[epoch {epoch} DONE] avg_loss {ep_loss / max(ep_n, 1):.4f}", flush=True)
        save_adapter(f"adapter-ep{epoch}")   # per-epoch checkpoint: progress proof + early-testable adapter

    save_adapter("adapter-final")


if __name__ == "__main__":
    train()
