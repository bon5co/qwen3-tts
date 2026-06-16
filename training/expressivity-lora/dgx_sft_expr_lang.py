# Targeted expressivity fine-tune of Qwen3-TTS CustomVoice: train ONLY L16-26 (mlp.gate_proj +
# self_attn) + text_projection, instruct-conditioned, voice-agnostic (random preset at the speaker
# slot). Based on QwenLM/Qwen3-TTS finetuning/sft_12hz.py. Output = a full CV checkpoint dir.
import argparse, json, os, random, shutil
import torch
from accelerate import Accelerator
from dgx_dataset_expr_lang import TTSDataset  # LANGUAGE-TAGGED builder
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

PRESET_IDS = [3066, 3065, 3010, 3061, 2861, 2873, 2864, 2875, 2878]  # serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan
TRAIN_LAYERS = set(range(16, 27))

def is_trainable(name):
    if "text_projection" in name:
        return True
    for L in TRAIN_LAYERS:
        if f"talker.model.layers.{L}." in name and ("mlp.gate_proj" in name or "self_attn" in name):
            return True
    return False

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", default="/root/qwen-ft/models/1.7B-CustomVoice")
    ap.add_argument("--output_model_path", default="/root/qwen-ft/out_expr")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--layers", default="16-26",
                    help="dense-FT layer band, e.g. 16-26 (default) | 0-27 (wide) | 0-12,16-26")
    args = ap.parse_args()
    global TRAIN_LAYERS
    def _parse_layers(spec):
        out=set()
        for part in spec.split(","):
            part=part.strip()
            if "-" in part:
                a,b=part.split("-"); out|=set(range(int(a),int(b)+1))
            elif part: out.add(int(part))
        return out
    TRAIN_LAYERS = _parse_layers(args.layers)
    print(f"[layers] DENSE full-rank FT on {len(TRAIN_LAYERS)} layers: {sorted(TRAIN_LAYERS)}", flush=True)

    acc = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
    qwen3tts = Qwen3TTSModel.from_pretrained(args.init_model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
    config = AutoConfig.from_pretrained(args.init_model_path)

    # (a) FREEZE all but L16-26 + text_projection
    for p in qwen3tts.model.parameters():
        p.requires_grad = False
    ntr = 0
    for n, p in qwen3tts.model.named_parameters():
        if is_trainable(n):
            p.requires_grad = True; ntr += p.numel()
    acc.print(f"[freeze] trainable params: {ntr/1e6:.1f}M (L16-26 gate_proj+attn + text_projection)")

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
                te = model.talker.model.text_embedding(input_ids[:, :, 0]) * tmask
                ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * cmask
                # (c) voice-agnostic: random preset speaker at the (shifted) speaker slot
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

    # save full CV checkpoint (instruct stays in weights; loadable by our C engine)
    if acc.is_main_process:
        out_dir = os.path.join(args.output_model_path, "checkpoint-final")
        shutil.copytree(args.init_model_path, out_dir, dirs_exist_ok=True)
        sd = {k: v.detach().to("cpu") for k, v in acc.unwrap_model(model).state_dict().items()}
        save_file(sd, os.path.join(out_dir, "model.safetensors"))
        acc.print(f"[save] -> {out_dir}")

if __name__ == "__main__":
    train()
