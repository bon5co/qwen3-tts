# LANGUAGE-TAGGED variant of dgx_dataset_expr.py (which is left UNTOUCHED).
#
# WHY: the original builder (and even upstream Qwen dataset.py) trains with the NO-language codec prefix
#   [NO_THINK, THINK_BOS, THINK_EOS, speaker, PAD, ...]
# i.e. it NEVER tells the model which language a sample is. That's fine for a SINGLE-language FT (the
# weights specialise implicitly), but on a MIXED multilingual set the dominant language wins and clobbers
# the minority one (measured: ESD+CREMA EN 97.7% vs EMOVO IT 2.3% -> the FT spoke great English and
# wrecked Italian on ryan). Inference, by contrast, DOES pass the language as a codec token:
#   CustomVoice WITH language: [THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]   (qwen_tts.c:962)
#
# This builder injects that SAME language_id token per sample, so (a) training matches inference, and
# (b) the model can route emotion per-language in ONE mixed FT (the goal). Each row must carry a
# `language` field (English/Italian/...); we map it to the codec token id used by the C engine.
#
# Token ids verified against qwen_tts.h + Qwen3TTSConfig.talker_config (2026-06-16):
#   THINK 2154  NO_THINK 2155  THINK_BOS 2156  THINK_EOS 2157  PAD 2148  BOS 2149  EOS 2150
#   language codec tokens (qwen_tts.c lang_table): English 2050, Italian 2070, Chinese 2055, ...
#
# Run the SELF-TEST (no model needed) to verify the prefix matches inference byte-for-byte:
#   python3 dgx_dataset_expr_lang.py --self-test
from typing import List
import torch
from torch.utils.data import Dataset

# language name -> codec token id (MUST match qwen_tts.c lang_table)
LANG2TOK = {
    "Chinese": 2055, "English": 2050, "Japanese": 2058, "Korean": 2064, "German": 2053,
    "French": 2061, "Russian": 2069, "Portuguese": 2071, "Spanish": 2054, "Italian": 2070,
}


class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config, lag_num=-1):
        self.data_list = data_list; self.processor = processor
        self.lag_num = lag_num; self.config = config

    def __len__(self): return len(self.data_list)

    def _build_assistant_text(self, text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct(self, instruct):
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tok(self, text):
        inp = self.processor(text=text, return_tensors="pt", padding=True)["input_ids"]
        return inp.unsqueeze(0) if inp.dim() == 1 else inp

    def _lang_tok(self, item):
        lang = item.get("language", "")
        if lang not in LANG2TOK:
            raise ValueError(f"row language={lang!r} not in {sorted(LANG2TOK)} — every row must be tagged")
        return LANG2TOK[lang]

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_ids = self._tok(self._build_assistant_text(item["text"]))[:, :-5]
        instruct = item.get("instruct", "") or ""
        instruct_ids = self._tok(self._build_instruct(instruct)) if instruct.strip() else torch.zeros((1, 0), dtype=torch.long)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        return {"instruct_ids": instruct_ids, "text_ids": text_ids, "audio_codes": audio_codes,
                "lang_tok": self._lang_tok(item)}

    def collate_fn(self, batch):
        assert self.lag_num == -1
        cfg, tk = self.config, self.config.talker_config
        # body now starts 1 LATER than the no-lang builder (the extra language token grows the prefix
        # from 5 special slots to 6): prefix codec = [think, think_bos, LANG, think_eos, spk, pad].
        lens = [b["instruct_ids"].shape[1] + b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        T = max(lens) + 10
        B = len(batch)
        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)
        tmask = torch.zeros((B, T), dtype=torch.bool)
        cmask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)

        for i, d in enumerate(batch):
            ins = d["instruct_ids"][0]; p = ins.shape[0]
            tids = d["text_ids"]; ac0 = d["audio_codes"][:, 0]; acs = d["audio_codes"]
            tlen = tids.shape[1]; clen = ac0.shape[0]
            L = d["lang_tok"]
            # --- instruct prefix (positions 0:p) ---
            if p > 0:
                input_ids[i, :p, 0] = ins
                input_ids[i, :p, 1] = tk.codec_pad_id
                tmask[i, :p] = True; cmask[i, :p] = True; attn[i, :p] = 1
            o = p
            # --- text channel (5 pads + bos, body at o+9) ---
            input_ids[i, o:o+3, 0] = tids[0, :3]
            input_ids[i, o+3:o+8, 0] = cfg.tts_pad_token_id          # 5 pads: o+3..o+7
            input_ids[i, o+8, 0] = cfg.tts_bos_token_id              # bos at o+8
            input_ids[i, o+9:o+9+tlen-3, 0] = tids[0, 3:]
            input_ids[i, o+9+tlen-3, 0] = cfg.tts_eos_token_id
            input_ids[i, o+9+tlen-2:o+9+tlen+clen, 0] = cfg.tts_pad_token_id
            tmask[i, :o+9+tlen+clen] = True
            # --- codec channel: [think, think_bos, LANG, think_eos, spk@o+7, pad] at o+3..o+8 ---
            input_ids[i, o+3:o+9, 1] = torch.tensor([tk.codec_think_id, tk.codec_think_bos_id, L,
                                                     tk.codec_think_eos_id, 0, tk.codec_pad_id])
            input_ids[i, o+9:o+9+tlen-3, 1] = tk.codec_pad_id
            input_ids[i, o+9+tlen-3, 1] = tk.codec_pad_id
            input_ids[i, o+9+tlen-2, 1] = tk.codec_bos_id
            input_ids[i, o+9+tlen-1:o+9+tlen-1+clen, 1] = ac0
            input_ids[i, o+9+tlen-1+clen, 1] = tk.codec_eos_token_id
            labels[i, o+9+tlen-1:o+9+tlen-1+clen] = ac0
            labels[i, o+9+tlen-1+clen] = tk.codec_eos_token_id
            codec_ids[i, o+9+tlen-1:o+9+tlen-1+clen, :] = acs
            cmask[i, o+3:o+9+tlen+clen] = True
            cmask[i, o+7] = False              # speaker-embedding slot (shifted +1 vs no-lang)
            codec_mask[i, o+9+tlen-1:o+9+tlen-1+clen] = True
            attn[i, :o+9+tlen+clen] = 1

        return {"input_ids": input_ids, "attention_mask": attn,
                "text_embedding_mask": tmask.unsqueeze(-1), "codec_embedding_mask": cmask.unsqueeze(-1),
                "codec_0_labels": labels, "codec_ids": codec_ids, "codec_mask": codec_mask,
                "spk_pos": torch.tensor([b["instruct_ids"].shape[1] + 7 for b in batch], dtype=torch.long)}


def _self_test():
    """Build a 1-item batch (no model) and assert the codec prefix == inference's WITH-language prefix."""
    from types import SimpleNamespace
    tk = SimpleNamespace(codec_think_id=2154, codec_nothink_id=2155, codec_think_bos_id=2156,
                         codec_think_eos_id=2157, codec_pad_id=2148, codec_bos_id=2149, codec_eos_token_id=2150)
    cfg = SimpleNamespace(tts_pad_token_id=9001, tts_bos_token_id=9002, tts_eos_token_id=9003, talker_config=tk)
    ds = TTSDataset([], processor=None, config=cfg)
    # fake item: no instruct (p=0), tiny text (5 toks), tiny codes (3 frames x16), Italian
    item = {"instruct_ids": torch.zeros((1, 0), dtype=torch.long),
            "text_ids": torch.arange(1, 6).view(1, 5),
            "audio_codes": torch.arange(100, 100 + 3 * 16).view(3, 16),
            "lang_tok": LANG2TOK["Italian"]}
    out = ds.collate_fn([item])
    codec = out["input_ids"][0, :, 1].tolist()
    spk_pos = out["spk_pos"][0].item()
    # codec positions 3..8 should be the language prefix (o=0): think,think_bos,LANG,think_eos,spk(0),pad
    got = codec[3:9]
    exp = [2154, 2156, 2070, 2157, 0, 2148]
    print("codec prefix [3:9] :", got)
    print("expected           :", exp, "  (THINK,THINK_BOS,Italian=2070,THINK_EOS,spk,PAD)")
    print("speaker slot spk_pos:", spk_pos, "(codec[spk_pos] =", codec[spk_pos], "-> masked to embedding)")
    assert got == exp, "PREFIX MISMATCH"
    assert spk_pos == 7 and codec[7] == 0, "speaker slot wrong"
    # codec_bos must precede the codes; codes start right after
    assert tk.codec_bos_id in codec, "no codec_bos"
    print("inference (qwen_tts.c) WITH-lang: [THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]")
    print("SELF-TEST PASS — training prefix matches inference (language token in the right slot).")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("use --self-test (this module is imported by the dense FT script)")
