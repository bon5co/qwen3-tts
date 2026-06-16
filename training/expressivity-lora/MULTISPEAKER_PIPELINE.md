# Multi-speaker emotion FT — reproducible pipeline + findings (2026-06-16)

Goal: make emotion **generalize** (across identities, and ideally to cloned voices) by fine-tuning the
dense expressivity band **L16-26** on emotion spoken by **many speakers × many emotions**, instead of
the EMOVO-only (6 Italian actors) set that produced a voice-specific result.

## TL;DR of what we learned (read before re-running)

1. **The mechanism works — in the dominant training language.** A dense L16-26 FT on
   ESD (10 EN speakers) + CREMA-D (91 EN actors) + EMOVO (6 IT actors) made **English** emotion
   clearly better on a preset (ryan): anger = articulate+nervous+angry, sad = slow/heavy. Ear-confirmed.
2. **Untagged language mixing CORRUPTS the minority language.** The set was **97.7% English**
   (24 942 EN vs 588 IT) and the FT builder passed **no language tag** → the shared L16-26 weights
   learned English-dominant emotion and **wrecked Italian** on ryan ("foreigner speaking poor Italian",
   dark timbre, slurring). EMOVO-only worked before precisely because it was a *single* language.
3. **Even upstream Qwen `finetuning/dataset.py` drops the language token** (reads `language`, never uses
   it) → the official recipe is implicitly **single-language FT**. Mixing languages is the deviation.
4. **The fix = condition on language** (this dir's `*_lang.py`): inject the same language codec token the
   C engine uses at inference — `[THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]`
   (`qwen_tts.c`) — so one mixed FT can route emotion per language. Alternative: per-language adapters.
5. **The CLONE problem is SEPARATE.** A cloned `.qvoice` graft still resists emotion (its x-vector is
   out-of-distribution); no amount of multilingual emotion data fixes that. Different lever
   (better speaker embedding / disentanglement). Don't conflate it with the FT-quality work above.
6. **Temp** ≤ 1.3 (T1.5 slurs Italian). Instruct in **English** (model is EN/ZH-centric).

## Files (this dir)

ORIGINALS (pulled from the DGX as a tracked trace — do NOT edit, they are the proven recipe):
- `dgx_sft_expr.py` — dense full-rank FT, trains L16-26 (`--layers`) + text_projection, voice-agnostic.
- `dgx_dataset_expr.py` — its dataset builder (instruct-conditioned, **no** language tag).
- `dgx_sft_expr_lora.py` — the LoRA variant (low-rank, same band).
- `dgx_emovo_prep.py` — EMOVO → train_raw schema (the schema everything else mirrors).
- `prepare_data.py` — codec-encode step (Qwen tokenizer, GPU). Original upstream.
- `prepare_esd.py` — ESD (HF `duanyu027/ESD`) → schema. English speakers 0001-0010.

NEW (this epic):
- `prepare_cremad.py` — CREMA-D (HF parquet mirror) → schema. 91 actors, 12 fixed sentences.
- `concat_manifests.py` — merge + validate manifests; `--langs` stamps a per-file `language`,
  `--repeat` oversamples a minority language.
- `dgx_dataset_expr_lang.py` — **language-tagged** dataset builder (injects the language codec token,
  matching inference). Run `python3 dgx_dataset_expr_lang.py --self-test` to verify the prefix.
- `dgx_sft_expr_lang.py` — fork of `dgx_sft_expr.py` that uses the tagged builder (1-line diff: import).
- `dgx_multi_emotion.sh` — end-to-end orchestrator (download → prep → encode → concat → FT), idempotent
  via `<stage>.DONE` markers, fail-loud `need_file` gates, timestamped `multi_emotion.log`.
- `docker/Dockerfile` + `docker/build_img.sh` — the `qwen-ft:latest` image (ubuntu 24.04 + torch +
  torchaudio + qwen-tts). **Use this image for encode AND train** (the nvcr pytorch image has a broken
  torchaudio → tokenizer crashes).
- `../../tests/emo_score.py` — automatic SER scorer (audeering wav2vec2 arousal/valence) to rank
  expressivity variants without listening to every clip. CPU-ok.

## Reproduce (on the DGX, from `~/qwen-ft`)

```bash
cd ~/qwen-ft/docker && bash build_img.sh          # one-time: qwen-ft:latest
bash dgx_multi_emotion.sh                          # download ESD+CREMA-D, prep, encode, concat, FT (untagged)
# -> out_multi_l1626/checkpoint-final/model.safetensors

# LANGUAGE-TAGGED variant (the fix):
python3 concat_manifests.py --out multi_emotion_tagged/train_with_codes.jsonl \
    --langs Italian,English,English \
    emovo/train_with_codes.jsonl esd/train_with_codes.jsonl cremad/train_with_codes.jsonl
docker run --rm --gpus all --ipc=host -v $HOME/qwen-ft:/root/qwen-ft -v $HOME/qwen-ft:$HOME/qwen-ft \
    qwen-ft:latest bash -c "cd /root/qwen-ft/Qwen3-TTS/finetuning && \
    python3 -u dgx_sft_expr_lang.py --train_jsonl /root/qwen-ft/multi_emotion_tagged/train_with_codes.jsonl \
    --output_model_path /root/qwen-ft/out_multi_l1626_tagged --layers 16-26 --num_epochs 5"
```

Then (locally) extract the `.expr` and A/B:
```bash
mkdir qwen3-tts-1.7b-expr-multi && scp dgx:.../checkpoint-final/model.safetensors qwen3-tts-1.7b-expr-multi/
python3 tests/expr_extract.py qwen3-tts-1.7b qwen3-tts-1.7b-expr-multi presets/expr/<name>.expr --lang Italian
# A/B preset ryan IT vs EN, and (separately) the clone — see tests/emo_score.py
```
