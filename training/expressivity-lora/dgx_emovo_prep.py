#!/usr/bin/env python3
# EMOVO -> train_raw.jsonl for the L16-26 expressivity fine-tune (voice-agnostic, instruct-conditioned).
# Deterministic transcript from filename (mapping VERIFIED from "LA STRUTTURA DI EMOVO.pdf" p.8).
# Resamples 48kHz stereo -> 24kHz mono (codec requirement). Maps emotion -> English instruct.
# Run on the DGX (needs librosa + soundfile). Output: ~/qwen-ft/emovo/train_raw.jsonl
import os, json, glob
from collections import Counter
import librosa, soundfile as sf

EMOVO     = os.path.expanduser("~/qwen-ft/emovo/raw/EMOVO")
OUT_AUDIO = os.path.expanduser("~/qwen-ft/emovo/wav24k")
OUT_JSONL = os.path.expanduser("~/qwen-ft/emovo/train_raw.jsonl")
os.makedirs(OUT_AUDIO, exist_ok=True)

# sentence-code -> transcript (verified, EMOVO struttura p.8)
SENT = {
  "b1":"Gli operai si alzano presto.",
  "b2":"I vigili sono muniti di pistola.",
  "b3":"La cascata fa molto rumore.",
  "l1":"L'autunno prossimo Tony partirà per la Spagna nella prima metà di ottobre.",
  "l2":"Ora prendo la felpa di là ed esco per fare una passeggiata.",
  "l3":"Un attimo dopo s'è incamminato ed è inciampato.",
  "l4":"Vorrei il numero telefonico del Signor Piatti.",
  "n1":"La casa forte vuole col pane.",
  "n2":"La forza trova il passo e l'aglio rosso.",
  "n3":"Il gatto sta scorrendo nella pera.",
  "n4":"Insalata pastasciutta coscia d'agnello limoncello.",
  "n5":"Uno quarantatré dieci mille cinquantasette venti.",
  "d1":"Sabato sera cosa farà?",
  "d2":"Porti con te quella cosa?",
}
# emotion code -> (label, English instruct). neutral = empty instruct (anchors the no-instruct case).
EMO = {
  "neu":("neutral",""),
  "gio":("joy",     "Speak happily, bright and warm, smiling through the words."),
  "rab":("anger",   "Speak with hot, furious anger, sharp and forceful."),
  "tri":("sadness", "Speak with a sad, sorrowful, downcast tone, voice low and heavy."),
  "pau":("fear",    "Speak with fear, tense and trembling, your voice wary."),
  "sor":("surprise","Speak with surprise, startled and taken aback, held through the whole sentence."),
  "dis":("disgust", "Speak with physical disgust, repulsed and recoiling."),
}

rows, skipped = [], 0
for wav in sorted(glob.glob(f"{EMOVO}/*/*.wav")):
    base = os.path.basename(wav)[:-4]
    parts = base.split("-")
    if len(parts) != 3:
        skipped += 1; continue
    emo, actor, sent = parts
    if emo not in EMO or sent not in SENT:
        skipped += 1; continue
    label, instruct = EMO[emo]
    out = f"{OUT_AUDIO}/{base}.wav"
    if not os.path.exists(out):
        y, _ = librosa.load(wav, sr=24000, mono=True)   # 48k stereo -> 24k mono
        sf.write(out, y, 24000, subtype="PCM_16")
    rows.append({"audio": out, "text": SENT[sent], "ref_audio": out,
                 "instruct": instruct, "emotion": label, "actor": actor})

with open(OUT_JSONL, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(rows)} rows (skipped {skipped}) -> {OUT_JSONL}")
print("emotions:", dict(Counter(r['emotion'] for r in rows)))
print("actors:  ", dict(Counter(r['actor']   for r in rows)))
print("sample:  ", json.dumps(rows[0], ensure_ascii=False))
