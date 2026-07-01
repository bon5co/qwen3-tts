# Paralinguistic experiments log — WIN / KO by onomatopoeia × seed × language × emotion × speaker

> **READ THIS BEFORE running any `[tag]` / paralinguistic experiment.** It is the durable, reproducible
> record of which inline onomatopoeia-triggers produce which event (or fail) — per seed, language, emotion
> and speaker. The goal: a `[tag]` maps to a **precise, deterministic** para event that **keeps the voice
> timbre and mixes into the sentence** (INLINE, one generation — never a separate span / "splice"). Do NOT
> re-run a KO combo; extend the table with new experiments and promote the WINs into the `[tag]` mapping.

## Method (the only method we keep)
- **INLINE native-trigger**: the onomatopoeia goes **inside the sentence as plain text**, ONE generation,
  so the event is produced in the active voice (preset or clone) — same timbre, mixes naturally.
- ❌ **NEVER the split-span / steering-span** ("splice"): generating the event as a separate cold-prefill
  span mixes voices (sounds like a different speaker) — rejected by ear 2026-07-01. Dead, do not reuse.
- The emotion comes from `--emotion` (STEER on presets / COMBINE on clones). Instruct is an **optional
  booster** to push a specific event ("here you laugh") — a `[tag]` must map its event **with or without** it.

## Reproducible command template
```bash
./qwen_tts -d qwen3-tts-1.7b <VOICE> -l <LANG> -T 1.1 --seed <SEED> --emotion <EMO> \
    --text "<carrier sentence with the ONOMATOPOEIA inline>" -o out.wav
# preset:  <VOICE> = -s ryan            clone: <VOICE> = --load-voice voices/galatea_graft.qvoice --icl-only
```
Carriers used below:
- joy/laugh: `"Non ci posso credere, <O>, è la notizia più bella della mia vita!"`
- sad/sigh:  `"Ho perso tutto quello che avevo, <O> e adesso non so più cosa fare."`

---

## Findings — galatea (clone), Italian, T1.1, inline (ear 2026-07-01)

### 😮‍💨 SIGH  (`--emotion sad`)  — the vocal family works inline, cross-voice
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `唉` (CN) | **42** | ✅ **WIN — best short** | ottimo `ehh` breve, controllo perfetto |
| `唉` (CN) | **7**  | ✅ **WIN — short/medium** | sospiro breve + `ehh`, buon controllo, poche pause |
| `唉` (CN) | **2024** | ✅ **WIN — long** | sospiro lungo `ahhhhh` e finisce la frase pulito |
| `ahh` | **2024** | ✅ **WIN — "defeated"** | sospira, pause, emozione, esce `ehhhh` sconfitto |
| `ahh` | 42 | 🟡 interesting | `ah!` sospirato, pausa, poi finisce la frase |
| `哈哈` (CN) | 7 | 🟡 KEEP as **sigh medium-long** | NON ride: sospiro lungo, molto bello (è un sigh, non un laugh) |
| `ahh` | 7 | ❌ KO | fa `eh eh eh` 3× (non un sigh) |
| `哈哈` (CN) | 42 | ❌ KO | svaria in tanti sospiri + `eueoeo`, non finisce la frase |
| `哈哈` (CN) | 2024 | ❌ KO | `ahhhh` svaria, rumore di fondo a fine frase, troppo lungo |

→ **Sigh mapping candidates (galatea IT):** short = `唉` s42 · medium = `唉` s7 (or `哈哈` s7) · long = `唉` s2024 / `ahh` s2024.

### 😄 LAUGH  (`--emotion joy`)  — ❌ NOT solved inline in Italian yet
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `hahaha` | 7 / 42 / 2024 | ❌ KO | svaria male (non ride) |
| `哈哈` (CN) | 42 / 2024 | ❌ KO | svaria / troppo lungo / rumore, non ride |
| `哈哈` (CN) | 7 | ↪ became a **sigh** | (logged above as a sigh WIN) |

**⚠️ KEY OPEN PROBLEM:** in **Italian the onomatopoeia SIGHS instead of laughing** ("EN laughs, IT sighs",
plan_emo_v3 §8.10/L760). Historically galatea-IT DID laugh — but **only via the `laugh−cry` steering vector**
(plan L783 "galatea lvc_w8 = TOP, ride"), which is the split-span/"splice" we rejected. So the **inline** laugh
in IT is still unsolved. NEXT experiments (do NOT repeat the KO plain-onomatopoeia sweep above):
1. inline `哈哈`/`ehehe`/`ihih` + an **explicit laugh instruct** ("Burst out laughing with bright joyful
   giggles.") to tip IT into laughing in one generation (the user's "sfrutta meglio l'instruct" lever);
2. confirm **ryan-EN** laughs inline on `hahaha` (English path — expected to work, gives a reference);
3. only if inline can't laugh in IT: revisit whether the vector can be applied **within** the single
   generation (positional, not a separate span) — research, not the split-span.

---

## Status legend
✅ WIN (promote to the `[tag]` map) · 🟡 interesting/partial (keep, needs a pick) · ❌ KO (do not re-run) · ↪ produced a different event.

_Extend this table with every new para experiment. When a WIN is stable, wire it into the `[tag]`→inline
mapping in main.c and note the commit here._
