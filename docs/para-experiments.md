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

### 😄 LAUGH — ✅ solved inline in **English**, ❌ still not in Italian (plain onomatopoeia)
**English (ryan, `--emotion joy`, no event-instruct):**
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `hahaha` | 42 | ✅ **WIN (EN)** | ride; solo un filo metallico a fine (over di poco) — la risata inline in inglese esce |
| `hahaha` + laugh-instruct | 42 | ❌ KO | l'instruct-laugh esplicito lo fa **svariare metallico** → NON aggiungere event-instruct al laugh |

**Italian (galatea clone, `--emotion joy`) — plain onomatopoeia (KO) AND laugh-instruct (KO):**
| onomatopoeia | seed | verdict | ear note |
|---|---|---|---|
| `hahaha` | 7/42/2024 | ❌ KO | svaria male, non ride |
| `哈哈` (CN) | 7 | ❌ KO (laugh) | `ah ah!` 2×, non ride |
| `ehehe` | 42/7 | ❌ KO | metallico |
| `ehehe` | 2024 | 🟡 partial | regge ma finisce metallico, sospiro medio `ehhhh` |

**⚠️ KEY (2026-07-01):** in **Italian the onomatopoeia SIGHS/derails, never laughs** ("EN laughs, IT sighs",
plan §8.10/L760). The explicit **laugh-instruct HURTS** (metallic) — for laugh use onomatopoeia + `--emotion`
ONLY, no event-instruct. Historically galatea-IT laughed **only via the `laugh−cry` steering vector**
(split-span = the rejected "splice"). So: **`[laugh]` ships INLINE for English (`hahaha`+joy); inline laugh
in Italian/clone stays UNSOLVED.** NEXT (don't repeat the KO sweeps above): fix the mild metallic tail on the
EN win (try `-T 1.0`, shorter `haha`, or trim the tail); for IT, explore other triggers only if a new idea appears.

### 🎭 SERENDIPITOUS NEW-TAG candidates (galatea IT, from the laugh sweep — keep for future tags)
These did NOT laugh but produced a **clean, distinct OTHER event in-voice** — promote to their own `[tag]` later:
| sound | trigger | seed | note |
|---|---|---|---|
| **scoff / sneer** (sbeffeggio) | `哈哈` (CN) + laugh-instruct | 42 | `AHH!` sospirato = risata breve sprezzante/di scherno — a real scornful scoff |
| **pant / aroused** (ansimo) | `哈哈` (CN) + laugh-instruct | 2024 | `ah ah ah` poi ansima — a panting/aroused vocalization |

---

## Status legend
✅ WIN (promote to the `[tag]` map) · 🟡 interesting/partial (keep, needs a pick) · ❌ KO (do not re-run) · ↪ produced a different event.

_Extend this table with every new para experiment. When a WIN is stable, wire it into the `[tag]`→inline
mapping in main.c and note the commit here._
