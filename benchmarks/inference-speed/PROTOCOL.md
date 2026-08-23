# Protocol: inference speed (v1.0)

Single-stream decode throughput on local hardware, measured against a running `llama-server` with the **pinned recipe** that your model group defines for your VRAM tier. For models that ship an MTP (multi-token-prediction) module (like qwen3.8-27b), the MTP speculative path is part of every pinned recipe: it is on for everyone, so each row already shows what the family does at its best on a given class of card. There are no per-contributor A/B experiments in this repo.

The leaderboard ranks rows **within a VRAM tier** (rows are comparable only against other rows using the same pinned recipe). Runs with different settings are still welcome and listed under *unranked*; they still do not rank.

## Tool

- `llama-server` from llama.cpp + your model group's **pinned probe client** (for `qwen3.8-27b`: [`probe.py`](qwen3.8-27b/probe.py) in its folder) plus your tier's **pinned prompt preset** from [`test-data/`](test-data/) (large-context code + task prompts; see [test-data/README.md](test-data/README.md)). Use the newest stable build and record its exact tag in `tool.version` (e.g. `b6500`). Record both in `tool.name`, e.g. `"name": "llama-server + qwen3.8-27b/probe.py"`.

**Automated path:** [`scripts/setup.py`](../../scripts/setup.py) fetches a prebuilt llama.cpp into the gitignored `tools/` folder, and [`scripts/run_benchmark.py`](../../scripts/run_benchmark.py) drives this exact flow (tier prompt, the pinned server command, the pinned probe), then writes your validated `<result_id>.json` + `.raw.txt` pair into your tier's results subfolder. The manual steps below remain authoritative; a scripted run must be indistinguishable from a manual one.

## Pinned recipes per VRAM tier

VRAM tiers for this type are defined in [`config/benchmark_types.json`](../../config/benchmark_types.json). Four tiers, numbered **T1** to **T4**, each pinned to one **marketed VRAM value**; going up a tier always means better quantization, longer context and less-quantized KV cache. Test conditions per tier (current model group: `qwen3.8-27b`; its [README](qwen3.8-27b/README.md#exact-commands) carries the exact server commands):

**T1 - low-end / 16 GB.** Entry class for this model on consumer cards (e.g. RTX 4080, RTX 4060 Ti with 16 GB): the best quantization that still fits in 16 GB including KV cache headroom.
- Model file: `Qwen3.8-27B-UD-Q2_K_XL.gguf` (quantization `UD-Q2_K_XL`)
- Context size (`-c`): 64000
- KV cache quantization: `q4_0` / `q4_0`
- Test prompt: [`Prompt_35_000_tkn.txt`](test-data/Prompt_35_000_tkn.txt) (~35k tokens, half of the context window)

**T2 - med-end / 24 GB.** Previous top consumer class (e.g. RTX 3090, RTX 4090): a clearly better quantization and double T1's context window.
- Model file: `Qwen3.8-27B-UD-Q4_K_M.gguf` (quantization `UD-Q4_K_M`)
- Context size (`-c`): 128000
- KV cache quantization: `q4_0` / `q4_0`
- Test prompt: [`Prompt_75_000_tkn.txt`](test-data/Prompt_75_000_tkn.txt) (~75k tokens)

**T3 - high-end / 32 GB.** Current top consumer class (e.g. RTX 5090): near-lossless quantization, a very large context window and an almost unquantized KV cache.
- Model file: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (quantization `UD-Q4_K_XL`)
- Context size (`-c`): 240000
- KV cache quantization: `q8_0` / `q8_0`
- Test prompt: [`Prompt_120_000_tkn.txt`](test-data/Prompt_120_000_tkn.txt) (~120k tokens, half of the context window)

**T4 - time-traveler / above 32 GB.** Beyond consumer cards: workstations (e.g. a single 48 GB card), multi-GPU rigs (record your total usable VRAM, e.g. dual RTX 5090 = 64), or unified memory such as the NVIDIA DGX Spark; full precision with no quantization anywhere in the pipeline.
- Model file: `Qwen3.8-27B-Q8_0.gguf` (quantization `Q8_0`)
- Context size (`-c`): 262144 (model maximum)
- KV cache quantization: `f16` / `f16` (no quantization)
- Test prompt: [`Prompt_120_000_tkn.txt`](test-data/Prompt_120_000_tkn.txt) (~120k tokens, about half of the context window)

Record `hardware.vram_gb` as the card's **marketed** size: a "32 GB" card that reports 31.9 counts as 32; validation snaps it. Sizes between pins, or below 16, still validate and are listed *unranked*.

Your model group's `README.md` pins one complete condition per tier: model file + quantization, context length, KV cache quantization, offload depth, flash attention, MTP flags and the preset prompt file; see the table in [`qwen3.8-27b/README.md`](qwen3.8-27b/README.md). **Run your tier's row exactly.** That is what makes rows within a tier apples-to-apples comparable.

- A result is *ranked* only when `hardware.vram_gb` snaps to a tier's pinned VRAM value (**above the highest pin ranks as `time-traveler`**) **and** it reproduces every pinned value (the settings plus `variant.file` / `quantization`).
- Deviating does not invalidate your run: CI still accepts the file, but the leaderboard lists it under *unranked*. If a pinned recipe is wrong for that hardware class (OOMs, file missing on Hugging Face…), fix the table via issue + PR to the model's README and config; do not improvise per run.

## How to measure

1. Start the server with **the exact command** of your tier row (`--parallel 1 --host 127.0.0.1 --port 8080`, no other GPU load running).
2. Wait for "server is listening", then run the probe with **your tier's pinned preset**: `python3 <model-folder>/probe.py --preset Prompt_75_000_tkn.txt`. It does an untimed warmup, then 3 measured runs, each sending the full preset file (a large-context code + task prompt from [`test-data/`](test-data/), sized at roughly half of your pinned context) as one message with at most 400 reply tokens. Each timed run does a real full prefill because every pinned server command carries `--no-cache-prompt`, which turns llama-server's prompt cache off; without it the first request warms the cache and runs 2-3 would get a near-zero TTFT from cached KV (measured on b10582 with the pinned high-end recipe: ~5 s fresh prefill vs ~0.2 s cached for a ~15k-token prompt). A suspiciously small TTFT at the bigger tiers means the flag is missing.
3. Per run the probe prints time-to-first-token, decode tok/s and a short answer preview; take the **OVERALL medians** into `metrics.token_generation_tps` (decode) and `metrics.ttft_s`.

One result file = one such server session + probe pass. The task is intentionally truncated at 400 reply tokens; we measure speed under realistic large-context load, not completion quality, so a partial answer in your raw output is expected. At the bigger tiers the first token can take minutes to arrive (that wait *is* the TTFT you record); do not kill the probe early. The presets and all probe parameters are part of protocol v1.0: no local edits without a version bump.

## Metrics (required)

| field | source | unit |
|---|---|---|
| `token_generation_tps` | OVERALL median decode speed of the pinned probe; **the ranking metric** for this type | tokens/s |
| `ttft_s` | OVERALL median time-to-first-token (≈ how long that tier's preset takes to process at its context size) | seconds |

Extra numeric metrics are allowed but not ranked on; propose them through a protocol issue if you want one to become a leaderboard column.

## Required settings fields (CI-enforced)

For every inference-speed v1.0 result these `settings` keys must be present. CI rejects files without them:

| setting | meaning |
|---|---|
| `n_gpu_layers` | offload depth (`999` = full GPU offload; record partial-offload values honestly) |
| `context_length` | context window size you ran with (the `-c` value) |
| `kv_cache_quant` | KV cache quantization: the `--cache-type-k/v` value, e.g. `q4_0`, `f16` |
| `mtp_enabled` | whether the MTP speculative path was active (`true` for every pinned qwen3.8-27b run) |

Tier recipes pin additional keys on top (for `qwen3.8-27b`: `spec_draft_n_max`, `flash_attn`). CI does not reject a file missing them, but without them the row can never be *ranked*; conformance is checked against [`config/benchmark_types.json`](../../config/benchmark_types.json).

Also recommended for every run: `variant.file` (the exact GGUF), `hardware.vram_gb`, and `settings.prompt_tokens` (nominal token size of your prompt preset in tokens, recorded with the result); a short `comment` appears as its own column in [LEADERBOARD.md](../../LEADERBOARD.md).

## Environment rules

- No other GPU workloads running (close browsers with hardware acceleration if possible).
- The technical minimum for a result is the exact GPU type in `hardware.gpu` plus its VRAM in `hardware.vram_gb` (those two decide which pinned tier it can rank in). Everything else about your machine (`cpu`, `ram_gb`, `os`) is optional and never affects ranking. Record VRAM at the card's **marketed size**: an RTX 5090 is recorded as `32` even when nvidia-smi reports ~31.9 GB (driver reserve), because tier pins refer to that marketed class.
- Note ambient conditions / background load / VRAM before-after in `comment` or `notes` when relevant.
- Report the number exactly as printed by the probe; no manual averaging across sessions. One file = one session; if you want more statistics, do that outside this repo.

## Submitting a result

1. Put your run under `benchmarks/inference-speed/<model_family>/results/<tier-or-unranked>/`, e.g. `qwen3.8-27b/results/high-end/` (or `.../unranked/` when the run does not match a pinned tier recipe; create the folder if missing, model id is lowercase with no quantization in the name).
2. File name must be `<result_id>.json`. Copy [`examples/2026-08-15-alice-qwen3.8-27b-a1b2c3.json`](examples/) as a starting point and fill in **every** field with values from your own run; keep the random 6-character suffix of the id (or generate your own). Follow your model's README for the pinned settings.
3. Attach complete, unedited output next to it as `<result_id>.raw.txt`: the server start lines (command + loaded model) followed by the full probe stdout. This is how other members can audit your numbers.
4. Validate locally before opening the PR:

   ```sh
   python scripts/validate_results.py
   ```

5. Open a pull request; CI re-runs validation and comments with a leaderboard preview showing where you'd rank (or that your run lands in the *unranked* section).

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the field-by-field reference.
