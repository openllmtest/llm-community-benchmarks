# Qwen3.8-27B (pinned test setup)

This is the first model group in this repo, benchmarked under the **inference speed** type. This family ships an **MTP (multi-token-prediction) module**, so every run here has MTP speculative decoding **on by default** (`--spec-type draft-mtp --spec-draft-n-max 2`); we compare what this model does at its best on each class of consumer GPU, apples-to-apples.

The approach (a fixed streaming chat workload against `llama-server` with MTP on) was inspired by the public qwen3.8-27b runs in [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp); the pinned probe code is original to this repo ([`probe.py`](probe.py)), and the large-context prompt presets are pinned workloads of this benchmark type (one preset per tier; see [`test-data/`](../test-data/)). No paired A/B experiments: one number per run, comparable because everything else is pinned.

The general rules live in [the protocol](../PROTOCOL.md); every run is still one JSON file per result, validated by CI.

## Pinned recipe per VRAM tier

Source of all files: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Record the exact file you used in `variant.file`. Verify current availability on the repo page before downloading.

| Tier | Available GPU VRAM  | Model file (record as quantization) | Ctx window size (`-c`) | KV cache quantization (`--cache-type-k/v`) | Test prompt size |
|---|---|---|---|---|---|
| **low-end** | 16 GB | [`Qwen3.8-27B-UD-Q2_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q2_K_XL.gguf) (`UD-Q2_K_XL`) | 64000 (64k) | `q4_0` / `q4_0` | [`Prompt_35_000_tkn.txt`](../test-data/Prompt_35_000_tkn.txt) (~35k tok = half of ctx) |
| **med-end** | 24 GB | [`Qwen3.8-27B-UD-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf) (`UD-Q4_K_M`) | 128000 (128k) | `q4_0` / `q4_0` | [`Prompt_75_000_tkn.txt`](../test-data/Prompt_75_000_tkn.txt) (~75k tok) |
| **high-end** | 32 GB | [`Qwen3.8-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf) (`UD-Q4_K_XL`) | 240000 | `q8_0` / `q8_0` | [`Prompt_120_000_tkn.txt`](../test-data/Prompt_120_000_tkn.txt) (~120k tok = half of ctx) |
| **time-traveler** | > 32 GB (anything beyond consumer cards: workstations / multi-GPU rigs; record total usable `vram_gb`, e.g. dual RTX 5090s = 64; unified memory such as the NVIDIA DGX Spark) | [`Qwen3.8-27B-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q8_0.gguf) (`Q8_0`) | 262144 (model max) | `f16` / `f16` (no quant) | [`Prompt_120_000_tkn.txt`](../test-data/Prompt_120_000_tkn.txt) (~120k tok ≈ half of ctx) |

Context windows are sized from the tier's model file + KV cache against its pinned VRAM value, so a card of that size runs close to, but never past, its limit: low-end measured at ≈ 15.7 GB in use on the 16 GB card, high-end ≈ 31.2 GB on 32 GB; time-traveler reserves KV for the full 262k window (≈ 45 GB, comfortable from 64 up).

Identical for every tier: full offload `-ngl 999`, flash attention on `-fa 1`, single stream `--parallel 1`, MTP on `--spec-type draft-mtp --spec-draft-n-max 2`, and prompt caching off `--no-cache-prompt` so every timed run does a real full prefill (the protocol explains why).

*The [test-data/](../test-data/) folder also carries `Prompt_250_000_tkn.txt`, which is **not pinned** to any tier; it's there for people experimenting with context extension beyond the model's native 262k trained range (e.g. YaRN). Runs using it are valid contributions but listed *unranked*.*

**The workload is a real large-context task, not a small chat question.** Each tier sends its pinned preset file from [`test-data/`](../test-data/) (code snippets + a numbered task list at the end) as one message, sized at roughly half of that tier's context so attention over the loaded KV cache is part of what gets measured. The answer is not scored: with a 400-token reply budget the task is deliberately left unfinished, because we measure *speed under realistic load* and everyone measures it with the same ruler (presets documented in [test-data/README.md](../test-data/README.md)).

If a pinned file or recipe turns out to be wrong for that hardware class (OOMs, missing on Hugging Face…), fix this table via issue + PR; do not improvise per run. Deviating runs are still accepted, but listed *unranked* with the reason shown.

## Exact commands

```sh
# low-end: UD-Q2_K_XL (~9 GiB file, ≈ 15.7 GB in use) at 64k context
llama-server -m Qwen3.8-27B-UD-Q2_K_XL.gguf \
  -c 64000 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --no-cache-prompt \
  --parallel 1 --host 127.0.0.1 --port 8080

python3 probe.py --preset Prompt_35_000_tkn.txt   # wait for the OVERALL line, then stop the server


# med-end: UD-Q4_K_M (~15 GB) at 128k context
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  -c 128000 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --no-cache-prompt \
  --parallel 1 --host 127.0.0.1 --port 8080

python3 probe.py --preset Prompt_75_000_tkn.txt


# high-end: UD-Q4_K_XL (~16 GB), q8_0 KV cache, at 240k context
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 240000 -ngl 999 -fa 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --no-cache-prompt \
  --parallel 1 --host 127.0.0.1 --port 8080

python3 probe.py --preset Prompt_120_000_tkn.txt


# time-traveler: Q8_0, f16 KV cache (no quant), at model-max 262k context
llama-server -m Qwen3.8-27B-Q8_0.gguf \
  -c 262144 -ngl 999 -fa 1 \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --no-cache-prompt \
  --parallel 1 --host 127.0.0.1 --port 8080

python3 probe.py --preset Prompt_120_000_tkn.txt
```

## What your result records (for a ranked run)

| Information | Field in the result JSON | Value for this model group |
|---|---|---|
| Speed | `metrics.token_generation_tps` | the probe's OVERALL median decode (tok/s): what your tier table ranks by |
| Time to first token (≈ how long the preset takes to process at this context) | `metrics.ttft_s` | the probe's OVERALL median TTFT, in seconds |
| Exact model file + quantization | `variant.file`, `variant.quantization` | your tier's row in the table above |
| GPU type + VRAM (it must snap to your tier's pin, see the table above) | `hardware.gpu`, `hardware.vram_gb` | e.g. `NVIDIA RTX 3090 (24 GB)` / `24` |
| Offload layers | `settings.n_gpu_layers` | `999` |
| Context window size | `settings.context_length` | per tier: 64000 / 128000 / 240000 / 262144 (model max) |
| KV cache quantization | `settings.kv_cache_quant` | per tier: `q4_0` / `q4_0` / `q8_0` / `f16` |
| MTP enabled | `settings.mtp_enabled` | `true` (always, for this model group) |
| Drafts per step | `settings.spec_draft_n_max` | `2` |
| Flash attention | `settings.flash_attn` | `1` |
| Your short comment (shown in the **Comment** column of [LEADERBOARD.md](../../../LEADERBOARD.md)) | `comment` | e.g. *"idle machine, room temp; slightly throttled after 2nd run"* |

## Ran it differently?

Fine, your file still validates and is listed under *unranked* for this model with the reason shown (which pinned value differs). It still earns contribution points. To get ranked: rerun your tier's exact recipe. If you believe the table itself is wrong (OOM on that card class, …), open an issue; the pinned table only changes through a PR to this README and `config/benchmark_types.json`.

## Submitting your run

1. Copy an example from [`examples/`](../examples/) into the tier subfolder that matches your run under `results/` (`unranked/` when it does not match a pinned recipe), and fill in **every** field with values from *your* run; see [CONTRIBUTING.md](../../../CONTRIBUTING.md).
2. Validate locally: `python scripts/validate_results.py` → must print `OK`.
3. Open a PR. CI validates it and posts the leaderboard preview including your row (ranked, or unranked with the reason). Results are append-only; corrections go in as new files with `"supersedes"`.
