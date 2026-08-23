# LLM Community Benchmarks

Community-collected benchmark results for **local** LLM inference. Everyone runs the same pinned protocols on their own hardware, submits one small JSON file per run via pull request, and the leaderboard below is assembled automatically from everything merged in this repo.

## Leaderboard

<!-- leaderboard:begin -->
### Inference speed

Single-stream decode throughput on local hardware (llama-server + the model group's pinned probe). One pinned recipe per VRAM tier; rows are ranked within a tier, everything else is listed unranked. Protocol: benchmarks/inference-speed/PROTOCOL.md

_No results yet. Be the first to submit one!_

<!-- leaderboard:end -->

Every row above is a single verified run; the full generated copy with superseded history lives in [LEADERBOARD.md](LEADERBOARD.md). Ready to add your own hardware? The quick start below covers the whole flow, and [CONTRIBUTING.md](CONTRIBUTING.md) has every detail.

## Prerequisites

- **OS:** Windows 10/11, macOS (Intel or Apple Silicon) or Linux (x64 or arm64).
- **Python ≥ 3.9** on your PATH (`python --version` should print it; `setup.py` checks and tells you if yours is missing or old). Standard library only, so there is nothing to `pip install`.
- **GPU runs:** an NVIDIA GPU with a current driver. The prebuilt llama.cpp assets carry the CUDA runtime, so no local CUDA toolkit is needed. Ranked tiers start at 16 GB of VRAM; smaller cards still run fine but their results are listed *unranked*. No NVIDIA GPU? A CPU-only build works too, but expect very slow runs.
- **Disk space:** ~2 GB for the llama.cpp tools folder plus one model file per tier you plan to run (~9-30 GB each, stored in the gitignored `tools/models/`); budget ~45 GB free if you want more than one tier on disk at once.
- **Internet** access to github.com and huggingface.co for the one-time downloads (partial downloads resume where they left off).

## Quick start (automated path)

Two scripts do most of the work (Python standard library only, nothing to install):

```sh
python scripts/setup.py            # once: checks Python, fetches a prebuilt llama.cpp into tools/ (CUDA/CPU auto)
python scripts/run_benchmark.py    # interactive: pick your VRAM tier -> runs the pinned recipe + probe,
                                   # downloads the model file if missing, writes <result_id>.json + .raw.txt
                                   # and validates them locally
```

**Step 1: `setup.py` (once per machine).** Checks your Python version, auto-detects GPU/VRAM (`nvidia-smi`, snapped to the marketed class) and downloads the newest prebuilt llama.cpp into gitignored `tools/llama.cpp/` (CUDA build when an NVIDIA GPU is detected, on Windows it also grabs the matching runtime pack; CPU build otherwise). Then it reports every pinned model file for you: already on disk, missing with its size + direct Hugging Face URL, or *stale* (the pin 404s → non-zero exit, fix via issue + PR), plus the preset sizes under `test-data/`.

**Step 2: `run_benchmark.py` (once per result).** Interactive, five steps:

1. pick model group + VRAM tier; the pinned recipe comes straight from `config/benchmark_types.json`, so nothing can drift from the protocol;
2. ask only your GPU type + VRAM (the technical minimum that decides your tier; auto-detected, both editable);
3. show your tier's exact server command + preset prompt, downloading your pinned GGUF into `tools/models/` if it is missing;
4. start `llama-server` with the pinned recipe (or reuse one you started yourself with `--attach`) and run the model group's probe: an untimed warmup, then 3 measured runs of the tier's large-context preset (≤400 reply tokens each); at bigger tiers the first token can take minutes;
5. ask only what a machine cannot know (GitHub username, date, optional comment), write `results/<your-tier>/<result_id>.json` + `<result_id>.raw.txt` into your tier's folder (`results/unranked/` when the run does not match a pinned tier), and validate them locally with `validate_results.py`.

Useful options: `--port <n>` (default 8080), `--model <path>` to use your own GGUF copy.

**Step 3: open the PR.** The two files under `results/` are everything CI needs ([CONTRIBUTING.md](CONTRIBUTING.md) has the full checklist). Nothing is ever committed or sent by the scripts themselves; the manual procedure below is exactly what step 2 drives, so a hand-made result and a scripted one are indistinguishable.

### Installing llama.cpp by hand (fallback)

`setup.py` fetches the newest *build-tag* release (`bNNN`) from the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases). If you'd rather do it yourself, or the automatic download fails: put a working `llama-server[.exe]` on **PATH** or under `tools/llama.cpp/` (both scripts look there) and re-run `setup.py`; it reports "already available", verifies everything else, and your `run_benchmark.py` run works unchanged.

Note the current Windows + NVIDIA packaging: each build ships as **two** archives that belong in the same folder (the build itself, `llama-bNNN-bin-win-cuda-<ver>-x64.zip`, plus a CUDA-runtime pack with the matching version, `cudart-llama-bin-win-cuda-<ver>-x64.zip`), so you don't need a local CUDA toolkit. Other platforms are a single archive for your platform/arch (e.g. `bin-macos-arm64`, `bin-ubuntu-x64`). Model files are not part of that install; `run_benchmark.py` fetches them for you, or you grab your tier's pinned GGUF from Hugging Face when running fully manually (below).

## Running it manually

Everything below is what `run_benchmark.py` does under the hood; a hand-made result is indistinguishable from a scripted one. Per-tier values always come from [the pinned recipe table](benchmarks/inference-speed/qwen3.8-27b/README.md), never improvised per run.

1. **Get `llama-server`.** Let `setup.py` fetch it, or install by hand (above) so a working binary is on your PATH or in `tools/llama.cpp/`.
2. **Download your tier's model file.** The pinned table links each exact GGUF straight from Hugging Face; run the server *from that folder* (the commands reference the file by bare name, or pass a full path to `-m`).
3. **Start the server** with *your tier's exact block* from [Exact commands](benchmarks/inference-speed/qwen3.8-27b/README.md#exact-commands): model, context, KV cache type and MTP flags are all pinned per tier; keep it running (no other GPU load). Wait for `server is listening` in the log.
4. **Run the probe** in a second terminal:

   ```sh
   python benchmarks/inference-speed/qwen3.8-27b/probe.py --preset <your-tier's-preset>
   # e.g. --preset Prompt_35_000_tkn.txt for low-end; bare names resolve against test-data/ automatically
   ```

   It does one untimed warmup, then 3 measured runs of the tier's pinned large-context prompt (≤400 reply tokens each). At bigger tiers the first token can take several minutes to arrive, so do not kill the probe early. Take the **OVERALL medians** printed at the end: those are `metrics.token_generation_tps` and `metrics.ttft_s`.
5. **Write your result file.** Copy [an example](benchmarks/inference-speed/examples/) into the tier subfolder for your run, e.g. `benchmarks/inference-speed/qwen3.8-27b/results/high-end/<result_id>.json` (or `results/unranked/` when the run does not match a pinned tier). The id is lowercase letters, digits, dots and hyphens and **must end with a 6-character random suffix** so ids stay unique without coordination (see the pattern in [`schema/result.schema.json`](schema/result.schema.json)); the filename must equal the id. Fill in *every* field from your actual run (hardware as-is, settings exactly as pinned, metrics from step 4); [CONTRIBUTING.md](CONTRIBUTING.md) explains each field.
6. **Save the raw log** next to it as `<result_id>.raw.txt`: the verbatim probe output plus the server's startup lines.
7. **Validate locally:** `python scripts/validate_results.py` must print `OK`.
8. **Open a PR** with both files; CI re-validates and previews your leaderboard row (ranked, or unranked with the reason shown).

## Layout

```
benchmarks/
  <benchmark-type>/            # e.g. inference-speed, context-retrieval, ...
    PROTOCOL.md                # the fixed test conditions for this type (tool, workloads, settings)
    examples/                  # a complete example result + raw output to copy from
    <model-family>/            # e.g. qwen3.8-27b; base model id, quantization is NOT in the name
      README.md                # optional: pinned test recipe per VRAM tier for this model group (see qwen3.8-27b/)
      results/                 # append-only, one JSON per run per contributor, never edited after merge
        low-end/  med-end/ high-end/ time-traveler/    # one folder per pinned VRAM tier (for inference-speed)
        unranked/              # valid runs that do not match a pinned tier recipe
config/
  benchmark_types.json         # per type: required metrics/settings, VRAM tiers, pinned per-model recipes, scoring
schema/
  result.schema.json           # machine-readable rules for a result file
scripts/
  setup.py                     # one-time: fetches llama.cpp into tools/, then checks the rest with exact URLs (Quick start)
  run_benchmark.py             # interactive runner: tier -> pinned recipe -> result files (Quick start)
  validate_results.py          # CI + local validation (stdlib-only Python)
  generate_leaderboard.py      # regenerates LEADERBOARD.md from all results
.github/workflows/ci.yml       # validates PRs, previews the leaderboard, publishes on merge
LEADERBOARD.md                 # regenerated on merge; the Leaderboard section of this README updates with it
```

## The rules (short version)

1. **One file = one run.** Never average, never "correct"; a second run is a new file.
2. **Results are immutable.** CI rejects any pull request that modifies or deletes an existing file under `results/`. Made a mistake? Add a new result with a `"supersedes": "<old-id>"` field; the old one stays in history and drops out of the ranking.
3. **Follow the protocol.** Each benchmark type pins its tool, workloads and settings in `PROTOCOL.md`. The version you followed goes into your result file, so future protocol changes will not break comparability with old runs.
4. **Show your raw output.** Every result should carry the verbatim tool log next to it (`<result_id>.raw.txt`) so anyone can audit your numbers.

## How automation works

- **On pull request** (anything under `benchmarks/`): CI validates schema, required metrics *and* settings, folder/name consistency of added files, enforces the append-only rule, and posts a comment showing the leaderboard with your run included, so you see where it ranks before merging.
- **On merge to main:** `LEADERBOARD.md` (and the leaderboard section of this README) are regenerated and committed automatically, only if they changed.

## Contributing

Shortest path: open [CONTRIBUTING.md](CONTRIBUTING.md), follow the 5 steps, validate locally with

```sh
python scripts/validate_results.py
```

and open a PR. No dependencies to install; everything is Python standard library.

Current benchmark types: **inference speed** (protocol v1.0: `llama-server` + pinned probe, one recipe per VRAM tier). New types start as an issue first; adding one means a new `benchmarks/<type>/` folder with its protocol, plus an entry in `config/benchmark_types.json`.

## Attribution

The code snippets embedded in the test prompt presets under [`benchmarks/inference-speed/test-data/`](benchmarks/inference-speed/test-data/) are taken from the [LongBench v2](https://arxiv.org/html/2412.15204v1) dataset (Bai et al., 2024).

This code was generatedb by AI -> Qwen3.8 27b Q4_K_XL UD
