# Contributing a benchmark result

You need: Python 3.9+ (nothing to install; stdlib only), the tool named in the protocol, and an account on GitHub.

## Steps

1. **Read the protocol** for your benchmark type (e.g. [benchmarks/inference-speed/PROTOCOL.md](benchmarks/inference-speed/PROTOCOL.md)) **and the README of your model folder if it has one** (e.g. [qwen3.8-27b/README.md](benchmarks/inference-speed/qwen3.8-27b/README.md) defines baseline settings per VRAM tier). Deviations from a tier's row don't count for that protocol version unless recorded and agreed in an issue.
2. **Run the test** on your hardware, following the protocol exactly. Keep the full terminal output; you'll submit it verbatim. (Shortcut: [`scripts/run_benchmark.py`](scripts/run_benchmark.py) automates this step and creates both files from step 3 for you; run [`scripts/setup.py`](scripts/setup.py) once beforehand.)
3. **Create your result file.** Copy the example from the type's `examples/` folder into the subfolder that matches your run:

   ```
   benchmarks/<type>/<model-family>/results/<tier-or-unranked>/<result_id>.json
   ```

   For tiered types like inference-speed, put it in your VRAM tier's folder (`low-end`, `med-end`, `high-end` or `time-traveler`). If the run does not match a pinned tier recipe (different settings, VRAM between pins, CPU-only), use `unranked/`. Create folders as needed. `<result_id>` is lowercase letters, digits and dashes (dots allowed inside segments) and **must end with a 6-character random suffix** so you never have to check for collisions by hand: generate one with `python -c "import secrets; print(secrets.token_hex(3))"`, e.g. `2026-08-21-alice-qwen3.8-27b-a1b2c3`. The file name must equal the `result_id` inside it; putting a file in the wrong folder fails validation. Next to it, put the unedited tool output in `<result_id>.raw.txt`.
4. **Validate locally:**

   ```sh
   python scripts/validate_results.py
   ```

   It checks schema rules, required metrics *and* required settings, folder/name consistency (including that the tier subfolder under `results/` matches your run), and that your raw-output file exists. Fix everything it reports until it prints `OK`.
5. **Open a pull request.** CI re-validates your file against the append-only rule (you may only *add* files under `results/`, never change or delete existing ones) and posts a comment with the leaderboard preview including your run. Once merged, `LEADERBOARD.md` and the generated leaderboard section of the main [README](README.md) update themselves automatically on main. Leaderboard tables are **upsert-style**: submitting another result for the same tier (or the same condition in the unranked table) replaces your previous row; older files stay in history and still count toward points.

## Result fields

Full reference: [schema/result.schema.json](schema/result.schema.json). Required:

| field | meaning |
|---|---|
| `result_id` | unique id = file name without `.json`; recommended `<date>-<contributor>-<model>` |
| `benchmark_type` | must equal the folder under `benchmarks/`, e.g. `inference-speed` |
| `model_family` | base model id, must equal its folder name; **no quantization** (e.g. `qwen3.8-27b`) |
| `contributor` | your GitHub username |
| `date` | when the run happened, `YYYY-MM-DD` |
| `tool` | `{name, version}`: exact tool and build/tag you used |
| `protocol_version` | version of the PROTOCOL.md you followed (e.g. `"1.0"`) |
| `hardware` | Technical minimum: `gpu` (exact type incl. VRAM, e.g. "NVIDIA RTX 3090 (24 GB)") and `vram_gb` at the card's marketed size; those place your run in a tier. `cpu`, `ram_gb`, `os` are optional extras, never used for ranking. |
| `metrics` | the numbers named by the protocol (required keys per type: [config/benchmark_types.json](config/benchmark_types.json)) |

Required **settings** per benchmark type are also defined in `config/benchmark_types.json`: inference-speed v1.0 requires `n_gpu_layers`, `context_length`, `kv_cache_quant`, `mtp_enabled`. Record the values you actually ran, not what you intended to run.

**Tier conformance decides ranking:** a result whose `vram_gb` snaps to one of the tiers' pinned VRAM values (low-end = 16 / med-end = 24 / high-end = 32 GB; above 32 ranks as time-traveler, i.e. workstations, unified memory or multi-GPU rigs with that much combined VRAM) and that reproduces that tier's pinned recipe exactly (file, quantization, context, KV cache, offload, MTP flags; see [the qwen3.8-27b README](benchmarks/inference-speed/qwen3.8-27b/README.md)) ranks on the leaderboard and lives in `results/<that-tier>/`. Anything else still validates but is listed under *unranked* with the reason shown, and belongs in `results/unranked/`. Contributors earn **1 pt per live result** plus a one-time **+2** for being first to submit on a given GPU within a tier (VRAM pin); see the Top contributors table in [LEADERBOARD.md](LEADERBOARD.md).

Recommended / optional: `variant` (`quantization`, `repo_id` where the GGUF came from, `file` = exact model file), `comment` (short note, shown as its own column in the leaderboard; max 200 chars), `raw_output_file`, `notes` (longer caveats / supersedes explanation).

## I made a mistake in my submitted result

Do **not** edit the file; CI blocks it. Submit a new correct run (in the right tier folder) and add:

```json
"supersedes": "<result_id of the wrong one>",
"notes": "corrects hardware field (was typed as 32 GB, is 64 GB)"
```

The old entry remains in the repo for provenance but leaves the leaderboard table.

## Adding a new benchmark type or model group

**New model group:** create `benchmarks/<type>/<model>/results/`, and (recommended) a `README.md` defining baseline settings tiers (quantization, KV cache, context length, MTP) per VRAM class so runs stay comparable, like [qwen3.8-27b's](benchmarks/inference-speed/qwen3.8-27b/README.md).

**New benchmark type:** open an issue describing what you want to measure and why; get agreement on a protocol draft. Then add `benchmarks/<type>/` with `PROTOCOL.md`, an `examples/` folder, and create model subfolders as needed, plus an entry in [config/benchmark_types.json](config/benchmark_types.json) (title, required metrics, leaderboard columns). Until this config entry exists, CI rejects results for that type; that's intentional.
