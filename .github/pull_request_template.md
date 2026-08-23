## What I measured

- Benchmark type: `____` (protocol version `__`), plus the VRAM tier you ran for it (low-end / med-end / high-end / time-traveler)
- Model + quantization + file: e.g. `qwen3.8-27b UD-Q4_K_XL`, file `Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF`
- Hardware in one line: CPU / GPU (VRAM) / RAM / OS
- Result file(s) added: e.g. `benchmarks/inference-speed/qwen3.8-27b/results/<tier-or-unranked>/<result_id>.json`

## Checklist

- [ ] I followed the exact commands/workloads from `PROTOCOL.md` and my model's baseline tier (if its README defines one); protocol version recorded in the result.
- [ ] Required settings recorded with values I actually ran: `n_gpu_layers`, `context_length`, `kv_cache_quant`, `mtp_enabled`; to rank, also my tier's pinned extras exactly (e.g. `spec_draft_n_max`, `flash_attn` for qwen3.8-27b).
- [ ] My run reproduces the pinned recipe of my VRAM tier in my model's README (file, context, KV cache, offload, MTP flags); otherwise it will be listed under *unranked* with the reason shown.
- [ ] Every required field is filled with true values, especially hardware GPU + VRAM and the model file/quantization.
- [ ] The verbatim tool output is included as `<result_id>.raw.txt`.
- [ ] This PR **only adds** files under `results/`; it does not modify or delete any existing result (corrections go in a new file with `"supersedes"`).
- [ ] I did not touch `LEADERBOARD.md` or the generated Leaderboard section of `README.md` by hand.
- [ ] `python scripts/validate_results.py` prints `OK` on my branch.
