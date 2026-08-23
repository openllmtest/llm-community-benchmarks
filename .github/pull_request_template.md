## What I measured

- Benchmark type: (eg inference-speed) 
- GPU type: (eg. RTX 5090)
- VRAM tier: (T1 low-end / T2 med-end / T3 high-end / T4 time-traveler)
- Model + KV quantization e.g. `qwen3.8-27b UD-Q4_K_XL / KV Q8_0`
- Hardware in one line: CPU / GPU (VRAM) / RAM / OS

## Checklist

- [ ] I followed the exact commands/workloads from `PROTOCOL.md` and my model's baseline tier (if its README defines one); protocol version recorded in the result.
- [ ] Required settings recorded with values I actually ran: `n_gpu_layers`, `context_length`, `kv_cache_quant`, `mtp_enabled`; to rank, also my tier's pinned extras exactly (e.g. `spec_draft_n_max`, `flash_attn` for qwen3.8-27b).
- [ ] My run reproduces the pinned recipe of my VRAM tier in my model's README (file, context, KV cache, offload, MTP flags); otherwise it will be listed under *unranked*.
- [ ] Every required field is filled with true values, especially hardware GPU + VRAM and the model file/quantization.
- [ ] The verbatim tool output is included as `<result_id>.raw.txt`.
- [ ] This PR **only adds** files under `results/`; it does not modify or delete any existing result (corrections go in a new file with `"supersedes"`).
- [ ] I did not touch `LEADERBOARD.md` or the generated Leaderboard section of `README.md` by hand.
- [ ] `python scripts/validate_results.py` prints `OK` on my branch.
