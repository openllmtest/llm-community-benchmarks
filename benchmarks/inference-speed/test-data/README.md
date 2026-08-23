# Pinned prompt presets (large-context workloads)

The measurement workload of the inference speed protocol. Each preset is a single chat message containing **code snippets with context plus a numbered task list at the end** (summarize each snippet, write unit tests, mark completion, self-evaluate). The model's answer is *not* scored: these exist so every contributor measures decode speed under realistic large-context load instead of trivial small prompts.

| File | Approx. size | Used by tier | Pinned ctx for that tier |
|---|---|---|---|
| `Prompt_35_000_tkn.txt` | ~35k tokens | low-end | 64000 |
| `Prompt_75_000_tkn.txt` | ~75k tokens | med-end | 128000 |
| `Prompt_120_000_tkn.txt` | ~120k tokens | high-end + time-traveler | 240000 / 262144 (preset ≈ half of both) |

Each preset is sized at roughly **half** of its tier's pinned context, so the KV cache / attention cost is part of what gets measured. The reply budget (`max_tokens`) truncates the task by design; a partial answer is expected.

**Not pinned:** `Prompt_250_000_tkn.txt` (~250k tokens) also lives in this folder but belongs to no tier; it's kept for experiments, e.g. running models beyond their native context range with context-extension methods (YaRN & co). It is not a protocol pin; runs using it are valid contributions but listed *unranked*.

Rules:

- These files are **protocol pins**: do not edit them, reflow them or rename them without a protocol issue + version bump (see [PROTOCOL.md](../PROTOCOL.md)).
- A new context size = add `Prompt_<tokens>_tkn.txt` here, extend the tier table in your model's README and `config/benchmark_types.json`, through a PR.
- Naming convention: `Prompt_<approx_token_count>_tkn.txt`.

Provided by the repo owner as the starting workload set for this benchmark type.
