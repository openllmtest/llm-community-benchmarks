#!/usr/bin/env python3
"""Fixed decode-speed probe for llama-server (inference speed protocol v1.0).

The workload is a REAL large-context task, not a small chat question: your
tier's pinned preset file from test-data/ (code snippets + tasks at the end,
sized at roughly half of your tier's context) is sent as ONE chat message.
Decode speed in real use depends on how much context is loaded; attention
over that KV cache is exactly what gets measured.

One measurement pass:
  1. an untimed warmup request settles kernel loading and GPU clocks;
  2. RUNS_PER_PROMPT measured runs each send the full preset file, streamed,
     with at most MAX_TOKENS reply tokens; the task is deliberately
     truncated by that budget because we measure speed under realistic load, not
     completion quality, so a partial answer is expected and fine;
  3. per run we record time-to-first-token (≈ how long the preset takes to
     process at that context) and the decode rate in tok/s over the window
     between the first and the last reply token;
  4. each run prints a short answer preview so your raw output file shows the
     server really generated something plausible for the task;
  5. the OVERALL line prints medians; record "median decode" as
     metrics.token_generation_tps and "median TTFT" as metrics.ttft_s.

All constants below are PINS of protocol v1.0 (see ../PROTOCOL.md): do not
edit them locally; they change only through a protocol version bump. The
approach was inspired by public qwen3.8-27b MTP runs in sudoingX/qwen38-mtp;
this implementation is original to this repo, and the preset files are pinned
workloads of this benchmark type (see test-data/README.md).

Usage:
    python3 probe.py --preset Prompt_120_000_tkn.txt [url]
Bare preset names resolve against benchmarks/inference-speed/test-data/, so
this works from any directory. url defaults to http://127.0.0.1:8080.
Expect long first-token waits at the bigger tiers (that wait is exactly what
TTFT measures); do not kill the probe early.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "test-data")

# --- protocol v1.0 pins -----------------------------------------------------
WARMUP_MAX_TOKENS = 40   # untimed warmup request before measuring starts
RUNS_PER_PROMPT = 3      # measured runs over the preset prompt
MAX_TOKENS = 400         # reply-token budget per run (task intentionally truncated)
TIMEOUT_S = 14400        # per-read socket timeout: a full-prompt prefill at the bigger tiers
                         # regularly exceeds one hour (b10582 measured ~475 s for only ~15k of the tokens)


def resolve_preset(arg):
    """A bare file name resolves against test-data/; anything else is used as-is."""
    if not (os.path.isabs(arg) or os.sep in arg or "/" in arg):
        candidate = os.path.join(TEST_DATA_DIR, arg)
        if os.path.isfile(candidate):
            return candidate
    return arg


def stream_chat(user_message, max_tokens, url):
    """POST one streaming chat completion.

    Returns (token_count, ttft_s, decode_tps, answer_text); the token count is
    the number of streamed content deltas and the decode window excludes TTFT."""
    payload = {
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "stream": True,
        # pinned off for this model family: thinking tokens would inflate the count
        "chat_template_kwargs": {"enable_thinking": False},
        # belt-and-braces against prompt-cache reuse: the real enforcement is the
        # server-side --no-cache-prompt flag (part of every pinned command), which
        # this build ignores per request; keep the field in case a newer honours it.
        # Without no-reuse, runs 2-3 hit the warm prompt cache and their TTFT
        # collapses to near zero (measured on b10582: ~5 s fresh prefill vs
        # ~0.2 s cached for a ~15k-token prompt)
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    counted = 0
    first_ts = last_ts = None
    answer_parts = []
    # TTFT spans the full request: t0 is taken BEFORE urlopen so server prefill
    # over the large preset is included (that wait IS the metric, per PROTOCOL.md)
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[len("data: "):])["choices"][0].get("delta") or {}
            content = delta.get("content") or ""
            if not content and not delta.get("reasoning_content"):
                continue
            now = time.perf_counter()
            first_ts = first_ts if first_ts is not None else now
            last_ts = now
            counted += 1
            answer_parts.append(content)
    ttft = (first_ts - t0) if (t0 is not None and first_ts is not None) else 0.0
    span = (last_ts - first_ts) if (counted and first_ts is not None) else 0.0
    speed = (counted / span) if span > 0 else 0.0
    return counted, ttft, speed, "".join(answer_parts).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", required=True,
                    help="your tier's pinned preset file name from test-data/ (or a path)")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8080",
                    help="llama-server base url (default http://127.0.0.1:8080)")
    args = ap.parse_args()

    preset_path = resolve_preset(args.preset)
    if not os.path.isfile(preset_path):
        names = os.listdir(TEST_DATA_DIR) if os.path.isdir(TEST_DATA_DIR) else []
        available = sorted(n for n in names if n.startswith("Prompt_") and n.endswith(".txt"))
        print("error: preset file not found: %s\navailable presets:\n  %s"
              % (preset_path, "\n  ".join(available) or "(none. Check the test-data/ folder.)"),
              file=sys.stderr)
        sys.exit(2)

    with open(preset_path, "r", encoding="utf-8-sig") as fh:
        user_message = fh.read().strip()

    print("preset: %s (%d chars)" % (os.path.basename(preset_path), len(user_message)))
    print()

    # untimed warmup before the clock starts
    stream_chat("warm up", WARMUP_MAX_TOKENS, args.url)

    ttfts, speeds = [], []
    for i in range(RUNS_PER_PROMPT):
        counted, ttft, speed, answer = stream_chat(user_message, MAX_TOKENS, args.url)
        ttfts.append(ttft)
        speeds.append(speed)
        preview = " ".join(answer.split())
        print("run %d/%d | tokens %d | TTFT %.1f s | decode %.1f tok/s"
              % (i + 1, RUNS_PER_PROMPT, counted, ttft, speed))
        if len(preview) > 240:
            preview = preview[:240].rstrip() + " ..."
        print("answer preview: %s" % (preview or "(empty answer)"))
        print()

    print(
        "OVERALL (%d measured runs): median decode %.1f tok/s · mean %.1f | median TTFT %.1f s"
        % (RUNS_PER_PROMPT, st.median(speeds), st.mean(speeds), st.median(ttfts))
    )
    print("Record the MEDIAN DECODE as metrics.token_generation_tps "
          "and the MEDIAN TTFT as metrics.ttft_s.")


if __name__ == "__main__":
    main()
