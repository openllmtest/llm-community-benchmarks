#!/usr/bin/env python3
"""Run the inference-speed benchmark end to end and write your result files.

Interactive flow (stdlib only, no install needed):

  [1] pick model group + VRAM tier - the pinned recipe comes straight from
      config/benchmark_types.json, so nothing can drift from the protocol;
  [2] auto-detect GPU type + VRAM via nvidia-smi (the technical minimum that
      decides your tier) and let you edit both values before anything runs;
  [3] show your tier's exact server command + preset prompt; download your
      tier's pinned GGUF from Hugging Face into tools/models/ if it is missing;
  [4] start llama-server with the pinned recipe (or --attach to reuse one you
      started yourself) and run the model group's pinned probe: untimed warmup,
      then 3 measured runs sending the tier preset as one message - identical
      output format to running benchmarks/<type>/<model>/probe.py by hand;
  [5] ask only what a machine cannot know (GitHub username, date, comment),
      write <result_id>.json + <result_id>.raw.txt into your tier's folder under
      results/ (results/unranked/ when the run does not match a pinned tier) and
      validate them locally with scripts/validate_results.py.

Nothing is committed or sent anywhere - you open the PR yourself afterwards
(see CONTRIBUTING.md). The manual protocol in PROTOCOL.md stays authoritative;
this script only automates exactly that flow.

Usage:
    python scripts/run_benchmark.py [--port 8080] [--model PATH] [--attach]
"""
import argparse
import datetime
import importlib.util
import io
import json
import math
import os
import re
import secrets
import shutil
import socket
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE_NAME = "inference-speed"
TOOLS_DIR = os.path.join(ROOT, "tools")
MODELS_DIR = os.path.join(TOOLS_DIR, "models")
IS_WIN = os.name == "nt"
SERVER_BIN = "llama-server.exe" if IS_WIN else "llama-server"

CONTRIBUTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
PROTOCOL_VERSION_RE = re.compile(r"\bv(\d+\.\d+)")


def _safe_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def step(title):
    print()
    print("== %s" % title)


# -------------------------------------------------------------- prompts ------

def ask(prompt, default=None, validate=None):
    """Prompt until a (validated) value is entered; Enter accepts the default.

    The default itself must pass validation when one is given - an invalid
    default must never be accepted silently."""
    suffix = ("  [default: %s]" % default) if default else ""
    while True:
        val = input("  %s%s: " % (prompt, suffix)).strip()
        if not val and default is not None:
            if validate and not validate(str(default)):
                print("    invalid - try again")
                continue
            return str(default)
        if validate and not validate(val):
            print("    invalid - try again")
            continue
        return val


def ask_yn(prompt, default_yes=True):
    hint = "Y/n" if default_yes else "y/N"
    while True:
        val = input("  %s [%s]: " % (prompt, hint)).strip().lower()
        if not val:
            return default_yes
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False


def ask_choice(prompt, options):
    """options: list of (value, label)"""
    print("  %s" % prompt)
    for i, (_v, label) in enumerate(options, start=1):
        print("   %d. %s" % (i, label))
    while True:
        val = ask("choose", None)
        if val.isdigit() and 1 <= int(val) <= len(options):
            return options[int(val) - 1][0]


# --------------------------------------------------------------- config ------

def load_type_config():
    cfg_path = os.path.join(ROOT, "config", "benchmark_types.json")
    with open(cfg_path, "r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)
    if TYPE_NAME not in raw:
        sys.exit("error: benchmark type '%s' has no entry in config/benchmark_types.json" % TYPE_NAME)
    return raw[TYPE_NAME]


def pick_model(type_cfg):
    models = {m: c for m, c in (type_cfg.get("models") or {}).items() if isinstance(c, dict) and c.get("recipes")}
    if not models:
        sys.exit("error: no model groups with pinned recipes configured yet")
    if len(models) == 1:
        return list(models)[0]
    label = lambda m: "%s (%d tier recipes)" % (m, len(models[m]["recipes"]))
    chosen = ask_choice("Model group:", [(m, label(m)) for m in sorted(models)])
    return chosen


def tier_of_vram(vram, tiers):
    """Same pin semantics as the validator: snap to the marketed VRAM class, then
    exact-match a tier's 'vram_gb'; a tier without one is open-ended and takes
    everything above the highest pin."""
    if not (isinstance(vram, (int, float)) and not isinstance(vram, bool) and vram > 0):
        return None
    s = snap_vram_gb(vram)

    def pinned(t):
        v = t.get("vram_gb")
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    pins = [pinned(t) for t in tiers]
    names = [t["name"] for t, p in zip(tiers, pins) if p is not None]
    values = [p for p in pins if p is not None]
    opens = [t["name"] for t, p in zip(tiers, pins) if p is None]
    for name, value in zip(names, values):
        if s == value:
            return name
    if values and opens and s > max(values):
        return opens[-1]
    return None


def band_label(t, tiers=None):
    """'32 GB' for a pinned tier; '>32 GB' (vs. the highest pin) for an open one."""
    v = t.get("vram_gb")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "%d GB" % v
    pins = [x.get("vram_gb") for x in (tiers or [])
            if isinstance(x.get("vram_gb"), (int, float))]
    return ">%d GB" % max(pins) if pins else "any VRAM"


def results_subfolder(vram, tiers, picked_tier_name, variant_conforms):
    """results/ subfolder this run belongs in: its ranked tier when the recorded
    VRAM snaps to exactly the picked tier and the pinned model file was used;
    'unranked' otherwise. Must stay in sync with validate_results.py."""
    if not variant_conforms:
        return "unranked"
    suggested = tier_of_vram(vram, tiers)
    return suggested if (suggested is not None and suggested == picked_tier_name) else "unranked"


def protocol_version(type_cfg):
    proto = os.path.join(ROOT, "benchmarks", TYPE_NAME, "PROTOCOL.md")
    try:
        with open(proto, "r", encoding="utf-8-sig") as fh:
            head = fh.read(400)
        m = PROTOCOL_VERSION_RE.search(head)
        if m:
            return m.group(1)
    except OSError:
        pass
    return ask("protocol version (see benchmarks/%s/PROTOCOL.md)" % TYPE_NAME, "1.0",
               validate=lambda v: bool(re.match(r"^\d+\.\d+$", v)))


# ---------------------------------------------------------- hardware detect --

def nvidia_smi_path():
    p = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if p:
        return p
    win_smi = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if IS_WIN and os.path.isfile(win_smi):
        return win_smi
    return None


def snap_vram_gb(gb):
    """Snap a framebuffer total up to the card's marketed VRAM class.

    nvidia-smi reports slightly less than the nominal size (driver reserve) -
    e.g. an 'RTX 5090 (32 GB)' shows ~31.9 GB, which would miss its pinned tier
    entirely. Snap up to the next whole GB when within 0.75 GB; other values pass
    through unchanged."""
    ceiling = int(math.ceil(gb))
    if ceiling - gb <= 0.75:  # includes exact integers (no change)
        return ceiling
    return round(gb, 1)


def detect_gpus():
    """List of {'index','name','vram_gb'} from nvidia-smi; [] when unavailable."""
    smi = nvidia_smi_path()
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        bits = [b.strip() for b in line.split(",")]
        if len(bits) >= 3 and bits[0].isdigit() and bits[2].isdigit():
            gpus.append({"index": int(bits[0]), "name": bits[1],
                         "vram_gb": snap_vram_gb(int(bits[2]) / 1024.0)})
    return gpus


def _powershell(command):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                         capture_output=True, text=True, timeout=60)
    for line in (out.stdout or "").splitlines():
        if line.strip():
            return line.strip()
    return None


def _num(value):
    """64 -> 64, '32.5' -> 32.5 (int when integral); None when not a number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f <= 0 or f != f:
        return None
    return int(f) if f.is_integer() else round(f, 2)


def collect_hardware():
    """Returns (hardware_dict, cuda_visible_devices|None). Every value editable."""
    gpus = detect_gpus()
    cuda_dev = None
    gpu_default = vram_default = None
    if len(gpus) == 1:
        g = gpus[0]
        gpu_default, vram_default = "%s (%g GB)" % (g["name"], g["vram_gb"]), g["vram_gb"]
    elif len(gpus) > 1:
        chosen = ask_choice(
            "Multiple NVIDIA GPUs detected - which one will llama-server use?",
            [(str(g["index"]), "GPU %d: %s (%g GB)" % (g["index"], g["name"], g["vram_gb"])) for g in gpus])
        g = [x for x in gpus if str(x["index"]) == chosen][0]
        cuda_dev, gpu_default, vram_default = \
            str(chosen), "%s (%g GB)" % (g["name"], g["vram_gb"]), g["vram_gb"]

    print("  Detected values - press Enter to accept a value or type a correction.")
    if not gpus:
        print("  no NVIDIA GPU detected via nvidia-smi")

    gpu = vram = None
    while True:
        gpu_raw = ask("GPU model (e.g. 'NVIDIA RTX 3090 (24 GB)'), or 'cpu' for CPU-only", gpu_default)
        if not gpu_raw and gpu_default is None:
            print("    blank = CPU-only run (no VRAM recorded - the row will be UNRANKED)")
            break
        if gpu_raw.strip().lower() in ("cpu", "cpu-only", "none"):
            break
        vram = _num(ask("VRAM in GB (marketed class - matches you to a pinned tier)", vram_default))
        if vram is None:
            print("    VRAM must be a positive number")
            continue
        gpu = gpu_raw.strip()
        break

    # The technical minimum a run needs to rank: GPU type + VRAM - this runner asks only for
    # those two. cpu/ram_gb/os remain valid optional schema fields (never used for ranking);
    # contributors who want them add them by hand before the PR.
    hardware = {"gpu": gpu}
    if vram is not None:
        hardware["vram_gb"] = vram
    return hardware, cuda_dev


# ------------------------------------------------------------- tooling -------

def find_llama_server():
    candidates = [shutil.which("llama-server"), shutil.which(SERVER_BIN)]
    llama_dir = os.path.join(TOOLS_DIR, "llama.cpp")
    if os.path.isdir(llama_dir):
        for dirpath, _d, files in os.walk(llama_dir):
            if SERVER_BIN in files:
                candidates.append(os.path.join(dirpath, SERVER_BIN))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def server_version(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            if "version" in line.lower() or any(c.isdigit() for c in line[:20]):
                return line.strip().split("\n")[0][:100]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def load_probe_module(model):
    path = os.path.join(ROOT, "benchmarks", TYPE_NAME, model, "probe.py")
    if not os.path.isfile(path):
        sys.exit("error: no pinned probe found at %s - run the protocol manually" % path)
    spec = importlib.util.spec_from_file_location("qcb_probe_%s" % model.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ----------------------------------------------------------- model download --

def hf_url(repo_id, filename):
    return "https://huggingface.co/%s/resolve/main/%s" % (repo_id, filename)


def hf_size_mb(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-community-benchmarks"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = int(resp.headers.get("Content-Length", 0))
            return length >> 20 if length else None
    except (urllib.error.URLError, OSError):
        return None


def download_model(url, dest):
    """Stream the GGUF into dest (.part while in flight); resume-safe."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    for attempt in range(1, 4):
        headers = {"User-Agent": "llm-community-benchmarks"}
        if have > 0:
            headers["Range"] = "bytes=%d-" % have
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                resumed = resp.status == 206 and have > 0
                total = int(resp.headers.get("Content-Length", 0)) + (have if resumed else 0)
                t0 = time.time()
                with open(part, "ab" if resumed else "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if total and now - t0 > 1.0:
                            sys.stdout.write("\r  model %8d/%d MB (%3d%%)"
                                             % (have >> 20, total >> 20, min(99, have * 100 // max(total, 1))))
                            sys.stdout.flush()
            print("\r  model downloaded: %d MB%s" % (have >> 20, " (resumed)" if resumed else "") + " " * 30)
            os.replace(part, dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("error: model file not found on Hugging Face (%s)" % url)
                return False
            raise
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            if attempt == 3:
                raise RuntimeError("model download failed after 3 attempts (%s); partial kept at %s for resume"
                                   % (exc, part))
            print("\n  retrying model download in %ds: %s" % (5 * attempt, exc))
            time.sleep(5 * attempt)


def resolve_model(recipe_file, repo_id, user_path):
    """Returns the model file path; exits cleanly when it cannot be provided."""
    if user_path:
        if not os.path.isfile(user_path):
            sys.exit("error: --model file not found: %s" % user_path)
        if os.path.basename(user_path) != recipe_file:
            print("  warning: file name differs from the pinned '%s' - this row will be UNRANKED." % recipe_file)
        return user_path
    dest = os.path.join(MODELS_DIR, recipe_file)
    if os.path.isfile(dest):
        return dest
    size_mb = hf_size_mb(hf_url(repo_id, recipe_file))
    print("  model file missing: %s (from %s%s)" % (recipe_file, repo_id,
                                                     ", ~%d MB" % size_mb if size_mb else ""))
    if not ask_yn("Download it now into tools/models/?"):
        sys.exit("aborted - no files written. Re-run with the file present in tools/models/ "
                 "(or pass --model PATH).")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        download_model(hf_url(repo_id, recipe_file), dest)
    except (RuntimeError, OSError) as exc:
        sys.exit("error: %s" % exc)
    if not os.path.isfile(dest):
        sys.exit("error: model file was not downloaded - re-run to resume, or pass --model PATH.")
    return dest


# ---------------------------------------------------------------- server -----

class ServerProc:
    """llama-server child process; streams its output live and into self.lines."""

    def __init__(self, cmd_args, env, workdir):
        self.proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, cwd=workdir)
        self.stream = io.TextIOWrapper(self.proc.stdout, encoding="utf-8", errors="replace")
        self.lines = []
        self._cancel = threading.Event()
        t = threading.Thread(target=self._watchdog, daemon=True)
        t.start()

    def _watchdog(self):
        if self._cancel.wait(45 * 60):  # safety net: stuck model load / CUDA init
            return
        try:
            self.proc.terminate()
        except OSError:
            pass

    def read_until_listening(self):
        for raw in self.stream:
            line = raw.rstrip("\n")
            print(line)
            self.lines.append(line)
            if "listening" in line.lower():
                self._cancel.set()
                return True
        self._cancel.set()
        return False

    def stop(self):
        self._cancel.set()
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(10)
            if self.proc.poll() is None:
                self.proc.kill()
        except OSError:
            pass


def build_server_cmd(server_bin, model_path, recipe, port):
    args = [server_bin, "-m", model_path,
            "-c", str(recipe["context_length"]),
            "-ngl", str(recipe["n_gpu_layers"]),
            "-fa", "1",
            "--cache-type-k", str(recipe["kv_cache_quant"]),
            "--cache-type-v", str(recipe["kv_cache_quant"])]
    if recipe.get("mtp_enabled"):
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(recipe["spec_draft_n_max"])]
    # prompt caching stays off so every timed run does a true full prefill (PROTOCOL.md)
    args += ["--no-cache-prompt"]
    args += ["--parallel", "1", "--host", "127.0.0.1", "--port", str(port)]
    return args


# ------------------------------------------------------------------- probe ---

def run_probe_pass(probe_mod, preset_path, url):
    """Warmup + measured runs with the EXACT output format of standalone
    probe.py; returns (client_lines, {'token_generation_tps':..., 'ttft_s':...})."""
    with open(preset_path, "r", encoding="utf-8-sig") as fh:
        user_message = fh.read().strip()

    lines = []
    line0 = "preset: %s (%d chars)" % (os.path.basename(preset_path), len(user_message))
    print(line0); lines.append(line0)
    print(); lines.append("")

    probe_mod.stream_chat("warm up", probe_mod.WARMUP_MAX_TOKENS, url)  # untimed warmup

    ttfts, speeds = [], []
    for i in range(probe_mod.RUNS_PER_PROMPT):
        counted, ttft, speed, answer = probe_mod.stream_chat(user_message, probe_mod.MAX_TOKENS, url)
        ttfts.append(ttft)
        speeds.append(speed)
        ln = "run %d/%d | tokens %d | TTFT %.1f s | decode %.1f tok/s" \
             % (i + 1, probe_mod.RUNS_PER_PROMPT, counted, ttft, speed)
        print(ln); lines.append(ln)
        preview = " ".join(answer.split())
        if len(preview) > 240:
            preview = preview[:240].rstrip() + " ..."
        ln2 = "answer preview: %s" % (preview or "(empty answer)")
        print(ln2); lines.append(ln2)
        print()

    overall = ("OVERALL (%d measured runs): median decode %.1f tok/s \u00b7 mean %.1f | median TTFT %.1f s"
               % (probe_mod.RUNS_PER_PROMPT, st.median(speeds), st.mean(speeds), st.median(ttfts)))
    print(overall); lines.append(overall)
    metrics = {
        "token_generation_tps": float("%.1f" % st.median(speeds)),   # exactly as printed above
        "ttft_s": float("%.1f" % st.median(ttfts)),
    }
    return lines, metrics


# ------------------------------------------------------------- result file ---

def existing_result_ids():
    """Every result_id under benchmarks/<type>/<model>/results[/<tier>] so a
    new id cannot collide with one in any tier subfolder."""
    ids = set()
    bench_dir = os.path.join(ROOT, "benchmarks")
    for dirpath, _d, files in os.walk(bench_dir):
        parts = [p for p in dirpath.replace(os.sep, "/").split("/") if p]
        try:
            i = parts.index("benchmarks")
        except ValueError:
            continue
        seg = parts[i + 1:]
        if not (len(seg) in (3, 4) and seg[2] == "results"):
            continue
        for f in files:
            if f.endswith(".json") and not f.startswith("_"):
                ids.add(os.path.splitext(f)[0])
    return ids


def choose_result_id(date_s, contributor, model):
    # Generated, never asked for: ids allow only lowercase a-z0-9/dots/hyphens -
    # GitHub usernames may carry capitals and model names vary, so sanitize the
    # whole base. The id ends in a 6-char random suffix (schema rule) so nobody
    # has to reason about collisions by hand; it is printed for the record only.
    raw_base = "%s-%s-%s" % (date_s, contributor.lower(), model)
    base = re.sub(r"[^a-z0-9.-]", "", raw_base) or "run"
    base = re.sub(r"[.-]{2,}", "-", base).strip("-.")  # usernames like 'foo-' must not break the pattern
    taken = existing_result_ids()
    rid = base + "-" + secrets.token_hex(3)
    while rid in taken:  # astronomically unlikely, but keep it collision-free
        rid = base + "-" + secrets.token_hex(3)
    print("  result id (generated): %s" % rid)
    return rid


def write_files(results_dir, rid, data, raw_text):
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, rid + ".json")
    raw_path = os.path.join(results_dir, rid + ".raw.txt")
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(raw_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(raw_text)
    return json_path, raw_path


def run_validator():
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "validate_results.py"), "--root", ROOT],
                          capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip() and proc.returncode != 0:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode


# ------------------------------------------------------------------- main ----

def _run(args):
    type_cfg = load_type_config()
    tiers = [t for t in type_cfg.get("tiers", []) if isinstance(t, dict) and t.get("name")]
    model = pick_model(type_cfg)
    model_cfg = type_cfg["models"][model]
    recipes, presets_map = model_cfg["recipes"], model_cfg.get("presets") or {}
    proto_ver = protocol_version(type_cfg)

    print("  model group : %s" % model)

    step("[1/5] VRAM tier")
    if len(tiers) == 1:
        tier_name = tiers[0]["name"]
        print("  (only tier configured: %s - %s)" % (tier_name, band_label(tiers[0], tiers)))
    else:
        tier_name = ask_choice("Tier:", [(t["name"], "%s (%s)" % (t["name"], band_label(t, tiers))) for t in tiers])

    step("[2/5] hardware (the technical minimum that decides your tier - both editable)")
    hardware, cuda_dev = collect_hardware()
    vram = hardware.get("vram_gb")
    suggested = tier_of_vram(vram, tiers) if vram else None
    if hardware.get("gpu") is None:
        print("  note: CPU-only run - results without GPU/VRAM are listed UNRANKED.")
    elif suggested and suggested != tier_name:
        band_txt = band_label([t for t in tiers if t["name"] == suggested][0], tiers)
        print("  your recorded VRAM (%s GB) maps to tier '%s' (%s)." % (vram, suggested, band_txt))
        if not ask_yn("You picked tier '%s' - that row will be UNRANKED. Continue?" % tier_name,
                      default_yes=False):
            return _aborted()
    elif vram and not suggested:  # no pinned tier matches (e.g. an 8/12 GB card, or a size between pins)
        values = sorted(t["vram_gb"] for t in tiers if isinstance(t.get("vram_gb"), (int, float)))
        print("  note: recorded VRAM (%s GB) matches no pinned tier (pinned values: %s) - this row will be UNRANKED."
              % (vram, " / ".join("%d" % v for v in values)))

    recipe = recipes.get(tier_name)
    if not isinstance(recipe, dict):
        sys.exit("error: no pinned recipe for model '%s' + tier '%s'" % (model, tier_name))
    preset_name = presets_map.get(tier_name)
    if not preset_name:
        sys.exit("error: config has no pinned preset for tier '%s' of '%s'" % (tier_name, model))
    preset_path = os.path.join(ROOT, "benchmarks", TYPE_NAME, "test-data", preset_name)
    if not os.path.isfile(preset_path):
        sys.exit("error: pinned preset file missing: %s" % preset_path)

    step("[3/5] tooling")
    server_bin = find_llama_server()
    if not server_bin:
        sys.exit("error: llama-server not found - run 'python scripts/setup.py' first "
                 "(or put llama.cpp on PATH).")
    ver_line = server_version(server_bin) or ""
    print("  llama-server : %s%s" % (server_bin, (" (%s)" % ver_line) if ver_line else ""))

    model_path = resolve_model(recipe["file"], model_cfg.get("model_repo", ""), args.model)
    print("  model file   : %s" % model_path)
    print("  preset prompt: %s (%d chars)" % (preset_name, os.path.getsize(preset_path)))
    subfolder = results_subfolder(vram, tiers, tier_name, os.path.basename(model_path) == recipe["file"])
    if subfolder != tier_name:
        print("  note: this run does not rank in '%s', so its files will go to results/%s/" % (tier_name, subfolder))

    cmd_args = build_server_cmd(server_bin, model_path, recipe, args.port)
    pretty = " ".join(cmd_args)
    wrapped = "\n".join(("    " + chunk) for chunk in _wrap(pretty))
    print("  server command (pinned):")
    print(wrapped if not args.attach else wrapped + "\n    (--attach: using your already-running server)")

    port = args.port
    while not port_free(port) and not args.attach:
        print()
        choice = ask_choice("Port %d is busy. What now?" % port,
                            [("new", "pick a different port (default 8081+)"),
                             ("attach", "--attach style: use the server already on that port")])
        if choice == "attach":
            args.attach = True
            break
        while True:
            val = ask("new port number", str(port + 1))
            if val.isdigit() and 1 <= int(val) <= 65535:
                port = int(val)
                break
            print("    enter a number between 1 and 65535")

    if ask_yn("Start the benchmark now?", default_yes=True) is False:
        return _aborted()

    raw_parts = ["llama-server build: %s" % (ver_line or "unknown"),
                 "command: %s" % " ".join(cmd_args), ""]
    server = None
    url = "http://127.0.0.1:%d" % port
    try:
        step("[4/5] running the probe")
        if args.attach:
            raw_parts.append("(server started manually by contributor - --attach)")
            print("  assuming your server already matches the pinned recipe above.")
        else:
            env = dict(os.environ)
            if cuda_dev is not None:
                env["CUDA_VISIBLE_DEVICES"] = cuda_dev
            server = ServerProc(cmd_args, env, ROOT)
            if not server.read_until_listening():
                print("\n  error: llama-server stopped before it was listening - last lines of its log:")
                for ln in server.lines[-25:]:
                    print("    " + ln)
                print("  Common cause on low-end tiers: the pinned recipe does not fit in VRAM (OOM).")
                print("  If that is a problem with the recipe itself, open an issue - fix via PR to the README/config.")
                return 1
            raw_parts.extend(server.lines)

        probe_mod = load_probe_module(model)
        client_lines, metrics = run_probe_pass(probe_mod, preset_path, url)
        zero = [ln for ln in client_lines if re.match(r"^run \d+/\d+ \| tokens 0 ", ln)]
        if zero:  # an empty run would poison the medians - never record partial data
            print("\n  error: a measured run produced no reply tokens (%s) - not recording this." % zero[0])
            print("  Check that the server on %s matches the pinned recipe, then re-run from scratch." % url)
            return 1

        step("[5/5] your info + result files")
        contributor = ask("GitHub username (contributor)", validate=lambda v: bool(CONTRIBUTOR_RE.match(v)))
        date_s = ask("run date (YYYY-MM-DD)", datetime.date.today().isoformat(),
                     validate=lambda v: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", v)))
        comment = ask("short comment for the leaderboard (max 200 chars, may be empty)", "")
        if len(comment) > 200:
            sys.exit("error: comment longer than 200 characters")

        rid = choose_result_id(date_s, contributor, model)
        data = {
            "result_id": rid,
            "benchmark_type": TYPE_NAME,
            "model_family": model,
            "contributor": contributor,
            "date": date_s,
            "tool": {"name": "llama-server + %s/probe.py" % model,
                     "version": _tool_version(ver_line) if ver_line else "unknown"},
            "protocol_version": proto_ver,
        }
        variant = {
            "quantization": recipe["quantization"],
            "file": recipe["file"],
        }
        repo_id = model_cfg.get("model_repo")
        if repo_id:
            variant["repo_id"] = repo_id
        data["variant"] = variant
        data["hardware"] = hardware
        data["settings"] = {k: v for k, v in recipe.items() if k not in ("file", "quantization")}
        pt = (model_cfg.get("preset_tokens") or {}).get(tier_name)
        if isinstance(pt, (int, float)) and pt > 0:
            data["settings"]["prompt_tokens"] = pt
        data["metrics"] = metrics
        if comment:
            data["comment"] = comment
        data["raw_output_file"] = rid + ".raw.txt"

        raw_text = "\n".join(raw_parts).rstrip("\n") + "\n\n" \
            + "--- probe.py output ---\n" + "\n".join(client_lines) + "\n" \
            "Record the MEDIAN DECODE as metrics.token_generation_tps " \
            "and the MEDIAN TTFT as metrics.ttft_s.\n"
        results_dir = os.path.join(ROOT, "benchmarks", TYPE_NAME, model, "results", subfolder)
        json_path, raw_path = write_files(results_dir, rid, data, raw_text)
        print("\n  wrote: %s" % json_path)
        print("  wrote: %s" % raw_path)

        print("\nLocal validation (scripts/validate_results.py):")
        rc = run_validator()
        if rc != 0:
            print("\nThe files were kept, but fix the problems above before opening a PR.")
            return rc
        print("\nDone. Next steps:")
        print("  - open a pull request adding these two files (see CONTRIBUTING.md);")
        print("  - results are append-only: never edit them later - corrections go in as a new file with 'supersedes'.")
        return 0
    finally:
        if server is not None:
            print("\nstopping llama-server ...")
            server.stop()


def _aborted():
    print("aborted - no files written.")
    return 1


def _wrap(text, width=100):
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        chunks.append(cur)
    return chunks


def _tool_version(ver_line):
    """'... version = b6500 (...)' / '... (build 10582, ...)' -> 'bNNNN'; else the raw line."""
    m = re.search(r"\bb(\d{4,})\b", ver_line) or re.search(r"\bbuild\s+(\d{4,})\b", ver_line)
    if m:
        return "b%s" % m.group(1)
    return (ver_line or "unknown").strip()[:60] or "unknown"


def main(argv=None):
    _safe_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--port", type=int, default=8080, help="llama-server port (default 8080)")
    ap.add_argument("--model", default=None,
                    help="path to an existing GGUF instead of auto-download into tools/models/")
    ap.add_argument("--attach", action="store_true",
                    help="reuse a server you already started on the port (you must match the pinned recipe yourself)")
    args = ap.parse_args(argv)

    try:
        return _run(args)
    except KeyboardInterrupt:
        print()
        print("aborted by user - no result files were written.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
