#!/usr/bin/env python3
"""One-time environment setup for this benchmark repo.

Installs the tooling a contributor needs so that scripts/run_benchmark.py can
be run with minimal manual work:

  1. checks your Python interpreter (3.9+, stdlib only - nothing is pip-installed);
  2. locates or downloads a prebuilt llama.cpp release (llama-server) into
     <repo>/tools/llama.cpp/   (gitignored; a few hundred-MB archive, CUDA build when an
     NVIDIA driver is detected, CPU build otherwise - override with --build);
  3. reports the state of the rest: nvidia-smi presence, the pinned prompt
     presets, and every pinned model file - with exact download URLs for any
     that are missing (those are fetched by run_benchmark.py on demand).

Windows and Linux are supported directly; on other systems it prints manual
instructions (and still verifies everything else). Model GGUF files are NOT
downloaded here - run_benchmark.py fetches your tier's file when you pick a
tier. Nothing is installed system-wide, nothing needs admin rights.

Usage:
    python scripts/setup.py [--build auto|cuda|cpu] [--force-download]
"""
import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
LLAMA_DIR = os.path.join(TOOLS_DIR, "llama.cpp")
MODELS_DIR = os.path.join(TOOLS_DIR, "models")
CONFIG_PATH = os.path.join(ROOT, "config", "benchmark_types.json")
TYPE_KEY = "inference-speed"  # single benchmark type in this repo (see config/)
# llama.cpp marks its build tags (bNNNN) as *pre-releases*, so GitHub's
# /releases/latest endpoint returns a stub tag without binaries - scan the
# recent release list and pick the newest one that has usable assets.
GITHUB_API_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"

IS_WIN = os.name == "nt"
SERVER_BIN = "llama-server.exe" if IS_WIN else "llama-server"


def _safe_streams():
    # Windows consoles often default to cp1252; degrade instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def step(title):
    print()
    print("== %s" % title)


# ------------------------------------------------------------- Python check --

def check_python():
    ver = "%d.%d.%d" % sys.version_info[:3]
    path = sys.executable or "(embedded)"
    if sys.version_info < (3, 9):
        print("Python %s at %s is too old - this repo needs Python 3.9+." % (ver, path))
        return False
    print("python: %s (%s) - ok" % (ver, path))
    if "WindowsApps" in path.replace("\\", "/"):
        print("warning: this looks like the Microsoft Store stub interpreter;")
        print("         a regular python.org install is more reliable for long downloads.")
    return True


# ------------------------------------------------------------- nvidia-smi ----

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

    nvidia-smi reports slightly less than nominal (driver reserve) - e.g. an
    'RTX 5090 (32 GB)' shows ~31.9 GB, which would miss its pinned tier entirely.
    Snap up to the next whole GB when within 0.75 GB; other values pass through."""
    ceiling = int(math.ceil(gb))
    if ceiling - gb <= 0.75:  # includes exact integers (no change)
        return ceiling
    return round(gb, 1)


def gpu_summary():
    """List of 'NAME (NN GB)' strings from nvidia-smi, or None if not usable."""
    smi = nvidia_smi_path()
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.splitlines():
        bits = [b.strip() for b in line.split(",")]
        if len(bits) >= 2 and bits[1].isdigit():
            gpus.append("%s (%g GB)" % (bits[0], snap_vram_gb(int(bits[1]) / 1024.0)))
    return gpus or None


# ----------------------------------------------------- llama.cpp discovery ---

def find_existing_server():
    """(path, version) of a usable llama-server: PATH first, then tools/."""
    candidates = [shutil.which("llama-server"), shutil.which(SERVER_BIN)]
    if os.path.isdir(LLAMA_DIR):
        for dirpath, _d, files in os.walk(LLAMA_DIR):
            if SERVER_BIN in files:
                candidates.append(os.path.join(dirpath, SERVER_BIN))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand, server_version(cand)
    return None, None


def server_version(path):
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            if "version" in line.lower() or any(c.isdigit() for c in line[:20]):
                return line.strip().split("\n")[0][:100]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown version"


# ------------------------------------------------------- release download ----

def http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "llm-community-benchmarks-setup"})
    last_err = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            last_err = exc
            time.sleep(3 * attempt)
    raise RuntimeError("cannot reach %s (%s)" % (url, last_err))


def _build_number(tag):
    m = re.match(r"^b(\d+)$", tag or "")
    return int(m.group(1)) if m else -1


def _match_version(name, pattern):
    """Match name against pattern. Returns the CUDA version tuple from group 1,
    an empty tuple when the pattern carries no version, None on no match."""
    m = re.match(pattern, name)
    if not m:
        return None
    if not m.lastindex:
        return ()
    return tuple(int(p) for p in m.group(1).split("."))


def pick_assets(releases, want_cuda):
    """Best (assets, release_tag, all_names) across recent releases.

    Windows + CUDA ships each build as TWO archives that belong together - the
    build itself plus a CUDA-runtime pack so no local toolkit is needed:
      llama-bNNN-bin-win-cuda-<ver>-x64.zip   build (llama-server.exe, ggml*.dll)
      cudart-llama-bin-win-cuda-<ver>-x64.zip runtime DLLs (cublas/cublasLt/cudart)
    Both are returned when present. Other platforms use one archive each:
      llama-bNNN-bin-win-cpu-x64.zip / bin-ubuntu-{x64,arm64}.tar.gz /
      bin-macos-{arm64,x64}.tar.gz (Metal). The flavour must match what was
    asked - a wrong CUDA/CPU asset is never silently substituted. Among matches
    the newest b-tag wins."""
    arch = "x64" if platform.machine() in ("AMD64", "x86_64") else "arm64"
    cv = r"\d+(?:\.\d+)?"
    if IS_WIN and want_cuda:
        main_pats = [r"^llama-b\d+-bin-win-cuda-(%s)-%s\.zip$" % (cv, arch)]
        comp_pat = r"^cudart-llama-bin-win-cuda-(%s)-%s\.zip$" % (cv, arch)
    elif IS_WIN:
        main_pats = [r"^llama-b\d+-bin-win-cpu-%s\.zip$" % arch]
        comp_pat = None
    elif sys.platform == "darwin":  # machine() is 'arm64' / 'x86_64'; assets say x64/arm64
        main_pats = [r"^llama-b\d+-bin-macos-(%s)\.tar\.gz$" % arch]
        comp_pat = None
    elif sys.platform.startswith("linux"):
        main_pats = [r"^llama-b\d+-bin-ubuntu-%s\.tar\.gz$" % arch,
                     r"^llama-b\d+-bin-linux-%s\.tar\.gz$" % arch]
        comp_pat = None
    else:
        return None, None, []  # no known prebuilt naming for this platform

    best_key, best_asset, best_rel = None, None, None
    all_names = set()
    for rel in releases:
        bnum = _build_number(rel.get("tag_name"))
        for asset in rel.get("assets", []):
            name = asset.get("name")
            if not name:
                continue
            all_names.add(name)
            for pat in main_pats:
                ver = _match_version(name, pat)
                if ver is None:
                    continue
                key = (bnum, ver)
                if best_key is None or key > best_key:
                    best_key, best_asset, best_rel = key, asset, rel
    if best_asset is None:
        return None, None, sorted(all_names)[:40]

    assets = [best_asset]
    if comp_pat and best_rel is not None:
        main_ver = ()
        for pat in main_pats:
            v = _match_version(best_asset["name"], pat)
            if v is not None:
                main_ver = v
                break
        companions = []
        for asset in best_rel.get("assets", []):
            cver = _match_version(asset.get("name") or "", comp_pat)
            if cver is not None:
                companions.append((cver, asset))
        companion = next((a for v, a in companions if v == main_ver), None)
        if companion is None and companions:
            companion = max(companions)[1]  # closest CUDA version within this release
        if companion is not None:
            assets.append(companion)
    return assets, (best_rel or {}).get("tag_name"), sorted(all_names)[:40]


def download_with_progress(url, dest):
    """Download url to dest with resume support and progress output."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    for attempt in range(1, 4):
        headers = {"User-Agent": "llm-community-benchmarks-setup"}
        if have > 0:
            headers["Range"] = "bytes=%d-" % have
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
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
                        if total and now - t0 > 0.5:
                            rate = have / max(now - t0, 1e-6) / (1 << 20)
                            sys.stdout.write("\r  %8d/%d MB (%3d%%)  %.1f MB/s   "
                                             % (have >> 20, total >> 20,
                                                min(99, have * 100 // max(total, 1)), rate))
                            sys.stdout.flush()
            print("\r  downloaded %d MB%s" % (have >> 20, " (resumed)" if resumed else "") + " " * 24)
            os.replace(part, dest)
            return
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            if attempt == 3:
                raise RuntimeError("download failed after 3 attempts (%s); partial file kept at %s for resume"
                                   % (exc, part))
            print("\n  retrying in %ds (resume from %d MB): %s" % (5 * attempt, have >> 20, exc))
            time.sleep(5 * attempt)


def extract_release(archive_path):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(LLAMA_DIR)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(LLAMA_DIR)


def install_llama_cpp(force, build_choice):
    step("llama.cpp (llama-server)")
    detected = gpu_summary()
    if build_choice == "auto":
        want_cuda = detected is not None
    else:
        want_cuda = build_choice == "cuda"
    if build_choice != "auto":
        print("--build %s selected -> %s build" % (build_choice, "CUDA" if want_cuda else "CPU"))
    elif detected:
        print("NVIDIA GPU(s) detected via nvidia-smi: %s -> CUDA build" % ", ".join(detected))
    else:
        print("no NVIDIA driver detected via nvidia-smi -> CPU build (override with --build cuda)")

    if not force:
        path, ver = find_existing_server()
        if path:
            print("llama-server already available: %s (%s)" % (path, ver))
            return True

    try:
        releases = http_get_json(GITHUB_API_RELEASES)
    except RuntimeError as exc:
        print("error: %s" % exc)
        print("Manual fallback: download a prebuilt from https://github.com/ggml-org/llama.cpp/releases,")
        print("put the binary on PATH (or under tools/llama.cpp/) and re-run this script.")
        return False

    assets, tag, all_names = pick_assets(releases, want_cuda)
    if assets is None:
        kind = "CUDA" if want_cuda else "CPU"
        print("error: no prebuilt %s asset for this platform in recent releases." % kind)
        if sys.platform == "darwin":
            print("hint (macOS): 'brew install llama.cpp' also works - or pick a matching")
            print("      macos-{arm64,x64}.tar.gz manually and put the binary on PATH, then re-run.")
        elif platform.machine() in ("aarch64", "arm64"):
            print("hint (ARM64 - e.g. NVIDIA DGX Spark): recent releases ship an 'ubuntu-arm64'")
            print("      tarball; if none matched, the naming changed again - pick manually from")
            print("      https://github.com/ggml-org/llama.cpp/releases, or build from source with")
            print("      CUDA support (GB10 needs a recent enough CUDA toolkit). Re-run after.")
        else:
            print("available assets:")
            for n in all_names[:40]:
                print("  - " + n)
            print("Pick one manually from https://github.com/ggml-org/llama.cpp/releases,")
            print("put the binary on PATH (or under tools/llama.cpp/) and re-run this script.")
        return False

    os.makedirs(LLAMA_DIR, exist_ok=True)
    for asset in assets:
        name = asset["name"]
        size_mb = int(asset.get("size", 0)) >> 20
        print("%s | %s (%d MB)" % (tag or "?", name, size_mb))
        archive_path = os.path.join(TOOLS_DIR, name)
        try:
            download_with_progress(asset["browser_download_url"], archive_path)
            extract_release(archive_path)
            if os.path.isfile(archive_path):
                os.remove(archive_path)
        except (RuntimeError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            print("\nerror: %s" % exc)
            print("Partial files (if any) are kept under tools/ for a resumed re-run.")
            return False

    path, ver = find_existing_server()
    if not path:
        print("error: extraction finished but llama-server was not found under tools/llama.cpp/")
        return False
    print("installed: %s (%s)" % (path, ver))
    return True


# ------------------------------------------------------------ repo sanity ----

def check_presets():
    step("pinned prompt presets")
    td = os.path.join(ROOT, "benchmarks", "inference-speed", "test-data")
    if not os.path.isdir(td):
        print("error: %s is missing - the repo checkout looks incomplete." % td)
        return False
    files = sorted(n for n in os.listdir(td) if n.startswith("Prompt_") and n.endswith(".txt"))
    if not files:
        print("warning: no Prompt_*_tkn.txt presets found under test-data/.")
        return False
    for n in files:
        size = os.path.getsize(os.path.join(td, n))
        shown = "%d MB" % (size >> 20) if size >= (1 << 20) else "%d KB" % (size >> 10)
        print("  %s (%s)" % (n, shown))
    print("%d preset(s) present - ok" % len(files))
    return True


# ------------------------------------------------------------ model files ----

def hf_check_file(url):
    """HEAD-check a Hugging Face URL.
    Returns ("ok", MB) | ("not-found", None) for an explicit 404 | ("unknown", None)
    when the network or headers gave no answer."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-community-benchmarks-setup"},
                                     method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return "ok", int(resp.headers.get("Content-Length", 0)) >> 20
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "not-found", None
        return "unknown", None
    except (urllib.error.URLError, socket.timeout, OSError, ValueError):
        return "unknown", None


def check_models():
    """Report the state of every pinned model file (single source: config).

    Nothing is downloaded here - run_benchmark.py fetches your tier's file on
    demand with resume support. This step tells you up front what is already in
    tools/models/ and exactly where to get the rest; a pin that 404s on
    Hugging Face is reported as a stale name (fix it via issue + PR)."""
    step("model files (pinned GGUFs)")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            tcfg = json.load(fh).get(TYPE_KEY, {})
    except (OSError, ValueError) as exc:
        print("error: cannot read %s (%s) - the repo checkout looks incomplete." % (CONFIG_PATH, exc))
        return False
    models = tcfg.get("models", {})
    if not models:
        print("warning: no model entries found in config/benchmark_types.json.")
        return True

    missing_any = False
    broken = []
    for model_id, mdef in models.items():
        repo = mdef.get("model_repo") or ""
        recipes = mdef.get("recipes", {})
        if not repo or not recipes:
            print("%s: config entry incomplete (needs 'model_repo' + per-tier 'file')." % model_id)
            missing_any = True
            continue
        print("%s - source: https://huggingface.co/%s" % (model_id, repo))
        seen = set()  # several tiers may share one file; report each once
        for tier in recipes:
            fname = (recipes[tier] or {}).get("file") or ""
            if not fname or fname in seen:
                continue
            seen.add(fname)
            path = os.path.join(MODELS_DIR, fname)
            if os.path.isfile(path):
                print("  [have   ] %s (%d MB)" % (fname, os.path.getsize(path) >> 20))
                continue
            url = "https://huggingface.co/%s/resolve/main/%s" % (repo, fname)
            status, size = hf_check_file(url)
            if status == "not-found":
                broken.append(fname)
                print("  [stale  ] %s - NOT FOUND on Hugging Face (HTTP 404); the pinned name is wrong." % fname)
                print("             Fix via issue + PR to config/benchmark_types.json. URL checked: %s" % url)
            else:
                missing_any = True
                print("  [missing] %s%s - downloaded automatically by run_benchmark.py when you pick that tier,"
                      % (fname, " (~%d MB)" % size if size else ""))
                print("             or grab it now: %s" % url)
    if missing_any:
        print("Those files are the only remaining downloads; they land in tools/models/ (gitignored).")
    elif not broken:
        print("All pinned model files present - you are fully set up.")
    return not broken


# -------------------------------------------------------------------- main ---

def main(argv=None):
    _safe_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--build", choices=["auto", "cuda", "cpu"], default="auto",
                    help="llama.cpp build to install (default: auto - CUDA if nvidia-smi is present)")
    ap.add_argument("--force-download", action="store_true",
                    help="download a fresh release even if llama-server is already available")
    args = ap.parse_args(argv)

    print("llm-community-benchmarks setup (repo root: %s)" % ROOT)
    ok_python = check_python()
    ok_llama = install_llama_cpp(args.force_download, args.build)
    ok_presets = check_presets()
    ok_models = check_models()

    print()
    if ok_python and ok_llama and ok_presets and ok_models:
        print("Setup complete. Next step:")
        print("    python scripts/run_benchmark.py")
        return 0
    print("Setup finished with problems - see notes above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
