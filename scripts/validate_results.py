#!/usr/bin/env python3
"""Validate benchmark result files.

Stdlib only. Two modes:

  1. Full validation (default): every JSON under benchmarks/**/results/**/*.json
     (including the per-tier subfolders) and benchmarks/*/examples/*.json is checked against the rules in
     schema/result.schema.json (implemented here, since no external
     jsonschema dependency is allowed). Also checks path consistency
     (benchmark_type / model_folder names match the file contents), that
     supersedes targets exist, and that every metric + setting key required
     by config/benchmark_types.json for the file's benchmark type is present.

  2. PR mode: --check-additions with a GitHub unified diff (--diff-file),
     enforcing the append-only rule: files under .../results/ may only be
     ADDED, never modified/deleted/renamed. LEADERBOARD.md is generated and
     must not be hand-edited in a PR. Every added result file is then fully
     validated as in mode 1.

Usage:
  python scripts/validate_results.py                       # validate repo at CWD
  python scripts/validate_results.py --root /path/to/repo
  python scripts/validate_results.py --check-additions --diff-file gh.diff
"""

import argparse
import datetime
import json
import math
import os
import re
import sys

# ids must end in a 6-char random suffix so contributors never reason about collisions by hand (see schema)
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9.]+)*-[0-9a-f]{6}$")
TYPE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
MODEL_RE = re.compile(r"^[a-z0-9]+([._-][a-z0-9]+)*$")
USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
PROTOCOL_RE = re.compile(r"^\d+\.\d+$")
RAW_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.txt$")

ALLOWED_FIELDS = {
    "result_id", "benchmark_type", "model_family", "contributor", "date",
    "tool", "protocol_version", "variant", "hardware", "settings",
    "metrics", "comment", "raw_output_file", "supersedes", "notes",
}

MAX_FUTURE_DAYS = 1


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def snap_vram_gb(gb):
    """Snap a framebuffer total up to the marketed VRAM class. Must stay in sync
    with setup.py / run_benchmark.py ('RTX 5090 (32 GB)' reports ~31.9 GB)."""
    ceiling = int(math.ceil(gb))
    if ceiling - gb <= 0.75:  # includes exact integers (no change)
        return ceiling
    return round(gb, 1)


def result_files(root):
    """Yield (path, kind) for every JSON to validate.

    kind is 'results' (benchmarks/{type}/{model}/results[/<tier-or-unranked>]/)
    or 'examples' (benchmarks/{type}/examples/).
    """
    bench_dir = os.path.join(root, "benchmarks")
    if not os.path.isdir(bench_dir):
        return
    for dirpath, _dirnames, filenames in os.walk(bench_dir):
        parts = [p for p in dirpath.replace(os.sep, "/").split("/") if p]
        try:
            i = parts.index("benchmarks")
        except ValueError:
            continue
        seg = parts[i + 1:]
        kind = None
        if len(seg) == 2 and seg[1] == "examples":
            kind = "examples"
        elif len(seg) in (3, 4) and seg[2] == "results":
            kind = "results"
        if not kind:
            continue
        for name in sorted(filenames):
            if name.endswith(".json") and not name.startswith("_"):
                yield os.path.join(dirpath, name), kind


def layout_parts(path):
    """(type, model, kind, band) for a well-formed path, else None.

    kind is 'results' or 'examples'; band is the tier subfolder under results/
    (e.g. 'high-end' or 'unranked'), or None when the file sits directly in
    results/ (allowed only for types without pinned tiers)."""
    parts = [p for p in path.replace(os.sep, "/").split("/") if p]
    try:
        i = parts.index("benchmarks")
    except ValueError:
        return None
    seg = parts[i + 1:]
    if len(seg) == 3 and seg[1] == "examples":
        return (seg[0], None, "examples", None)
    if len(seg) in (4, 5) and seg[2] == "results":
        band = seg[3] if len(seg) == 5 else None
        return (seg[0], seg[1], "results", band)
    return None


def check_structure(path, data):
    """Path must agree with benchmark_type / model_family inside the file."""
    errors = []
    lay = layout_parts(path)
    if lay is None:
        return ["%s: not under benchmarks/{type}/{model}/results[/<tier>]/ or benchmarks/{type}/examples/"
                % os.path.basename(path)]
    type_name, model_name, kind, _band = lay
    btype = data.get("benchmark_type", "")
    if kind == "results" and btype != type_name:
        errors.append(
            "%s: benchmark_type '%s' does not match folder '%s'" % (data.get("result_id", "?"), btype, type_name)
        )
    model = data.get("model_family", "")
    if kind == "results" and model != model_name:
        errors.append(
            "%s: model_family '%s' does not match folder '%s'" % (data.get("result_id", "?"), model, model_name)
        )
    return errors


def load_type_requirements(root):
    """type name -> {'metrics': [keys], 'settings': [keys], 'tiers': [...],
    'models': {model: {'recipes': {tier_name: pinned-settings-dict}}}}
    from config/benchmark_types.json."""
    cfg_path = os.path.join(root, "config", "benchmark_types.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = {}
    for tname, tcfg in raw.items():
        metric_keys = [m.get("key") for m in tcfg.get("metrics", []) if isinstance(m, dict) and m.get("key")]
        setting_keys = [s.get("key") for s in tcfg.get("settings_columns", []) if isinstance(s, dict) and s.get("key")]
        tiers = [t for t in tcfg.get("tiers", []) if isinstance(t, dict) and t.get("name")]
        models_raw = tcfg.get("models") if isinstance(tcfg.get("models"), dict) else {}
        out[tname] = {"metrics": metric_keys, "settings": setting_keys,
                      "tiers": tiers, "models": models_raw}
    return out


def tier_of(vram, tiers):
    """Tier name for a VRAM size. Tiers pin a marketed VRAM value ('vram_gb'):
    snap to the marketed class first, then match exactly. A tier without a
    pinned value is open-ended and takes everything above the highest pin."""
    if not (is_number(vram) and vram > 0):
        return None
    s = snap_vram_gb(vram)
    pins = [(t["vram_gb"], t["name"]) for t in tiers if is_number(t.get("vram_gb"))]
    opens = [t["name"] for t in tiers if not is_number(t.get("vram_gb"))]
    for value, name in pins:
        if s == value:
            return name
    if pins and opens and s > max(v for v, _ in pins):
        return opens[-1]
    return None


def no_tier_reason(vram, tiers):
    """Human-readable reason why a VRAM size matches no pinned tier."""
    pins = sorted(t["vram_gb"] for t in tiers if is_number(t.get("vram_gb")))
    opens = [t["name"] for t in tiers if not is_number(t.get("vram_gb"))]
    if not pins:
        return "no pinned VRAM tiers defined"
    s = snap_vram_gb(vram)
    shown = " / ".join("%d" % p for p in pins)
    if s < pins[0]:
        return "vram %g GB is below the lowest pinned tier (%d GB)" % (vram, pins[0])
    tail = "; above %d ranks as '%s'" % (pins[-1], opens[0]) if opens else ""
    return "no pinned tier matches vram %g GB (pinned values: %s%s)" % (vram, shown, tail)


def check_tier_conformance(data, model, tiers, recipes):
    """(tier_name, reason): does this result exactly match one pinned tier recipe?

    A ranked row must snap to a pinned VRAM value and reproduce every pinned
    setting (plus variant.file/quantization). Anything else is still a valid
    contribution; it just lands on the leaderboard's unranked table.
    """
    hardware = data.get("hardware") if isinstance(data.get("hardware"), dict) else {}
    vram = hardware.get("vram_gb")
    if not (is_number(vram) and vram > 0):
        return None, "hardware.vram_gb missing or invalid"
    band = tier_of(vram, tiers)
    if band is None:
        return None, no_tier_reason(vram, tiers)
    rec = recipes.get(band)
    if not isinstance(rec, dict):
        return None, "tier '%s' has no pinned recipe for model '%s'" % (band, model)
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    variant = data.get("variant") if isinstance(data.get("variant"), dict) else {}
    problems = []
    for key, want in rec.items():
        if key == "file":
            got = variant.get("file")
        elif key == "quantization":
            got = variant.get("quantization")
        else:
            got = settings.get(key)
        if got != want:
            prefix = "variant." if key in ("file", "quantization") else "settings."
            problems.append("%s%s %r, pinned %r" % (prefix, key, got, want))
    if problems:
        shown = "; ".join(problems[:2]) + ("; +%d more" % (len(problems) - 2)
                                           if len(problems) > 2 else "")
        return None, "tier '%s': %s" % (band, shown)
    return band, ""


def check_type_requirements(data, path, type_req):
    """Require every metric + setting key configured for the file's benchmark type.

    Returns (errors, warnings). For types with pinned tiers this also enforces
    that the results/ subfolder agrees with what the file's own data says:
    a run that matches a tier's recipe lives in that tier's folder, every
    other valid run lives in 'unranked/'."""
    errors = []
    _t, _m, kind, folder_band = layout_parts(path) or (None, None, None, None)
    tname = _t
    if not tname or kind != "results":
        return errors, []  # broken layout is reported by other checks; examples are exempt
    if tname not in (type_req or {}):
        errors.append("%s: benchmark type '%s' has no entry in config/benchmark_types.json" % (
            data.get("result_id", "?"), tname))
        return errors, []
    req = type_req[tname]

    metrics = data.get("metrics") or {}
    for key in req["metrics"]:
        if not is_number(metrics.get(key)):
            errors.append("%s: metric '%s' required for benchmark type '%s' (see PROTOCOL.md)" % (
                data.get("result_id", "?"), key, tname))

    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    for key in req["settings"]:
        if key not in settings or settings[key] is None:
            errors.append(
                "%s: setting '%s' required for benchmark type '%s'. Record the value you actually ran "
                "(see PROTOCOL.md and the model's README)" % (data.get("result_id", "?"), key, tname)
            )

    # folder placement must agree with the data: a run that snaps to a pinned
    # VRAM value AND reproduces that tier's recipe exactly goes into that
    # tier's subfolder; every other valid run goes into 'unranked/'
    warnings = []
    tiers = req.get("tiers") or []
    if tiers:
        model_cfg = (req.get("models") or {}).get(data.get("model_family", "")) or {}
        recipes = model_cfg.get("recipes") if isinstance(model_cfg, dict) else None
        band, reason = check_tier_conformance(
            data, data.get("model_family", ""), tiers, recipes or {}) \
            if recipes else (None, "")
        tier_names = ", ".join(t["name"] for t in tiers)
        rid_s = data.get("result_id", "?")
        if folder_band is None:
            errors.append("%s: put the file in a tier subfolder of results/: %s (or unranked/)"
                          % (rid_s, tier_names))
        elif folder_band == "unranked":
            if band is not None:
                errors.append("%s: matches the pinned recipe of tier '%s' exactly; put it under results/%s/ so it can rank"
                              % (rid_s, band, band))
        else:
            if not any(t["name"] == folder_band for t in tiers):
                errors.append("%s: '%s' is not a configured tier of this type (%s); put the file under results/unranked/"
                              % (rid_s, folder_band, tier_names))
            elif band != folder_band:
                errors.append("%s: does not match the pinned recipe of tier '%s' (%s); put it under results/unranked/, or fix the run so it matches"
                              % (rid_s, folder_band, reason))
    elif folder_band is not None:
        errors.append("%s: benchmark type '%s' has no pinned tiers; put the file directly under results/"
                      % (data.get("result_id", "?"), tname))
    return errors, warnings


def validate_file(path, today=None):
    """Return (errors, parsed_data_or_None) for one result file."""
    fid = os.path.basename(path)
    errors = []
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return ["%s: cannot parse JSON: %s" % (fid, exc)], None
    if not isinstance(data, dict):
        return ["%s: top level must be a JSON object" % fid], None

    _lt, _lm, kind, _lb = layout_parts(path) or (None, None, None, None)

    stem = os.path.splitext(os.path.basename(path))[0]
    rid = data.get("result_id")
    if not isinstance(rid, str) or not ID_RE.match(rid or ""):
        errors.append("%s: 'result_id' missing/invalid (lowercase letters, digits, dots and hyphens; must end with a 6-char random hex suffix like -a1b2c3)" % fid)
    elif kind == "results" and rid != stem:
        errors.append("%s: file name must be <result_id>.json (got '%s.json', result_id is '%s')" % (fid, stem, rid))

    def require_str(key, regex=None, empty_ok=False):
        v = data.get(key)
        if not isinstance(v, str) or (not v and not empty_ok):
            errors.append("%s: required string field '%s' is missing/invalid" % (fid, key))
        elif regex and not regex.match(v):
            errors.append("%s: field '%s' does not match pattern %s" % (fid, key, regex.pattern))

    require_str("benchmark_type", TYPE_RE)
    require_str("model_family", MODEL_RE)
    require_str("contributor", USER_RE)
    require_str("protocol_version", PROTOCOL_RE)

    date_s = data.get("date")
    if not isinstance(date_s, str):
        errors.append("%s: required field 'date' (YYYY-MM-DD) is missing" % fid)
    else:
        m = DATE_RE.match(date_s)
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
            if d is None:
                errors.append("%s: 'date' must be YYYY-MM-DD" % fid)
            elif today and d > today + datetime.timedelta(days=MAX_FUTURE_DAYS):
                errors.append("%s: 'date' %s is in the future" % (fid, date_s))
        except ValueError:
            errors.append("%s: 'date' %s is not a real calendar date" % (fid, date_s))

    tool = data.get("tool")
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
        errors.append("%s: required field 'tool.name' is missing/invalid" % fid)
    elif "version" in tool and (not isinstance(tool["version"], str) or not tool["version"]):
        errors.append("%s: 'tool.version', when present, must be a non-empty string" % fid)

    hardware = data.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("%s: required object field 'hardware' is missing/invalid" % fid)
    else:
        # only the GPU type + VRAM matter for ranking; cpu/ram/os are optional extras
        gpu = hardware.get("gpu")
        if "gpu" not in hardware or (gpu is not None and not isinstance(gpu, str)):
            errors.append("%s: 'hardware.gpu' must be a string, or null for CPU-only runs" % fid)
        cpu = hardware.get("cpu")
        if "cpu" in hardware and (not isinstance(cpu, str) or not cpu):
            errors.append("%s: 'hardware.cpu', when present, must be a non-empty string" % fid)
        ram = hardware.get("ram_gb")
        if "ram_gb" in hardware and (not is_number(ram) or ram <= 0):
            errors.append("%s: 'hardware.ram_gb', when present, must be a positive number" % fid)
        vram = hardware.get("vram_gb")
        if "vram_gb" in hardware and (not is_number(vram) or vram <= 0):
            errors.append("%s: 'hardware.vram_gb', when present, must be a positive number" % fid)
        ostext = hardware.get("os")
        if "os" in hardware and (not isinstance(ostext, str) or not ostext):
            errors.append("%s: 'hardware.os', when present, must be a non-empty string" % fid)

    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("%s: required field 'metrics' must be a non-empty object" % fid)
    else:
        for k, v in metrics.items():
            if not is_number(v):
                errors.append("%s: metric '%s' must be a number (got %r)" % (fid, k, type(v).__name__))

    settings = data.get("settings")
    if settings is not None and not isinstance(settings, dict):
        errors.append("%s: 'settings', when present, must be an object" % fid)

    variant = data.get("variant")
    if variant is not None:
        if not isinstance(variant, dict) or not isinstance(variant.get("quantization"), str) or not variant["quantization"]:
            errors.append("%s: 'variant.quantization', when present, must be a non-empty string" % fid)
        elif "file" in variant and (not isinstance(variant["file"], str) or not variant["file"]):
            errors.append("%s: 'variant.file', when present, must be a non-empty string" % fid)

    comment = data.get("comment")
    if comment is not None and (not isinstance(comment, str) or len(comment) > 200):
        errors.append("%s: 'comment', when present, must be a string of at most 200 chars" % fid)

    raw = data.get("raw_output_file")
    if raw is not None:
        if not isinstance(raw, str) or not RAW_FILE_RE.match(raw):
            errors.append("%s: 'raw_output_file' must look like '<result_id>.raw.txt'" % fid)
        elif kind == "results" and not os.path.isfile(os.path.join(os.path.dirname(path), raw)):
            errors.append("%s: raw output file '%s' not found next to the result" % (fid, raw))

    sup = data.get("supersedes")
    if sup is not None:
        if not isinstance(sup, str) or not ID_RE.match(sup):
            errors.append("%s: 'supersedes' must be a valid result_id" % fid)
        elif kind == "results":
            # the superseded file may live in any tier subfolder of this model's results/ tree
            parts = [p for p in path.replace(os.sep, "/").split("/") if p]
            found = False
            if "results" in parts:
                root_dir = os.path.join(*parts[:parts.index("results") + 1])
                candidates = [root_dir]
                try:
                    candidates += [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
                                   if os.path.isdir(os.path.join(root_dir, d))]
                except OSError:
                    pass
                found = any(os.path.isfile(os.path.join(d, sup + ".json")) for d in candidates)
            if not found:
                errors.append("%s: superseded result '%s' does not exist anywhere under this model's results/ folder" % (fid, sup))

    notes = data.get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 500):
        errors.append("%s: 'notes', when present, must be a string of at most 500 chars" % fid)

    for key in data:
        if key not in ALLOWED_FIELDS:
            errors.append("%s: unknown field '%s' (not allowed by schema)" % (fid, key))

    return errors + check_structure(path, data), data


# ---------------------------------------------------------------- PR mode ----

DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def parse_diff(text):
    """Return list of (status, path) from a GitHub unified diff.

    status: A/M/D/R/T/U (best effort; anything with content changes but no
    explicit marker is conservatively reported as M).
    """
    entries = []
    cur = None
    for line in text.splitlines():
        m = DIFF_HEADER_RE.match(line)
        if m:
            if cur is not None:
                entries.append((cur[0], cur[1]))
            status = "M"  # conservative default, refined by mode markers below
            cur = [status, m.group(2)]
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur[0] = "A"
        elif line.startswith("deleted file mode"):
            cur[0] = "D"
        elif line.startswith("rename to "):
            cur[0] = "R"
            cur[1] = line[len("rename to "):].strip()
        # other lines (content hunks) carry no status info; default stands
    if cur is not None:
        entries.append((cur[0], cur[1]))
    return entries


def check_additions(diff_text, root, type_req=None):
    """Enforce append-only rule + validate added files.

    Returns (errors, checked_count, warnings)."""
    errors = []
    warnings = []
    checked = []
    for status, path in parse_diff(diff_text):
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        lay = layout_parts(path)
        under_results = lay is not None and lay[2] == "results"
        if path == "LEADERBOARD.md" and status != "A":
            errors.append(
                "LEADERBOARD.md: generated file, do not edit it directly "
                "(it is regenerated automatically on merge)"
            )
        if under_results:
            if status in ("M", "D", "R", "T", "U"):
                errors.append(
                    "%s: existing results are immutable (action '%s'). Add a new file with 'supersedes' instead"
                    % (path, status)
                )
            elif path.endswith(".json") and not path.startswith("_"):
                fp = os.path.join(root, *parts)
                if not os.path.isfile(fp):
                    errors.append("%s: marked as added but not present in the checkout" % path)
                else:
                    checked.append(fp)
    for fp in sorted(set(checked)):
        errs, data = validate_file(fp)
        errors.extend(errs)
        if data is not None:
            req_errors, req_warnings = check_type_requirements(data, fp, type_req)
            errors.extend(req_errors)
            warnings.extend(req_warnings)
    return errors, len(checked), warnings


# ------------------------------------------------------------------- main ----

def _safe_streams():
    # Windows consoles often default to cp1252 and would crash when printing
    # '✗' on failure, which is exactly when the error report matters most.
    # Degrade unencodable characters instead of losing the output entirely.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def duplicate_id_errors(root):
    """A result_id must be unique across the whole repo (all types, models and tier
    subfolders). The random suffix makes collisions unlikely; this catches any that
    happen so two files can never shadow each other."""
    seen = {}
    for fp, kind in result_files(root):
        if kind != "results":
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        seen.setdefault(stem, []).append(os.path.relpath(fp, root))
    return ["duplicate result_id '%s' (%s): an id must appear once across the whole repo" % (rid, ", ".join(rels))
            for rid, rels in sorted(seen.items()) if len(rels) > 1]


def main(argv=None):
    _safe_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--check-additions", action="store_true",
                    help="PR mode: use the diff to enforce append-only results")
    ap.add_argument("--diff-file", default=None,
                    help="path to a GitHub unified .diff (required with --check-additions)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    errors = []
    warnings = []
    type_req = load_type_requirements(root)

    if args.check_additions:
        if not args.diff_file:
            print("error: --check-additions requires --diff-file", file=sys.stderr)
            return 2
        with open(args.diff_file, "r", encoding="utf-8") as fh:
            errors, n, warnings = check_additions(fh.read(), root, type_req)
        print("PR mode: checked %d added result file(s), append-only rule verified" % n)
    else:
        files = [fp for fp, _kind in result_files(root)]
        for fp in files:
            errs, data = validate_file(fp)
            errors.extend(errs)
            if data is not None:
                req_errors, req_warnings = check_type_requirements(data, fp, type_req)
                errors.extend(req_errors)
                warnings.extend(req_warnings)
        print("Validated %d result file(s)" % len(files))

    errors.extend(duplicate_id_errors(root))

    if errors:
        print("\nFAIL: validation problems")
        for e in errors:
            print("  ✗ " + e)
        return 1
    if warnings:
        print("\nOK (with non-blocking notes):")
        for w in warnings:
            print("  ! " + w)
    else:
        print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
