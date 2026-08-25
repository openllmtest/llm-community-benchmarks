#!/usr/bin/env python3
"""Generate LEADERBOARD.md from all committed result files.

Stdlib only. Deterministic output (no timestamps) so the file is stable:
the publish workflow re-runs this on every merge to main and commits it
only when the content actually changes.

Rules implemented here (also documented in CONTRIBUTING.md):
  * one section per benchmark type configured in config/benchmark_types.json,
    with a sub-table per model family
  * for tiered types each model gets one numbered table PER VRAM TIER (T1..Tn in
    config order): only rows that snap to a tier's pinned VRAM value AND reproduce
    its pinned recipe exactly are ranked (by the type's primary metric); every other
    row is still shown, in a separate "unranked" table with the exact same columns
  * types without "tiers" fall back to one flat per-model table ranked by the
    type's primary metric, with the configured settings/metric columns
  * tables are upsert-style: within a tier only each contributor's latest run
    shows; in unranked/flat tables the same holds per (contributor, conditions).
    Older files stay in results/ and still score points. Result files live in
    per-tier subfolders of results/ (plus 'unranked/'); folder placement is
    enforced by validate_results.py, ranking itself stays data-driven
  * results whose id is referenced by another result's 'supersedes' field are
    excluded from tables and listed under "Superseded"
  * a Top contributors section scores live results: N points per result plus
    a one-time bonus for first submission on a given GPU within a tier
  * when writing to <root>/LEADERBOARD.md the Leaderboard section of README.md
    (between marker comments) is refreshed with the same sections; '-o' runs
    never touch the README

Usage:
  python scripts/generate_leaderboard.py                # writes ./LEADERBOARD.md
  python scripts/generate_leaderboard.py --root /path   # uses that repo root
  python scripts/generate_leaderboard.py -o out.md      # custom output path
"""

import argparse
import json
import math
import os
import re
import sys


def collect_results(root):
    """Return list of (results_dir, data) for every parsed result JSON file.

    Result files live at benchmarks/<type>/<model>/results[/<tier-or-unranked>]/*.json."""
    out = []
    bench_dir = os.path.join(root, "benchmarks")
    if not os.path.isdir(bench_dir):
        return out
    for dirpath, _dirnames, filenames in os.walk(bench_dir):
        parts = [p for p in dirpath.replace(os.sep, "/").split("/") if p]
        try:
            i = parts.index("benchmarks")
        except ValueError:
            continue
        seg = parts[i + 1:]
        # benchmarks/<type>/<model>/results or .../results/<band>
        if not (len(seg) in (3, 4) and seg[2] == "results"):
            continue
        for name in sorted(filenames):
            if not name.endswith(".json") or name.startswith("_"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                with open(fp, "r", encoding="utf-8-sig") as fh:
                    out.append((dirpath, json.load(fh)))
            except (OSError, ValueError) as exc:
                print("error: %s: cannot parse JSON: %s" % (fp, exc), file=sys.stderr)
                sys.exit(2)
    return out


def load_config(root):
    cfg_path = os.path.join(root, "config", "benchmark_types.json")
    if not os.path.isfile(cfg_path):
        print("error: missing config/benchmark_types.json", file=sys.stderr)
        sys.exit(2)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def folder_parts(results_dir):
    """Return (type_name, model_name) for a results/ dir, or (None, None)."""
    parts = [p for p in results_dir.replace(os.sep, "/").split("/") if p]
    try:
        i = parts.index("benchmarks")
        return parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        return None, None


def is_rankable(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt_num(v):
    return ("%g" % v) if is_rankable(v) else ""


def hardware_cell(h):
    gpu = h.get("gpu") or ""
    if gpu:
        return gpu
    cpu = h.get("cpu") or ""
    return "CPU only%s" % ((" (%s)" % cpu) if cpu else "")


def comment_cell(comment, limit=48):
    """Short one-line note for the table. GitHub squeezes the last column of a
    wide markdown table, so a long comment wraps one word per line and wrecks
    the row. Truncate on a word boundary; the full text stays in the JSON."""
    if not comment:
        return ""
    s = " ".join(str(comment).split())
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(";,:-")
    if len(cut) < max(12, limit // 3):
        cut = s[:limit].rstrip()
    return cut + "…"


def md_escape(s):
    return str(s).replace("|", "\\|")


def html_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_table(header, rows):
    """HTML table with nowrap cells so GitHub cannot collapse columns.

    Pipe tables on github.com wrap every cell to the container width; a long
    Comment then stacks one word per line. nowrap keeps a row on one line and
    the page scrolls horizontally instead."""
    out = ["<table>", "<thead><tr>"]
    for h in header:
        out.append("<th nowrap>%s</th>" % html_escape(h))
    out.extend(["</tr></thead>", "<tbody>"])
    n = len(header)
    for row in rows:
        cells = list(row) + [""] * (n - len(row))
        out.append("<tr>")
        for cell in cells[:n]:
            out.append("<td nowrap>%s</td>" % html_escape(cell))
        out.append("</tr>")
    out.extend(["</tbody></table>"])
    return out


def ctx_cell(d):
    s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    return fmt_num(s.get("context_length"))


def kv_cell(d):
    s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    v = s.get("kv_cache_quant")
    return str(v) if isinstance(v, str) and v else ""


def condition_key(e):
    """Identity of a run's conditions for upsert: same contributor + settings + variant."""
    d = e["data"]
    s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    v = d.get("variant") if isinstance(d.get("variant"), dict) else {}
    cond = json.dumps({"s": {k: s[k] for k in sorted(s)}, "file": v.get("file"),
                       "q": v.get("quantization")}, sort_keys=True, default=str)
    return (str(d.get("contributor", "?")).lower(), cond)


def upsert(entries, keyfn):
    """One row per identity key; the latest run by (date, id) wins. Older files stay in results/."""
    best = {}
    for e in entries:
        k = keyfn(e)
        cur = best.get(k)
        if cur is None or ((e["data"].get("date") or "", e["id"]) > (cur["data"].get("date") or "", cur["id"])):
            best[k] = e
    return list(best.values())


# ------------------------------------------------------- tier conformance ----

def snap_vram_gb(gb):
    """Snap a framebuffer total up to the marketed VRAM class. Must stay in sync
    with setup.py / run_benchmark.py ('RTX 5090 (32 GB)' reports ~31.9 GB)."""
    ceiling = int(math.ceil(gb))
    if ceiling - gb <= 0.75:  # includes exact integers (no change)
        return ceiling
    return round(gb, 1)


def tier_of(vram, tiers):
    """Tier name for a VRAM size. Tiers pin a marketed VRAM value ('vram_gb'):
    snap to the marketed class first, then match exactly. A tier without a
    pinned value is open-ended and takes everything above the highest pin."""

    def pinned(t):
        v = t.get("vram_gb")
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    s = snap_vram_gb(vram)
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


def band_label(tier, tiers=None):
    """'32 GB' for a pinned tier; '>32 GB' (vs. the highest pin) for an open one."""
    v = tier.get("vram_gb")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "%d GB" % v
    pins = [x.get("vram_gb") for x in (tiers or [])
            if isinstance(x.get("vram_gb"), (int, float))]
    return ">%d GB" % max(pins) if pins else "any VRAM"


def conformance(data, model, tcfg):
    """(tier_name|None, reason): does this live entry exactly match one pinned
    tier recipe of its type? Reason is a short display string for the table."""
    tiers = tcfg.get("tiers") or []
    if not tiers:
        return None, ""  # untiered type; caller renders the flat fallback
    hardware = data.get("hardware") if isinstance(data.get("hardware"), dict) else {}
    vram = hardware.get("vram_gb")
    if not (is_rankable(vram) and vram > 0):
        return None, "no vram_gb recorded"
    band = tier_of(vram, tiers)
    if band is None:
        pins = sorted(t["vram_gb"] for t in tiers
                      if isinstance(t.get("vram_gb"), (int, float)))
        opens = [t["name"] for t in tiers if not isinstance(t.get("vram_gb"), (int, float))]
        shown = " / ".join("%d" % p for p in pins)
        tail = "; above %d ranks as '%s'" % (pins[-1], opens[0]) if (pins and opens) else ""
        return None, "VRAM %g GB matches no pinned tier (%s%s)" % (vram, shown, tail)
    model_cfg = (tcfg.get("models") or {}).get(model) or {}
    recipes = model_cfg.get("recipes") if isinstance(model_cfg, dict) else None
    if not isinstance(recipes, dict) or not recipes:
        return None, "no pinned recipe for this model yet"
    rec = recipes.get(band)
    if not isinstance(rec, dict):
        return None, "no pinned recipe for tier '%s'" % band
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    variant = data.get("variant") if isinstance(data.get("variant"), dict) else {}
    problems = []
    for key in rec:
        got = variant.get(key) if key in ("file", "quantization") else settings.get(key)
        if got != rec[key]:
            shown = "missing" if got is None else repr(got)
            problems.append("%s=%s (pinned %r)" % (key, shown, rec[key]))
    if problems:
        return None, "%s: %s%s" % (band, ", ".join(problems[:2]),
                                   " …" if len(problems) > 2 else "")
    return band, ""


def norm_gpu(gpu):
    return re.sub(r"[^a-z0-9]", "", str(gpu or "").lower())


# ------------------------------------------------------------- rendering ----

def resolve_live(grouped):
    """Split grouped (type -> dir -> [entries]) into (live, dead) entries.

    An entry is "dead" when another entry in the same model's results tree
    (any tier subfolder) references it via 'supersedes'."""
    live, dead = [], []
    by_model = {}
    for tname in sorted(grouped):
        for dirpath in sorted(grouped[tname]):
            _t, model = folder_parts(dirpath)
            for e in grouped[tname][dirpath]:
                rec = dict(e)
                rec["_type"], rec["_model"] = tname, (model or "?")
                rec["_dir"] = dirpath
                by_model.setdefault((tname, model or "?"), []).append(rec)

    superseded_by = {}  # (type, model) -> {superseded id: replacing record}
    for key in sorted(by_model):
        repl = {}
        for r in by_model[key]:
            sup = (r["data"].get("supersedes") or "").strip()
            if sup:
                repl[sup] = r
        superseded_by[key] = repl

    for key in sorted(by_model):
        for rec in by_model[key]:
            r = superseded_by[key].get(rec["id"])
            if r is not None:
                rec["_replaced_by"] = r["id"]
                rec["_replaced_by_dir"] = r["_dir"]
                dead.append(rec)
            else:
                live.append(rec)
    return live, dead


def rank_key(e, primary):
    m = e["data"].get("metrics") or {}
    v = m.get(primary)
    return (0 if is_rankable(v) else 1, -(v if is_rankable(v) else 0.0),
            e["data"].get("date") or "9999", str(e["data"].get("contributor", "~")).lower())


def metric_defs(tcfg):
    return [m for m in tcfg.get("metrics", []) if isinstance(m, dict) and m.get("key")]


def flat_setting_cell(v):
    """One settings-column value for the untiered fallback table."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "on" if v else "off"
    if is_rankable(v):
        return "%g" % v
    return str(v)


def render_type(tname, live_entries, config_entry):
    """Render one benchmark-type section. Returns (lines, warnings)."""
    lines = ["## %s" % (config_entry.get("title", tname) if config_entry else tname), ""]
    warnings = []

    if not config_entry:
        warnings.append(
            "benchmark type '%s' has results but no entry in config/benchmark_types.json: table skipped" % tname
        )
        lines += ["_No leaderboard configured for this type yet._", ""]
        return lines, warnings

    description = config_entry.get("description")
    if description:
        lines.append(description)
        lines.append("")

    tiers = [t for t in config_entry.get("tiers", []) if isinstance(t, dict) and t.get("name")]
    primary = config_entry.get("primary_metric", "")
    mdefs = metric_defs(config_entry)

    # tier conformance decides ranked vs unranked (tiered types only)
    for e in live_entries:
        band, reason = (None, "") if not tiers else conformance(e["data"], e["_model"], config_entry)
        e["_band"], e["_reason"] = band, reason

    if not live_entries:
        lines.append("_No results yet. Be the first to submit one!_")
        lines.append("")
        return lines, warnings

    groups = {}
    for e in live_entries:
        groups.setdefault(e["_model"], []).append(e)

    def best_of(model):
        vals = [(e["data"].get("metrics") or {}).get(primary) for e in groups[model]
                if is_rankable((e["data"].get("metrics") or {}).get(primary))]
        return max(vals, default=0.0)

    for model in sorted(groups, key=lambda k: (-best_of(k), k)):
        entries = groups[model]
        ranked = [e for e in entries if tiers and e["_band"]]
        unranked = [e for e in entries if not (tiers and e["_band"])]

        if tiers:
            for ti, t in enumerate(tiers):  # fixed config order; T1..Tn label the tables
                rows = upsert([e for e in ranked if e["_band"] == t["name"]],
                              lambda e: str(e["data"].get("contributor", "?")).lower())
                rows = sorted(rows, key=lambda e: rank_key(e, primary))
                lines.append("### %s / T%d - %s (%s)" % (md_escape(model), ti + 1, md_escape(t["name"]), band_label(t, tiers)))
                lines.append("")
                if not rows:
                    lines.append("_No ranked results in this tier yet._")
                    lines.append("")
                    continue
                header = ["#", "Quant", "Contributor", "Hardware"] + [m["header"] for m in mdefs] \
                    + ["Ctx", "KV", "Date", "Comment"]
                table_rows = []
                rank = 0
                for e in rows:
                    d = e["data"]
                    m = d.get("metrics") or {}
                    if is_rankable(m.get(primary)):
                        rank += 1
                    table_rows.append(
                        [str(rank) if is_rankable(m.get(primary)) else "",
                         (d.get("variant") or {}).get("quantization", ""),
                         d.get("contributor", "?"),
                         hardware_cell(d.get("hardware") or {})]
                        + [fmt_num(m.get(x["key"])) for x in mdefs]
                        + [ctx_cell(d), kv_cell(d), d.get("date", ""), comment_cell(d.get("comment"))]
                    )
                lines.extend(render_table(header, table_rows))
                lines.append("")

            if unranked:
                # Same columns as the ranked tier tables, nothing else: the section
                # title tells the story and no rank is assigned (empty "#" cell).
                rows = sorted(upsert(unranked, condition_key), key=lambda e: rank_key(e, primary))
                lines.append("### %s / unranked" % md_escape(model))
                lines.append("")
                header = ["#", "Quant", "Contributor", "Hardware"] + [m["header"] for m in mdefs] \
                    + ["Ctx", "KV", "Date", "Comment"]
                table_rows = []
                for e in rows:
                    d = e["data"]
                    m = d.get("metrics") or {}
                    table_rows.append(
                        ["",
                         (d.get("variant") or {}).get("quantization", ""),
                         d.get("contributor", "?"),
                         hardware_cell(d.get("hardware") or {})]
                        + [fmt_num(m.get(x["key"])) for x in mdefs]
                        + [ctx_cell(d), kv_cell(d), d.get("date", ""), comment_cell(d.get("comment"))]
                    )
                lines.extend(render_table(header, table_rows))
                lines.append("")
        else:
            # flat fallback for types without tier definitions
            rows = sorted(upsert(entries, condition_key), key=lambda e: rank_key(e, primary))
            setting_defs = [s for s in config_entry.get("settings_columns", [])
                            if isinstance(s, dict) and s.get("key")]
            lines.append("### %s" % md_escape(model))
            lines.append("")
            header = ["#", "Quant", "Contributor", "Hardware"] \
                + [s["header"] for s in setting_defs] + [m["header"] for m in mdefs] \
                + ["Date", "Comment"]
            table_rows = []
            rank = 0
            for e in rows:
                d = e["data"]
                m = d.get("metrics") or {}
                s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
                if is_rankable(m.get(primary)):
                    rank += 1
                table_rows.append(
                    [str(rank) if is_rankable(m.get(primary)) else "",
                     (d.get("variant") or {}).get("quantization", ""),
                     d.get("contributor", "?"),
                     hardware_cell(d.get("hardware") or {})]
                    + [flat_setting_cell(s.get(x["key"])) for x in setting_defs]
                    + [fmt_num(m.get(x["key"])) for x in mdefs]
                    + [d.get("date", ""), comment_cell(d.get("comment"))]
                )
            lines.extend(render_table(header, table_rows))
            lines.append("")

    return lines, warnings


def result_rel_path(rec):
    """Repo-relative path of a record's file, built from the folder it lives in."""
    parts = [p for p in rec["_dir"].replace(os.sep, "/").split("/") if p]
    try:
        i = parts.index("benchmarks")
    except ValueError:
        return "results/" + rec["id"] + ".json"
    return "/".join(parts[i + 1:] + [rec["id"] + ".json"])


def render_superseded(dead):
    if not dead:
        return []
    lines = ["**Superseded** (excluded from ranking):"]
    for e in sorted(dead, key=lambda x: (x["_type"], x["_model"], x["id"])):
        repl_dir = {"_dir": e.get("_replaced_by_dir", e["_dir"]), "id": e["_replaced_by"]}
        lines.append("- ~~%s~~ - replaced by [%s](%s)" % (e["id"], e["_replaced_by"], result_rel_path(repl_dir)))
    lines.append("")
    return lines


def scoring_block(config):
    """(per_result_points, first_gpu_bonus) from the first configured block."""
    for tcfg in config.values():
        sc = (tcfg or {}).get("top_contributors") if isinstance(tcfg, dict) else None
        if isinstance(sc, dict):
            return sc.get("per_result_points", 1), sc.get("first_gpu_bonus", 0)
    return 1, 2


def render_top_contributors(live, config):
    """Score live results and render the Top contributors section."""
    per_result, first_gpu_bonus = scoring_block(config)
    tiers = next(((tcfg or {}).get("tiers") for tcfg in config.values() if (tcfg or {}).get("tiers")), None)

    points, counts, names, gpus_per_contrib = {}, {}, {}, {}
    seen_gpu_band = set()
    ordered = sorted(live, key=lambda e: (e["data"].get("date") or "", e["id"]))
    for e in ordered:
        d = e["data"]
        c = str(d.get("contributor", "?")).lower()
        names.setdefault(c, str(d.get("contributor", "?")))  # display casing; c is the match key
        points[c] = points.get(c, 0) + per_result
        counts[c] = counts.get(c, 0) + 1
        gpu = norm_gpu((d.get("hardware") or {}).get("gpu"))
        if gpu:
            gpus_per_contrib.setdefault(c, set()).add(gpu)
        vram = (d.get("hardware") or {}).get("vram_gb")
        band = tier_of(vram, tiers) if is_rankable(vram) and vram > 0 else None
        if first_gpu_bonus and band and gpu and (band, gpu) not in seen_gpu_band:
            seen_gpu_band.add((band, gpu))
            points[c] += first_gpu_bonus

    lines = ["## Top contributors", ""]
    rule_bits = ["%d pt per live result" % per_result]
    if first_gpu_bonus:
        rule_bits.append("+%d one-time for the first submission on a given GPU within a tier (VRAM pin)" % first_gpu_bonus)
    lines.append(" · ".join(rule_bits) + "; superseded results score no points.")
    lines.append("")
    header = ["#", "Contributor", "Points", "Results", "Unique GPUs"]
    table_rows = []
    for i, c in enumerate(sorted(points, key=lambda k: (-points[k], -counts.get(k, 0), k)), start=1):
        table_rows.append(
            [str(i), names[c], str(points[c]), str(counts.get(c, 0)),
             str(len(gpus_per_contrib.get(c, ())))]
        )
    lines.extend(render_table(header, table_rows))
    lines.append("")
    return lines


def render(config, root):
    collected = collect_results(root)

    grouped = {}
    for dirpath, data in collected:
        tname, _model = folder_parts(dirpath)
        if not tname:
            continue  # malformed layout; validator reports it separately
        rid = data.get("result_id") or os.path.basename(dirpath)
        grouped.setdefault(tname, {}).setdefault(dirpath, []).append({"id": rid, "data": data})

    live, dead = resolve_live(grouped)

    header = [
        "# LLM Community Benchmarks: Leaderboard",
        "",
        "Community-contributed results for local LLM inference. Every row is a single run "
        "submitted via pull request; this file is regenerated automatically on merge.",
        "",
        "Tables are upsert-style: each contributor shows their latest run per tier (or per condition in the unranked table). "
        "Submitting the same setup again replaces that row, while older files stay under results/ and still count toward points.",
        "",
        'Want to add your hardware? Read [CONTRIBUTING.md](CONTRIBUTING.md).',
        "",
    ]

    body = []
    warnings = []
    for tname in sorted(grouped):
        entries = [e for e in live if e["_type"] == tname]
        lines, warns = render_type(tname, entries, config.get(tname))
        body.extend(lines)
        warnings.extend(warns)

    # configured types that have no results yet still get a section
    for tname in sorted(set(config) - set(grouped)):
        lines, warns = render_type(tname, [], config[tname])
        body.extend(lines)
        warnings.extend(warns)

    body.extend(render_superseded(dead))

    if live:
        body.extend(render_top_contributors(live, config))

    ranked = sum(1 for e in live if e.get("_band"))
    contributors = {str(e["data"].get("contributor", "?")).lower() for e in live}
    footer = [
        "---",
        "",
        "_%d live result(s) (%d ranked, %d unranked), %d contributor(s). Generated automatically by "
        "`scripts/generate_leaderboard.py`. Do not edit this file directly._"
        % (len(live), ranked, len(live) - ranked, len(contributors)),
    ]

    text = "\n".join(header + body + footer) + "\n"

    # Same sections for the README's Leaderboard section: one heading level deeper
    # so they nest under '## Leaderboard' (tier sub-tables keep their relative levels).
    embed = ["#" + line if line.startswith("#") else line for line in body]

    if warnings:
        print("Leaderboard warnings:", file=sys.stderr)
        for w in warnings:
            print("  ! " + w, file=sys.stderr)
    return text, embed, bool(warnings)


README_BEGIN = "<!-- leaderboard:begin -->"
README_END = "<!-- leaderboard:end -->"


def update_readme_section(readme_path, body_lines):
    """Replace the lines between the leaderboard marker comments with body_lines.

    Returns True when the file content changed and was rewritten. Missing markers
    only print a warning (never fail CI): this is a cosmetic sync, not validation."""
    try:
        with open(readme_path, "r", encoding="utf-8") as fh:
            original = fh.read()
    except OSError as exc:
        print("warning: cannot read %s (%s); README leaderboard section left untouched"
              % (readme_path, exc), file=sys.stderr)
        return False
    lines = original.splitlines()
    try:
        bi = next(i for i, l in enumerate(lines) if l.strip() == README_BEGIN)
        ei = next(i for i, l in enumerate(lines[bi + 1:]) if l.strip() == README_END) + bi + 1
    except StopIteration:
        print("warning: leaderboard marker comments not found in %s; section left untouched"
              % readme_path, file=sys.stderr)
        return False
    # Replace the lines strictly BETWEEN the markers; both markers are preserved.
    new_text = "\n".join(lines[:bi + 1] + body_lines + lines[ei:]) + "\n"
    if new_text == original:
        return False
    with open(readme_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)
    return True


def disk_types(root):
    bench_dir = os.path.join(root, "benchmarks")
    if not os.path.isdir(bench_dir):
        return set()
    return {d for d in os.listdir(bench_dir)
            if os.path.isdir(os.path.join(bench_dir, d)) and not d.startswith("_")}


def _safe_streams():
    # Same guard as validate_results.py: don't crash on cp1252 consoles if any
    # warning text carries characters the codepage cannot encode.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def main(argv=None):
    _safe_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    ap.add_argument("-o", "--output", default=None, help="output path (default: <root>/LEADERBOARD.md)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    out_path = args.output or os.path.join(root, "LEADERBOARD.md")
    config = load_config(root)
    text, embed, had_warnings = render(config, root)

    unknown = disk_types(root) - set(config.keys())
    if unknown:
        print("error: benchmark type(s) missing from config/benchmark_types.json: %s"
              % ", ".join(sorted(unknown)), file=sys.stderr)

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("Wrote %s" % out_path)

    # The README section is a by-product of the publish flow only; '-o' runs (e.g. CI
    # previews into /tmp) must never touch it.
    if args.output is None and update_readme_section(os.path.join(root, "README.md"), embed):
        print("Updated the Leaderboard section of README.md")

    return 2 if (had_warnings or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
