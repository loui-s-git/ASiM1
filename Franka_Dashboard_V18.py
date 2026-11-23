#!/usr/bin/env python3
"""
Franka_Dashboard_V18.py - web dashboard for franka_sort.py
(hand-guided teach-in, tare and weighing at a weigh spot, force-guarded rest on the
testing bench and sorting of HDDs/SSDs into containers with a Franka Emika Panda
and Franka Hand, panda-py / libfranka 0.9.2, FCI).

    python3 Franka_Dashboard_V18.py      # robot unlocked in Desk, FCI active

The browser opens http://localhost:8080 automatically.
Uses the same waypoints.json as franka_sort.py, so both tools can be used side by side.
Worker view (button "Worker view" or http://localhost:8080/?view=worker): big digital twin, detected
drive, one-button Restart and STOP. Input limits follow the Franka manual (technical data): payload 3 kg incl. Hand ≈ 1 kg,
Hand grasping force up to 70 N, travel 80 mm at 30 mm/s.

Logging (since V5): every session writes to the log folder (default ./logs), file prefix dashboard_<stamp>:
  .log            everything incl. debug details, ms timestamps, header with versions, settings and waypoints
  .csv            one row per part (same first columns as franka_sort.py)
  _steps.csv      start, end and duration of every step of every part
  _weighing.csv   raw force samples of the tare and load measurement
  _bench.csv      bench descent curve (descent vs. rest load)
  _summary.md     KPIs for the evaluation: cycle times, throughput, accuracy, confusion matrix, interventions

Two files (keep them in the same folder; start this one):
  Franka_Dashboard_V18.py  digital twin and evaluation: 3D twin, telemetry, charts, confusion matrix, KPIs,
                           results table, session summary, worker view; builds the page and starts it
  Franka_Robot_V18.py      sorting and robot setup: constants, settings, kinematics, classification,
                           controller (robot, jobs, sorting cycle, teach-in) and their UI panels

Sections:  1 display constants · 2 evaluation (metrics, session summary)
           3 web UI (digital twin, charts, confusion matrix, page) · 4 main
"""
from __future__ import annotations

import contextlib
import json
import math
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nicegui import app, ui

from Franka_Robot_V18 import (  # sorting and robot setup (second file of this dashboard)
    ALL_PATHS, APP_TITLE, card, classify, Controller, DH, DRIVE_MODEL, DRIVE_MODEL_NOTE, EMPTY, EXPECTED, fk, FLANGE,
    HOME_CLASS, LABEL, MEASURED, PATH_CLASS, PATH_LABEL, PRED_COLS, Q_LOWER, Q_UPPER, RobotPanels,
    Settings, SETTINGS_FILE, SORTED, STATUS, STATUS_TEXT, STEP_SETS, TRUTH_ROWS,
)


# ═══════════════════════════════════════════════════════════════════ 1. display constants
BENCH_DRAW_DROP = 0.08     # twin: bench surface drawn this far below the taught approach pose

# Validated categorical palette (slots 1-3 pass all-pairs CVD checks in both modes); reject = neutral gray.
THEMES = {
    "dark": dict(page="#0d0d0d", surface="#1a1a19", raised="#242422", ink="#f0efec", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
                 ssd="#3987e5", hdd25="#d95926", hdd35="#199e70", other="#898781", accent="#9085e9",
                 warn_ink="#fab219", good_ink="#0ca30c", scene="#141413", link="#e4e4df", joint="#3c3c39"),
    "light": dict(page="#f4f4f1", surface="#fcfcfb", raised="#efeee9", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
                  ssd="#2a78d6", hdd25="#eb6834", hdd35="#1baf7a", other="#898781", accent="#6250d6",
                  warn_ink="#8a5a00", good_ink="#006300", scene="#e6e6e1", link="#fbfbf9", joint="#4a4a46"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
_REGION_CACHE: Dict[Tuple, List[List[Any]]] = {}   # V18: model decision regions of the scatter chart


# ═══════════════════════════════════════════════════════════════════ 2. evaluation (metrics, session summary)
def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confusion matrix (actual x decision), accuracy, per-class / macro F1 over labelled cycles."""
    cm = {t: {p: 0 for p in PRED_COLS} for t in TRUTH_ROWS}
    n = correct = 0
    for r in results:
        t = r.get("truth")
        if r.get("mode") == "dry" or t not in cm:
            continue
        if r["status"] in MEASURED and r.get("pred") in PRED_COLS:
            p = r["pred"]
        elif r["status"] == EMPTY:
            p = "none"
        else:
            continue
        cm[t][p] += 1
        n += 1
        correct += int(EXPECTED[t] == p)
    f1: Dict[str, Optional[float]] = {}
    for c in ["ssd", "hdd25", "hdd35", "reject"]:
        t = "other" if c == "reject" else c
        tp = cm[t][c]
        fp = sum(cm[x][c] for x in TRUTH_ROWS if x != t)
        fn = sum(cm[t][p] for p in PRED_COLS if p != c)
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    vals = [v for v in f1.values() if v is not None]
    recall = {t: (cm[t][EXPECTED[t]] / sum(cm[t].values()) if sum(cm[t].values()) else None) for t in TRUTH_ROWS}
    return dict(cm=cm, n=n, correct=correct, acc=correct / n if n else None, f1=f1,
                macro_f1=sum(vals) / len(vals) if vals else None, recall=recall,
                security=cm["hdd25"]["ssd"] + cm["hdd35"]["ssd"])


def _stats(vals: List[float]) -> Optional[Tuple[float, float, float, float, float]]:
    """mean, sd, median, min, max — None for an empty list."""
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, float(np.median(a)), float(a.min()), \
        float(a.max())


class DashboardController(Controller):
    """The Controller from Franka_Robot_V18.py plus the session evaluation (KPIs, confusion matrix)."""

    def summary_md(self) -> str:
        """KPIs of this session as Markdown (→ dashboard_<stamp>_summary.md, rewritten after every part)."""
        s = self.s
        rs = [r for r in self.results if r.get("mode") != "dry"]
        done = [r for r in rs if r.get("mode") == "sort" and r["status"] in SORTED]

        def num(v: Optional[float], nd: int = 1) -> str:
            return "—" if v is None else f"{v:.{nd}f}"

        def pct(a: int, b: int) -> str:
            return f"{a}/{b} ({a / b * 100:.0f} %)" if b else "—"

        def epoch(iso: str) -> Optional[float]:
            try:
                return datetime.fromisoformat(iso).timestamp()
            except (TypeError, ValueError):
                return None

        st_count = {k: sum(1 for r in rs if r["status"] == k) for k in STATUS_TEXT}
        dry = len({x["part"] for x in self.step_log if x["mode"] == "dry"})
        tags = sorted({r["tag"] for r in rs if r.get("tag")})
        L = [f"# Session dashboard_{self.session} — summary", "",
             f"{APP_TITLE} · session start {datetime.fromtimestamp(self.session_t0):%Y-%m-%d %H:%M:%S} · "
             f"updated {datetime.now():%Y-%m-%d %H:%M:%S} · robot {s.robot_ip}"
             + (f" · trial tags {', '.join(tags)}" if tags else ""), "",
             "## Overview", "", "| Metric | Value |", "|---|---|",
             f"| Parts handled (sort + measure only) | {len(rs)} (sort {sum(r['mode'] == 'sort' for r in rs)}, "
             f"measure only {sum(r['mode'] == 'measure' for r in rs)}) |",
             f"| Sorted into a container | {len(done)} |",
             f"| … with bench contact | {pct(sum(r['status'] == 'ok' for r in done), len(done))} |",
             f"| Holder empty (no grasp) | {st_count.get(EMPTY, 0)} |",
             f"| Lost at bench / error / stopped | {st_count.get('part_lost_at_bench', 0)} / "
             f"{st_count.get('error', 0)} / {st_count.get('aborted', 0)} |",
             f"| Dry runs (not in the results) | {dry} |", ""]
        # ---- cycle time & throughput
        ct = _stats([float(r["cycle_s"]) for r in done if r.get("cycle_s") is not None])
        L += ["## Cycle time (sorted parts)", ""]
        if ct:
            L += ["| | mean | sd | median | min | max |", "|---|---|---|---|---|---|",
                  "| Cycle time [s] | " + " | ".join(num(v) for v in ct) + " |", ""]
            t_a = [epoch(r["ts"]) for r in done]
            t_b = [epoch(r.get("t_end", "")) for r in done]
            span = (max(x for x in t_b if x) - min(x for x in t_a if x)) if any(t_a) and any(t_b) else 0.0
            L.append(f"Throughput: **{3600 / ct[0]:.0f} parts/h** from the mean cycle time"
                     + (f" · {len(done) / (span / 3600):.0f} parts/h over the session (first start → last end, "
                        f"incl. pauses)" if span > 0 else ""))
        else:
            L.append("No sorted parts yet.")
        L.append("")
        # ---- time per step
        L += ["## Time per step (sorted parts)", ""]
        per: Dict[str, List[float]] = {}
        for x in self.step_log:
            if x["mode"] == "sort" and x["part_status"] in SORTED and x["status"] == "done" and x["dur"] is not None:
                per.setdefault(x["step"], []).append(float(x["dur"]))
        if per and ct:
            L += ["| Step | mean [s] | sd [s] | share of the cycle |", "|---|---|---|---|"]
            for key, lbl in STEP_SETS["sort"]:
                st = _stats(per.get(key, []))
                if st:
                    L.append(f"| {lbl} | {num(st[0], 2)} | {num(st[1], 2)} | {st[0] / ct[0] * 100:.0f} % |")
        else:
            L.append("No step times yet.")
        L.append("")
        # ---- classification
        mt = compute_metrics(self.results)
        L += ["## Classification (parts with an 'Actual' label)", ""]
        if mt["n"]:
            L += [f"Accuracy **{pct(mt['correct'], mt['n'])}** · macro F1 {num(mt['macro_f1'], 2)} · "
                  f"HDDs classified as SSD: {mt['security']}", "",
                  "| Actual ↓ / Decision → | " + " | ".join(LABEL[p] for p in PRED_COLS) + " | Recall |",
                  "|---|" + "---|" * (len(PRED_COLS) + 1)]
            for t in TRUTH_ROWS:
                if sum(mt["cm"][t].values()):
                    rc = mt["recall"][t]
                    L.append(f"| {LABEL[t]} | " + " | ".join(str(mt["cm"][t][p]) for p in PRED_COLS)
                             + f" | {'—' if rc is None else f'{rc * 100:.0f} %'} |")
        else:
            L.append("No labelled parts yet — pick the actual part in 'Actual' (results table) to get accuracy and "
                     "the confusion matrix.")
        L.append("")
        # ---- measurements
        meas = [r for r in rs if r["status"] in MEASURED and r.get("mass_g") is not None]
        L += ["## Measurements", ""]
        if meas:
            L += ["| Group | n | mass [g] mean ± sd (min–max) | width [mm] mean ± sd (min–max) |",
                  "|---|---|---|---|"]
            groups = [("Decision " + LABEL[c], [r for r in meas if r["pred"] == c]) for c in PRED_COLS] + \
                     [("Actual " + LABEL[t], [r for r in meas if r.get("truth") == t]) for t in TRUTH_ROWS]
            for name, grp in groups:
                m_ = _stats([r["mass_g"] for r in grp])
                w_ = _stats([r["width_mm"] for r in grp if r.get("width_mm") is not None])
                if m_:
                    L.append(f"| {name} | {len(grp)} | {m_[0]:.1f} ± {m_[1]:.1f} ({m_[3]:.0f}–{m_[4]:.0f}) | "
                             + (f"{w_[0]:.2f} ± {w_[1]:.2f} ({w_[3]:.1f}–{w_[4]:.1f})" if w_ else "—") + " |")
            fs = _stats([r["f_std"] for r in meas if r.get("f_std") is not None])
            if fs:
                L += ["", f"Force noise while weighing (max σ of tare/load): mean {fs[0]:.3f} N, max {fs[4]:.3f} N"]
        else:
            L.append("No measured parts yet.")
        L.append("")
        # ---- interventions
        c = self.counters
        unplanned = c.get("stop", 0) + c.get("worker_restart", 0) + c.get("recover", 0)
        L += ["## Interventions & events", "", "| Event | Count |", "|---|---|",
              f"| STOP pressed | {c.get('stop', 0)} |",
              f"| Restart in the worker view | {c.get('worker_restart', 0)} |",
              f"| … part put back into the holder after a restart | {c.get('put_back_after_restart', 0)} |",
              f"| Recover errors (manual) | {c.get('recover', 0)} |",
              f"| Automatic recovery after an error | {c.get('auto_recover', 0)} |",
              f"| Bench 'Continue' (worker confirms the test) | {c.get('bench_continue', 0)} |",
              f"| Worker feedback 'correct' / 'wrong' | {c.get('feedback_correct', 0)} / "
              f"{c.get('feedback_wrong', 0)} |",
              f"| Auto mode started | {c.get('auto_start', 0)} |",
              f"| Tare measurements (empty gripper at the weigh spot) | {c.get('tare', 0)} |",
              f"| Robot connections | {c.get('connect', 0)} |",
              f"| Warnings / errors in the log | {c.get('warnings', 0)} / {c.get('errors', 0)} |", "",
              f"Unplanned interventions (STOP + Restart + Recover): **{unplanned}**"
              + (f" → {unplanned / len(done) * 10:.1f} per 10 sorted parts" if done else ""), ""]
        # ---- settings
        L += ["## Key settings (end of session — all settings are in the .log header)", "",
              f"speed {s.speed:g} · home speed {s.home_speed:g} · grasp force {s.grasp_force_n:g} N · "
              f"weighing {s.measure_s:g} s (settle {s.settle_s:g} s, {s.force_mode}) at the weigh spot, tare "
              + ("before every part" if s.tare_every_part else
                 f"stored (renewed after {s.tare_max_age_min:g} min)" if s.tare_max_age_min > 0 else "stored")
              + f" · bench contact ≤ {s.contact_g:g} g and ≤ {s.contact_pct:g} % of the part, "
              f"descent {s.descent_speed_mms:g} mm/s ({s.descent_method}), "
              + ("gripper opened during the test" if s.bench_release else "part held during the test") + ", "
              + ("test until Continue" if s.bench_wait else f"test {s.bench_pause_s:g} s")
              + f" · classifier: trained model ({DRIVE_MODEL_NOTE}), SSD only from {s.ssd_min_conf_pct:g} % "
              "model probability"
              + (" · reject check on" if s.reject_enabled else "")
              + " · taught exit path out of each bench · own way home from every container", "",
              "## Files of this session", "",
              f"- `dashboard_{self.session}.log` — full event log incl. debug details and settings header",
              f"- `dashboard_{self.session}.csv` — one row per part",
              f"- `dashboard_{self.session}_steps.csv` — start, end and duration of every step",
              f"- `dashboard_{self.session}_weighing.csv` — raw force samples (tare / load)",
              f"- `dashboard_{self.session}_bench.csv` — bench descent curve (descent vs. rest load)", ""]
        return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════ 3. web UI (digital twin, charts, page)
CSS = r"""
html, body { background: var(--page) !important; }
body { color: var(--ink); font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       font-size: 13px; -webkit-font-smoothing: antialiased; }
.nicegui-content { padding: 0 !important; gap: 0 !important; }
.q-header.topbar { background: var(--surface) !important; color: var(--ink) !important; border-bottom: 1px solid var(--border);
       padding: 10px 18px; gap: 14px; flex-wrap: nowrap; }
.brand-mark { width: 36px; height: 36px; border-radius: 11px; background: var(--ink); color: var(--page);
       display: flex; align-items: center; justify-content: center; flex: none; }
.brand-title { font-size: 15.5px; font-weight: 650; letter-spacing: -.01em; line-height: 1.15; }
.brand-sub { font-size: 11.5px; color: var(--muted); line-height: 1.2; }
.pill { display: inline-flex; align-items: center; gap: 7px; height: 30px; padding: 0 12px; border-radius: 999px;
       border: 1px solid var(--border); background: var(--raised); font-size: 12px; color: var(--ink2); white-space: nowrap; }
.pill .q-icon { font-size: 16px; color: var(--muted); }
.pill.chip-real { color: var(--ink); font-weight: 650; letter-spacing: .04em; }
.pill.busy .q-icon { color: var(--accent); animation: spin 1.1s linear infinite; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex: none; }
.dot.good { background: var(--good); box-shadow: 0 0 0 3px color-mix(in srgb, var(--good) 25%, transparent); }
.dot.warn { background: var(--warning); animation: pulse 1.2s ease-in-out infinite; }
.dot.bad { background: var(--critical); }
@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .3 } }
@keyframes spin { to { transform: rotate(360deg) } }
.btn-stop { background: var(--critical) !important; color: #fff !important; font-weight: 750; letter-spacing: .06em;
       border-radius: 11px; height: 40px; padding: 0 18px; box-shadow: 0 0 0 4px color-mix(in srgb, var(--critical) 22%, transparent); }
.icon-btn { color: var(--ink2) !important; }

.layout { display: grid; grid-template-columns: 340px minmax(0, 1fr) 440px; gap: 16px; padding: 16px;
       width: 100%; max-width: 2400px; margin: 0 auto; align-items: start; }
.dcol { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.dcol > * { flex-shrink: 0; }
.col-left { position: sticky; top: 76px; max-height: calc(100vh - 92px); overflow-y: auto; overflow-x: hidden;
       scrollbar-width: thin; padding-right: 2px; }
@media (max-width: 1580px) {
  .layout { grid-template-columns: 330px minmax(0, 1fr); }
  .col-right { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); align-items: start; }
}
@media (max-width: 1050px) {
  .layout { grid-template-columns: minmax(0, 1fr); padding: 10px; }
  .col-left { position: static; max-height: none; overflow: visible; }
  .col-right { grid-template-columns: minmax(0, 1fr); }
  .hdr-pills { display: none !important; }
}

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px 16px;
       min-width: 0; box-shadow: 0 1px 2px rgba(0,0,0,.05); }
.card-head { gap: 10px; margin-bottom: 12px; flex-wrap: nowrap; }
.card-icon { font-size: 18px; color: var(--muted); }
.card-title { font-size: 13.5px; font-weight: 650; color: var(--ink); line-height: 1.2; }
.card-sub { font-size: 11.5px; color: var(--muted); line-height: 1.25; }
.muted { color: var(--muted); }
.hint { font-size: 11.5px; color: var(--muted); line-height: 1.35; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
.field-label { font-size: 11.5px; color: var(--muted); font-weight: 550; }
.field-value { font-size: 12px; color: var(--ink); }
.sub-head { font-size: 11px; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); margin: 6px 0 2px; }
.ellipsis { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.grow { flex: 1 1 auto; min-width: 0; }

.section.q-expansion-item { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
.section .q-expansion-item__container > .q-item { min-height: 48px; padding: 6px 14px; }
.section .q-item__label { font-weight: 650; font-size: 13.5px; color: var(--ink); }
.section .q-item__label--caption { font-weight: 400; font-size: 11.5px; color: var(--muted); }
.section .q-item__section--avatar { min-width: 30px; color: var(--muted); }
.section .q-expansion-item__content { padding: 2px 14px 14px; }
.fields { width: 100%; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px 10px; align-items: start; }
.fields > * { width: 100%; min-width: 0; }
.fields > .full { grid-column: 1 / -1; }
.q-field--outlined .q-field__control:before { border-color: var(--border) !important; }
.q-field--dense .q-field__label { font-size: 12px; }
.q-field__suffix { color: var(--muted) !important; font-size: 11.5px; }
.slider-row .q-slider { margin-top: -2px; }

.seg { border: 1px solid var(--border); border-radius: 11px; overflow: hidden; background: var(--raised); }
.seg .q-btn { color: var(--ink2); font-weight: 550; }
.seg .q-btn.bg-primary { color: #fff !important; }
.seg-lg .q-btn { min-height: 40px; font-size: 13px; }

.btn-primary { background: var(--ink) !important; color: var(--page) !important; border-radius: 11px; font-weight: 650; }
.btn-accent { background: var(--accent) !important; color: #fff !important; border-radius: 11px; font-weight: 650; }
.btn-outline { color: var(--ink) !important; border-radius: 11px; font-weight: 550; }
.btn-outline:before { border-color: var(--border) !important; }
.btn-xl { height: 48px; padding: 0 20px; font-size: 14.5px; border-radius: 13px; }
.btn-lg { height: 40px; }
.q-btn.disabled { opacity: .38 !important; }
.action-grid .q-btn { width: 100%; justify-content: flex-start; }

.cycle-actions { flex-wrap: wrap; }
.elapsed { font-size: 20px; font-weight: 650; color: var(--ink); }
.ready { display: flex; flex-wrap: wrap; gap: 6px; }
.rchip { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; padding: 3px 10px 3px 7px;
       border-radius: 999px; border: 1px solid var(--border); color: var(--ink2); background: var(--surface); }
.rchip .q-icon { font-size: 15px; }
.rchip.ok .q-icon { color: var(--good); }
.rchip.bad { color: var(--ink); border-color: color-mix(in srgb, var(--warning) 60%, transparent);
       background: color-mix(in srgb, var(--warning) 9%, var(--surface)); }
.rchip.bad .q-icon { color: var(--warn-ink); }
.rchip .rdetail { color: var(--muted); }
.stepper { display: flex; flex-wrap: wrap; gap: 6px; }
.step { display: flex; align-items: center; gap: 6px; padding: 7px 11px; border-radius: 11px; border: 1px solid var(--border);
       background: var(--raised); color: var(--muted); font-size: 12px; transition: all .25s ease; }
.step .q-icon { font-size: 16px; }
.step .step-t { font-variant-numeric: tabular-nums; font-size: 11px; color: var(--muted); }
.step.active { color: var(--ink); border-color: var(--accent); background: color-mix(in srgb, var(--accent) 16%, var(--surface));
       box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent); }
.step.active .q-icon { color: var(--accent); animation: spin 1s linear infinite; }
.step.done { color: var(--ink2); }
.step.done .q-icon { color: var(--good); }
.step.failed { color: var(--ink); border-color: var(--critical); }
.step.failed .q-icon { color: var(--critical); }
.step.skipped { opacity: .4; }
.step.skipped span:not(.step-t) { text-decoration: line-through; }
.progress .q-linear-progress__track { opacity: .25; }

.result-grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: 22px; }
@media (max-width: 1250px) { .result-grid { grid-template-columns: minmax(0, 1fr); } }
.hero { font-size: 52px; font-weight: 680; letter-spacing: -.03em; line-height: 1; color: var(--ink); }
.hero-swatch { width: 14px; height: 50px; border-radius: 5px; background: var(--grid); flex: none; }
.stat-label { font-size: 11.5px; color: var(--muted); }
.stat-value { font-size: 26px; font-weight: 620; letter-spacing: -.01em; color: var(--ink); line-height: 1.15; }
.stat-unit { font-size: 13px; color: var(--muted); margin-left: 3px; font-weight: 500; }
.reason { font-size: 12.5px; color: var(--ink2); line-height: 1.4; }
.check { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink); }
.check .q-icon { font-size: 18px; }
.measure-chart { height: 230px; width: 100%; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 10px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; min-width: 0; }
.kpi-label { font-size: 11.5px; color: var(--muted); display: flex; align-items: center; gap: 5px; }
.kpi-label .q-icon { font-size: 14px; }
.kpi-value { font-size: 25px; font-weight: 620; letter-spacing: -.01em; margin-top: 2px; color: var(--ink); line-height: 1.2; }
.kpi-sub { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi.alert { border-color: color-mix(in srgb, var(--critical) 60%, transparent); }
.kpi.alert .q-icon { color: var(--critical); }

.analytics { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; }
.scatter-chart { height: 340px; width: 100%; }
.cm { display: grid; gap: 5px; font-size: 12px; align-items: center; }
.cm-corner { font-size: 10.5px; color: var(--muted); line-height: 1.2; }
.cm-head { font-size: 11px; color: var(--ink2); display: flex; flex-direction: column; align-items: center; gap: 4px; text-align: center; line-height: 1.15; }
.cm-rowlab { font-size: 12px; color: var(--ink2); display: flex; align-items: center; gap: 7px; }
.cm-cell { height: 44px; border-radius: 9px; display: flex; align-items: center; justify-content: center; gap: 3px;
       font-weight: 650; font-size: 14px; font-variant-numeric: tabular-nums; background: var(--raised); position: relative; }
.cm-cell.expected { box-shadow: inset 0 0 0 1.5px var(--ink2); }
.cm-cell .q-icon { font-size: 14px; color: var(--critical); }
.cm-recall { font-size: 12px; color: var(--ink2); text-align: right; font-variant-numeric: tabular-nums; }
.swatch-sm { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: none; }
.swatch-ring { box-shadow: inset 0 0 0 1.5px var(--muted); background: transparent !important; }

.results-table { background: transparent !important; color: var(--ink); }
.results-table thead tr th { color: var(--muted); font-weight: 600; font-size: 11.5px; }
.results-table tbody td { font-size: 12.5px; font-variant-numeric: tabular-nums; }
.results-table .q-table__bottom { color: var(--muted); }
.tbadge { display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border); }

.twin-wrap { position: relative; border-radius: 13px; overflow: hidden; border: 1px solid var(--border); background: var(--scene); }
.twin-scene { width: 100% !important; height: 310px; display: block; }
.twin-overlay { position: absolute; left: 10px; bottom: 9px; right: 10px; display: flex; flex-wrap: wrap; gap: 6px; pointer-events: none; }
.tag { font-size: 11px; padding: 3px 9px; border-radius: 999px; background: color-mix(in srgb, var(--surface) 82%, transparent);
       border: 1px solid var(--border); color: var(--ink2); font-variant-numeric: tabular-nums; backdrop-filter: blur(6px); }
.tag.live { color: var(--ink); border-color: color-mix(in srgb, var(--accent) 60%, transparent); }

.jrow { width: 100%; display: grid; grid-template-columns: 24px minmax(0, 1fr) 62px; gap: 10px; align-items: center; height: 21px; }
.jname { font-size: 11.5px; color: var(--muted); font-weight: 650; }
.meter { position: relative; height: 6px; border-radius: 999px; background: var(--raised); }
.meter .zone { position: absolute; top: 0; bottom: 0; width: 9%; background: color-mix(in srgb, var(--warning) 28%, transparent); }
.meter .zone.lo { left: 0; border-radius: 999px 0 0 999px; }
.meter .zone.hi { right: 0; border-radius: 0 999px 999px 0; }
.meter .mark { position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%; background: var(--ink);
       transform: translate(-50%, -50%); border: 2px solid var(--surface); transition: left .12s linear; }
.jrow.warn .mark { background: var(--warning); }
.jrow.crit .mark { background: var(--critical); }
.jval { font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.mini-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.mini { background: var(--raised); border-radius: 10px; padding: 7px 10px; }
.mini-l { font-size: 10.5px; color: var(--muted); }
.mini-v { font-size: 15px; font-weight: 620; font-variant-numeric: tabular-nums; color: var(--ink); }
.gbar { position: relative; height: 10px; border-radius: 999px; background: var(--raised); overflow: hidden; }
.gbar .gfill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--ink2); border-radius: 999px; transition: width .2s ease; }
.gbar .gfill.held { background: var(--accent); }
.force-chart { height: 150px; width: 100%; }

.teach-banner { display: flex; gap: 10px; align-items: center; border-radius: 12px; padding: 10px 12px; font-size: 12.5px;
       background: color-mix(in srgb, var(--warning) 14%, var(--surface)); border: 1px solid color-mix(in srgb, var(--warning) 55%, transparent); }
.teach-banner .q-icon { color: var(--warn-ink); font-size: 20px; }
.path-row { width: 100%; box-sizing: border-box; display: grid; grid-template-columns: 12px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 8px 11px;
       border-radius: 11px; border: 1px solid var(--border); cursor: pointer; transition: background .15s ease; }
.path-row:hover { background: var(--raised); }
.path-row.sel { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 10%, var(--surface)); }
.path-dot { width: 10px; height: 10px; border-radius: 50%; }
.path-name { font-weight: 650; font-size: 12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
.path-hint { font-size: 11px; color: var(--muted); line-height: 1.25; }
.badge { font-size: 11px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--ink2); white-space: nowrap; }
.badge.ok { color: var(--ink); }
.badge.miss { color: var(--warn-ink); border-color: color-mix(in srgb, var(--warning) 55%, transparent); }
.draft { border-radius: 12px; border: 1px dashed var(--border); padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
.draft-title { font-size: 12.5px; font-weight: 600; color: var(--ink); }
.draft-pts { max-height: 92px; overflow-y: auto; white-space: pre; font-size: 11px; }

.logview { height: 270px; background: var(--raised) !important; border-radius: 12px; border: 1px solid var(--border);
       font-size: 11.5px; padding: 8px 10px; line-height: 1.5; }
.lv-debug { color: var(--muted); }
.lv-info { color: var(--ink2); }
.lv-ok { color: var(--good-ink); }
.lv-warn { color: var(--warn-ink); }
.lv-error { color: var(--critical); font-weight: 600; }
.bench-chart { height: 170px; width: 100%; }
.bench-wait { border-radius: 12px; padding: 10px 12px; border: 1px solid var(--accent); font-size: 12.5px;
       background: color-mix(in srgb, var(--accent) 12%, var(--surface)); }
.bench-wait .q-icon { color: var(--accent); font-size: 20px; }
.q-notification { border-radius: 12px !important; }
body.twin-max { overflow: hidden; }
body.twin-max .twin-wrap { position: fixed; inset: 0; z-index: 5000; border-radius: 0; border: none; margin: 0; }
body.twin-max .twin-scene { width: 100vw !important; height: 100vh !important; }
.twin-hud { display: none; }
body.twin-max .twin-hud { display: flex; position: absolute; top: 18px; left: 18px; right: 18px; z-index: 3;
       justify-content: space-between; align-items: flex-start; gap: 16px; pointer-events: none; }
body.twin-max .twin-hud > * { pointer-events: auto; }
body.twin-max .twin-overlay { left: 18px; bottom: 16px; }
body.twin-max .tag { font-size: 13px; padding: 5px 12px; }
.hud-card { background: color-mix(in srgb, var(--surface) 86%, transparent); border: 1px solid var(--border);
       border-radius: 16px; padding: 14px 18px; backdrop-filter: blur(10px); min-width: 300px; }
.hud-title { font-size: 11.5px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.hud-line { font-size: 13px; color: var(--ink2); margin-top: 2px; }
.hud-class { font-size: 42px; font-weight: 680; letter-spacing: -.02em; color: var(--ink); line-height: 1.1; }
.hud-swatch { width: 12px; height: 40px; border-radius: 5px; background: var(--grid); flex: none; }
.hud-exit { background: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--border); }
body.worker-mode { overflow: hidden; }
body.worker-mode .twin-wrap { position: fixed; inset: 0; z-index: 5000; border-radius: 0; border: none; margin: 0; }
body.worker-mode .twin-scene { width: 100vw !important; height: 100vh !important; }
body.worker-mode .twin-hud, body.worker-mode .twin-overlay { display: none !important; }
.worker-ui { display: none; }
body.worker-mode .worker-ui { display: block; position: absolute; inset: 0; z-index: 4; pointer-events: none; }
body.worker-mode .wk-top, body.worker-mode .wk-drive, body.worker-mode .wk-actions { pointer-events: auto; }
.wk-top { position: absolute; top: 22px; left: 24px; right: 24px; display: flex; align-items: center;
       justify-content: space-between; gap: 16px; }
.wk-status { display: inline-flex; align-items: center; gap: 14px; padding: 14px 22px; border-radius: 999px;
       font-size: 22px; font-weight: 650; color: var(--ink); background: color-mix(in srgb, var(--surface) 88%, transparent);
       border: 1px solid var(--border); backdrop-filter: blur(10px); }
.wk-status .dot { width: 14px; height: 14px; }
.wk-drive { position: absolute; left: 24px; bottom: 24px; min-width: 440px; max-width: 52vw; padding: 22px 28px 24px;
       border-radius: 26px; background: color-mix(in srgb, var(--surface) 90%, transparent); border: 1px solid var(--border);
       backdrop-filter: blur(12px); }
.wk-drive-label { font-size: 15px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.wk-drive-class { font-size: clamp(56px, 7vw, 108px); font-weight: 720; letter-spacing: -.03em; line-height: 1.04; color: var(--ink); }
.wk-drive-swatch { width: 18px; border-radius: 8px; background: var(--grid); flex: none; }
.wk-drive-sub { font-size: 22px; color: var(--ink2); margin-top: 6px; }
.wk-actions { position: absolute; right: 24px; bottom: 24px; display: flex; flex-direction: column; gap: 14px;
       width: min(380px, 40vw); }
.wk-btn { height: 92px !important; border-radius: 22px !important; font-size: 28px !important; font-weight: 720 !important;
       letter-spacing: .02em; width: 100%; }
.wk-btn .q-icon { font-size: 36px !important; }
.wk-restart { background: var(--good) !important; color: #fff !important; }
.wk-stop { background: var(--critical) !important; color: #fff !important; }
.wk-continue { background: var(--accent) !important; color: #fff !important; }
.wk-switch { background: color-mix(in srgb, var(--surface) 88%, transparent) !important; color: var(--ink2) !important;
       border: 1px solid var(--border); border-radius: 14px !important; min-height: 48px; }
.wk-fb { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border); }          /* V18: feedback */
.wk-fb-q { font-size: 19px; font-weight: 600; color: var(--ink2); margin-bottom: 12px; }
.wk-fb-row { gap: 12px; }
.wk-fb-btn { height: 64px !important; border-radius: 18px !important; font-size: 22px !important;
       font-weight: 700 !important; flex: 1 1 0; min-width: 120px; }
.wk-fb-btn .q-icon { font-size: 28px !important; }
.wk-fb-ok { background: var(--good) !important; color: #fff !important; }
.wk-fb-wrong { background: var(--raised) !important; color: var(--ink) !important; border: 2px solid var(--critical); }
.wk-fb-type { background: var(--raised) !important; color: var(--ink) !important; border: 1px solid var(--border); }
.wk-fb-back { background: transparent !important; color: var(--ink2) !important; border: 1px solid var(--border);
       flex: 0 0 64px; min-width: 64px; }
@media (max-width: 900px) {
  .wk-status { font-size: 16px; }
  .wk-drive { left: 12px; right: 12px; bottom: 236px; min-width: 0; max-width: none; }
  .wk-actions { left: 12px; right: 12px; bottom: 12px; width: auto; }
}
"""


# Digital twin full screen: page fullscreen + the twin card covers it (dialogs and toasts stay visible).
# Esc leaves full screen; the server is told late (300 ms) so that Esc is never taken as STOP.
FULLSCREEN_JS = """<script>
window.franka_resize = function () {
  [80, 400].forEach(function (t) { setTimeout(function () { window.dispatchEvent(new Event('resize')); }, t); });
};
window.pageFullscreen = function (on) {
  var el = document.documentElement;
  if (on && el.requestFullscreen && !document.fullscreenElement) { el.requestFullscreen().catch(function () {}); }
  if (!on && document.fullscreenElement && document.exitFullscreen) { document.exitFullscreen().catch(function () {}); }
  window.franka_resize();
  if (on) { emitEvent('twin_fs', true); } else { setTimeout(function () { emitEvent('twin_fs', false); }, 300); }
};
window.twinFullscreen = function (on) {
  document.body.classList.toggle('twin-max', !!on);
  window.pageFullscreen(on);
};
window.setWorkerMode = function (on) {                       // worker view; remembered across reloads
  document.body.classList.toggle('worker-mode', !!on);
  try { localStorage.setItem('franka-worker-mode', on ? '1' : '0'); } catch (e) {}
  window.franka_resize();
  emitEvent('worker_mode', !!on);
};
document.addEventListener('DOMContentLoaded', function () {
  var on = false;
  try { on = localStorage.getItem('franka-worker-mode') === '1'; } catch (e) {}
  if (/[?&]view=worker\\b/.test(location.search)) { on = true; }
  if (on) {
    document.body.classList.add('worker-mode');
    window.franka_resize();
    setTimeout(function () { emitEvent('worker_mode', true); }, 1200);
  }
});
document.addEventListener('fullscreenchange', function () {
  if (!document.fullscreenElement) {
    if (document.body.classList.contains('twin-max')) { window.twinFullscreen(false); }
    else { setTimeout(function () { emitEvent('twin_fs', false); }, 300); }
  }
  setTimeout(function () { window.dispatchEvent(new Event('resize')); }, 80);
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && document.body.classList.contains('twin-max')) { window.twinFullscreen(false); }
});
</script>"""


def _css_vars(th: Dict[str, str]) -> str:
    items = {**{k.replace("_", "-"): v for k, v in th.items()}, **STATUS}
    return ":root{" + ";".join(f"--{k}:{v}" for k, v in items.items()) + "}"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _rgb01(hex_color: str) -> List[float]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _align_y(u: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the y axis (three.js cylinder axis) onto unit vector u."""
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(y, u), float(np.dot(y, u))
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class Twin:
    """Live 3D twin: forward kinematics of the Panda as nested groups, gripper, taught paths and bins."""

    def __init__(self, scene: Any, th: Dict[str, str]):
        self.scene, self.th = scene, th
        self.joints: List[Any] = []
        self.fingers: List[Any] = []
        self.path_objs: List[Any] = []
        with scene:
            self._build()
        self.reset_camera()

    def reset_camera(self) -> None:
        self.scene.move_camera(x=1.02, y=-0.88, z=0.78, look_at_x=0.33, look_at_y=0.0, look_at_z=0.26,
                               up_x=0, up_y=0, up_z=1, duration=0.6)

    def _seg(self, p0, p1, r: float, color: str) -> None:
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L > 1e-6:
            self.scene.cylinder(r, r, L, 24).move(*((p0 + p1) / 2)).rotate_R(_align_y(v / L).tolist()).material(color)

    def _build(self) -> None:
        sc, th = self.scene, self.th
        link, joint = th["link"], th["joint"]
        sc.cylinder(0.12, 0.13, 0.04, 40).rotate(math.pi / 2, 0, 0).move(0, 0, 0.02).material(joint)
        self._seg([0, 0, 0.04], [0, 0, 0.333], 0.062, link)
        with contextlib.ExitStack() as stack:
            for i, (a, d, alpha) in enumerate(DH):
                g_fixed = sc.group()
                stack.enter_context(g_fixed)
                g_fixed.move(a, 0, 0).rotate(alpha, 0, 0)
                g_joint = sc.group()
                stack.enter_context(g_joint)
                g_joint.move(0, 0, d)
                self.joints.append(g_joint)
                sc.cylinder(0.066, 0.066, 0.13, 28).rotate(math.pi / 2, 0, 0).material(joint if i % 2 == 0 else link)
                if i < 6:                                 # link geometry towards the next joint frame
                    a2, d2, al2 = DH[i + 1]
                    self._seg([0, 0, 0], [a2, 0, 0], 0.055, link)
                    self._seg([a2, 0, 0], [a2, -math.sin(al2) * d2, math.cos(al2) * d2], 0.055, link)
                else:
                    self._seg([0, 0, 0], [0, 0, FLANGE], 0.05, joint)
            g_hand = sc.group()
            stack.enter_context(g_hand)
            g_hand.move(0, 0, FLANGE).rotate(0, 0, -math.pi / 4)
            sc.box(0.06, 0.20, 0.065).move(0, 0, 0.035).material(link)
            for sgn in (1, -1):
                self.fingers.append(sc.box(0.022, 0.012, 0.045).move(0, sgn * 0.04, 0.086).material(joint))
            self.part = sc.box(0.07, 0.01, 0.10).move(0, 0, 0.125).material(th["accent"], 0.9)
            self.part.visible(False)

    def set_q(self, q) -> None:
        for g, th in zip(self.joints, q):
            g.rotate(0, 0, float(th))

    def set_gripper(self, width: float, holding: bool) -> None:
        w = max(0.0, min(0.08, float(width)))
        for sgn, f in zip((1, -1), self.fingers):
            f.move(0, sgn * (w / 2 + 0.006), 0.086)
        self.part.visible(bool(holding))
        if holding:
            self.part.scale(1.0, max(w, 0.002) / 0.01, 1.0)

    def set_paths(self, waypoints: Dict[str, List[List[float]]], draft_name: str, draft: List[List[float]],
                  show: bool = True) -> None:
        for o in self.path_objs:
            with contextlib.suppress(Exception):
                o.delete()
        self.path_objs = []
        if not show:
            return
        th = self.th
        with self.scene:
            for name in ALL_PATHS:
                qs = waypoints.get(name)
                if not qs:
                    continue
                cls = PATH_CLASS.get(name) or HOME_CLASS.get(name)
                color = th["ink2"] if cls is None else th[cls if cls != "reject" else "other"]
                kind = "home" if name in HOME_CLASS or name == "start_to_weigh" else ("bin" if cls else name)
                self._draw_path(qs, color, PATH_LABEL[name], kind)
            if draft:
                self._draw_path(draft, th["accent"], f"draft · {draft_name}", "draft", big=True)

    def _draw_path(self, qs, color: str, label: str, kind: str, big: bool = False) -> None:
        sc, th = self.scene, self.th
        qs = [np.asarray(q, dtype=float) for q in qs]
        pts: List[List[float]] = []
        for a, b in zip(qs, qs[1:]):
            for t in np.linspace(0.0, 1.0, 14, endpoint=False):
                pts.append(fk(a + (b - a) * t)[:3, 3].tolist())
        ends = [fk(q)[:3, 3] for q in qs]
        pts.append(ends[-1].tolist())
        if len(pts) > 1:
            self.path_objs.append(sc.point_cloud(points=pts, colors=[_rgb01(color)] * len(pts),
                                                 point_size=0.011 if big else 0.008))
        for p in ends:
            self.path_objs.append(sc.sphere(0.016 if big else 0.011).move(*p).material(color))
        x, y, z = (float(v) for v in ends[-1])
        if kind == "bin":                               # container below the drop pose
            self.path_objs.append(sc.box(0.16, 0.12, 0.07).move(x, y, max(0.035, z - 0.17)).material(color, 0.28))
        elif kind == "to_holder":                       # sliding holder below the grasp pose
            self.path_objs.append(sc.box(0.12, 0.07, 0.05).move(x, y, max(0.025, z - 0.09))
                                  .material(th["muted"], 0.4))
        elif kind.startswith("to_bench"):               # testing bench below the approach pose
            top = max(0.02, z - BENCH_DRAW_DROP)
            self.path_objs.append(sc.box(0.26, 0.18, top).move(x, y, top / 2).material(th["accent"], 0.22))
        if kind == "home":                              # ways back: no label (would pile up at the start point)
            return
        style = (f"color:{th['ink']};background:{_rgba(th['surface'][:7], 0.85)};border:1px solid {th['border']};"
                 f"padding:1px 7px;border-radius:7px;font:600 11px {FONT};white-space:nowrap")
        self.path_objs.append(sc.text(label, style).move(x, y, z + 0.07))


class Page(RobotPanels):
    """One browser tab. Reads the shared Controller via timers; never talks to the robot directly.
    The robot panels (parameters, sorting cycle, teach-in, test drive, log) come from RobotPanels."""

    def __init__(self, ctrl: Controller, worker: bool = False):
        self.c = ctrl
        self.th = THEMES["dark" if ctrl.s.dark else "light"]
        self._cache: Dict[str, Any] = {}
        self._seen = dict(log=max(0, ctrl.log_seq - 150), toast=ctrl.toast_seq)
        self._steps_key: Any = None
        self._step_els: List[Tuple[Any, Any, Any, Any]] = []
        self._q_shown: Optional[np.ndarray] = None
        self._errors_reported: set = set()
        self._inputs: List[Tuple[Any, str, Dict[str, bool]]] = []
        self._fullscreen = False
        self._worker = worker
        self._fb_n: Optional[int] = None              # V18: part the worker feedback buttons refer to
        self._fb_pick: Optional[int] = None           # … and the part whose "what was it really?" row is open
        self._build()
        ui.timer(0.1, self._tick_fast)
        ui.timer(0.5, self._tick_slow)
        ui.keyboard(on_key=self._on_key, ignore=["input", "select", "textarea"])
        ui.on("twin_fs", self._on_fullscreen)
        ui.on("worker_mode", self._on_worker_mode)

    # ------------------------------------------------------------ small helpers (the input helpers are in RobotPanels)
    def _stat(self, label: str, unit: str) -> Tuple[Any, Any]:
        with ui.column().classes("gap-0"):
            ui.label(label).classes("stat-label")
            with ui.row().classes("items-baseline gap-0 no-wrap"):
                v = ui.label("—").classes("stat-value")
                u = ui.label(unit).classes("stat-unit")
        return v, u

    def _axis(self, **kw: Any) -> Dict[str, Any]:
        th = self.th
        base = {"axisLine": {"lineStyle": {"color": th["axis"]}}, "axisTick": {"show": False},
                "axisLabel": {"color": th["muted"], "fontSize": 11},
                "splitLine": {"lineStyle": {"color": th["grid"]}},
                "nameTextStyle": {"color": th["muted"], "fontSize": 11}}
        base.update(kw)
        return base

    def _tooltip(self, **kw: Any) -> Dict[str, Any]:
        th = self.th
        base = {"backgroundColor": th["surface"], "borderColor": th["axis"], "borderWidth": 1,
                "textStyle": {"color": th["ink"], "fontSize": 12, "fontFamily": FONT},
                "extraCssText": "border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.22);"}
        base.update(kw)
        return base

    # ------------------------------------------------------------ build
    def _build(self) -> None:
        th, c = self.th, self.c
        ui.dark_mode(bool(c.s.dark))
        ui.colors(primary=th["accent"], secondary=th["ink2"], accent=th["accent"], positive=STATUS["good"],
                  negative=STATUS["critical"], warning=STATUS["warning"], info=th["ssd"])
        ui.add_head_html("<style>" + _css_vars(th) + CSS + "</style>" + FULLSCREEN_JS)
        self._build_header()
        with ui.element("div").classes("layout"):
            with ui.element("div").classes("dcol col-left"):
                self._build_settings()
            with ui.element("div").classes("dcol col-center"):
                self._build_cycle()
                self._build_result()
                self._build_kpis()
                self._build_analytics()
                self._build_table()
            with ui.element("div").classes("dcol col-right"):
                self._build_twin()
                self._build_telemetry()
                self._build_teach()
                self._build_drive()
                self._build_log()

    # ------------------------------------------------------------ centre column: last result, KPIs, charts, confusion matrix, results
    def _build_result(self) -> None:
        with card("Last result", "insights") as (_, right):
            with right:
                self.l_res_meta = ui.label("").classes("card-sub")
            with ui.element("div").classes("result-grid"):
                with ui.column().classes("gap-3 min-w-0"):
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        self.r_swatch = ui.element("span").classes("hero-swatch")
                        self.r_class = ui.label("—").classes("hero")
                    with ui.row().classes("gap-7 no-wrap"):
                        self.r_mass, _ = self._stat("Weight", "g")
                        self.r_width, _ = self._stat("Width", "mm")
                        self.r_cycle, _ = self._stat("Cycle time", "s")
                    self.r_reason = ui.label("Run a cycle — the measurement and decision appear here.").classes("reason")
                    with ui.element("div").classes("check"):
                        self.r_check_icon = ui.icon("help_outline")
                        self.r_check = ui.label("")
                with ui.column().classes("gap-1 min-w-0"):
                    ui.label("Weighing signal — Fz during tare (empty) and loaded").classes("field-label")
                    self.ch_measure = ui.echart(self._measure_options()).classes("measure-chart")
                    self.l_measure = ui.label("").classes("hint mono")
                    ui.label("Bench descent — load still hanging in the gripper").classes("field-label q-mt-sm")
                    self.ch_bench = ui.echart(self._bench_options()).classes("bench-chart")
                    self.l_bench = ui.label("").classes("hint mono")

    def _build_kpis(self) -> None:
        self.k: Dict[str, Tuple[Any, Any, Any]] = {}
        with ui.element("div").classes("kpis"):
            for key, label, icon in [("cycles", "Parts", "loop"), ("acc", "Accuracy", "verified"),
                                     ("f1", "Macro F1", "functions"), ("avg", "Avg cycle", "timer"),
                                     ("rate", "Throughput", "speed"), ("bench", "Bench contact", "fact_check"),
                                     ("empty", "Holder empty", "back_hand"), ("sec", "HDD → SSD", "gpp_maybe")]:
                with ui.element("div").classes("kpi") as box:
                    with ui.element("div").classes("kpi-label"):
                        ui.icon(icon)
                        ui.label(label)
                    v = ui.label("—").classes("kpi-value")
                    s = ui.label("").classes("kpi-sub")
                self.k[key] = (box, v, s)

    def _build_analytics(self) -> None:
        with ui.element("div").classes("analytics"):
            with card("Width × weight", "scatter_plot",
                      "Colour = actual part (hollow = unlabelled → decision) · ◆ = misclassified · "
                      "shading = current decision regions"):
                self.ch_scatter = ui.echart(self._scatter_options()).classes("scatter-chart")
            with card("Confusion matrix", "grid_on", "Rows = actual part · columns = robot decision · "
                                                     "outlined = correct"):
                self.cm_box = ui.element("div").classes("cm")
                self.l_cm = ui.label("").classes("hint q-mt-sm")

    def _build_table(self) -> None:
        with card("Results", "table_rows", "Pick the actual part in ‘Actual’ to correct a label · "
                                           "saved to the log folder (.log, CSVs, summary)") as (_, right):
            with right:
                self._button("CSV", "download", self._download, "outline", "Download this session as CSV")
                self._button("Summary", "summarize", self._download_summary, "outline",
                             "Download the KPIs of this session (cycle times, accuracy, interventions) as Markdown")
                ui.button(icon="delete_sweep", on_click=self._clear_results, color=None) \
                    .props("flat round dense").classes("icon-btn").tooltip("Start a new session")
            cols = [
                {"name": "n", "label": "#", "field": "n", "align": "left", "sortable": True},
                {"name": "time", "label": "Time", "field": "time", "align": "left"},
                {"name": "mode", "label": "Mode", "field": "mode", "align": "left"},
                {"name": "truth", "label": "Actual", "field": "truth", "align": "left"},
                {"name": "pred", "label": "Decision", "field": "pred_label", "align": "left"},
                {"name": "mass", "label": "Weight g", "field": "mass", "align": "right", "sortable": True},
                {"name": "width", "label": "Width mm", "field": "width", "align": "right", "sortable": True},
                {"name": "bench", "label": "Bench", "field": "bench", "align": "left"},
                {"name": "cycle", "label": "Cycle s", "field": "cycle", "align": "right"},
                {"name": "status", "label": "Status", "field": "status", "align": "left"},
                {"name": "tag", "label": "Tag", "field": "tag", "align": "left"},
                {"name": "note", "label": "Note", "field": "note", "align": "left"},
            ]
            self.table = ui.table(columns=cols, rows=[], row_key="n",
                                  pagination={"rowsPerPage": 8, "sortBy": "n", "descending": True}) \
                .props("flat dense separator=horizontal").classes("results-table w-full")
            opts = json.dumps([{"label": LABEL[k], "value": k} for k in ["", "ssd", "hdd25", "hdd35", "other"]])
            self.table.add_slot("body-cell-truth", r"""
                <q-td :props="props">
                  <q-select dense borderless options-dense emit-value map-options v-model="props.row.truth"
                    :options='""" + opts + r"""' style="min-width:104px"
                    @update:model-value="() => $parent.$emit('set_truth', props.row)" />
                </q-td>""")
            self.table.add_slot("body-cell-pred", r"""
                <q-td :props="props">
                  <span style="display:inline-flex;align-items:center;gap:7px">
                    <span :style="{width:'10px',height:'10px',borderRadius:'3px',background:props.row.pred_color}"></span>
                    {{ props.row.pred_label }}
                  </span>
                </q-td>""")
            self.table.add_slot("body-cell-status", r"""
                <q-td :props="props">
                  <span class="tbadge" :style="{borderColor: props.row.status_color}">{{ props.row.status }}</span>
                </q-td>""")
            self.table.add_slot("body-cell-note", r"""
                <q-td :props="props" style="max-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
                      :title="props.row.note">{{ props.row.note }}</q-td>""")
            self.table.on("set_truth", self._on_set_truth)

    # ------------------------------------------------------------ right column: digital twin (incl. worker view) and telemetry
    def _build_twin(self) -> None:
        with card("Digital twin", "view_in_ar", "Live arm pose · paths · holder, bench, containers (drag to orbit)") as (_, right):
            with right:
                self.w_paths = ui.switch("Paths", value=True).props("dense")
                ui.button(icon="center_focus_strong", on_click=lambda: self.twin.reset_camera(), color=None) \
                    .props("flat round dense").classes("icon-btn").tooltip("Reset view")
                ui.button(icon="fullscreen", color=None).props("flat round dense").classes("icon-btn") \
                    .tooltip("Full screen (Esc or ✕ to leave)").on("click", js_handler="() => twinFullscreen(true)")
            with ui.element("div").classes("twin-wrap"):
                self.scene = ui.scene(width=406, height=310, grid=(2, 20), background_color=self.th["scene"]) \
                    .classes("twin-scene")
                with ui.element("div").classes("twin-hud"):
                    with ui.element("div").classes("hud-card"):
                        ui.label(APP_TITLE).classes("hud-title")
                        self.hud_job = ui.label("").classes("hud-line")
                        with ui.row().classes("items-center gap-3 no-wrap"):
                            self.hud_swatch = ui.element("span").classes("hud-swatch")
                            self.hud_class = ui.label("—").classes("hud-class")
                        self.hud_meta = ui.label("").classes("hud-line mono")
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.button("STOP", icon="pan_tool", on_click=self._stop, color=None) \
                            .props("unelevated no-caps no-wrap").classes("btn-stop")
                        ui.button(icon="close", color=None).props("round unelevated").classes("hud-exit") \
                            .tooltip("Leave full screen").on("click", js_handler="() => twinFullscreen(false)")
                with ui.element("div").classes("worker-ui"):
                    with ui.element("div").classes("wk-top"):
                        with ui.element("div").classes("wk-status"):
                            self.wk_dot = ui.element("span").classes("dot")
                            self.wk_status = ui.label("")
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.button(icon="fullscreen", color=None).props("unelevated").classes("wk-switch") \
                                .tooltip("Full screen on / off") \
                                .on("click", js_handler="() => pageFullscreen(!document.fullscreenElement)")
                            ui.button("Engineer view", icon="tune", on_click=self._leave_worker, color=None) \
                                .props("unelevated no-caps no-wrap").classes("wk-switch")
                    with ui.element("div").classes("wk-drive"):
                        ui.label("Detected drive").classes("wk-drive-label")
                        with ui.row().classes("items-stretch gap-4 no-wrap q-mt-xs"):
                            self.wk_swatch = ui.element("span").classes("wk-drive-swatch")
                            with ui.column().classes("gap-0"):
                                self.wk_class = ui.label("—").classes("wk-drive-class")
                                self.wk_sub = ui.label("").classes("wk-drive-sub")
                        with ui.element("div").classes("wk-fb") as self.wk_fb:     # V18: was the decision right?
                            self.wk_fb_q = ui.label("").classes("wk-fb-q")
                            with ui.row().classes("wk-fb-row no-wrap") as self.wk_fb_btns:
                                ui.button("Correct", icon="check", on_click=lambda: self._feedback(True), color=None) \
                                    .props("unelevated no-caps no-wrap").classes("wk-fb-btn wk-fb-ok")
                                ui.button("Wrong", icon="close", on_click=self._feedback_wrong, color=None) \
                                    .props("unelevated no-caps no-wrap").classes("wk-fb-btn wk-fb-wrong")
                            with ui.row().classes("wk-fb-row") as self.wk_fb_pick:
                                self.wk_fb_types = {
                                    t: ui.button(LABEL[t], on_click=lambda t=t: self._feedback(False, t), color=None)
                                    .props("unelevated no-caps no-wrap").classes("wk-fb-btn wk-fb-type")
                                    for t in TRUTH_ROWS}
                                ui.button(icon="undo", on_click=self._feedback_back, color=None) \
                                    .props("unelevated").classes("wk-fb-btn wk-fb-back").tooltip("Back")
                        self.wk_fb.set_visibility(False)
                        self.wk_fb_pick.set_visibility(False)
                    with ui.element("div").classes("wk-actions"):
                        self.wk_continue = ui.button("Continue", icon="play_arrow", on_click=self.c.bench_continue,
                                                     color=None).props("unelevated no-caps no-wrap") \
                            .classes("wk-btn wk-continue")
                        self.wk_continue.set_visibility(False)
                        self.wk_restart = ui.button("Restart", icon="restart_alt", on_click=self._worker_restart,
                                                    color=None).props("unelevated no-caps no-wrap") \
                            .classes("wk-btn wk-restart")
                        ui.button("STOP", icon="pan_tool", on_click=self._stop, color=None) \
                            .props("unelevated no-caps no-wrap").classes("wk-btn wk-stop")
                with ui.element("div").classes("twin-overlay"):
                    self.l_ee = ui.label("").classes("tag mono")
                    self.l_grip = ui.label("").classes("tag mono")
                    self.l_motion = ui.label("").classes("tag live")
            self.twin = Twin(self.scene, self.th)

    def _build_telemetry(self) -> None:
        with card("Telemetry", "monitor_heart", "Joint positions vs limits · TCP · gripper · external force") as (_, right):
            with right:
                self.l_rate = ui.label("").classes("card-sub mono")
            self.j_rows: List[Tuple[Any, Any, Any]] = []
            with ui.column().classes("gap-1 w-full"):
                for i in range(7):
                    with ui.element("div").classes("jrow") as row:
                        ui.label(f"J{i + 1}").classes("jname")
                        with ui.element("div").classes("meter"):
                            ui.element("div").classes("zone lo")
                            ui.element("div").classes("zone hi")
                            mark = ui.element("div").classes("mark")
                        val = ui.label("—").classes("jval")
                    self.j_rows.append((row, mark, val))
            with ui.element("div").classes("mini-stats q-mt-md"):
                self.l_xyz = []
                for axis in "xyz":
                    with ui.element("div").classes("mini"):
                        ui.label(f"TCP {axis}").classes("mini-l")
                        self.l_xyz.append(ui.label("—").classes("mini-v"))
            with ui.column().classes("gap-1 w-full q-mt-md"):
                with ui.row().classes("w-full justify-between items-center no-wrap"):
                    ui.label("Gripper opening").classes("field-label")
                    self.l_gw = ui.label("—").classes("field-value mono")
                with ui.element("div").classes("gbar w-full"):
                    self.g_fill = ui.element("div").classes("gfill")
            ui.label("External force O_F_ext_hat_K — last 10 s").classes("field-label q-mt-md")
            self.ch_force = ui.echart(self._force_options()).classes("force-chart")

    # ------------------------------------------------------------ chart options
    def _measure_options(self) -> Dict[str, Any]:
        th = self.th
        return {
            "animation": False, "textStyle": {"fontFamily": FONT},
            "grid": {"left": 50, "right": 16, "top": 30, "bottom": 30},
            "legend": {"top": 0, "right": 0, "icon": "roundRect", "itemWidth": 14, "itemHeight": 3,
                       "textStyle": {"color": th["ink2"], "fontSize": 11}},
            "tooltip": self._tooltip(trigger="axis", **{":valueFormatter": "v => v == null ? '' : v.toFixed(3) + ' N'"}),
            "xAxis": self._axis(type="value", name="s", nameGap=6, min=0, splitLine={"show": False}),
            "yAxis": self._axis(type="value", name="Fz  N", scale=True, nameGap=10),
            "series": [
                {"name": "Tare (empty)", "type": "line", "showSymbol": False, "data": [],
                 "lineStyle": {"width": 1.5, "color": th["muted"]}, "itemStyle": {"color": th["muted"]}},
                {"name": "Loaded", "type": "line", "showSymbol": False, "data": [],
                 "lineStyle": {"width": 2, "color": th["accent"]}, "itemStyle": {"color": th["accent"]}},
            ],
        }

    def _bench_options(self) -> Dict[str, Any]:
        th = self.th
        return {
            "animation": False, "textStyle": {"fontFamily": FONT},
            "grid": {"left": 50, "right": 16, "top": 14, "bottom": 30},
            "tooltip": self._tooltip(trigger="axis", **{":valueFormatter": "v => v == null ? '' : v.toFixed(0) + ' g'"}),
            "xAxis": self._axis(type="value", name="mm", nameGap=6, min=0, splitLine={"show": False}),
            "yAxis": self._axis(type="value", name="g", nameGap=8),
            "series": [{"name": "rest load", "type": "line", "showSymbol": False, "data": [],
                        "lineStyle": {"width": 2, "color": th["accent"]}, "itemStyle": {"color": th["accent"]}}],
        }

    def _force_options(self) -> Dict[str, Any]:
        th = self.th
        return {
            "animation": False, "textStyle": {"fontFamily": FONT},
            "grid": {"left": 44, "right": 12, "top": 26, "bottom": 22},
            "legend": {"top": 0, "right": 0, "icon": "roundRect", "itemWidth": 14, "itemHeight": 3,
                       "textStyle": {"color": th["ink2"], "fontSize": 11}},
            "tooltip": self._tooltip(trigger="axis", **{":valueFormatter": "v => v == null ? '' : v.toFixed(2) + ' N'"}),
            "xAxis": self._axis(type="value", min=-10, max=0, splitLine={"show": False},
                                axisLabel={"color": th["muted"], "fontSize": 10, ":formatter": "v => v + ' s'"}),
            "yAxis": self._axis(type="value", scale=True, name="N", nameGap=8),
            "series": [
                {"name": "Fz", "type": "line", "showSymbol": False, "data": [],
                 "lineStyle": {"width": 2, "color": th["accent"]}, "itemStyle": {"color": th["accent"]}},
                {"name": "Fx", "type": "line", "showSymbol": False, "data": [],
                 "lineStyle": {"width": 1, "color": th["ink2"]}, "itemStyle": {"color": th["ink2"]}},
                {"name": "Fy", "type": "line", "showSymbol": False, "data": [],
                 "lineStyle": {"width": 1, "color": th["muted"], "type": "dotted"}, "itemStyle": {"color": th["muted"]}},
            ],
        }

    def _scatter_options(self) -> Dict[str, Any]:
        th = self.th
        return {
            "animation": False, "textStyle": {"fontFamily": FONT},
            "grid": {"left": 58, "right": 18, "top": 34, "bottom": 44},
            "legend": {"top": 0, "left": 0, "icon": "circle", "itemWidth": 9, "itemHeight": 9,
                       "textStyle": {"color": th["ink2"], "fontSize": 11},
                       "data": ["SSD", "HDD 2.5″", "HDD 3.5″", "Other / reject", "Datasheet (training)"]},
            "tooltip": self._tooltip(trigger="item", **{
                ":formatter": "p => (p.data && p.data.n) ? `<b>#${p.data.n}</b> · actual: ${p.data.truth}<br/>"
                              "decision: <b>${p.data.pred}</b><br/>${p.data.m} g · ${p.data.w} mm` : ''"}),
            "xAxis": self._axis(type="value", name="width  [mm]", nameLocation="middle", nameGap=28, min=0, max=35),
            "yAxis": self._axis(type="log", logBase=10, name="weight  [g]", nameLocation="middle", nameGap=42,
                                min=10, max=1000),
            "series": self._scatter_series(),
        }

    def _model_regions(self, x_max: float, y_lo: float, y_hi: float) -> List[List[Dict[str, Any]]]:
        """V18: where the trained model (incl. SSD guard and reject windows) decides what — classify() on a grid
        (linear width × log weight), equal cells merged into rectangles; cached per classification settings."""
        th, s = self.th, self.c.s
        key = (self._thr_sig(), x_max, y_lo, y_hi)
        rects = _REGION_CACHE.get(key)
        if rects is None:
            nx, ny = 105, 50
            ly0, ly1 = math.log10(y_lo), math.log10(y_hi)
            rects, open_ = [], {}
            for j in range(ny):
                ya, yb = 10 ** (ly0 + (ly1 - ly0) * j / ny), 10 ** (ly0 + (ly1 - ly0) * (j + 1) / ny)
                cells = [classify(math.sqrt(ya * yb), x_max * (i + 0.5) / nx, s)[0] for i in range(nx)]
                row_open, i0 = {}, 0
                for i in range(1, nx + 1):
                    if i == nx or cells[i] != cells[i0]:
                        k = (cells[i0], i0, i)
                        if k in open_:                   # same run as in the row below → make it taller
                            rects[open_[k]][4] = yb
                            row_open[k] = open_[k]
                        else:
                            row_open[k] = len(rects)
                            rects.append([cells[i0], x_max * i0 / nx, x_max * i / nx, ya, yb])
                        i0 = i
                open_ = row_open
            if len(_REGION_CACHE) > 16:
                _REGION_CACHE.clear()
            _REGION_CACHE[key] = rects
        biggest: Dict[str, Tuple[float, int]] = {}
        for k, (cls, x0, x1, ya, yb) in enumerate(rects):
            area = (x1 - x0) * math.log10(yb / ya)
            if area > biggest.get(cls, (0.0, -1))[0]:
                biggest[cls] = (area, k)
        lab = {"color": th["muted"], "fontSize": 11, "fontWeight": 600, "position": "inside"}
        out = []
        for k, (cls, x0, x1, ya, yb) in enumerate(rects):
            first: Dict[str, Any] = {"coord": [round(x0, 3), round(ya, 3)],
                                     "itemStyle": {"color": _rgba(self._col(cls), 0.09)}}
            if biggest[cls][1] == k:
                first.update(name=LABEL.get(cls, cls), label=lab)
            out.append([first, {"coord": [round(x1, 3), round(yb, 3)]}])
        return out

    def _scatter_series(self) -> List[Dict[str, Any]]:
        c, th = self.c, self.th
        x_max, y_lo, y_hi = 35.0, 10.0, 1000.0
        series: List[Dict[str, Any]] = [{                # V18: decision regions + datasheet drives of the model
            "name": "regions", "type": "scatter", "data": [], "silent": True,
            "markArea": {"silent": True, "data": self._model_regions(x_max, y_lo, y_hi)},
        }, {
            "name": "Datasheet (training)", "type": "scatter", "silent": True, "symbol": "rect", "symbolSize": 6,
            "itemStyle": {"color": th["muted"]},
            "data": [{"value": [round(min(w, x_max - 0.2), 2), round(min(max(g, y_lo * 1.05), y_hi * 0.95), 1)],
                      "itemStyle": {"color": _rgba(self._col(cls), 0.55)}} for w, g, cls in DRIVE_MODEL["points"]],
        }]
        groups: Dict[str, List[Dict[str, Any]]] = {"ssd": [], "hdd25": [], "hdd35": [], "other": []}
        for r in c.results:
            if r["status"] not in MEASURED or r.get("mass_g") is None or r["mode"] == "dry":
                continue
            labelled = bool(r["truth"])
            key = r["truth"] if labelled else ("other" if r["pred"] == "reject" else r["pred"])
            wrong = labelled and r["pred"] != EXPECTED[r["truth"]]
            col = self._col(key)
            item: Dict[str, Any] = {
                "value": [round(min(max(r["width_mm"], 0.0), x_max - 0.2), 2),
                          round(min(max(r["mass_g"], y_lo * 1.05), y_hi * 0.95), 1)],
                "n": r["n"], "pred": LABEL.get(r["pred"], "—"), "m": f"{r['mass_g']:.1f}", "w": f"{r['width_mm']:.2f}",
                "truth": LABEL[r["truth"]] if labelled else "unlabelled",
                "symbol": "diamond" if wrong else "circle", "symbolSize": 13 if wrong else 10}
            if not labelled:
                item["itemStyle"] = {"color": th["surface"], "borderColor": col, "borderWidth": 2}
            groups[key].append(item)
        names = {"ssd": "SSD", "hdd25": "HDD 2.5″", "hdd35": "HDD 3.5″", "other": "Other / reject"}
        for key, items in groups.items():
            series.append({"name": names[key], "type": "scatter", "data": items,
                           "itemStyle": {"color": self._col(key), "borderColor": th["surface"], "borderWidth": 2},
                           "emphasis": {"scale": 1.4}})
        return series

    # ------------------------------------------------------------ actions
    async def _clear_results(self) -> None:
        if await self._confirm("Start a new session?", "Clears the table and charts. All files of the current "
                                                       "session (.log, CSVs, summary) stay in the log folder.",
                               ok_text="New session"):
            self.c.clear_results()

    def _download(self) -> None:
        if not self.c.results:
            ui.notify("No results yet", type="info")
            return
        ui.download.content(self.c.results_csv(), f"dashboard_{self.c.session}.csv")

    def _download_summary(self) -> None:
        ui.download.content(self.c.summary_md(), f"dashboard_{self.c.session}_summary.md")

    def _on_set_truth(self, e: Any) -> None:
        row = e.args if isinstance(e.args, dict) else {}
        if "n" in row:
            self.c.set_truth(int(row["n"]), row.get("truth") or "")

    def _on_key(self, e: Any) -> None:
        if not e.action.keydown or e.action.repeat:
            return
        if getattr(e.modifiers, "ctrl", False) or getattr(e.modifiers, "meta", False) or getattr(e.modifiers, "alt", False):
            return
        name = getattr(e.key, "name", str(e.key))
        if name == "Escape":
            if self._fullscreen:              # Esc leaves the full-screen twin — never taken as STOP
                return
            self._stop()
        elif name in ("w", "W") and self.c.connected:
            self.c.teach_add()

    def _on_worker_mode(self, e: Any) -> None:
        v = e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args
        self._worker = bool(v)

    async def _leave_worker(self) -> None:
        if await self._confirm("Switch to the engineer view?", "The full dashboard with all parameters opens.",
                               ok_text="Engineer view"):
            ui.run_javascript("setWorkerMode(false)")

    def _feedback(self, correct: bool, truth: str = "") -> None:
        """V18: worker feedback on the shown part (→ 'Actual' label, confusion matrix, CSV, summary)."""
        n = self._fb_n
        self._fb_pick = None
        if n is None:
            return
        self.c.worker_feedback(n, correct, truth)
        ui.notify(f"#{n}: saved — " + ("correct" if correct else f"actually {LABEL.get(truth, truth)}"),
                  type="positive", position="top", timeout=1800)

    def _feedback_wrong(self) -> None:
        self._fb_pick = self._fb_n

    def _feedback_back(self) -> None:
        self._fb_pick = None

    async def _worker_restart(self) -> None:
        c = self.c
        if c.busy:
            ui.notify(f"Robot is busy ({c.job})", type="warning", position="top")
            return
        if await self._confirm("Restart the robot?", "The robot moves to its start position and then sorts "
                               "automatically. Keep your hands out of the robot's working area.", ok_text="Restart"):
            c.worker_restart()

    def _on_fullscreen(self, e: Any) -> None:
        v = e.args[0] if isinstance(e.args, (list, tuple)) and e.args else e.args
        self._fullscreen = bool(v)

    # ------------------------------------------------------------ periodic updates
    def _safe(self, fns: List[Callable]) -> None:
        for fn in fns:
            try:
                fn()
            except Exception as e:  # keep the page alive, report once per function
                if fn.__name__ not in self._errors_reported:
                    self._errors_reported.add(fn.__name__)
                    self.c.error(f"UI update {fn.__name__} failed: {type(e).__name__}: {e}")
                    self.c.debug("traceback:\n" + traceback.format_exc().rstrip())

    def _tick_fast(self) -> None:
        self._safe([self._upd_header, self._upd_buttons, self._upd_steps, self._upd_tele, self._upd_twin,
                    self._upd_log, self._upd_toasts, self._upd_teach, self._upd_hud,
                    self._upd_worker])

    def _tick_slow(self) -> None:
        self._safe([self._upd_ready, self._upd_force_chart, self._upd_measure, self._upd_bench, self._upd_results,
                    self._upd_twin_paths, self._upd_paths, self._sync_inputs])

    def _upd_worker(self) -> None:
        """Worker view: one status line, the detected drive, Restart / STOP (and Continue at the bench)."""
        c, th = self.c, self.th
        mode = c.tele.get("mode", "")
        robot_err = c.connected and (mode in ("kReflex", "kUserStopped", "kOther") or bool(c.tele.get("errors")))
        last_err = c.last is not None and c.last.get("status") in ("error", "part_lost_at_bench")
        if c.job == "restart":
            txt, dot = "Restarting …", "dot warn"
        elif c.bench.get("waiting"):
            txt, dot = "Drive rests on the test bench — press Continue when the test is done", "dot warn"
        elif c.auto_on:
            txt, dot = "Sorting — place the next drive on the holder", "dot good"
        elif c.job:
            txt, dot = f"Robot busy — {c.job}", "dot warn"
        elif not c.connected:
            txt, dot = "Robot not connected — press Restart", "dot bad"
        elif c.teaching:
            txt, dot = "Teaching mode is on — press Restart to sort again", "dot warn"
        elif robot_err or last_err:
            txt, dot = "Stopped after an error — press Restart", "dot bad"
        else:
            txt, dot = "Stopped — press Restart to continue", "dot warn"
        self.wk_status.set_text(txt)
        if self._changed("wk_dot", dot):
            self.wk_dot.classes(replace=dot)
        r = c.last
        running = c.job is not None and bool(c.steps) and c.cycle_t1 is None
        if running and (r is None or r.get("n") != c.cycle_n):
            prev = f" · last drive: {LABEL[r['pred']]}" if r is not None and r.get("pred") else ""
            title, color, sub = "Measuring …", th["grid"], "weighing and measuring the drive" + prev
        elif r is None:
            title, color, sub = "No drive yet", th["grid"], "place a drive on the holder"
        elif r["status"] == EMPTY:
            title, color, sub = "No drive found", STATUS["warning"], "the holder was empty"
        elif r.get("pred") and r.get("mass_g") is not None:
            dest = "the reject container" if r["pred"] == "reject" else f"the {LABEL[r['pred']]} container"
            if r["mode"] == "measure":
                where = "measured, put back into the holder"
            elif r["status"] == "sorting":
                where = f"→ going into {dest}"
            elif r["status"] in SORTED:
                where = f"sorted into {dest}"
            elif r["status"] == "part_lost_at_bench":
                where = "lost at the test bench"
            else:
                where = "not finished"
            title, color = LABEL[r["pred"]], self._col(r["pred"])
            sub = f"{r['mass_g']:.0f} g · {r['width_mm']:.1f} mm · {where}"
        else:
            title, color, sub = "—", th["grid"], "last part not finished"
        self.wk_class.set_text(title)
        self.wk_sub.set_text(sub)
        if self._changed("wk_col", color):
            self.wk_swatch.style(f"background:{color}")
        if self._changed("wk_restart_en", not c.busy):
            self.wk_restart.set_enabled(not c.busy)
        waiting = bool(c.bench.get("waiting"))
        if self._changed("wk_cont", waiting):
            self.wk_continue.set_visibility(waiting)
        # V18: feedback on the last decision (stays for the previous part while the next one is measured)
        t = c.feedback_target()
        n = None if t is None else t["n"]
        self._fb_n = n
        if self._fb_pick is not None and self._fb_pick != n:
            self._fb_pick = None
        truth = "" if t is None else (t.get("truth") or "")
        if self._changed("wk_fb", (n, None if t is None else t["pred"], truth, self._fb_pick)):
            self.wk_fb.set_visibility(t is not None)
            if t is not None:
                pred, picking = t["pred"], self._fb_pick == n
                if picking:
                    q = f"What was drive #{n} really?"
                elif truth:
                    q = (f"#{n} is labelled {LABEL[truth]} — decision {LABEL[pred]} "
                         + ("correct ✓" if EXPECTED[truth] == pred else "wrong ✗") + " · tap to change")
                else:
                    q = f"Was drive #{n} classified correctly as {LABEL[pred]}?"
                self.wk_fb_q.set_text(q)
                self.wk_fb_btns.set_visibility(not picking)
                self.wk_fb_pick.set_visibility(picking)
                for tr, b in self.wk_fb_types.items():
                    b.set_visibility(EXPECTED[tr] != pred)

    def _upd_hud(self) -> None:
        """Status card shown on top of the full-screen digital twin."""
        c, th = self.c, self.th
        step = next((st["label"] for st in c.steps if st["status"] == "active"), "")
        job = f"{c.job} · {time.time() - c.job_t0:.0f} s" + (f" · {step}" if step else "") if c.job else "Idle"
        self.hud_job.set_text(("● Connected · " if c.connected else "○ Disconnected · ") + job)
        r = c.last
        if r and r.get("pred") and r["status"] in MEASURED + ("sorting",):
            title, color = LABEL[r["pred"]], self._col(r["pred"])
        elif r and r["status"] == EMPTY:
            title, color = "Holder empty", STATUS["warning"]
        elif r and r["status"] in ("error", "aborted"):
            title, color = ("Error" if r["status"] == "error" else "Stopped"), STATUS["critical"]
        else:
            title, color = "—", th["grid"]
        self.hud_class.set_text(title)
        if self._changed("hud_col", color):
            self.hud_swatch.style(f"background:{color}")
        parts = [] if r is None else [f"#{r['n']}"]
        if r is not None:
            if r.get("mass_g") is not None:
                parts.append(f"{r['mass_g']:.1f} g")
            if r.get("width_mm") is not None:
                parts.append(f"{r['width_mm']:.2f} mm")
            if r.get("bench_contact") is not None:
                parts.append((r.get("bench") or "bench") + (" ✓ " if r["bench_contact"] else " ✗ ")
                             + f"{r.get('bench_descent_mm') or 0:.0f} mm")
        self.hud_meta.set_text(" · ".join(parts) if parts else "no part yet")

    def _upd_tele(self) -> None:
        c = self.c
        if not self._changed("tele_v", (c.ver["tele"], c.connected)):
            return
        q = np.asarray(c.tele["q"], dtype=float)
        for i, (row, mark, val) in enumerate(self.j_rows):
            lo, hi = float(Q_LOWER[i]), float(Q_UPPER[i])
            frac = (float(q[i]) - lo) / (hi - lo)
            edge = min(frac, 1.0 - frac)
            state = "crit" if edge < 0.03 else "warn" if edge < 0.09 else ""
            pct = round(max(0.0, min(1.0, frac)) * 100, 1)
            if self._changed(f"j{i}", pct):
                mark.style(f"left:{pct}%")
            if self._changed(f"js{i}", state):
                row.classes(replace=f"jrow {state}".strip())
            val.set_text(f"{math.degrees(float(q[i])):7.1f}°" if c.connected else "—")
        ee = c.tele["ee"]
        for lab, v in zip(self.l_xyz, ee):
            lab.set_text(f"{float(v):.3f} m" if c.connected else "—")
        w, mw, held = c.tele["width"], c.tele["max_width"] or 0.08, c.tele["grasped"]
        self.l_gw.set_text(f"{w * 1000:.1f} / {mw * 1000:.1f} mm" + ("  · holding" if held else "")
                           if c.connected else "—")
        pct = round(max(0.0, min(1.0, w / mw)) * 100, 1)
        if self._changed("gfill", (pct, held)):
            self.g_fill.style(f"width:{pct}%")
            self.g_fill.classes(replace="gfill held" if held else "gfill")
        self.l_rate.set_text(f"{c.tele['hz']:.0f} Hz" if c.connected else "offline")
        self.l_ee.set_text(f"TCP {ee[0]:+.3f} {ee[1]:+.3f} {ee[2]:+.3f} m")
        self.l_grip.set_text(f"gripper {w * 1000:.1f} mm")

    def _upd_twin(self) -> None:
        c = self.c
        q = np.asarray(c.tele["q"], dtype=float)
        if self._q_shown is None or float(np.max(np.abs(q - self._q_shown))) > 0.002:
            self._q_shown = q.copy()
            self.twin.set_q(q)
        holding = bool(c.tele["grasped"]) and c.tele["width"] * 1000 >= c.s.empty_width_mm
        if self._changed("twin_g", (round(c.tele["width"], 4), holding)):
            self.twin.set_gripper(c.tele["width"], holding)
        label = c.motion_label or ("teaching — guide by hand" if c.teaching else "")
        self.l_motion.set_text(("▶ " + label) if label else "")
        if self._changed("motion_vis", bool(label)):
            self.l_motion.set_visibility(bool(label))

    def _upd_twin_paths(self) -> None:
        c = self.c
        sig = (c.ver["wp"], c.ver["draft"], bool(self.w_paths.value), c.draft_path)
        if self._changed("twin_paths_v", sig):
            self.twin.set_paths(c.waypoints, c.draft_path, c.draft, bool(self.w_paths.value))

    def _upd_force_chart(self) -> None:
        c = self.c
        if not self._changed("force_v", (c.ver["tele"], c.ver["measure"], len(c.force_hist))):
            return
        now = time.time()
        pts = [p for p in list(c.force_hist) if now - p[0] <= 10.0]
        if len(pts) > 400:
            pts = pts[:: math.ceil(len(pts) / 400)]
        series = self.ch_force.options["series"]
        for k, idx in ((0, 3), (1, 1), (2, 2)):
            series[k]["data"] = [[round(p[0] - now, 3), round(p[idx], 3)] for p in pts]
        self.ch_force.update()

    def _upd_measure(self) -> None:
        c = self.c
        if not self._changed("meas_v", c.ver["measure"]):
            return
        m, th = c.measure, self.th
        series = self.ch_measure.options["series"]
        for k, phase in enumerate(("tare", "load")):
            data = list(m[phase])
            series[k]["data"] = [[round(p[0], 3), round(p[3], 4)] for p in data]
            mean = m.get(f"{phase}_mean")
            col = th["muted"] if phase == "tare" else th["accent"]
            series[k]["markLine"] = {"silent": True, "symbol": "none", "data": [{"yAxis": round(mean[2], 4)}],
                                     "lineStyle": {"type": "dashed", "color": col, "width": 1},
                                     "label": {"formatter": f"{mean[2]:.3f} N", "color": th["ink2"], "fontSize": 11,
                                               "position": "insideEndTop"}} if mean else {"data": []}
        self.ch_measure.update()
        if m.get("tare_mean") and m.get("load_mean"):
            d = np.asarray(m["load_mean"]) - np.asarray(m["tare_mean"])
            self.l_measure.set_text(f"ΔF = ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}) N   |ΔF| = {np.linalg.norm(d):.3f} N")
        elif m.get("phase"):
            self.l_measure.set_text(f"measuring … ({'tare, gripper empty' if m['phase'] == 'tare' else 'loaded'})")

    def _upd_bench(self) -> None:
        c = self.c
        if not self._changed("bench_v", (c.ver["bench"], c.s.contact_g)):
            return
        b, th = c.bench, self.th
        pts = list(b["samples"])
        lim = float(b["limit"]) if b.get("limit") is not None else float(c.s.contact_g)
        series = self.ch_bench.options["series"][0]
        series["data"] = [[round(x, 2), round(y, 1)] for x, y in pts]
        series["markLine"] = {"silent": True, "symbol": "none", "data": [{"yAxis": lim}],
                              "lineStyle": {"type": "dashed", "color": th["ink2"], "width": 1},
                              "label": {"formatter": f"contact ≤ {lim:g} g", "color": th["ink2"], "fontSize": 11,
                                        "position": "insideEndTop"}}
        self.ch_bench.update()
        if b.get("waiting"):
            txt = "resting on the bench — waiting for ‘Continue’"
        elif b.get("phase") == "descent":
            txt = (f"lowering … {pts[-1][0]:.1f} mm, rest load {pts[-1][1]:.0f} g" if pts
                   else "holding still for the reference force …")
        elif b.get("contact") is not None:
            txt = (f"contact after {b['descent'] * 1000:.1f} mm · rest load {b['rest'] * 1000:.0f} g" if b["contact"]
                   else f"NO contact within {c.s.max_descent_mm:g} mm (rest {b['rest'] * 1000:.0f} g) — check bench "
                        "height / bench path")
            if b.get("phase") == "resting":
                txt = "resting · " + txt
        else:
            txt = "no bench descent yet"
        self.l_bench.set_text(txt)

    def _thr_sig(self) -> Tuple:
        s = self.c.s
        return (s.reject_enabled, s.ssd_mass_min_g, s.hdd25_mass_max_g, s.w25_min_mm, s.w25_max_mm,
                s.hdd35_mass_min_g, s.hdd35_mass_max_g, s.hdd35_w_max_mm, s.ssd_min_conf_pct)

    def _upd_results(self) -> None:
        c = self.c
        if not self._changed("res_v", (c.ver["results"], self._thr_sig())):
            return
        metrics = compute_metrics(c.results)
        self._upd_hero()
        self._upd_kpis(metrics)
        self.ch_scatter.options["series"] = self._scatter_series()
        self.ch_scatter.update()
        self._render_cm(metrics)
        self._upd_table()

    def _upd_hero(self) -> None:
        c, th = self.c, self.th
        r = c.last
        if r is None:
            self.r_class.set_text("—")
            self.r_swatch.style(f"background:{th['grid']}")
            for lab in (self.r_mass, self.r_width, self.r_cycle):
                lab.set_text("—")
            self.r_reason.set_text("Run a cycle — the measurement and decision appear here.")
            self.r_check.set_text("")
            self.r_check_icon.set_visibility(False)
            self.l_res_meta.set_text("")
            return
        status = r["status"]
        if r.get("pred") and status in MEASURED + ("sorting",):
            title, color = LABEL[r["pred"]], self._col(r["pred"])
        else:
            title, color = {EMPTY: ("Holder empty", STATUS["warning"]), "error": ("Error", STATUS["critical"]),
                            "aborted": ("Stopped", STATUS["serious"])}.get(status, ("—", th["grid"]))
        self.r_class.set_text(title)
        self.r_swatch.style(f"background:{color}")
        self.r_mass.set_text(f"{r['mass_g']:.1f}" if r.get("mass_g") is not None else "—")
        self.r_width.set_text(f"{r['width_mm']:.2f}" if r.get("width_mm") is not None else "—")
        self.r_cycle.set_text(f"{r['cycle_s']:.1f}" if r.get("cycle_s") else "…")
        mode = {"sort": "sorted", "measure": "measured, part put back"}.get(r["mode"], r["mode"])
        note = {"sorting": " · bench & container …", "ok_no_bench_contact": " · no bench contact",
                "part_lost_at_bench": " · part LOST at the bench"}.get(status, "")
        self.l_res_meta.set_text(f"#{r['n']} · {r['time']} · {mode}{note}")
        self.r_reason.set_text(r.get("reason") or "")
        if r.get("truth") and status in MEASURED + ("sorting", EMPTY):
            pred = r["pred"] if status != EMPTY else "none"
            right = pred == EXPECTED[r["truth"]]
            self.r_check.set_text(("Correct" if right else "Wrong") + f" — actual part: {LABEL[r['truth']]}")
            self.r_check_icon.name = "check_circle" if right else "cancel"
            self.r_check_icon.style(f"color:{STATUS['good'] if right else STATUS['critical']}")
            self.r_check_icon.set_visibility(True)
        else:
            self.r_check.set_text("Unlabelled part — pick the actual part to evaluate accuracy"
                                  if status in MEASURED + ("sorting",) else "")
            self.r_check_icon.set_visibility(False)

    def _upd_kpis(self, M: Dict[str, Any]) -> None:
        c = self.c
        res = [r for r in c.results if r["mode"] != "dry"]
        ok = [r for r in res if r["status"] in SORTED]
        times = [r["cycle_s"] for r in ok if r["mode"] == "sort" and r.get("cycle_s")]
        avg = sum(times) / len(times) if times else None
        n_fail = sum(1 for r in res if r["status"] in ("error", "aborted", "part_lost_at_bench"))
        n_empty = sum(1 for r in res if r["status"] == EMPTY)
        benched = [r for r in res if r.get("bench_contact") is not None]
        n_contact = sum(1 for r in benched if r["bench_contact"])
        desc = [r["bench_descent_mm"] for r in benched if r["bench_contact"] and r.get("bench_descent_mm") is not None]
        per = " · ".join(f"{LABEL[k].replace('HDD ', '')} {v:.2f}" for k, v in M["f1"].items() if v is not None)

        def put(key: str, value: str, sub: str, alert: bool = False) -> None:
            box, v, s = self.k[key]
            v.set_text(value)
            s.set_text(sub)
            if self._changed("kalert_" + key, alert):
                box.classes(replace="kpi alert" if alert else "kpi")

        put("cycles", str(len(res)), f"{len(ok)} sorted ok · {n_fail} failed/stopped")
        put("acc", f"{M['acc'] * 100:.0f}%" if M["acc"] is not None else "—",
            f"{M['correct']}/{M['n']} labelled" if M["n"] else "label parts to evaluate")
        put("f1", f"{M['macro_f1']:.2f}" if M["macro_f1"] is not None else "—", per or "per class")
        put("avg", f"{avg:.1f} s" if avg else "—", f"last {times[-1]:.1f} s" if times else "sort cycles")
        put("rate", f"{60.0 / avg:.1f}/min" if avg else "—", "parts per minute")
        put("bench", f"{n_contact}/{len(benched)}" if benched else "—",
            f"avg descent {sum(desc) / len(desc):.0f} mm" if desc else "force-guarded rest",
            bool(benched) and n_contact < len(benched))
        put("empty", str(n_empty), "no part at the holder")
        put("sec", str(M["security"]), "data-security risk" if M["security"] else "none — good", M["security"] > 0)

    def _render_cm(self, M: Dict[str, Any]) -> None:
        th = self.th
        cm = M["cm"]
        self.cm_box.clear()
        self.cm_box.style(f"grid-template-columns: 86px repeat({len(PRED_COLS)}, minmax(0, 1fr)) 50px")
        ink = th["ink"]
        with self.cm_box:
            ui.label("actual ↓  decision →").classes("cm-corner")
            for p in PRED_COLS:
                with ui.element("div").classes("cm-head"):
                    sw = ui.element("span").classes("swatch-sm" + (" swatch-ring" if p == "none" else ""))
                    sw.style(f"background:{self._col(p)}")
                    ui.label(LABEL[p])
            ui.label("recall").classes("cm-head")
            for t in TRUTH_ROWS:
                tot = sum(cm[t].values())
                with ui.element("div").classes("cm-rowlab"):
                    ui.element("span").classes("swatch-sm").style(f"background:{self._col(t)}")
                    ui.label(LABEL[t])
                for p in PRED_COLS:
                    v = cm[t][p]
                    alpha = (0.10 + 0.80 * v / tot) if v and tot else 0.0
                    bg = _rgba(ink, alpha) if v else th["raised"]
                    fg = th["page"] if alpha > 0.5 else th["ink"] if v else th["muted"]
                    cls = "cm-cell" + (" expected" if EXPECTED[t] == p else "")
                    with ui.element("div").classes(cls).style(f"background:{bg};color:{fg}"):
                        ui.label(str(v) if v else "·")
                        if v and t in ("hdd25", "hdd35") and p == "ssd":
                            ui.icon("warning").tooltip("HDD sorted as SSD — data-security risk")
                rec = M["recall"][t]
                ui.label(f"{rec * 100:.0f}%" if rec is not None else "—").classes("cm-recall")
        n = M["n"]
        self.l_cm.set_text(f"{n} labelled cycles · accuracy {M['acc'] * 100:.0f}% · macro F1 {M['macro_f1']:.2f}"
                           if n and M["macro_f1"] is not None else
                           "Pick the part label before each cycle (or in the table) to fill the matrix.")

    def _upd_table(self) -> None:
        th = self.th
        status_col = {"ok": STATUS["good"], "ok_no_bench_contact": STATUS["warning"], EMPTY: STATUS["warning"],
                      "part_lost_at_bench": STATUS["critical"], "error": STATUS["critical"],
                      "aborted": STATUS["serious"]}
        rows = []
        for r in self.c.results:
            pred = r["pred"] if r["status"] in MEASURED else ("none" if r["status"] == EMPTY else None)
            bc = r.get("bench_contact")
            bench = "—" if bc is None else (r.get("bench", "").replace("Bench ", "") + (" ✓ " if bc else " ✗ ")
                                            + f"{r.get('bench_descent_mm') or 0:.0f} mm").strip()
            rows.append(dict(
                n=r["n"], time=r["time"], tag=r["tag"], mode=r["mode"], truth=r["truth"] or "",
                pred_label=LABEL.get(pred, "—") if pred else "—", pred_color=self._col(pred) if pred else th["grid"],
                mass=f"{r['mass_g']:.1f}" if r.get("mass_g") is not None else "—",
                width=f"{r['width_mm']:.2f}" if r.get("width_mm") is not None else "—", bench=bench,
                cycle=f"{r['cycle_s']:.1f}" if r.get("cycle_s") else "—",
                status=STATUS_TEXT.get(r["status"], r["status"]),
                status_color=status_col.get(r["status"], th["axis"]), note=r.get("reason") or ""))
        self.table.rows = rows
        self.table.update()


# ═══════════════════════════════════════════════════════════════════ 4. main
def main() -> None:
    ctrl = DashboardController(Settings.load(SETTINGS_FILE))
    app.on_shutdown(ctrl.shutdown)

    @ui.page("/", title=APP_TITLE)
    def index(view: str = "") -> None:
        Page(ctrl, worker=view == "worker")

    ctrl.info(f"{APP_TITLE} started — robot {ctrl.s.robot_ip}")
    ui.run(host="127.0.0.1", title=APP_TITLE, reload=False, show_welcome_message=False)


if __name__ == "__main__":
    main()
