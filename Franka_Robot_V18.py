#!/usr/bin/env python3
"""
Franka_Robot_V18.py - sorting and robot setup for Franka_Dashboard_V18.py
(hand-guided teach-in, tare and weighing at a weigh spot, force-guarded rest on the testing bench and sorting
of HDDs/SSDs into containers with a Franka Emika Panda and Franka Hand, panda-py / libfranka 0.9.2, FCI).

Not started on its own: Franka_Dashboard_V18.py imports this module, so keep both files in the same folder.

    python3 Franka_Dashboard_V18.py      # robot unlocked in Desk, FCI active

This file holds everything that moves, sorts or configures the robot:
  constants       path model, step sets, CSV columns, input limits from the Franka manual
  settings        every adjustable variable (saved to dashboard_settings.json)
  kinematics      DH geometry of the 3D twin (poses come from panda_py.fk)
  classification  classify() with the trained sklearn model from drive_model.joblib (train_drive_model.py), plus
                  the rest load on the bench
  controller      robot connection, jobs, sorting cycle, teach-in, event log and session files
  robot panels    web UI for connection, parameters, sorting cycle, teach-in, test drive and event log

The digital twin, telemetry, charts, confusion matrix, KPIs, results table and session summary are in
Franka_Dashboard_V18.py.

Sections:  1 constants · 2 settings · 3 kinematics · 4 classification · 5 controller · 6 robot panels
"""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import logging
import math
import platform
import random
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import panda_py
from nicegui import ui
from panda_py import controllers, fk, libfranka
from panda_py.constants import JOINT_POSITION_START


# ═══════════════════════════════════════════════════════════════════ 1. constants
APP_TITLE = "Franka Dashboard V18"
SETTINGS_FILE = Path("dashboard_settings.json")

# V6: one test bench per form factor (franka_sort.py still has a single 'to_bench'); V8: weigh spot 'to_weigh'
# V9: own way back to the start point from every container (home_*), so the arm does not cut through obstacles
# V11: short taught exit path out of each bench (out_bench*) instead of a fixed straight lift
PATHS = ["to_holder", "to_weigh", "to_bench25", "to_bench35", "out_bench25", "out_bench35", "to_hdd35", "to_hdd25",
         "to_ssd", "home_hdd35", "home_hdd25", "home_ssd"]            # required
# V18: to_reject25/35 take parts that failed the (simulated) bench test from the bench exit to the reject container
OPTIONAL_PATHS = ["start_to_weigh", "to_reject", "home_reject", "to_reject25", "to_reject35"]
ALL_PATHS = ["start_to_weigh"] + PATHS + ["to_reject25", "to_reject35", "to_reject", "home_reject"]  # route order
CLASS_TO_PATH = {"hdd35": "to_hdd35", "hdd25": "to_hdd25", "ssd": "to_ssd", "reject": "to_reject"}
BENCH_PATH = {"ssd": "to_bench25", "hdd25": "to_bench25", "hdd35": "to_bench35"}   # SSDs (2.5″) use the 2.5″ bench
HOME_PATH = {"hdd35": "home_hdd35", "hdd25": "home_hdd25", "ssd": "home_ssd", "reject": "home_reject"}
HOME_CLASS = {v: k for k, v in HOME_PATH.items()}
BENCH_OUT = {"to_bench25": "out_bench25", "to_bench35": "out_bench35"}
BENCH_REJECT = {"to_bench25": "to_reject25", "to_bench35": "to_reject35"}   # V18: defective after the bench test
SEAT_XY_TOL_MM, SEAT_ANG_TOL_DEG = 5.0, 4.0   # V18: exit path must start straight below the end of the bench path
# the taught paths in driving order (LIFT = straight lift out of the holder) — used by 'Drive to end of previous path'
LIFT = "<lift>"
CHAIN_BEFORE = {
    "start_to_weigh": [], "to_holder": [], "to_weigh": ["to_holder", LIFT], "to_bench25": ["to_holder", LIFT, "to_weigh"],
    "to_bench35": ["to_holder", LIFT, "to_weigh"], "out_bench25": ["to_holder", LIFT, "to_weigh", "to_bench25"],
    "out_bench35": ["to_holder", LIFT, "to_weigh", "to_bench35"],
    "to_hdd25": ["to_holder", LIFT, "to_weigh", "to_bench25", "out_bench25"],
    "to_ssd": ["to_holder", LIFT, "to_weigh", "to_bench25", "out_bench25"],
    "to_hdd35": ["to_holder", LIFT, "to_weigh", "to_bench35", "out_bench35"],
    "home_hdd25": ["to_holder", LIFT, "to_weigh", "to_bench25", "out_bench25", "to_hdd25"],
    "home_ssd": ["to_holder", LIFT, "to_weigh", "to_bench25", "out_bench25", "to_ssd"],
    "home_hdd35": ["to_holder", LIFT, "to_weigh", "to_bench35", "out_bench35", "to_hdd35"],
    "to_reject": ["to_holder", LIFT, "to_weigh"], "home_reject": ["to_holder", LIFT, "to_weigh", "to_reject"],
    "to_reject25": ["to_holder", LIFT, "to_weigh", "to_bench25", "out_bench25"],
    "to_reject35": ["to_holder", LIFT, "to_weigh", "to_bench35", "out_bench35"],
}
ROUTES = {   # presets for 'Test drive' (driven in ALL_PATHS order)
    "ssd": ["to_holder", "to_weigh", "to_bench25", "out_bench25", "to_ssd", "home_ssd"],
    "hdd25": ["to_holder", "to_weigh", "to_bench25", "out_bench25", "to_hdd25", "home_hdd25"],
    "hdd35": ["to_holder", "to_weigh", "to_bench35", "out_bench35", "to_hdd35", "home_hdd35"],
    "reject": ["to_holder", "to_weigh", "to_reject", "home_reject"],
    "bad25": ["to_holder", "to_weigh", "to_bench25", "out_bench25", "to_reject25", "home_reject"],
    "bad35": ["to_holder", "to_weigh", "to_bench35", "out_bench35", "to_reject35", "home_reject"],
}
PATH_CLASS = {**{v: k for k, v in CLASS_TO_PATH.items()}, "to_reject25": "reject", "to_reject35": "reject"}
PATH_LABEL = {"start_to_weigh": "Start → weigh spot", "to_holder": "Holder", "to_weigh": "Weigh spot", "out_bench25": "Exit 2.5″ bench",
              "out_bench35": "Exit 3.5″ bench", "to_bench25": "Bench 2.5″", "to_bench35": "Bench 3.5″", "to_hdd35": "HDD 3.5″",
              "to_hdd25": "HDD 2.5″", "to_ssd": "SSD", "to_reject": "Reject", "home_hdd35": "HDD 3.5″ → start",
              "home_hdd25": "HDD 2.5″ → start", "home_ssd": "SSD → start", "home_reject": "Reject → start",
              "to_reject25": "2.5″ exit → reject", "to_reject35": "3.5″ exit → reject"}
PATH_HINT = {
    "start_to_weigh": "optional · start point → … → weigh spot: the empty trip for the tare, driven backwards "
                      "afterwards (→ start → to_holder). Without it the arm uses to_weigh both ways",
    "to_holder": "start point → … → grasp pose at the sliding holder (LAST point = grasp). The empty arm always "
                 "approaches the holder along this path",
    "to_weigh": "above the holder (after the lift) → … → weigh spot in free air (≥ 2 points; LAST point = "
                "weigh spot). Driven with the part for weighing (and backwards when a part is put back)",
    "to_bench25": "weigh spot → … → approach pose 2–5 cm ABOVE the 2.5″ test bench (HDD 2.5″ and SSD; "
                  "descent is force-guarded)",
    "to_bench35": "weigh spot → … → approach pose 2–5 cm ABOVE the 3.5″ test bench (descent is force-guarded)",
    "out_bench25": "short exit from the 2.5″ bench: FIRST point = where the part sits in the bench (gripper on it) "
                   "→ … → clear of the bench. Driven with the part after the re-grasp",
    "out_bench35": "short exit from the 3.5″ bench: FIRST point = where the part sits in the bench (gripper on it) "
                   "→ … → clear of the bench. Driven with the part after the re-grasp",
    "to_hdd35": "end of the 3.5″ bench exit → … → drop pose over the 3.5″ HDD container",
    "to_hdd25": "end of the 2.5″ bench exit → … → drop pose over the 2.5″ HDD container",
    "to_ssd": "end of the 2.5″ bench exit → … → drop pose over the SSD container",
    "to_reject": "optional · weigh spot → … → drop pose over a reject container (rejects skip the bench)",
    "home_hdd35": "drop pose over the 3.5″ HDD container → … around obstacles … → start point "
                  "(≥ 2 points; the start point is added automatically at the end)",
    "home_hdd25": "drop pose over the 2.5″ HDD container → … around obstacles … → start point "
                  "(≥ 2 points; the start point is added automatically at the end)",
    "home_ssd": "drop pose over the SSD container → … around obstacles … → start point "
                "(≥ 2 points; the start point is added automatically at the end)",
    "home_reject": "needed for to_reject and for the bench test · reject container → … → start point",
    "to_reject25": "needed when the bench test can fail · end of the 2.5″ bench exit → … → drop pose over the reject "
                   "container (same pose as the end of to_reject)",
    "to_reject35": "needed when the bench test can fail · end of the 3.5″ bench exit → … → drop pose over the reject "
                   "container (same pose as the end of to_reject)",
}
SORTED = ("ok", "ok_no_bench_contact")                 # franka_sort.py status values
MEASURED = SORTED + ("part_lost_at_bench",)            # weight / width / class exist
EMPTY = "no_part_at_holder"
STATUS_TEXT = {"ok": "ok", "ok_no_bench_contact": "no bench contact", EMPTY: "holder empty",
               "part_lost_at_bench": "lost at bench", "error": "error", "aborted": "stopped", "running": "running",
               "sorting": "running"}
LABEL = {"ssd": "SSD", "hdd25": "HDD 2.5″", "hdd35": "HDD 3.5″", "reject": "Reject", "other": "Other",
         "none": "No grasp", "": "Unlabeled"}
TRUTH_ROWS = ["ssd", "hdd25", "hdd35", "other"]
PRED_COLS = ["ssd", "hdd25", "hdd35", "reject", "none"]
EXPECTED = {"ssd": "ssd", "hdd25": "hdd25", "hdd35": "hdd35", "other": "reject"}
NICE_MODE = {"kIdle": "Idle", "kMove": "Moving", "kGuiding": "Guiding", "kReflex": "Reflex",
             "kUserStopped": "User stop", "kAutomaticErrorRecovery": "Recovering", "kOther": "Other"}

# Limits from the Franka Emika manual (technical data, "Exercise Unterlagen/Franka Manual.pdf"):
# payload 3 kg, Hand ≈ 1 kg, Hand grasping force up to 70 N, travel 80 mm at 30 mm/s, Cartesian speed ≤ 2 m/s.
PAYLOAD_KG, HAND_KG = 3.0, 1.0
MAX_PART_G = (PAYLOAD_KG - HAND_KG) * 1000.0
HAND_FORCE_MAX_N, HAND_TRAVEL_MM, HAND_SPEED_MAX_MMS = 70.0, 80.0, 30.0

STEP_SETS = {
    # V12: open → empty to the weigh spot → tare → to holder → grasp → lift → to_weigh → weigh → bench …
    # ("to_weigh" and "tare" are skipped when the stored tare is reused — 'tare_every_part' off)
    "sort": [("open", "Open"), ("to_weigh", "Empty to weigh spot"), ("tare", "Tare"), ("to_holder", "To holder"),
             ("grasp", "Grasp"), ("lift", "Lift & to weigh spot"), ("weigh", "Weigh"),
             ("classify", "Classify"), ("to_bench", "To bench"),
             ("bench", "Lower onto bench"), ("bench_open", "Open at bench"), ("bench_test", "Test"),
             ("regrasp", "Re-grasp"), ("bench_out", "Exit path"), ("to_bin", "To container"), ("release", "Release"),
             ("home", "Home")],
    "measure": [("open", "Open"), ("to_weigh", "Empty to weigh spot"), ("tare", "Tare"), ("to_holder", "To holder"),
                ("grasp", "Grasp"), ("lift", "Lift & to weigh spot"), ("weigh", "Weigh"),
                ("classify", "Classify"), ("put_back", "Put back"), ("home", "Home")],
    "dry": [("open", "Open"), ("to_holder", "To holder"), ("lift", "Lift & to weigh spot"), ("to_bench", "To bench"),
            ("bench_out", "Exit path"), ("to_bin", "To container"), ("home", "Home")],
}
# first 17 columns = franka_sort.py logs/sort_<stamp>.csv, then dashboard extras
CSV_FIELDS = ["part", "t_start", "t_end", "cycle_s", "status", "label", "class", "correct",
              "mass_g", "width_mm", "dFx_N", "dFy_N", "dFz_N",
              "bench_contact", "bench_descent_mm", "bench_rest_load_g", "error",
              "mode", "tag", "force_std_N", "note", "bench", "regrasp_width_mm",
              "tare_age_s", "confidence", "bench_test"]       # V18: model probability, simulated bench test result

# Input ranges: UI fields, the settings file and every cycle are clamped to these (Hand / payload from the manual)
LIMITS: Dict[str, Tuple[float, float]] = {
    "speed": (0.05, 1.0), "home_speed": (0.05, 1.0), "lift_mm": (5.0, 200.0), "goal_tol_rad": (0.005, 0.5),
    "grasp_force_n": (1.0, HAND_FORCE_MAX_N), "grasp_speed_mms": (5.0, HAND_SPEED_MAX_MMS),
    "open_speed_mms": (5.0, HAND_SPEED_MAX_MMS), "grasp_width_mm": (0.0, HAND_TRAVEL_MM),
    "eps_inner_mm": (0.0, HAND_TRAVEL_MM), "eps_outer_mm": (0.0, HAND_TRAVEL_MM), "empty_width_mm": (0.0, 20.0),
    "settle_s": (0.0, 5.0), "measure_s": (0.1, 10.0), "gravity": (9.0, 10.5), "mass_scale": (0.5, 2.0),
    "mass_offset_g": (-200.0, 200.0), "contact_g": (1.0, 500.0),
    "descent_speed_mms": (1.0, 50.0), "bench_hold_s": (0.3, 3.0),
    "bench_avg_ms": (5.0, 500.0), "bench_pause_s": (0.0, 120.0), "bench_open_mm": (1.0, HAND_TRAVEL_MM),
    "contact_pct": (5.0, 90.0), "ssd_min_conf_pct": (0.0, 99.0), "defect_pct": (0.0, 100.0),
    "tare_max_age_min": (0.0, 480.0),
    "ssd_mass_min_g": (0.0, MAX_PART_G), "hdd25_mass_max_g": (0.0, MAX_PART_G),
    "w25_min_mm": (0.0, HAND_TRAVEL_MM), "w25_max_mm": (0.0, HAND_TRAVEL_MM), "hdd35_mass_min_g": (0.0, MAX_PART_G),
    "hdd35_mass_max_g": (0.0, MAX_PART_G), "hdd35_w_max_mm": (0.0, HAND_TRAVEL_MM),
    "auto_max_empty": (1, 50), "auto_pause_s": (0.0, 30.0),
}
_TRAVEL = f"Franka manual: Hand travel {HAND_TRAVEL_MM:g} mm"
_PAYLOAD = f"Franka manual: payload {PAYLOAD_KG:g} kg incl. Hand ≈ {HAND_KG:g} kg → ≤ {MAX_PART_G:.0f} g per part"
DOC_NOTE = {
    "grasp_force_n": f"Franka manual: Hand grasping force up to {HAND_FORCE_MAX_N:g} N",
    "grasp_speed_mms": f"Franka manual: Hand travel speed {HAND_SPEED_MAX_MMS:g} mm/s",
    "open_speed_mms": f"Franka manual: Hand travel speed {HAND_SPEED_MAX_MMS:g} mm/s",
    **{k: _TRAVEL for k in ("grasp_width_mm", "eps_inner_mm", "eps_outer_mm", "bench_open_mm", "w25_min_mm",
                            "w25_max_mm", "hdd35_w_max_mm")},
    **{k: _PAYLOAD for k in ("ssd_mass_min_g", "hdd25_mass_max_g", "hdd35_mass_min_g", "hdd35_mass_max_g")},
}

# Status colours (also the dashboard's chart and badge colours)
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")


# ═══════════════════════════════════════════════════════════════════ 2. settings
@dataclass
class Settings:
    """Every adjustable variable. Units as shown in the UI (mm, g, s); defaults = franka_sort.py."""
    # connection
    robot_ip: str = "172.16.0.2"
    # motion            (franka_sort.py: --speed 0.2, --lift 0.05, go_home speed 0.15)
    speed: float = 0.20
    home_speed: float = 0.15
    lift_mm: float = 50.0
    goal_tol_rad: float = 0.05
    confirm_moves: bool = True
    auto_recover: bool = True
    # gripper           (franka_sort.py: grasp(0.0, 0.05, --force 40, 0.005, max_w), move(max_w, 0.1));
    #                   speeds capped at the manual's Hand travel speed of 30 mm/s
    grasp_force_n: float = 40.0
    grasp_speed_mms: float = 30.0
    grasp_width_mm: float = 0.0
    eps_inner_mm: float = 5.0
    eps_outer_mm: float = 0.0          # 0 = max opening width (as in franka_sort.py)
    open_speed_mms: float = 30.0
    empty_width_mm: float = 2.0
    homing_on_connect: bool = False
    # weighing          (franka_sort.py: mean_force(duration=1.0, settle=0.5), G = 9.81)
    settle_s: float = 1.0              # V13: 0.5 s was too short on the robot (tare σ 0.28 N instead of 0.07 N)
    measure_s: float = 1.0
    gravity: float = 9.81
    force_mode: str = "norm"           # "norm" = |ΔF| (franka_sort.py) · "z" = |ΔFz|
    mass_scale: float = 1.0
    mass_offset_g: float = 0.0
    tare_every_part: bool = True       # V8: empty trip to the weigh spot + tare before every part; False = reuse
    tare_max_age_min: float = 30.0     # only if tare_every_part is off: renew the stored tare when older; 0 = never
    # testing bench     (franka_sort.py: --contact-g 40, --max-descent 0.08, --descent-speed 0.01, --descent)
    contact_g: float = 40.0
    contact_pct: float = 35.0          # V17: contact also needs rest load ≤ this % of the part (60 g SSD → 21 g)
    descent_speed_mms: float = 10.0
    bench_hold_s: float = 0.7
    bench_avg_ms: float = 50.0
    bench_pause_s: float = 10.0        # test time on the bench (V6 default 10 s; franka_sort.py: 3 s)
    bench_wait: bool = False           # keep the part on the bench until 'Continue' (franka_sort.py --bench-wait)
    bench_release: bool = True         # V6: open the gripper on the bench during the test, then re-grasp
    bench_open_mm: float = 10.0        # open this much wider than the measured part width (fits narrow benches)
    bench_open_full: bool = True       # V17: open fully when the part is placed (else part width + bench_open_mm)
    defect_pct: float = 50.0           # V18: simulated bench test — chance that a part is defective (→ reject)
    # teach-in          (franka_sort.py: --teach-buttons)
    teach_buttons: bool = False
    # classification    (V18: trained model, drive_model.json)
    ssd_min_conf_pct: float = 60.0     # V18: model must be this sure for SSD, else the likelier HDD (0 = argmax)
    reject_enabled: bool = False
    ssd_mass_min_g: float = 30.0
    hdd25_mass_max_g: float = 180.0
    w25_min_mm: float = 5.0
    w25_max_mm: float = 16.0
    hdd35_mass_min_g: float = 300.0
    hdd35_mass_max_g: float = 900.0
    hdd35_w_max_mm: float = 30.0
    # auto mode
    auto_max_empty: int = 3
    auto_pause_s: float = 1.5
    # files & data
    waypoints_file: str = "waypoints.json"
    results_dir: str = "logs"
    trial_tag: str = ""
    # ui
    dark: bool = True
    settings_version: int = 6

    @property
    def grasp_speed(self) -> float:        # m/s for libfranka
        return self.grasp_speed_mms / 1000.0

    @property
    def open_speed(self) -> float:         # m/s for libfranka
        return self.open_speed_mms / 1000.0

    def clamp(self) -> "Settings":
        """Clamp every numeric value to LIMITS (in place)."""
        defaults = Settings()
        for name, (lo, hi) in LIMITS.items():
            v = min(max(float(getattr(self, name)), lo), hi)
            setattr(self, name, int(round(v)) if isinstance(getattr(defaults, name), int) else float(v))
        return self

    @classmethod
    def load(cls, path: Path) -> "Settings":
        s = cls()
        data = json.loads(path.read_text()) if path.exists() else {}
        for f in fields(cls):
            if f.name in data:
                setattr(s, f.name, type(getattr(s, f.name))(data[f.name]))
        s.settings_version = cls().settings_version
        return s.clamp()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


# ═══════════════════════════════════════════════════════════════════ 3. kinematics
# Panda modified DH parameters (a_{i-1}, d_i, alpha_{i-1}) and flange length: geometry of the 3D twin.
# Poses (Franka Hand TCP in the base frame) come from panda_py.fk.
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
FLANGE = 0.107


# ═══════════════════════════════════════════════════════════════════ 4. classification
# V18: trained classifier — sklearn pipeline (StandardScaler + LogisticRegression) saved by train_drive_model.py with
# joblib as {"model", "points", "note"} next to this file, read once at start
DRIVE_MODEL = joblib.load(Path(__file__).resolve().with_name("drive_model.joblib"))
MODEL = DRIVE_MODEL["model"]
DRIVE_MODEL_NOTE = DRIVE_MODEL["note"]


def model_proba(width_mm: Any, mass_g: Any) -> np.ndarray:
    """Class probabilities (columns = MODEL.classes_) for one or many parts. Features as in training: gripper
    opening (= drive thickness) and ln(weight)."""
    return MODEL.predict_proba(np.column_stack([np.atleast_1d(width_mm),
                                                np.log(np.clip(np.atleast_1d(mass_g), 1.0, None))]))


def classify(mass_g: float, width_mm: float, s: Settings,
             p: Optional[Dict[str, float]] = None) -> Tuple[str, str, float]:
    """Decision for a measured part → (class, reason, model probability of that class). V18: trained model; an SSD
    needs ssd_min_conf_pct, otherwise the likelier HDD class wins (HDD → SSD = data-security risk). Then the optional
    plausibility windows: a part outside the window of its class becomes 'reject'. p: probabilities if already known."""
    p = p or dict(zip(MODEL.classes_, model_proba(width_mm, mass_g)[0]))
    cls = max(p, key=p.get)
    guarded = cls == "ssd" and p["ssd"] * 100.0 < s.ssd_min_conf_pct
    if guarded:
        cls = "hdd25" if p["hdd25"] >= p["hdd35"] else "hdd35"
    conf = p[cls]
    why = (f"model: {LABEL[cls]} {conf * 100:.0f} % at width {width_mm:.1f} mm, weight {mass_g:.0f} g ("
           + " · ".join(f"{LABEL[c]} {v * 100:.0f} %" for c, v in p.items()) + ")"
           + (f" — SSD needs ≥ {s.ssd_min_conf_pct:g} %" if guarded else ""))
    if not s.reject_enabled:
        return cls, why, conf
    issues = []
    if cls == "hdd35":
        if not s.hdd35_mass_min_g <= mass_g <= s.hdd35_mass_max_g:
            issues.append(f"weight {mass_g:.0f} g outside {s.hdd35_mass_min_g:g}–{s.hdd35_mass_max_g:g} g")
        if width_mm > s.hdd35_w_max_mm:
            issues.append(f"width {width_mm:.1f} mm > {s.hdd35_w_max_mm:g} mm")
    else:
        if not s.w25_min_mm <= width_mm <= s.w25_max_mm:
            issues.append(f"width {width_mm:.1f} mm outside 2.5″ range {s.w25_min_mm:g}–{s.w25_max_mm:g} mm")
        if cls == "ssd" and mass_g < s.ssd_mass_min_g:
            issues.append(f"weight {mass_g:.0f} g < SSD minimum {s.ssd_mass_min_g:g} g")
        if cls == "hdd25" and mass_g > s.hdd25_mass_max_g:
            issues.append(f"weight {mass_g:.0f} g > 2.5″ HDD maximum {s.hdd25_mass_max_g:g} g")
    if issues:
        return "reject", f"looks like {LABEL[cls]}, but " + "; ".join(issues) + f" ({why})", conf
    return cls, why, conf


def rest_load(f_now, f_hold, d_f, mass_kg: float, g: float = 9.81) -> float:
    """franka_sort.rest_load(): load still hanging in the gripper [kg]. f_hold = force while holding the part
    still above the bench (local offset, so pose-dependent model errors cancel); projecting on the measured
    weight direction d_f ignores lateral noise and sign."""
    n = float(np.linalg.norm(d_f))
    if n < 1e-9:
        return mass_kg
    w = np.asarray(d_f, dtype=float) / n
    return mass_kg - float((np.asarray(f_hold, dtype=float) - np.asarray(f_now, dtype=float)) @ w) / g


# ═══════════════════════════════════════════════════════════════════ 5. controller
class Abort(Exception):
    """Raised inside a job when STOP was pressed."""


class MotionError(RuntimeError):
    """A motion did not reach its goal (reflex, user stop, collision, ...)."""


def _errs(errors: Any) -> str:
    return repr(errors).strip("[] ").replace('"', "") if errors else ""


def _fmt(v: Any, nd: int) -> str:
    return "" if v is None else f"{float(v):.{nd}f}"


_LEVEL = {"debug": "DEBUG", "info": "INFO", "ok": "INFO", "warn": "WARNING", "error": "ERROR"}


def _iso(t: Optional[float]) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds") if t else ""


def _short(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else repr(v)


class _PandaLogHandler(logging.Handler):
    """Routes panda_py's own log messages (logger 'panda') into the dashboard log."""

    def __init__(self, ctrl: "Controller"):
        super().__init__(logging.DEBUG)
        self.ctrl = ctrl

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        lvl = "error" if record.levelno >= logging.ERROR else "warn" if record.levelno >= logging.WARNING else "debug"
        self.ctrl._log(lvl, f"panda_py: {msg}")
        if "Control loop interrup" in msg:       # panda_py spells it "interruped"
            self.ctrl._loop_err_t = time.time()
            if self.ctrl.teaching and self.ctrl.job is None:
                self.ctrl.teaching = False       # V10: the dashboard showed "ON" although the robot had stopped it
                self.ctrl.warn("Teaching mode was stopped by the robot (reflex — guided too fast or against "
                               "something?). Press 'Recover errors', then enable teaching again.")
                self.ctrl.toast("Teaching mode stopped by the robot — Recover errors, then enable again", "negative")


class Controller:
    """Robot connection, jobs (teach-in, sorting cycles), telemetry and results - shared by all browser tabs.

    Thread model: one job thread at a time issues robot commands, always while holding hw_lock;
    the telemetry thread reads state only when that is safe for the installed panda_py version."""

    def __init__(self, settings: Settings, settings_path: Path = SETTINGS_FILE):
        self.s = settings
        self.settings_path = settings_path
        # hardware
        self.panda: Any = None
        self.robot: Any = None
        self.gripper: Any = None
        self.hw_lock = threading.RLock()
        self.connected = False
        self.lib_info = ""
        self.teaching = False
        self.gripper_busy = False
        self.max_width = 0.0
        self.motion_label: Optional[str] = None
        # jobs
        self._job_lock = threading.Lock()
        self.job: Optional[str] = None
        self.job_t0 = 0.0
        self.abort = threading.Event()
        self.auto_on = False
        # cycles & results
        self.steps: List[Dict[str, Any]] = []
        self.cycle_n = 0
        self.cycle_t0 = 0.0
        self.cycle_t1: Optional[float] = None
        self.measure: Dict[str, Any] = dict(tare=[], load=[], t0=0.0, tare_mean=None, load_mean=None, phase=None)
        self.tare: Optional[Dict[str, Any]] = None    # stored empty-gripper force at the weighing pose (V7)
        self._loop_err_t = 0.0                        # time of the last panda_py "Control loop interrupted"
        self.lift_ref: Optional[Dict[str, Any]] = None  # V12: remembered pose above the holder (+ file)
        self._gaps_logged = -1
        self._last_add_t = 0.0                        # debounce for 'Add point (W)'

        self.results: List[Dict[str, Any]] = []
        self.last: Optional[Dict[str, Any]] = None
        self.truth = ""
        self._pending_truth: Dict[int, str] = {}     # V18: worker feedback for the part that is still sorting
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_t0 = time.time()
        self.step_log: List[Dict[str, Any]] = []      # every step of every part (→ _steps.csv, summary)
        self.counters: Dict[str, int] = {}            # STOPs, restarts, recoveries, warnings … (→ summary)
        self._rec_lock = threading.Lock()
        self.bench: Dict[str, Any] = dict(samples=[], phase=None, contact=None, descent=None, rest=None,
                                          waiting=False, limit=None)
        self._bench_go = threading.Event()
        self._logfile: Any = None
        self._logfile_path: Optional[Path] = None
        # teach-in
        self.waypoints: Dict[str, List[List[float]]] = {}
        self.draft_path = "to_holder"
        self.draft: List[List[float]] = []
        self._backed_up: set = set()
        # telemetry
        self.tele: Dict[str, Any] = dict(q=JOINT_POSITION_START.copy(), ee=fk(JOINT_POSITION_START)[:3, 3].copy(), f=np.zeros(3), mode="—",
                                         errors="", t=0.0, width=0.0, max_width=0.0, grasped=False, temp=None, hz=0.0)
        self.force_hist: deque = deque(maxlen=900)
        self._tele_stop = threading.Event()
        self._tele_thread: Optional[threading.Thread] = None
        # log & notifications
        self._log_lock = threading.Lock()
        self.log: deque = deque(maxlen=1000)
        self.log_seq = 0
        self.toasts: deque = deque(maxlen=50)
        self.toast_seq = 0
        # change counters polled by the UI
        self.ver = dict(steps=0, results=0, wp=0, draft=0, measure=0, tele=0, bench=0)
        self.load_waypoints(quiet=True)
        threading.Thread(target=self._persist_loop, daemon=True, name="settings").start()

    # ------------------------------------------------------------ logging
    def _log(self, level: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._log_lock:
            self.log_seq += 1
            self.log.append((self.log_seq, ts, level, str(msg)))
            if level in ("warn", "error"):
                self.count("warnings" if level == "warn" else "errors")
            # the file gets everything (also debug) with ms timestamps; the UI hides debug unless 'verbose'
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._write_logfile(f"{stamp}  {_LEVEL[level]:7s} {msg}")

    def _write_logfile(self, line: str) -> None:
        """Session log next to the CSV (like franka_sort.py's logs/sort_<stamp>.log); a new file starts with
        a header (versions, robot, waypoints, all settings) so every session can be reproduced later."""
        path = self.log_file()
        if self._logfile is None or self._logfile_path != path:
            if self._logfile is not None:
                self._logfile.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists() or path.stat().st_size == 0
            self._logfile, self._logfile_path = open(path, "a", encoding="utf-8"), path
            if new:
                self._logfile.write("\n".join(self._session_header()) + "\n")
        self._logfile.write(line + "\n")
        self._logfile.flush()

    def _session_header(self) -> List[str]:
        return [f"# {APP_TITLE} — session {self.session}",
                f"# started {datetime.now():%Y-%m-%d %H:%M:%S} on {socket.gethostname()} · Python "
                f"{platform.python_version()} · numpy {np.__version__} · nicegui {metadata.version('nicegui')} · "
                f"panda-python {metadata.version('panda-python')} · scikit-learn {metadata.version('scikit-learn')}",
                f"# robot {self.s.robot_ip} · trial tag '{self.s.trial_tag}' · log folder "
                f"{Path(self.s.results_dir).expanduser().resolve()}",
                f"# waypoints {self.wp_path().resolve()}: "
                + (", ".join(f"{k} {len(v)} pts" for k, v in self.waypoints.items() if v) or "none"),
                "# settings " + json.dumps(asdict(self.s), sort_keys=True),
                f"# model drive_model.joblib: {DRIVE_MODEL_NOTE}",
                f"# files of this session: dashboard_{self.session}.csv (one row per part), _steps.csv "
                "(every step), _weighing.csv (raw tare/load forces), _bench.csv (descent curve), _summary.md (KPIs)"]

    def count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def debug(self, m: str) -> None:
        self._log("debug", m)

    def info(self, m: str) -> None:
        self._log("info", m)

    def ok(self, m: str) -> None:
        self._log("ok", m)

    def warn(self, m: str) -> None:
        self._log("warn", m)

    def error(self, m: str) -> None:
        self._log("error", m)

    def toast(self, msg: str, kind: str = "info") -> None:
        with self._log_lock:
            self.toast_seq += 1
            self.toasts.append((self.toast_seq, msg, kind))

    # ------------------------------------------------------------ status helpers
    @property
    def busy(self) -> bool:
        return self.job is not None

    def wp_path(self) -> Path:
        return Path(self.s.waypoints_file).expanduser()

    def session_file(self) -> Path:
        return Path(self.s.results_dir).expanduser() / f"dashboard_{self.session}.csv"

    def log_file(self) -> Path:
        return Path(self.s.results_dir).expanduser() / f"dashboard_{self.session}.log"

    def session_path(self, suffix: str) -> Path:
        return Path(self.s.results_dir).expanduser() / f"dashboard_{self.session}{suffix}"

    def path_span(self, name: str) -> float:
        qs = np.asarray(self.waypoints.get(name) or [[0.0] * 7], dtype=float)
        return float(np.max(np.abs(qs - qs[0])))

    def path_issues(self) -> List[str]:
        """Checks of franka_sort.py run(): all paths taught, ≥ 2 waypoints, robot actually moves."""
        miss = self.missing_paths()
        if miss:
            return ["missing paths: " + ", ".join(miss)]
        out = [f"'{p}' needs ≥ 2 waypoints" for p in ("to_holder", "to_weigh", "to_bench25", "to_bench35")
               if len(self.waypoints[p]) < 2]
        flat = [p for p in PATHS if len(self.waypoints[p]) >= 2 and self.path_span(p) < 0.01]
        if flat:
            out.append("waypoints (almost) identical in " + ", ".join(flat) + " — robot would not move, teach again")
        if self.waypoints.get("to_reject") and not self.waypoints.get("home_reject"):
            out.append("'to_reject' is taught → teach 'home_reject' (way back from the reject container) too")
        need = [p for p in ("to_reject25", "to_reject35", "home_reject") if not self.waypoints.get(p)]
        if self.s.defect_pct > 0 and need:                  # V18: defective parts go from the bench to the reject bin
            out.append(f"bench test defect rate {self.s.defect_pct:g} % → teach {', '.join(need)} (or set it to 0)")
        return out

    @staticmethod
    def seat_offset(q_bench_end, q_seat) -> Tuple[float, float, float]:
        """V18: seat (first point of out_bench*) relative to the end of to_bench*: beside [mm], turned [°],
        below [mm] — from the taught joints via FK."""
        Ta, Ts = fk(np.asarray(q_bench_end, dtype=float)), fk(np.asarray(q_seat, dtype=float))
        dxy = float(np.linalg.norm(Ts[:2, 3] - Ta[:2, 3])) * 1000.0
        ang = math.degrees(math.acos(float(np.clip((np.trace(Ta[:3, :3].T @ Ts[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        return dxy, ang, float(Ta[2, 3] - Ts[2, 3]) * 1000.0

    def bench_geometry(self, bench_path: str) -> Optional[Tuple[bool, str]]:
        """V18: the part goes straight down into the bench and is picked up there, so out_bench* has to start
        straight below the end of to_bench*. None while one of the two paths is not taught."""
        out_name = BENCH_OUT[bench_path]
        a, o = self.waypoints.get(bench_path), self.waypoints.get(out_name)
        if not a or not o:
            return None
        dxy, ang, dz = self.seat_offset(a[-1], o[0])
        if dxy <= SEAT_XY_TOL_MM and ang <= SEAT_ANG_TOL_DEG and -1.0 <= dz <= 200.0:
            return True, f"{out_name} starts {max(dz, 0.0):.0f} mm straight below the end of {bench_path}"
        where = f"{dxy:.0f} mm beside, {ang:.0f}° turned and " + (f"{dz:.0f} mm below" if dz >= 0 else f"{-dz:.0f} mm ABOVE")
        return False, (f"{out_name} starts {where} the end of {bench_path} — it must start straight below "
                       f"(≤ {SEAT_XY_TOL_MM:g} mm, ≤ {SEAT_ANG_TOL_DEG:g}°). Re-teach {out_name}: 'Drive to end of "
                       "previous path', then ▼ until the part sits in the bench, W, then the way out")

    def bench_blockers(self, mode: str, target: Optional[str] = None) -> List[str]:
        """V18: benches this cycle can visit whose exit path does not start straight below the bench path."""
        if mode == "measure" or (mode == "dry" and target == "reject"):
            return []
        benches = [BENCH_PATH[target]] if mode == "dry" and target in BENCH_PATH else list(BENCH_OUT)
        out = []
        for b in dict.fromkeys(benches):
            g = self.bench_geometry(b)
            if g is not None and not g[0]:
                out.append(g[1])
        return out

    def _link_targets(self, name: str) -> Tuple[Optional[Tuple[np.ndarray, str, float]],
                                                 Optional[Tuple[np.ndarray, str, float]]]:
        """V13: where a path must start / end so that no untaught move happens: (joints, label, tolerance [rad])."""
        w = {k: [np.asarray(x, dtype=float) for x in v] for k, v in self.waypoints.items() if v}
        start = (w["to_holder"][0], "start point", 0.15) if "to_holder" in w else None
        weigh = (w["to_weigh"][-1], "weigh spot", 0.15) if "to_weigh" in w else None

        def end_of(n: str, tol: float = 0.15):
            return (w[n][-1], f"end of {n}", tol) if n in w else None
        st: Any = None
        en: Any = None
        if name == "start_to_weigh":
            st, en = start, weigh
        elif name == "to_weigh":
            r = self._lift_ref(w["to_holder"][-1], self.s) if "to_holder" in w else None   # also loads the file
            st = (np.asarray(r["q_up"]), "lift pose above the holder", 0.15) if r else None
        elif name in ("to_bench25", "to_bench35", "to_reject"):
            st = weigh
        elif name in ("out_bench25", "out_bench35"):     # V17: approach → seat is the bench descent (no joint jump)
            st = None
        elif name in ("to_hdd25", "to_ssd"):
            st = end_of("out_bench25")
        elif name == "to_hdd35":
            st = end_of("out_bench35")
        elif name == "to_reject25":                      # V18: all reject paths end at the same drop pose
            st, en = end_of("out_bench25"), end_of("to_reject")
        elif name == "to_reject35":
            st, en = end_of("out_bench35"), end_of("to_reject") or end_of("to_reject25")
        elif name == "home_reject":
            st, en = end_of("to_reject") or end_of("to_reject25") or end_of("to_reject35"), start
        elif name in HOME_CLASS:
            st, en = end_of(CLASS_TO_PATH[HOME_CLASS[name]]), start
        return st, en

    def path_gaps(self) -> List[Tuple[str, float]]:
        """V13: untaught moves between paths (the program drives them as one joint interpolation) — largest first."""
        out: List[Tuple[str, float]] = []
        for name in ALL_PATHS:
            qs = self.waypoints.get(name)
            if not qs:
                continue
            st, en = self._link_targets(name)
            if st is not None:
                d = float(np.max(np.abs(np.asarray(qs[0]) - st[0])))
                if d > st[2]:
                    out.append((f"{st[1]} → {name}", d))
            if en is not None:
                d = float(np.max(np.abs(np.asarray(qs[-1]) - en[0])))
                if d > en[2]:
                    out.append((f"{name} → {en[1]}", d))
        return sorted(out, key=lambda x: -x[1])

    def missing_paths(self) -> List[str]:
        return [p for p in PATHS if not self.waypoints.get(p)]

    def cycle_blockers(self, ignore_busy: bool = False) -> List[str]:
        out = []
        if not self.connected:
            out.append("not connected")
        if self.busy and not ignore_busy:
            out.append(f"busy ({self.job})")
        if self.teaching:
            out.append("teaching mode is on")
        out += self.path_issues()
        if self.connected and self.max_width <= 0.01:
            out.append("gripper not homed (max width 0) → Home gripper")
        return out

    def readiness(self) -> List[Tuple[str, bool, str]]:
        miss = self.missing_paths()
        issues = self.path_issues()
        gaps = self.path_gaps()
        geo = [g for g in (self.bench_geometry(b) for b in BENCH_OUT) if g is not None]
        mode, errs = self.tele.get("mode", "—"), self.tele.get("errors", "")
        robot_ok = self.connected and mode not in ("kReflex", "kUserStopped", "kOther") and not errs
        return [
            ("Connected", self.connected, self.s.robot_ip if self.connected else "press Connect"),
            (f"Paths {len(PATHS) - len(miss)}/{len(PATHS)}", not issues,
             ("missing: " + ", ".join(miss)) if miss else (issues[0] if issues else "taught")),
            ("Paths connected", not gaps, "every path starts where the previous one ends" if not gaps else
             f"{len(gaps)} gap(s) — worst: {gaps[0][0]} {gaps[0][1]:.2f} rad"),
            ("Bench seats", all(g[0] for g in geo), "not taught yet" if not geo else
             next((g[1] for g in geo if not g[0]), " · ".join(g[1] for g in geo))),
            ("Gripper", self.connected and self.max_width > 0.01,
             f"opens {self.max_width * 1000:.1f} mm" if self.max_width > 0.01 else "not homed"),
            ("Robot " + (NICE_MODE.get(mode, mode) if self.connected else "—"), robot_ok, errs or "no errors"),
            ("Teaching off", not self.teaching, "arm is free" if self.teaching else "arm is stiff"),
        ]

    def near_home(self, tol: float = 0.05) -> bool:
        if not self.waypoints.get("to_holder"):
            return True
        return float(np.max(np.abs(np.asarray(self.tele["q"]) - np.asarray(self.waypoints["to_holder"][0])))) < tol

    # ------------------------------------------------------------ jobs
    def run_job(self, name: str, fn: Callable, *args) -> bool:
        with self._job_lock:
            if self.job is not None:
                self.toast(f"Busy ({self.job}) — wait or press STOP", "warning")
                return False
            self.job, self.job_t0 = name, time.time()
            self.abort.clear()
        self.debug(f"job '{name}' started")
        threading.Thread(target=self._job_main, args=(name, fn, args), daemon=True, name="job").start()
        return True

    def _job_main(self, name: str, fn: Callable, args: tuple) -> None:
        t0, outcome = time.time(), "done"
        try:
            fn(*args)
        except Abort:
            outcome = "stopped"
            self.warn(f"{name}: stopped by user — the robot holds its position. "
                      "Use 'Recover errors' and 'Go to start' when the workspace is clear.")
            self.toast(f"{name} stopped", "warning")
        except Exception as e:
            outcome = "failed"
            self.error(f"{name} failed: {type(e).__name__}: {e}")
            self.debug("traceback:\n" + traceback.format_exc().rstrip())
            self.toast(f"{name} failed: {e}", "negative")
        finally:
            self.motion_label = None
            self.gripper_busy = False
            self.job = None
            self.debug(f"job '{name}' {outcome} after {time.time() - t0:.2f} s")
            if self.results or self.step_log:
                self._write_summary()

    def _check_abort(self) -> None:
        if self.abort.is_set():
            raise Abort()

    def _sleep(self, seconds: float) -> None:
        if seconds > 0 and self.abort.wait(seconds):
            raise Abort()

    def stop(self) -> None:
        """Software stop: preempts the running motion, ends auto mode / teaching. NOT a safety function."""
        self.auto_on = False
        self.abort.set()
        self.count("stop")
        self.warn("STOP pressed — stopping motion (software stop, not a safety function)"
                  + (f" · job '{self.job}'" if self.job else "")
                  + (f" · motion '{self.motion_label}'" if self.motion_label else ""))
        threading.Thread(target=self._hw_stop, daemon=True, name="stop").start()

    def _hw_stop(self) -> None:
        robot, gripper = self.robot, self.gripper
        if robot is not None:
            try:
                robot.stop()
            except Exception as e:
                self.debug(f"robot.stop(): {e}")
        if gripper is not None and self.gripper_busy:
            try:
                gripper.stop()
            except Exception as e:
                self.debug(f"gripper.stop(): {e}")
        if self.teaching and self.panda is not None:
            try:
                with self.hw_lock:
                    self.panda.teaching_mode(False)
            except Exception as e:
                self.debug(f"teaching_mode(False): {e}")
            self.teaching = False
            self.info("Teaching mode switched off by STOP")

    # ------------------------------------------------------------ connection
    def connect(self) -> bool:
        if self.connected:
            return False
        return self.run_job("connect", self._connect)

    def _connect(self) -> None:
        host = self.s.robot_ip.strip()
        self.info(f"Connecting to robot {host} (FCI) …")
        lg = logging.getLogger("panda")
        if not any(isinstance(h, _PandaLogHandler) for h in lg.handlers):
            lg.addHandler(_PandaLogHandler(self))
        lg.setLevel(logging.INFO)
        lg.propagate = False
        try:
            panda = panda_py.Panda(host)
            gripper = libfranka.Gripper(host)
        except Exception as e:
            raise RuntimeError(f"{e} — check IP/cable, that FCI is activated in Desk and the robot is unlocked") from e
        self.panda, self.gripper, self.robot = panda, gripper, panda.get_robot()
        self.lib_info = f"panda_py {metadata.version('panda-python')}"
        self.connected, self.teaching = True, False
        self.tare = None                               # new connection → tare again before the first part
        self.load_waypoints(quiet=True)
        if self.s.homing_on_connect:
            self._gripper_homing()
        gs = self._gripper_state()
        if float(gs.max_width) < 0.01:
            self.warn("Gripper reports max width 0 — run 'Home gripper' before sorting")
        self._ingest_state(self._fresh_state())
        self._start_telemetry()
        self.count("connect")
        self.debug(f"robot at connect: mode {self.tele['mode']}, errors '{self.tele['errors']}', "
                   f"q = {np.round(self.tele['q'], 3).tolist()} rad, gripper max width {gs.max_width * 1000:.1f} mm, "
                   f"width {gs.width * 1000:.1f} mm, grasped {bool(gs.is_grasped)}")
        self.ok(f"Connected to {host} · {self.lib_info}")
        self.toast(f"Connected to {host}", "positive")

    def disconnect(self) -> bool:
        return self.run_job("disconnect", self._disconnect)

    def _disconnect(self) -> None:
        self._stop_telemetry()
        if self.teaching and self.panda is not None:
            with contextlib.suppress(Exception):
                self.panda.teaching_mode(False)
        self.panda = self.robot = self.gripper = None
        self.connected = self.teaching = False
        self.max_width = 0.0
        self.tele.update(mode="—", errors="")
        self.info("Disconnected")

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            if self.busy:
                self.stop()
                time.sleep(0.3)
            self._stop_telemetry()
            if self.teaching and self.panda is not None:
                self.panda.teaching_mode(False)
        with contextlib.suppress(Exception):
            self.s.save(self.settings_path)
        with contextlib.suppress(Exception):
            if self.results or self.step_log:
                self._write_summary()
            self.info(f"Dashboard closed — session files: {self.session_path('*').resolve()}")

    def _persist_loop(self) -> None:
        last = None
        while True:
            time.sleep(1.0)
            d = asdict(self.s)
            if d != last:
                if last is not None:
                    self.s.save(self.settings_path)
                    changes = [f"{k} {_short(last.get(k))} → {_short(v)}" for k, v in d.items() if last.get(k) != v]
                    self.debug("settings changed: " + ", ".join(changes))
                last = d

    def reset_settings(self) -> None:
        keep = ("robot_ip", "dark", "waypoints_file", "results_dir", "trial_tag", "settings_version")
        d = Settings()
        for f in fields(Settings):
            if f.name not in keep:
                setattr(self.s, f.name, getattr(d, f.name))
        self.ok("All parameters reset to the franka_sort.py defaults")

    # ------------------------------------------------------------ low-level robot helpers
    def _fresh_state(self):
        """Fresh robot state; only call when no motion runs and teaching mode is off."""
        with self.hw_lock:
            return self.robot.read_once()

    def _pose(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.hw_lock:
            p = np.array(self.panda.get_position(), dtype=float).copy()
            o = np.array(self.panda.get_orientation(), dtype=float).copy()
        return p, o

    def _move_joints(self, waypoints, speed: float, label: str) -> None:
        self._check_abort()
        wps = [np.asarray(q, dtype=float) for q in waypoints]
        if not wps:
            return
        self.motion_label = label
        t0 = time.time()
        try:
            with self.hw_lock:
                ok = self.panda.move_to_joint_position(wps, speed_factor=float(min(max(speed, 0.01), 1.0)))
        finally:
            self.motion_label = None
        self.debug(f"move '{label}': {len(wps)} waypoint(s), speed {float(speed):.2f} → {ok} "
                   f"in {time.time() - t0:.2f} s")
        self._check_abort()
        self._after_motion(ok, label, q_goal=wps[-1])

    def _move_pose(self, position: np.ndarray, orientation: np.ndarray, speed: float, label: str) -> None:
        self._check_abort()
        self.motion_label = label
        t0 = time.time()
        try:
            with self.hw_lock:
                ok = self.panda.move_to_pose(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float),
                                             speed_factor=float(min(max(speed, 0.01), 1.0)))
        finally:
            self.motion_label = None
        self.debug(f"move '{label}': to {np.round(np.asarray(position, dtype=float) * 1000, 1).tolist()} mm, "
                   f"speed {float(speed):.2f} → {ok} in {time.time() - t0:.2f} s")
        self._check_abort()
        self._after_motion(ok, label, p_goal=np.asarray(position, dtype=float))

    def _after_motion(self, ok: Any, label: str, q_goal=None, p_goal=None) -> None:
        """panda_py returns False instead of raising when a motion fails - check it (franka_sort.py does not)."""
        st = self._fresh_state()
        self._ingest_state(st)
        if ok is not False:
            return
        mode = st.robot_mode.name
        errs = _errs(st.current_errors) or _errs(st.last_motion_errors)
        if mode in ("kReflex", "kUserStopped", "kOther") or _errs(st.current_errors):
            raise MotionError(f"{label}: motion aborted — robot mode {NICE_MODE.get(mode, mode)}"
                              + (f", errors: {errs}" if errs else ""))
        if q_goal is not None:
            dev = float(np.max(np.abs(np.asarray(st.q, dtype=float) - q_goal)))
            if dev > float(self.s.goal_tol_rad):
                raise MotionError(f"{label}: stopped {dev:.3f} rad from its goal" + (f" ({errs})" if errs else ""))
            self.warn(f"{label}: panda_py reported failure, but the arm is {dev:.3f} rad from the goal "
                      f"(within tolerance) — continuing")
        if p_goal is not None:
            T = np.asarray(st.O_T_EE, dtype=float).reshape(4, 4, order="F")
            dev = float(np.linalg.norm(T[:3, 3] - p_goal))
            if dev > 0.01:
                raise MotionError(f"{label}: stopped {dev * 1000:.0f} mm from its goal" + (f" ({errs})" if errs else ""))
            self.warn(f"{label}: panda_py reported failure, but the arm is {dev * 1000:.1f} mm from the goal — continuing")

    def _grip_move(self, width: float, speed: float) -> bool:
        self._check_abort()
        self.gripper_busy = True
        t0 = time.time()
        try:
            ok = self.gripper.move(float(width), float(speed))
        finally:
            self.gripper_busy = False
        self.debug(f"gripper move to {width * 1000:.1f} mm at {speed * 1000:.0f} mm/s → {ok} "
                   f"in {time.time() - t0:.2f} s")
        self._check_abort()
        if ok is False:
            self.warn(f"gripper.move({width * 1000:.1f} mm) returned False")
        self._gripper_state()
        return ok

    def _grasp(self, s: Settings, max_w: float) -> bool:
        self._check_abort()
        eps_out = s.eps_outer_mm / 1000.0 if s.eps_outer_mm > 0 else max_w
        self.gripper_busy = True
        t0 = time.time()
        try:
            ok = self.gripper.grasp(s.grasp_width_mm / 1000.0, float(s.grasp_speed), float(s.grasp_force_n),
                                    s.eps_inner_mm / 1000.0, eps_out)
        finally:
            self.gripper_busy = False
        self.debug(f"gripper grasp: width {s.grasp_width_mm:g} mm, {s.grasp_speed_mms:g} mm/s, "
                   f"{s.grasp_force_n:g} N, eps inner {s.eps_inner_mm:g} mm / outer {eps_out * 1000:.1f} mm → {ok} "
                   f"in {time.time() - t0:.2f} s")
        self._check_abort()
        return bool(ok)

    def _gripper_state(self):
        gs = self.gripper.read_once()
        self._ingest_gripper(gs)
        return gs

    def _gripper_homing(self) -> None:
        self.info("Homing gripper …")
        self.gripper_busy = True
        try:
            self.gripper.homing()
        finally:
            self.gripper_busy = False
        gs = self._gripper_state()
        self.ok(f"Gripper homed — max width {gs.max_width * 1000:.1f} mm")

    def _open_width(self) -> float:
        return self.max_width if self.max_width > 0.01 else 0.08

    def _home_q(self) -> np.ndarray:
        if not self.waypoints.get("to_holder"):
            raise RuntimeError("no 'to_holder' path taught — the start point is its first waypoint")
        return np.asarray(self.waypoints["to_holder"][0], dtype=float)

    def _mean_force(self, phase: str, s: Settings) -> Tuple[np.ndarray, float]:
        """franka_sort.mean_force(): settle, then average O_F_ext_hat_K over 'measure_s' seconds."""
        self._sleep(float(s.settle_s))
        m = self.measure
        if phase == "tare" or float(m.get("t0") or 0.0) < self.cycle_t0:   # load with the stored tare → fresh chart
            m.update(tare=[], load=[], t0=time.time(), tare_mean=None, load_mean=None)
            if phase == "load" and self.tare is not None:
                m["tare_mean"] = [float(x) for x in self.tare["f"]]
        m["phase"] = phase
        buf, samples, last_push = m[phase], [], 0.0
        t_end = time.time() + max(0.05, float(s.measure_s))
        with self.hw_lock:
            while True:
                now = time.time()
                if now >= t_end and samples:
                    break
                f = np.asarray(self.robot.read_once().O_F_ext_hat_K[:3], dtype=float)
                samples.append(f)
                if now - last_push >= 0.01:
                    buf.append((now - m["t0"], float(f[0]), float(f[1]), float(f[2])))
                    self.force_hist.append((now, float(f[0]), float(f[1]), float(f[2])))
                    self.ver["measure"] += 1
                    last_push = now
                if self.abort.is_set():
                    m["phase"] = None
                    raise Abort()
        arr = np.asarray(samples)
        mean, std = arr.mean(axis=0), float(arr.std(axis=0).max())
        m[f"{phase}_mean"], m["phase"] = [float(x) for x in mean], None
        self.ver["measure"] += 1
        self.debug(f"{phase}: {len(samples)} samples, F = {np.round(mean, 3).tolist()} N, σ = {std:.3f} N")
        return mean, std

    @staticmethod
    def _path(*segments) -> List[np.ndarray]:
        """Join taught segments into one joint path, dropping waypoints (almost) identical to the previous one."""
        out: List[np.ndarray] = []
        for seg in segments:
            for q in seg:
                q = np.asarray(q, dtype=float)
                if not out or float(np.max(np.abs(q - out[-1]))) > 1e-3:
                    out.append(q)
        return out

    def _lift_file(self) -> Path:
        wpf = self.wp_path()
        return wpf.with_name(wpf.stem + "_lift.json")

    def _lift_ref(self, q_grasp: np.ndarray, s: Settings) -> Optional[Dict[str, Any]]:
        """V12: remembered pose straight above the grasp pose (from an earlier lift). V14: only used to check that
        to_weigh starts there (path links); the empty arm always takes to_holder. Invalid when to_holder or the
        lift height change."""
        r = self.lift_ref
        if r is None and self._lift_file().exists():
            d = json.loads(self._lift_file().read_text())
            r = self.lift_ref = {k: np.asarray(v, dtype=float) if isinstance(v, list) else v for k, v in d.items()}
        if r is None:
            return None
        if float(np.max(np.abs(np.asarray(r["grasp_q"]) - q_grasp))) > 1e-3 or abs(float(r["lift_mm"]) - s.lift_mm) > 0.5:
            return None
        return r

    def _store_lift_ref(self, q_up: np.ndarray, p0: np.ndarray, o0: np.ndarray, q_grasp: np.ndarray,
                        s: Settings) -> None:
        if self._lift_ref(q_grasp, s) is not None:
            return
        self.lift_ref = dict(q_up=np.asarray(q_up, dtype=float), p0=np.asarray(p0, dtype=float),
                             o0=np.asarray(o0, dtype=float), grasp_q=np.asarray(q_grasp, dtype=float),
                             lift_mm=float(s.lift_mm))
        self._lift_file().write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                 for k, v in self.lift_ref.items()}, indent=1))
        self.debug("  lift pose above the holder stored (to check that to_weigh starts there)")

    def _tare_for(self, q_weigh: np.ndarray, s: Settings) -> Optional[Dict[str, Any]]:
        """The stored tare if it may be reused, else None (→ empty trip to the weigh spot and tare).
        O_F_ext_hat_K has a pose-dependent bias, so tare and weighing must happen at the same weigh spot."""
        t = self.tare
        if s.tare_every_part or t is None:
            return None
        if float(np.max(np.abs(np.asarray(t["q"]) - q_weigh))) > 0.01:
            self.info("  stored tare is from another weigh spot (to_weigh re-taught) → new tare")
            return None
        age_min = (time.time() - t["t"]) / 60.0
        if s.tare_max_age_min > 0 and age_min > s.tare_max_age_min:
            self.info(f"  stored tare is {age_min:.0f} min old (> {s.tare_max_age_min:g} min) → new tare")
            return None
        return t

    def _store_tare(self, f: np.ndarray, sd: float, q_weigh: np.ndarray, where: str) -> None:
        self.tare = dict(f=np.asarray(f, dtype=float).copy(), sd=float(sd), q=np.asarray(q_weigh, dtype=float).copy(),
                         t=time.time(), where=where)
        self.count("tare")
        self.info(f"  tare stored ({where}): F = {np.round(np.asarray(f), 3).tolist()} N, σ = {sd:.3f} N")

    def retare(self) -> None:
        self.tare = None
        self.info("Tare cleared — the next part starts with an empty trip to the weigh spot (tare)")
        self.toast("The next part is tared first", "info")

    # ------------------------------------------------------------ robot actions
    def go_home(self) -> bool:
        return self.run_job("go to start", self._go_home)

    def _go_home(self) -> None:
        self._drive_home()
        self.ok("At the start point")

    def _route_home(self, q: np.ndarray) -> Tuple[List[np.ndarray], str]:
        """V10: way back to the start point along TAUGHT paths, starting at the waypoint nearest to q.
        Holder / weigh spot / bench paths are driven backwards, container and home paths forwards (→ home_*)."""
        wp = {k: [np.asarray(x, dtype=float) for x in v] for k, v in self.waypoints.items() if v}
        home = self._home_q()
        if float(np.max(np.abs(q - home))) < 0.02:
            return [], "already at the start point"
        to_holder = wp.get("to_holder", [home])
        back_home = list(reversed(to_holder[:-1])) or [home]
        to_weigh = wp.get("to_weigh", [])
        best: Optional[Tuple[float, str, int]] = None
        for name, qs in wp.items():
            for i, w in enumerate(qs):
                d = float(np.max(np.abs(q - w)))
                if best is None or d < best[0]:
                    best = (d, name, i)
        if best is None or best[0] > 0.3:
            near = f" (nearest: {best[1]} point {best[2] + 1}, {best[0]:.2f} rad away)" if best else ""
            raise RuntimeError("the arm is not close to any taught path" + near + " — guide it by hand (teaching "
                               "mode) next to a taught waypoint or the start point, then try again")
        d, name, i = best
        qs = wp[name]
        if name == "to_holder":
            route = list(reversed(qs[:i + 1]))
        elif name == "to_weigh":
            route = list(reversed(qs[:i + 1])) + back_home
        elif name in ("to_bench25", "to_bench35"):
            route = list(reversed(qs[:i + 1])) + list(reversed(to_weigh)) + back_home
        elif name in ("out_bench25", "out_bench35"):     # V13: forward like a part: exit → container path → home
            cont = next((c for c in (("to_hdd25", "to_ssd") if name.endswith("25") else ("to_hdd35",))
                         if c in wp and HOME_PATH[PATH_CLASS[c]] in wp), None)
            if cont:
                route = qs[i:] + wp[cont] + wp[HOME_PATH[PATH_CLASS[cont]]]
            else:
                bench = wp.get("to_bench25" if name.endswith("25") else "to_bench35", [])
                route = qs[i:] + list(reversed(bench)) + list(reversed(to_weigh)) + back_home
        elif name in HOME_CLASS:
            route = qs[i:]
        elif name in PATH_CLASS and HOME_PATH.get(PATH_CLASS[name]) in wp:     # container path → its way home
            route = qs[i:] + wp[HOME_PATH[PATH_CLASS[name]]]
        else:
            route = list(reversed(qs[:i + 1]))
        return self._path(route, [home]), f"via {name} (point {i + 1}, {d:.2f} rad away)"

    # ------------------------------------------------------------ test drive & teach helper (V11)
    def _segments(self, names: List[str], backwards: bool = False) -> List[Tuple[str, Any, str]]:
        """Selected paths in route order (ALL_PATHS); between to_holder and to_weigh the straight lift out of the
        holder is inserted, as in the cycle. Backwards: last path first, each one reversed (no lift)."""
        names = [n for n in ALL_PATHS if n in set(names)]
        missing = [n for n in names if not self.waypoints.get(n)]
        if missing:
            raise RuntimeError("not taught yet: " + ", ".join(missing))
        segs: List[Tuple[str, Any, str]] = []
        for k, n in enumerate(names):
            if n == "to_weigh" and k > 0 and names[k - 1] == "to_holder":
                segs.append((LIFT, None, "lift out of the holder"))
            w = [np.asarray(q, dtype=float) for q in self.waypoints[n]]
            if n in HOME_CLASS:                       # as in the cycle: the start point closes every way home
                w = self._path(w, [self._home_q()])
            segs.append((n, w, n))
        if backwards:
            segs = [(n, list(reversed(w)), f"{n} backwards") for n, w, _ in reversed(segs) if n != LIFT]
        return segs

    def plan_drive(self, segs: List[Tuple[str, Any, str]]) -> List[str]:
        """Direct (untaught) joint moves a test drive would make: to its first point and between the paths."""
        out, prev, prev_name = [], np.asarray(self.tele["q"], dtype=float), "current pose"
        for n, w, _ in segs:
            if n == LIFT:
                prev = None
                continue
            if prev is not None:
                d = float(np.max(np.abs(np.asarray(w[0]) - prev)))
                if d > 0.05:
                    out.append(f"{prev_name} → {n}: {d:.2f} rad")
            prev, prev_name = np.asarray(w[-1]), f"end of {n}"
        return out

    def _exec_segments(self, segs: List[Tuple[str, Any, str]]) -> None:
        s = copy.copy(self.s).clamp()
        for n, w, label in segs:
            if n == LIFT:
                p, o = self._pose()
                self._move_pose(p + np.array([0.0, 0.0, s.lift_mm / 1000.0]), o, s.speed, label)
            else:
                self._move_joints(w, s.speed, label)

    def drive_paths(self, names: List[str], backwards: bool = False) -> bool:
        if not self.connected:
            self.toast("Connect first", "warning")
            return False
        return self.run_job("test drive" + (" backwards" if backwards else ""), self._drive_paths, list(names),
                            bool(backwards))

    def _drive_paths(self, names: List[str], backwards: bool) -> None:
        segs = self._segments(names, backwards)
        if not segs:
            raise RuntimeError("no path selected")
        if self.teaching:
            self._set_teaching(False)
        q = np.asarray(self._fresh_state().q, dtype=float)
        d = float(np.max(np.abs(q - segs[0][1][0])))
        if d > 0.6:
            raise RuntimeError(f"the arm is {d:.2f} rad away from the first point of {segs[0][2]} — use 'Go to start' "
                               "or 'Drive to end of previous path' first")
        self.info("Test drive: " + " → ".join(lbl for _, _, lbl in segs))
        self._exec_segments(segs)
        self.ok("Test drive finished")

    def goto_path_start(self, name: str) -> bool:
        if not self.connected:
            self.toast("Connect first", "warning")
            return False
        return self.run_job(f"drive to the start of {name}", self._goto_path_start, name)

    def jog_z(self, mm: float) -> bool:
        """V18: move the gripper straight down (mm < 0) or up, keeping x, y and orientation — e.g. from the end of
        to_bench* straight into the bench for the first point of out_bench*."""
        if not self.connected:
            self.toast("Connect first", "warning")
            return False
        return self.run_job(f"straight {'down' if mm < 0 else 'up'} {abs(mm):g} mm", self._jog_z, float(mm))

    def _jog_z(self, mm: float) -> None:
        was_teaching = self.teaching
        if was_teaching:
            self._set_teaching(False)
        p, o = self._pose()
        self._move_pose(p + np.array([0.0, 0.0, mm / 1000.0]), o, 0.05,
                        f"straight {'down' if mm < 0 else 'up'} {abs(mm):g} mm")
        p1, _ = self._pose()
        self.info(f"Gripper now at z = {p1[2] * 1000:.1f} mm — press W to record this pose")
        if was_teaching:
            self._set_teaching(True)

    def chain_before(self, name: str) -> List[str]:
        """Taught paths that lead to the start of 'name' (V18: home_reject via to_reject25 if to_reject is missing)."""
        if name == "home_reject" and not self.waypoints.get("to_reject") and self.waypoints.get("to_reject25"):
            return CHAIN_BEFORE["to_reject25"] + ["to_reject25"]
        return CHAIN_BEFORE.get(name, [])

    def _goto_path_start(self, name: str) -> None:
        """Drive along the taught paths to where 'name' starts (= end of the previous path), e.g. to re-teach it."""
        chain = self.chain_before(name)
        missing = [n for n in chain if n != LIFT and not self.waypoints.get(n)]
        if missing:
            raise RuntimeError(f"teach {', '.join(missing)} first — '{name}' starts where they end")
        was_teaching = self.teaching
        if self.teaching:
            self._set_teaching(False)
        q = np.asarray(self._fresh_state().q, dtype=float)
        wps = {n: [np.asarray(x, dtype=float) for x in self.waypoints[n]] for n in chain if n != LIFT}
        start = None                          # already standing at the end of a chain path? continue from there
        for k in range(len(chain) - 1, -1, -1):
            n = chain[k]
            if n != LIFT and float(np.max(np.abs(q - wps[n][-1]))) < 0.03:
                start = k + 1
                break
        if start is None:
            self._drive_home()
            start = 0
        segs = [(n, wps.get(n), "lift out of the holder" if n == LIFT else n) for n in chain[start:]]
        if segs:
            self.info(f"Driving to the start of {name}: " + " → ".join(lbl for _, _, lbl in segs))
            self._exec_segments(segs)
        self.ok(f"At the start of {name} — teach it from here")
        if was_teaching:
            self._set_teaching(True)

    def _drive_home(self, speed: Optional[float] = None) -> None:
        q = np.asarray(self._fresh_state().q, dtype=float)
        route, how = self._route_home(q)
        if not route:
            self.info(f"Go to start: {how}")
            return
        self.info(f"Go to start {how} — {len(route)} waypoint(s) along taught paths")
        self._move_joints(route, float(speed if speed is not None else self.s.home_speed), f"to start {how}")

    def open_gripper(self) -> bool:
        return self.run_job("open gripper", lambda: self._grip_move(self._open_width(), self.s.open_speed))

    def home_gripper(self) -> bool:
        return self.run_job("home gripper", self._gripper_homing)

    def test_grasp(self) -> bool:
        return self.run_job("test grasp", self._test_grasp)

    def _test_grasp(self) -> None:
        ok = self._grasp(self.s, self._open_width())
        gs = self._gripper_state()
        w = gs.width * 1000
        verdict = "EMPTY" if w < self.s.empty_width_mm else "object"
        self.info(f"Test grasp: grasp() → {ok}, is_grasped = {bool(gs.is_grasped)}, width {w:.2f} mm → {verdict}")
        self.toast(f"Test grasp: {w:.2f} mm · is_grasped={bool(gs.is_grasped)} → {verdict}", "info")

    def recover(self) -> bool:
        return self.run_job("recover", self._recover)

    def _recover(self) -> None:
        self.count("recover")
        with self.hw_lock:
            self.panda.recover()
        self._ingest_state(self._fresh_state())
        self.ok(f"Error recovery done — robot mode {NICE_MODE.get(self.tele['mode'], self.tele['mode'])}")

    # ------------------------------------------------------------ teach-in
    def set_teaching(self, on: bool) -> bool:
        if not self.connected:
            self.toast("Connect first", "warning")
            return False
        return self.run_job("teaching mode on" if on else "teaching mode off", self._set_teaching, bool(on))

    def _set_teaching(self, on: bool) -> None:
        if on:                                   # V10: the robot rejects teaching mode while the user stop is pressed
            mode = self._fresh_state().robot_mode.name
            if mode == "kUserStopped":
                raise RuntimeError("the user stop is pressed (lights white) — release it (lights blue) for teaching "
                                   "mode, or switch on 'Guide with the wrist buttons'")
            if mode == "kReflex":
                with self.hw_lock:
                    self.panda.recover()
                self.info("Robot errors cleared before teaching mode")
        t0 = time.time()
        with self.hw_lock:
            self.panda.teaching_mode(on)
        if on:
            time.sleep(0.3)                      # a rejected controller reports within ~50 ms (panda_py log)
            if self._loop_err_t > t0:
                with self.hw_lock:
                    self.panda.teaching_mode(False)
                self.teaching = False
                raise RuntimeError("the robot rejected teaching mode (see the panda_py error above) — is the user "
                                   "stop released (lights blue)? Otherwise press 'Recover errors' and try again")
        self.teaching = on
        if on:
            self.ok("Teaching mode ON — gravity compensation, guide the arm by hand")
            self.toast("Teaching mode ON — the arm can be guided by hand", "warning")
        else:
            self.info("Teaching mode OFF — arm holds its position. Clear the workspace, then 'Go to start'.")

    def _read_q(self) -> np.ndarray:
        """franka_sort.current_q(): read_once() when no controller runs, else the controller's state."""
        return np.asarray((self.panda.get_state() if self.teaching else self._fresh_state()).q, dtype=float)

    def teach_add(self) -> None:
        if not self.connected:
            self.toast("Connect first", "warning")
            return
        if self.busy:
            self.toast(f"Busy ({self.job})", "warning")
            return
        now = time.time()
        if now - self._last_add_t < 0.6:       # V10: one key press was recorded twice (30–400 ms apart) on the robot
            self.debug("Add point ignored — pressed again within 0.6 s")
            return
        self._last_add_t = now
        try:
            q = self._read_q()
        except Exception as e:
            self.error(f"Could not read joint positions: {e}")
            return
        if not self.draft:                     # V13: does the new path start where it has to?
            st = self._link_targets(self.draft_path)[0]
            if st is not None and float(np.max(np.abs(q - st[0]))) > st[2]:
                d = float(np.max(np.abs(q - st[0])))
                self.warn(f"{self.draft_path}: first point is {d:.2f} rad from the {st[1]} — the robot would make "
                          f"an untaught move there. Better: 'Drive to end of previous path', then teach")
                self.toast(f"First point {d:.2f} rad from the {st[1]} — use 'Drive to end of previous path'", "warning")
        if self.draft and float(np.max(np.abs(q - np.asarray(self.draft[-1])))) < 0.005:
            self.warn("Pose barely changed since the last waypoint — not added. Move the arm further (if it did "
                      "move, the joint reading is not updating → try 'Guide with the wrist buttons').")
            self.toast("Same pose as the last waypoint — not added", "warning")
            return
        self.draft.append([round(float(x), 6) for x in q])
        self.ver["draft"] += 1
        self.info(f"{self.draft_path}: waypoint {len(self.draft)}  q = {np.round(q, 3).tolist()}")
        if not self.teaching and not self.s.teach_buttons:
            self.toast("Recorded the current pose (teaching mode is off)", "info")

    def teach_end_target(self) -> None:
        """V13: append the exact pose the path has to end at (start point for home_*, weigh spot for
        start_to_weigh), so there is no gap. Guide the arm close to it first."""
        en = self._link_targets(self.draft_path)[1]
        if en is None:
            self.toast("Only for start_to_weigh (→ weigh spot) and the home_* paths (→ start point)", "info")
            return
        if self.draft and float(np.max(np.abs(en[0] - np.asarray(self.draft[-1])))) < 0.005:
            self.toast(f"The draft already ends at the {en[1]}", "info")
            return
        self.draft.append([round(float(x), 6) for x in en[0]])
        self.ver["draft"] += 1
        self.info(f"{self.draft_path}: waypoint {len(self.draft)} = {en[1]} (exact pose)")

    def teach_undo(self) -> None:
        if self.draft:
            self.draft.pop()
            self.ver["draft"] += 1

    def teach_clear(self) -> None:
        self.draft = []
        self.ver["draft"] += 1

    def teach_select(self, name: str) -> bool:
        if self.draft and name != self.draft_path:
            self.toast(f"Save or clear the draft for {self.draft_path} first", "warning")
            return False
        self.draft_path = name
        self.ver["draft"] += 1
        return True

    def teach_save(self) -> None:
        name = self.draft_path
        if not self.draft:
            self.toast("Record waypoints first", "warning")
            return
        if name == "to_holder" and len(self.draft) < 2:
            self.toast("to_holder needs ≥ 2 waypoints: start point … grasp pose", "warning")
            return
        self.waypoints[name] = [list(q) for q in self.draft]
        try:
            self.save_waypoints()
        except Exception as e:
            self.error(f"Could not write {self.wp_path()}: {e}")
            return
        self.ok(f"Saved path {name} ({len(self.draft)} waypoints) → {self.wp_path().resolve()}")
        self.toast(f"Saved {name} ({len(self.draft)} waypoints)", "positive")
        gaps = [(lbl, d) for lbl, d in self.path_gaps() if name in lbl.split(" → ")]
        for lbl, d in gaps:                    # V13: tell right away when the new path leaves a gap
            self.warn(f"Gap after saving {name}: {lbl} {d:.2f} rad (untaught move)")
        if gaps:
            self.toast(f"{name}: {len(gaps)} gap(s) to the neighbouring paths — see the log", "warning")
        bench = name if name in BENCH_OUT else next((b for b, o in BENCH_OUT.items() if o == name), None)
        g = self.bench_geometry(bench) if bench else None
        if g is not None:                      # V18: seat straight below the end of the bench path?
            (self.ok if g[0] else self.warn)(("Bench seat OK: " if g[0] else "Bench seat NOT OK: ") + g[1])
            if not g[0]:
                self.toast(f"{BENCH_OUT[bench]} does not start straight below {bench} — see the log", "warning")
        self.draft = []
        self.ver["draft"] += 1

    def teach_delete(self, name: str) -> None:
        if self.waypoints.pop(name, None) is not None:
            self.save_waypoints()
            self.warn(f"Deleted path {name}")

    def load_waypoints(self, quiet: bool = False) -> None:
        path = self.wp_path()
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
            self.waypoints = {k: [[float(x) for x in q] for q in v] for k, v in data.items() if isinstance(v, list)}
            if not quiet:
                self.info(f"Loaded {path.resolve()}: " +
                          (", ".join(f"{k} ({len(v)})" for k, v in self.waypoints.items()) or "no paths"))
        except Exception as e:
            self.waypoints = {}
            self.error(f"Could not read {path}: {e}")
        self.ver["wp"] += 1

    def save_waypoints(self) -> None:
        path = self.wp_path()
        key = str(path.resolve())
        if path.exists() and key not in self._backed_up:
            backup = path.with_name(path.name + ".bak")
            backup.write_text(path.read_text())
            self._backed_up.add(key)
            self.info(f"Backup of the previous file → {backup.name}")
        path.write_text(json.dumps(self.waypoints, indent=2))     # same format as franka_sort.py
        self.ver["wp"] += 1

    # ------------------------------------------------------------ sorting cycle
    def run_cycle(self, mode: str = "sort", target: Optional[str] = None) -> bool:
        blockers = self.cycle_blockers()
        if mode == "dry" and target == "reject" and not self.waypoints.get("to_reject"):
            blockers.append("no 'to_reject' path taught")
        if mode == "dry" and target == "reject" and not self.waypoints.get("home_reject"):
            blockers.append("no 'home_reject' path taught")
        blockers += self.bench_blockers(mode, target)
        if blockers:
            self.toast("Cannot start: " + "; ".join(blockers), "warning")
            return False
        name = {"sort": "sort cycle", "measure": "measure-only cycle"}.get(mode, f"dry run → {LABEL.get(target, '')}")
        return self.run_job(name, self._cycle, mode, target)

    def _steps_init(self, mode: str) -> None:
        self.steps = [dict(key=k, label=lbl, status="pending", t0=None, t1=None) for k, lbl in STEP_SETS[mode]]
        self.ver["steps"] += 1

    @contextlib.contextmanager
    def _step(self, key: str, label: Optional[str] = None):
        st = next((x for x in self.steps if x["key"] == key), None)
        if st is None:
            st = dict(key=key, label=label or key, status="pending", t0=None, t1=None)
            self.steps.append(st)
        if label:
            st["label"] = label
        st.update(status="active", t0=time.time())
        self.ver["steps"] += 1
        try:
            yield
        except BaseException:
            st.update(status="failed", t1=time.time())
            self.ver["steps"] += 1
            raise
        st.update(status="done", t1=time.time())
        self.ver["steps"] += 1

    def _skip_pending(self, keep: Tuple[str, ...] = ()) -> None:
        for st in self.steps:
            if st["status"] == "pending" and st["key"] not in keep:
                st["status"] = "skipped"
        self.ver["steps"] += 1

    def _cycle(self, mode: str, target: Optional[str] = None) -> Dict[str, Any]:
        """One part (V8): start → weigh spot (empty: tare) → holder (grasp = width) → lift → weigh spot (weigh,
        classify) → test bench of its form factor (lower, open, test, re-grasp) → container → start.
        'measure' puts the part back into the holder; 'dry' drives the paths without a part."""
        bad = self.bench_blockers(mode, target)    # V18: also for auto mode — refuse before anything moves
        if bad:
            raise RuntimeError("; ".join(bad))
        s = copy.copy(self.s).clamp()             # parameter changes apply from the next part
        wp = {k: [np.asarray(q, dtype=float) for q in v] for k, v in self.waypoints.items() if v}
        to_holder = wp["to_holder"]
        back_home = list(reversed(to_holder[:-1]))  # same taught path back, without the grasp pose
        to_weigh = wp["to_weigh"]                    # above the holder → … → weigh spot (last point)
        q_weigh = to_weigh[-1]
        weigh_back = list(reversed(to_weigh[:-1]))   # weigh spot → … → first to_weigh point (above the holder)
        home_q = to_holder[0]

        def home_from(c: str) -> List[np.ndarray]:   # V9: taught way back from a container (+ start point)
            return self._path(wp[HOME_PATH[c]], [home_q])
        max_w = self._open_width()
        self.cycle_n += 1
        n = self.cycle_n
        self.cycle_t0, self.cycle_t1 = time.time(), None
        self._steps_init(mode)
        self.bench.update(samples=[], phase=None, contact=None, descent=None, rest=None, waiting=False, limit=None)
        self.ver["bench"] += 1
        truth = "" if mode == "dry" else (self.truth or "")
        res: Dict[str, Any] = dict(
            n=n, ts=datetime.now().isoformat(timespec="seconds"), t_end="", time=time.strftime("%H:%M:%S"),
            mode=mode, tag=s.trial_tag.strip(), truth=truth, pred=None, status="running", mass_g=None,
            width_mm=None, df=None, f_std=None, cycle_s=None, reason="",
            bench_contact=None, bench_descent_mm=None, bench_rest_g=None, bench="", regrasp_mm=None,
            tare_age_s=None, conf=None, test=None)
        self.info(f"--- Part #{n} started ({mode})" + (f" · label {LABEL[truth]}" if truth else ""))
        if self._gaps_logged != self.ver["wp"]:
            self._gaps_logged = self.ver["wp"]
            gaps = self.path_gaps()
            if gaps:
                self.warn("Untaught moves between paths (re-teach with 'Drive to end of previous path'): "
                          + "; ".join(f"{lbl} {d:.2f} rad" for lbl, d in gaps))
        try:
            with self._step("open"):
                self._grip_move(max_w, s.open_speed)
            # ---- 0) empty gripper: to the weigh spot, tare there, then to the holder (V12 order)
            tare = None if mode == "dry" else self._tare_for(q_weigh, s)
            tared_now = mode != "dry" and tare is None
            if tared_now:
                empty_way = "start_to_weigh" if "start_to_weigh" in wp else "to_weigh"
                empty_path = self._path(wp[empty_way], [q_weigh])   # V13: always ends exactly at the weigh spot
                with self._step("to_weigh", "Empty to weigh spot"):
                    self._move_joints(empty_path, s.speed, f"{empty_way} (empty)")
                with self._step("tare", "Tare (empty)"):
                    f_empty, sd_empty = self._mean_force("tare", s)
                    self._store_tare(f_empty, sd_empty, q_weigh, "empty gripper at the weigh spot")
                # V14: always the taught way to the holder — the empty trip backwards to the start point, then
                # to_holder. (V12/V13 took a shortcut: to_weigh backwards → above the holder → straight down.)
                with self._step("to_holder", "To holder (back to start → to_holder)"):
                    back = list(reversed(empty_path[:-1]))
                    self._move_joints(self._path(back, [home_q], to_holder), s.speed,
                                      f"{empty_way} backwards → start → to_holder")
            else:
                if mode != "dry":
                    f_empty, sd_empty = tare["f"], tare["sd"]
                    for k in ("to_weigh", "tare"):
                        self._skip_step(k)
                    self.debug(f"  stored tare reused ({(time.time() - tare['t']) / 60:.1f} min old, "
                               f"{tare['where']})")
                with self._step("to_holder"):
                    self._move_joints(to_holder, s.speed, "to_holder")
            if mode != "dry":
                res["tare_age_s"] = 0.0 if tared_now else time.time() - self.tare["t"]
            p0, o0 = self._pose()                      # grasp pose at the holder
            p_up = p0 + np.array([0.0, 0.0, s.lift_mm / 1000.0])

            def lift_to_weigh(what: str) -> np.ndarray:
                """Straight up out of the holder, then the taught to_weigh path. Returns the joints above the holder
                (end point of the way back) and remembers them for the next part."""
                self._move_pose(p_up, o0, s.speed, f"lift ({what})")
                q_up = np.asarray(self._fresh_state().q, dtype=float).copy()
                self._store_lift_ref(q_up, p0, o0, to_holder[-1], s)
                self._move_joints(to_weigh, s.speed, f"to_weigh ({what})")
                return q_up

            holder_from_weigh = lambda q_up: self._path(weigh_back, [q_up])   # noqa: E731 - weigh spot → above holder
            if mode == "dry":
                path_name = CLASS_TO_PATH[target]
                with self._step("lift"):
                    lift_to_weigh("dry run")
                if target == "reject":                 # rejects skip the bench
                    self._skip_pending(keep=("to_bin", "home"))
                else:
                    bench_path = BENCH_PATH[target]
                    with self._step("to_bench", f"To {PATH_LABEL[bench_path]}"):
                        self._move_joints(wp[bench_path], s.speed, bench_path)
                    out_name = BENCH_OUT[bench_path]
                    with self._step("bench_out", f"Exit path {PATH_LABEL[out_name]}"):   # no part → no force check
                        out_wps = wp[out_name]
                        pa, oa = self._pose()           # V18: same way as with a part: straight down to the seat
                        seat_z = self._bench_seat_z(out_name, out_wps[0], pa)
                        if pa[2] - seat_z > 0.001:
                            self._move_pose(np.array([pa[0], pa[1], seat_z]), oa, min(float(s.speed), 0.1),
                                            "straight down to the seat (dry run)")
                        self._move_joints(out_wps, s.speed, f"{out_name} (dry run)")
                with self._step("to_bin", f"To {LABEL[target]} container"):
                    self._move_joints(wp[path_name], s.speed, path_name)
                with self._step("home"):
                    self._move_joints(home_from(target), s.speed, HOME_PATH[target])
                res.update(status="ok", pred=target, reason="dry run — no part, no measurement, no bench descent")
                self.ok(f"#{n} dry run to the {LABEL[target]} container finished in {time.time() - self.cycle_t0:.1f} s")
                return res
            # ---- 1) holder: grasp (finger width = size)
            with self._step("grasp"):
                ok = self._grasp(s, max_w)
                gs = self._gripper_state()
                width_mm = float(gs.width) * 1000.0
                grasped = ok and bool(gs.is_grasped) and width_mm >= s.empty_width_mm
            if not grasped:
                why = (f"grasp() → {ok}, is_grasped = {bool(gs.is_grasped)}, finger width {width_mm:.2f} mm"
                       + (f" < {s.empty_width_mm:g} mm" if width_mm < s.empty_width_mm else ""))
                self._skip_pending(keep=("home",))
                with self._step("home", "Back to start"):
                    self._grip_move(max_w, s.open_speed)
                    self._move_pose(p_up, o0, s.speed, "lift")
                    self._move_joints(back_home, s.speed, "back to start")
                res.update(status=EMPTY, width_mm=width_mm, reason="grasp at holder failed — holder empty? " + why)
                self.warn(f"#{n} grasp at holder FAILED — holder empty? ({why})")
                self.toast(f"#{n}: nothing grasped — holder empty?", "warning")
                return res
            # ---- 2) lift, to the weigh spot, weigh (same pose as the tare)
            with self._step("lift"):
                q_up = lift_to_weigh("with part")
            with self._step("weigh"):
                f_loaded, sd_loaded = self._mean_force("load", s)
            d = f_loaded - f_empty
            raw_kg = (abs(float(d[2])) if s.force_mode == "z" else float(np.linalg.norm(d))) / s.gravity
            mass_g = raw_kg * 1000.0 * float(s.mass_scale) + float(s.mass_offset_g)
            if mass_g > MAX_PART_G:
                self.warn(f"#{n} measured {mass_g:.0f} g — more than the {MAX_PART_G:.0f} g a Panda can carry with the "
                          f"Hand ({PAYLOAD_KG:g} kg payload incl. ≈ {HAND_KG:g} kg Hand). Check the measurement.")
            with self._step("classify"):
                cls, why, conf = classify(mass_g, width_mm, s)
            res.update(pred=cls, mass_g=mass_g, width_mm=width_mm, df=[float(x) for x in d],
                       f_std=float(max(sd_empty, sd_loaded)), reason=why, conf=conf)
            self.last = dict(res, status="sorting")       # show the decision while the robot keeps moving
            self.ver["results"] += 1
            verdict = f" (label {LABEL[truth]}, {'OK' if EXPECTED[truth] == cls else 'WRONG'})" if truth else ""
            self.ok(f"#{n} measured at the weigh spot: {mass_g:.1f} g, {width_mm:.2f} mm -> {cls}{verdict}")
            if cls == "reject":
                self.warn(f"#{n} reject: {why}")
            # ---- measure only, or a reject without reject container: put the part back into the holder
            if mode == "measure" or (cls == "reject" and "to_reject" not in wp):
                if mode != "measure":
                    self.warn(f"#{n} no 'to_reject' path taught → putting the part back into the holder")
                key = "put_back" if mode == "measure" else "to_bin"
                self._skip_pending(keep=(key, "home"))
                with self._step(key, "Put back"):
                    self._move_joints(holder_from_weigh(q_up), s.speed, "to_weigh backwards → above holder")
                    self._move_pose(p0, o0, s.speed, "lower into holder")
                    self._grip_move(max_w, s.open_speed)
                    self._move_pose(p_up, o0, s.speed, "lift")
                with self._step("home"):
                    self._move_joints(back_home, s.speed, "back to start")
                res["status"] = "ok"
                return res
            if cls == "reject":                         # rejects skip the bench
                self._skip_pending(keep=("to_bin", "release", "home"))
                with self._step("to_bin", "To reject container"):
                    self._move_joints(wp["to_reject"], s.speed, "to_reject")
                with self._step("release"):
                    self._grip_move(max_w, s.open_speed)
                self.info(f"#{n} dropped into the reject container")
                with self._step("home"):
                    self._move_joints(home_from("reject"), s.speed, "home_reject")
                res["status"] = "ok"
                return res
            # ---- 2) test bench of its form factor: lower until the part rests, open, test, re-grasp, lift
            bench_path = BENCH_PATH[cls]
            res["bench"] = PATH_LABEL[bench_path]
            with self._step("to_bench", f"To {PATH_LABEL[bench_path]}"):
                self._move_joints(wp[bench_path], s.speed, bench_path)
            contact, descent, rest, held, w2 = self._rest_on_bench(s, raw_kg, d, width_mm, max_w,
                                                                   BENCH_OUT[bench_path], wp[BENCH_OUT[bench_path]])
            res.update(bench_contact=bool(contact), bench_descent_mm=descent * 1000.0, bench_rest_g=rest * 1000.0,
                       regrasp_mm=w2)
            if not held or not bool(self._gripper_state().is_grasped):
                self._skip_pending(keep=("home",))
                with self._step("home", "Back to start"):     # V13: empty via the container path and its way home —
                    self._move_joints(self._path(wp[CLASS_TO_PATH[cls]], home_from(cls)), s.speed,   # never back
                                      f"{CLASS_TO_PATH[cls]} (empty) → {HOME_PATH[cls]}")            # past the bench
                why = ("re-grasp at the bench failed — the part is still in the bench" if not held
                       else "part no longer in the gripper after the bench")
                res.update(status="part_lost_at_bench", reason=why)
                self.error(f"#{n} {why} — take it out of the {PATH_LABEL[bench_path]}")
                self.toast(f"#{n}: {why}", "negative")
                return res
            # ---- 3) V18: simulated bench test result — defective → reject container, working → its own container
            broken = random.random() * 100.0 < s.defect_pct
            res["test"] = self.last["test"] = "defective" if broken else "ok"
            self.ver["results"] += 1
            bin_cls = "reject" if broken else cls
            path_name = BENCH_REJECT[bench_path] if broken else CLASS_TO_PATH[cls]
            (self.warn if broken else self.ok)(f"#{n} bench test (simulated, {s.defect_pct:g} % defect rate): "
                                               + ("DEFECTIVE → reject container" if broken else
                                                  f"works → {LABEL[cls]} container"))
            # ---- 4) container: drop, then straight back to the start point
            with self._step("to_bin", f"To {LABEL[bin_cls]} container"):
                self._move_joints(wp[path_name], s.speed, path_name)
            with self._step("release"):
                self._grip_move(max_w, s.open_speed)
            self.info(f"#{n} dropped into container '{bin_cls}'")
            with self._step("home"):
                self._move_joints(home_from(bin_cls), s.speed, HOME_PATH[bin_cls])
            res["status"] = "ok" if contact else "ok_no_bench_contact"
            self.toast(f"#{n} → {LABEL[bin_cls]}" + (" (bench test: defective)" if broken else "")
                       + f" · {mass_g:.0f} g · {width_mm:.1f} mm" + ("" if contact else " · no bench contact"),
                       "positive" if contact and not broken else "warning")
            return res
        except Abort:
            res.update(status="aborted", reason="stopped by user")
            raise
        except Exception as e:
            if self.abort.is_set():
                res.update(status="aborted", reason="stopped by user")
                raise Abort() from e
            res.update(status="error", reason=f"{type(e).__name__}: {e}")
            self.error(f"Part #{n}: {type(e).__name__}: {e}")
            self.toast(f"Part #{n} failed: {e}", "negative")
            if s.auto_recover:                     # V10: clear the error only — never drive off by itself (after a
                self.count("auto_recover")         # collision the direct move to start collided again on the robot)
                try:
                    with self.hw_lock:
                        self.panda.recover()
                    self.warn("Robot errors cleared — the arm stays where it is. Check the cause, clear the way, then "
                              "'Go to start' (drives back along the taught paths)")
                except Exception as e2:
                    self.error(f"Recovery failed: {type(e2).__name__}: {e2} — check the robot in Desk, "
                               "then 'Recover errors'")
            return res
        finally:
            self.cycle_t1 = time.time()
            res["cycle_s"] = self.cycle_t1 - self.cycle_t0
            res["t_end"] = datetime.now().isoformat(timespec="seconds")
            self.bench.update(waiting=False, phase=None)
            self._skip_pending()
            self._record_cycle(res)
            if mode != "dry":
                self._add_result(res)
            self.info(f"--- Part #{n} {res['status']} in {res['cycle_s']:.1f} s")

    # ------------------------------------------------------------ testing bench (franka_sort.rest_on_bench)
    def _skip_step(self, key: str) -> None:
        for st in self.steps:
            if st["key"] == key and st["status"] == "pending":
                st["status"] = "skipped"
        self.ver["steps"] += 1

    def _bench_seat_z(self, out_name: str, q_seat: np.ndarray, p0: np.ndarray) -> float:
        """V18: height of the seat (first point of the exit path, taught with the part in the bench). The arm only
        moves straight down/up at the bench — never sideways (V17's line-up move hit the 2.5″ bench)."""
        z = float(fk(np.asarray(q_seat, dtype=float))[2, 3])
        self.info(f"  seat (first point of {out_name}) {(p0[2] - z) * 1000:.0f} mm straight below")
        return z

    def _rest_on_bench(self, s: Settings, mass_kg: float, d_f: np.ndarray, width_mm: float, max_w: float,
                       out_name: str, out_wps: List[np.ndarray]) -> Tuple[bool, float, float, bool, Optional[float]]:
        """At the test bench (V18, only straight down/up): lower the part force-guarded until it rests or reaches
        the seat height (first point of the exit path), open the gripper fully, 'test' for bench_pause_s
        (or until Continue), straight down to the seat with open jaws, grasp there, then the exit path.
        Returns contact, descent [m], rest load [kg], part held again, finger width after re-grasp [mm]."""
        p0, o0 = self._pose()                       # approach pose above the bench (end of to_bench*)
        seat_z = self._bench_seat_z(out_name, out_wps[0], p0)
        slow = min(s.speed, 0.1)
        thr_g = min(s.contact_g, s.contact_pct / 100.0 * mass_kg * 1000.0)
        with self._step("bench", "Lower onto bench"):
            max_dz = max(float(p0[2] - seat_z), 0.0)
            self.info(f"  lowering onto bench (stop at ≤ {thr_g:.0f} g rest load or at the seat {max_dz * 1000:.0f} mm "
                      "lower)")
            self.bench.update(samples=[], phase="descent", limit=thr_g)
            self.ver["bench"] += 1
            contact, descent, rest, reached = self._descend_impedance(s, mass_kg, d_f, p0, o0, max_dz, thr_g)
            at_seat = reached and not contact
            self.bench.update(contact=contact, descent=descent, rest=rest, phase="resting")
            self.ver["bench"] += 1
            if contact:
                self.ok(f"  resting on bench after {descent * 1000:.1f} mm (rest load {rest * 1000:.0f} g)")
            elif at_seat:
                self.ok(f"  at the taught seat after {descent * 1000:.1f} mm (rest load {rest * 1000:.0f} g) — "
                        "the part is where you taught it")
            else:
                self.warn(f"  NO bench contact within {max_dz * 1000:.0f} mm (rest load {rest * 1000:.0f} g) "
                          "— check bench height / bench path")
        # open only when the part really rests on the bench (or sits at the taught seat) — otherwise it would drop
        release = bool(s.bench_release) and (bool(contact) or at_seat)
        w_open = max_w if s.bench_open_full else min(max_w, width_mm / 1000.0 + float(s.bench_open_mm) / 1000.0)
        if release:
            with self._step("bench_open", "Open at bench"):
                self._grip_move(w_open, s.open_speed)
                self.info(f"  gripper opened {'fully' if s.bench_open_full else 'to'} {w_open * 1000:.1f} mm — "
                          "the part sits in the bench")
        else:
            self._skip_step("bench_open")
            if s.bench_release:
                self.warn("  part NOT released (no bench contact — it would drop); holding it during the test")
        test_lbl = "Test (Continue)" if s.bench_wait else f"Test {s.bench_pause_s:g} s"
        with self._step("bench_test", test_lbl):
            if s.bench_wait:
                self._wait_bench()
            else:
                self.info(f"  testing on the bench ({s.bench_pause_s:g} s) …")
                self._sleep(float(s.bench_pause_s))
        held, w2 = True, None
        exit_wps = list(out_wps)
        with self._step("regrasp", "Down to the seat, re-grasp" if release else "Hold"):
            if release:
                p_now, _ = self._pose()                 # the part sinks into the bench when released →
                if p_now[2] - seat_z > 0.001:           # straight down to where it sits, jaws open
                    self._move_pose(np.array([p0[0], p0[1], seat_z]), o0, slow,
                                    f"straight down {(p_now[2] - seat_z) * 1000:.0f} mm to the seat, jaws open")
                ok = self._grasp(s, max_w)
                gs = self._gripper_state()
                w2 = float(gs.width) * 1000.0
                held = bool(ok) and bool(gs.is_grasped) and w2 >= s.empty_width_mm
                (self.ok if held else self.error)(
                    f"  re-grasp at the bench: grasp() → {ok}, width {w2:.2f} mm (at the holder {width_mm:.2f} mm)")
                if not held:
                    self._grip_move(max_w, s.open_speed)    # leave the part in the bench, do not drag it along
            self.bench["phase"] = None
            self.ver["bench"] += 1
        with self._step("bench_out", f"Exit path {PATH_LABEL[out_name]}"):   # V11: taught short way out of the bench
            self._move_joints(exit_wps, s.speed, out_name + ("" if held else " (empty)"))
        return bool(contact) or at_seat, descent, rest, held, w2     # V17: at the taught seat counts as placed

    def _descend_impedance(self, s: Settings, mass_kg: float, d_f: np.ndarray, p0: np.ndarray,
                           o0: np.ndarray, max_dz: float, thr_g: float) -> Tuple[bool, float, float, bool]:
        """franka_sort.descend_impedance(): a CartesianImpedance controller lowers the part at the descent
        speed until the load still hanging in the gripper is ≤ thr_g, or the setpoint has been at max_dz
        (the seat) for 0.5 s. Returns contact, descent [m], rest load [kg], seat reached."""
        ctrl = controllers.CartesianImpedance()
        g, v, t_hold = s.gravity, s.descent_speed_mms / 1000.0, s.bench_hold_s
        buf: deque = deque(maxlen=int(round(s.bench_avg_ms)))   # ~1 sample per ms
        hold: List[np.ndarray] = []
        f_hold: Optional[np.ndarray] = None
        contact, reached, rest, last_push = False, False, mass_kg, -1.0
        t_bottom: Optional[float] = None
        panda = self.panda
        with self.hw_lock:
            self.motion_label = "lowering onto the bench"
            panda.start_controller(ctrl)
            try:
                with panda.create_context(frequency=1000, max_runtime=t_hold + max_dz / v + 2.0) as ctx:
                    t0 = time.time()
                    relax_t: Optional[float] = None
                    p_rel = p0
                    while ctx.ok() and not self.abort.is_set():
                        t = time.time() - t0
                        if relax_t is not None:   # V13: contact → setpoint = actual pose, so the arm stops pushing
                            ctrl.set_control(p_rel, o0)              # the part into the bench before it is released
                            if t - relax_t >= 0.3:
                                break
                            continue
                        f = np.asarray(panda.get_state().O_F_ext_hat_K[:3], dtype=float)
                        if t < t_hold:                    # hold still: loaded reference force f_hold
                            ctrl.set_control(p0, o0)
                            if t > 0.2:
                                hold.append(f)
                            continue
                        if f_hold is None:
                            f_hold = np.mean(hold, axis=0)
                        dz = min(v * (t - t_hold), max_dz)
                        ctrl.set_control(p0 - np.array([0.0, 0.0, dz]), o0)
                        buf.append(f)
                        if len(buf) == buf.maxlen:
                            rest = rest_load(np.mean(buf, axis=0), f_hold, d_f, mass_kg, g)
                            if t - last_push >= 0.04:
                                self.bench["samples"].append((dz * 1000.0, rest * 1000.0))
                                self.ver["bench"] += 1
                                last_push = t
                            if rest <= thr_g / 1000.0:
                                contact = True
                                p_rel = np.asarray(panda.get_position(), dtype=float).copy()
                                relax_t = t
                                continue
                        if dz >= max_dz:          # V17: at the seat — let the arm settle there (contact still counts)
                            t_bottom = t if t_bottom is None else t_bottom
                            if t - t_bottom >= 0.5:
                                reached = True
                                break
            finally:
                panda.stop_controller()
                self.motion_label = None
            descent = float(p0[2] - np.asarray(panda.get_position(), dtype=float)[2])
        self._check_abort()
        return contact, descent, rest, reached

    def _wait_bench(self) -> None:
        self._bench_go.clear()
        self.bench["waiting"] = True
        self.ver["bench"] += 1
        self.info("  part rests on the bench — press 'Continue' when the test is done")
        self.toast("Part rests on the bench — press Continue", "info")
        try:
            while not self._bench_go.wait(0.1):
                self._check_abort()
        finally:
            self.bench["waiting"] = False
            self.ver["bench"] += 1

    def bench_continue(self) -> None:
        if self.bench.get("waiting"):
            self.count("bench_continue")
            self.info("Continue — test done, re-grasping the part" if self.s.bench_release
                      else "Continue — lifting the part off the bench")
            self._bench_go.set()

    # ------------------------------------------------------------ worker view: one-button restart
    def worker_restart(self) -> bool:
        return self.run_job("restart", self._worker_restart)

    def _worker_restart(self) -> None:
        """Worker 'Restart': connect if needed, end teaching, recover errors, drive to the start point
        (a part still in the gripper is put back into the holder first), then sort automatically."""
        self.auto_on = False
        self.count("worker_restart")
        self.info("Restart pressed in the worker view")
        if not self.connected:
            self._connect()
        if self.teaching:
            with self.hw_lock:
                self.panda.teaching_mode(False)
            self.teaching = False
            self.info("Restart: teaching mode switched off")
        if self.max_width <= 0.01:
            self._gripper_homing()
        blockers = self.cycle_blockers(ignore_busy=True)
        if blockers:
            raise RuntimeError("cannot start sorting — " + "; ".join(blockers))
        with self.hw_lock:
            self.panda.recover()
        held = self._holding_part()
        self._drive_home()
        if held:
            self._put_back_held_part()
        self.truth = ""
        self.ok("Restart done — sorting automatically")
        self.auto_on = True
        self.job = "auto mode"
        self._auto_loop()

    def _holding_part(self) -> bool:
        try:
            gs = self._gripper_state()
            return bool(gs.is_grasped) and float(gs.width) * 1000.0 >= self.s.empty_width_mm
        except Exception:
            return False

    def _put_back_held_part(self) -> None:
        """After a STOP the gripper may still hold a drive: put it back into the holder so it is sorted again."""
        s = copy.copy(self.s).clamp()
        to_holder = [np.asarray(q, dtype=float) for q in self.waypoints["to_holder"]]
        self.count("put_back_after_restart")
        self.warn("Restart: the gripper still holds a part — putting it back into the holder")
        self._move_joints(to_holder, s.speed, "to_holder (put back)")
        self._grip_move(self._open_width(), s.open_speed)
        p0, o0 = self._pose()
        self._move_pose(p0 + np.array([0.0, 0.0, s.lift_mm / 1000.0]), o0, s.speed, "lift")
        self._move_joints(list(reversed(to_holder[:-1])), s.speed, "back to start")

    def start_auto(self) -> bool:
        blockers = self.cycle_blockers()
        if blockers:
            self.toast("Cannot start: " + "; ".join(blockers), "warning")
            return False
        self.auto_on = True
        if not self.run_job("auto mode", self._auto_loop):
            self.auto_on = False
            return False
        return True

    def stop_auto(self) -> None:
        if self.auto_on:
            self.auto_on = False
            self.info("Auto mode: finishing the current cycle, then stopping")

    def _auto_loop(self) -> None:
        self.count("auto_start")
        self.ok(f"Auto mode started — stops after {int(self.s.auto_max_empty)} empty grasps in a row or STOP")
        empty = 0
        try:
            while self.auto_on:
                res = self._cycle("sort")
                if res["status"] in ("error", "part_lost_at_bench"):
                    self.warn(f"Auto mode stopped ({STATUS_TEXT[res['status']]}) — check the robot and the bench")
                    break
                empty = empty + 1 if res["status"] == EMPTY else 0
                if empty >= max(1, int(self.s.auto_max_empty)):
                    self.warn(f"Auto mode stopped: {empty} empty grasps in a row (holder empty?)")
                    self.toast("Auto mode stopped — holder empty?", "warning")
                    break
                if self.auto_on:
                    self._sleep(float(self.s.auto_pause_s))
        finally:
            self.auto_on = False
        self.info("Auto mode finished")

    # ------------------------------------------------------------ results
    def _add_result(self, res: Dict[str, Any]) -> None:
        if res["n"] in self._pending_truth:               # V18: worker feedback given while the part was sorting
            res["truth"] = self._pending_truth.pop(res["n"])
        self.results.append(res)
        self.last = res
        self.ver["results"] += 1
        self._write_session_csv()
        self._write_summary()

    # ------------------------------------------------------------ detail files & session summary
    def _append_csv(self, suffix: str, header: List[str], rows: List[List[Any]]) -> None:
        if not rows:
            return
        try:
            p = self.session_path(suffix)
            p.parent.mkdir(parents=True, exist_ok=True)
            new = not p.exists()
            with open(p, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(header)
                w.writerows(rows)
        except Exception as e:
            self.warn(f"Could not write dashboard_{self.session}{suffix}: {e}")

    def _record_cycle(self, res: Dict[str, Any]) -> None:
        """Detail files per part: start/end/duration of every step, raw weighing samples, bench descent curve."""
        try:
            n, mode, status = res["n"], res["mode"], res["status"]
            rows = []
            for st in self.steps:
                t0, t1 = st.get("t0"), st.get("t1")
                dur = t1 - t0 if t0 and t1 else None
                self.step_log.append(dict(part=n, mode=mode, part_status=status, step=st["key"], label=st["label"],
                                          status=st["status"], dur=dur))
                rows.append([n, mode, status, st["key"], st["label"], st["status"], _iso(t0), _iso(t1), _fmt(dur, 3)])
            self._append_csv("_steps.csv", ["part", "mode", "part_status", "step", "label", "step_status",
                                            "t_start", "t_end", "duration_s"], rows)
            m = self.measure
            if float(m.get("t0") or 0.0) >= self.cycle_t0:      # weighed in this cycle (never in dry runs)
                rows = [[n, ph, _fmt(t, 3), _fmt(fx, 4), _fmt(fy, 4), _fmt(fz, 4)]
                        for ph in ("tare", "load") for t, fx, fy, fz in (m.get(ph) or [])]
                if not m.get("tare") and m.get("tare_mean"):     # weighed with the stored tare
                    rows.insert(0, [n, "tare_stored", ""] + [_fmt(x, 4) for x in m["tare_mean"][:3]])
                self._append_csv("_weighing.csv", ["part", "phase", "t_s", "fx_N", "fy_N", "fz_N"], rows)
            lim = self.bench.get("limit")
            rows = [[n, i, _fmt(dz, 2), _fmt(rest, 1), _fmt(lim, 0)]
                    for i, (dz, rest) in enumerate(list(self.bench.get("samples") or []))]
            self._append_csv("_bench.csv", ["part", "sample", "descent_mm", "rest_load_g", "contact_threshold_g"],
                             rows)
            self.debug(f"#{n} step times: " + ", ".join(f"{r['step']} {r['dur']:.2f} s" for r in self.step_log
                                                        if r["part"] == n and r["dur"] is not None))
        except Exception as e:
            self.warn(f"Could not write the detail files of part #{res.get('n')}: {type(e).__name__}: {e}")

    def summary_md(self) -> str:
        """KPIs of this session as Markdown (→ dashboard_<stamp>_summary.md). Part of the evaluation, so it is
        implemented by DashboardController in Franka_Dashboard_V18.py, the controller class the dashboard runs."""
        raise NotImplementedError("summary_md() is implemented by DashboardController in Franka_Dashboard_V18.py")

    def _write_summary(self) -> None:
        try:
            with self._rec_lock:
                p = self.session_path("_summary.md")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.summary_md(), encoding="utf-8")
        except Exception as e:
            self.warn(f"Could not write the session summary: {type(e).__name__}: {e}")

    def results_csv(self) -> str:
        """Same columns as franka_sort.py's logs/sort_<stamp>.csv, plus a few dashboard extras."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(CSV_FIELDS)
        for r in self.results:
            df = r.get("df") or [None, None, None]
            p = r["pred"] if r["status"] in MEASURED else "none" if r["status"] == EMPTY else None
            correct = "" if not r["truth"] or p is None else int(p == EXPECTED[r["truth"]])
            bc = r.get("bench_contact")
            w.writerow([r["n"], r["ts"], r.get("t_end", ""), _fmt(r["cycle_s"], 1), r["status"], r["truth"],
                        r["pred"] or "", correct, _fmt(r["mass_g"], 1), _fmt(r["width_mm"], 2), _fmt(df[0], 3),
                        _fmt(df[1], 3), _fmt(df[2], 3), "" if bc is None else int(bc),
                        _fmt(r.get("bench_descent_mm"), 1), _fmt(r.get("bench_rest_g"), 1),
                        r["reason"] if r["status"] in ("error", "aborted") else "",
                        r["mode"], r["tag"], _fmt(r["f_std"], 3),
                        r["reason"], r.get("bench", ""), _fmt(r.get("regrasp_mm"), 2), _fmt(r.get("tare_age_s"), 0),
                        _fmt(r.get("conf"), 3), r.get("test") or ""])
        return buf.getvalue()

    def _write_session_csv(self) -> None:
        try:
            p = self.session_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.results_csv())
        except Exception as e:
            self.warn(f"Could not write the results CSV: {e}")

    def set_truth(self, n: int, truth: str) -> None:
        for r in self.results:
            if r["n"] == n:
                r["truth"] = truth or ""
                self.ver["results"] += 1
                self.info(f"#{n}: label set to {LABEL.get(truth or '', truth)}")
                self._write_session_csv()
                self._write_summary()
                return
        if self.last is not None and self.last.get("n") == n:   # V18: still sorting → applied when it finishes
            self._pending_truth[n] = truth or ""
            self.last["truth"] = truth or ""
            self.ver["results"] += 1
            self.info(f"#{n}: label set to {LABEL.get(truth or '', truth)} (part still on its way)")

    def feedback_target(self) -> Optional[Dict[str, Any]]:
        """V18: the part the worker can rate — the last one with a decision, also while it is still being sorted."""
        r = self.last
        if r is None or r.get("mode") == "dry" or not r.get("pred") or r.get("mass_g") is None:
            return None
        return r if r["status"] in MEASURED + ("sorting",) else None

    def worker_feedback(self, n: int, correct: bool, truth: str = "") -> None:
        """V18: worker says whether part #n was classified correctly (→ 'Actual' label, confusion matrix)."""
        r = self.feedback_target()
        if r is None or r["n"] != n:
            self.warn(f"#{n}: feedback ignored — no decision of this part to rate")
            return
        pred = r["pred"]
        truth = ("other" if pred == "reject" else pred) if correct else truth
        if truth not in TRUTH_ROWS or (not correct and EXPECTED[truth] == pred):
            self.warn(f"#{n}: feedback ignored — '{truth}' does not fit 'wrong' for decision {LABEL[pred]}")
            return
        self.count("feedback_correct" if correct else "feedback_wrong")
        self.ok(f"#{n}: worker feedback — " + (f"correct ({LABEL[pred]})" if correct else
                                                 f"wrong, decision {LABEL[pred]}, actually {LABEL[truth]}"))
        self.set_truth(n, truth)

    def clear_results(self) -> None:
        if self.results or self.step_log:
            self._write_summary()
        self.info(f"Session {self.session} closed")
        self.results, self.last = [], None
        self.step_log, self.counters, self._pending_truth = [], {}, {}
        self.session_t0 = time.time()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ver["results"] += 1
        self.info(f"New session — results will be written to {self.session_file()}")

    # ------------------------------------------------------------ telemetry
    def _start_telemetry(self) -> None:
        self._stop_telemetry()
        self._tele_stop.clear()
        self._tele_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="telemetry")
        self._tele_thread.start()

    def _stop_telemetry(self) -> None:
        self._tele_stop.set()
        t = self._tele_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._tele_thread = None

    def _telemetry_read(self):
        if self.panda is None:
            return None
        if self.hw_lock.acquire(blocking=False):      # panda_py >= 1.0: get_state() may call readOnce()
            try:
                return self.panda.get_state()
            finally:
                self.hw_lock.release()
        return None

    def _telemetry_loop(self) -> None:
        last_g, n, t_rate, fails = 0.0, 0, time.time(), 0
        while not self._tele_stop.is_set():
            t = time.time()
            try:
                st = self._telemetry_read()
                if st is not None:
                    self._ingest_state(st, hist=self.measure.get("phase") is None)
                    n += 1
                fails = 0
            except Exception as e:
                fails += 1
                if fails in (1, 50):
                    self.debug(f"telemetry: {type(e).__name__}: {e}")
            if t - last_g > 0.25 and self.gripper is not None:
                last_g = t
                with contextlib.suppress(Exception):
                    self._ingest_gripper(self.gripper.read_once())
            if t - t_rate >= 1.0:
                self.tele["hz"], n, t_rate = n / (t - t_rate), 0, t
            self._tele_stop.wait(max(0.0, 0.05 - (time.time() - t)))

    def _ingest_state(self, st: Any, hist: bool = True) -> None:
        q = np.asarray(st.q, dtype=float)
        T = np.asarray(st.O_T_EE, dtype=float).reshape(4, 4, order="F")
        f = np.asarray(st.O_F_ext_hat_K, dtype=float)[:3]
        now = time.time()
        self.tele.update(q=q, ee=T[:3, 3].copy(), f=f, mode=st.robot_mode.name,
                         errors=_errs(st.current_errors), t=now)
        if hist:
            self.force_hist.append((now, float(f[0]), float(f[1]), float(f[2])))
        self.ver["tele"] += 1

    def _ingest_gripper(self, gs: Any) -> None:
        self.tele.update(width=float(gs.width), max_width=float(gs.max_width), grasped=bool(gs.is_grasped),
                         temp=gs.temperature)
        if float(gs.max_width) > 0.01:
            self.max_width = float(gs.max_width)



# ═══════════════════════════════════════════════════════════════════ 6. robot panels (web UI)
@contextlib.contextmanager
def card(title: str, icon: Optional[str] = None, subtitle: Optional[str] = None, classes: str = ""):
    with ui.element("div").classes(f"card {classes}") as c:
        with ui.row().classes("card-head items-center w-full"):
            if icon:
                ui.icon(icon).classes("card-icon")
            with ui.column().classes("gap-0 grow"):
                ui.label(title).classes("card-title")
                if subtitle:
                    ui.label(subtitle).classes("card-sub")
            right = ui.row().classes("items-center gap-2 no-wrap")
        yield c, right


class RobotPanels:
    """Web UI for running and setting up the robot: header, all parameters (left column), sorting cycle,
    teach-in, test drive and event log. Mixed into Page (Franka_Dashboard_V18.py), which creates the timers and
    the attributes used here (self.c = Controller, self.th = theme colours, self._cache, self._inputs, …)."""

    # ------------------------------------------------------------ small helpers
    def _changed(self, key: str, value: Any) -> bool:
        if key not in self._cache or self._cache[key] != value:
            self._cache[key] = value
            return True
        return False

    def _col(self, key: Optional[str]) -> str:
        if key in ("ssd", "hdd25", "hdd35"):
            return self.th[key]
        return self.th["other"] if key in ("reject", "other") else self.th["grid"]

    def _num(self, label: str, attr: str, lo: float, hi: float, step: float, suffix: str = "",
             tip: str = "", fmt: Optional[str] = None, full: bool = False, integer: bool = False):
        """Number field that commits on Enter or when it is left — never while typing. The value is clamped to
        LIMITS (from the Franka manual where it gives one); a note appears when a value had to be limited."""
        lo, hi = LIMITS.get(attr, (lo, hi))
        s = self.c.s
        integer = integer or isinstance(getattr(Settings(), attr), int)
        el = ui.number(label, value=getattr(s, attr), min=lo, max=hi, step=step, suffix=suffix or None, format=fmt) \
            .props("dense outlined hide-bottom-space")
        if full:
            el.classes("full")
        note = DOC_NOTE.get(attr, "")
        el.tooltip(" · ".join(x for x in (tip, f"range {lo:g}–{hi:g} {suffix}".strip(), note) if x))
        state = {"focus": False}

        def commit() -> None:
            try:
                v = float(el.value)
            except (TypeError, ValueError):
                v = float(getattr(s, attr))               # empty / invalid → keep the current value
            val = min(max(v, lo), hi)
            val = int(round(val)) if integer else val
            if abs(val - v) > 1e-9:
                ui.notify(f"{label}: limited to {val:g} {suffix}".rstrip() + (f" — {note}" if note else
                          f" (allowed {lo:g}–{hi:g})"), type="warning", position="bottom-right")
            setattr(s, attr, val)
            if el.value != val:
                el.set_value(val)

        def leave() -> None:
            state["focus"] = False
            commit()

        el.on("focus", lambda: state.update(focus=True))
        el.on("blur", leave)
        el.on("keydown.enter", commit)
        self._inputs.append((el, attr, state))
        return el

    def _text(self, label: str, attr: str, tip: str = "", full: bool = True):
        """Text field that commits on Enter or when it is left."""
        s = self.c.s
        el = ui.input(label, value=str(getattr(s, attr))).props("dense outlined hide-bottom-space")
        if full:
            el.classes("full")
        if tip:
            el.tooltip(tip)
        state = {"focus": False}

        def commit() -> None:
            val = str(el.value or "").strip()
            if not val and attr != "trial_tag":
                val = str(getattr(s, attr))               # IP and paths must not be empty
            setattr(s, attr, val)
            if el.value != val:
                el.set_value(val)

        def leave() -> None:
            state["focus"] = False
            commit()

        el.on("focus", lambda: state.update(focus=True))
        el.on("blur", leave)
        el.on("keydown.enter", commit)
        self._inputs.append((el, attr, state))
        return el

    def _sync_inputs(self) -> None:
        """Settings → fields for values changed elsewhere (reset, suggestions, other tabs) — never while typing."""
        s = self.c.s
        for el, attr, state in self._inputs:
            if state["focus"]:
                continue
            v, cur = getattr(s, attr), el.value
            if isinstance(v, str):
                if cur != v:
                    el.set_value(v)
                continue
            try:
                same = cur is not None and abs(float(cur) - float(v)) < 1e-9
            except (TypeError, ValueError):
                same = False
            if not same:
                el.set_value(v)

    def _slider(self, label: str, attr: str, lo: float, hi: float, step: float, fmt: str = "{:.2f}", tip: str = ""):
        with ui.column().classes("full gap-0 slider-row") as col:
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label(label).classes("field-label")
                ui.label().classes("field-value mono").bind_text_from(self.c.s, attr, backward=lambda v: fmt.format(v))
            ui.slider(min=lo, max=hi, step=step).props("dense").bind_value(self.c.s, attr)
        if tip:
            col.tooltip(tip)
        return col

    def _switch(self, label: str, attr: str, tip: str = ""):
        sw = ui.switch(label).props("dense").classes("full").bind_value(self.c.s, attr)
        if tip:
            sw.tooltip(tip)
        return sw

    def _button(self, label: str, icon: str, handler: Callable, style: str = "outline", tip: str = "", classes: str = ""):
        props = "unelevated no-caps no-wrap" if style in ("primary", "accent") else "outline no-caps no-wrap"
        b = ui.button(label, icon=icon, on_click=handler, color=None).props(props) \
            .classes(f"btn-{style} {classes}".strip())
        if tip:
            b.tooltip(tip)
        return b

    # ------------------------------------------------------------ header
    def _build_header(self) -> None:
        c = self.c
        with ui.header(elevated=False).classes("topbar items-center"):
            with ui.row().classes("items-center no-wrap gap-3"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("precision_manufacturing", size="21px")
                with ui.column().classes("gap-0"):
                    ui.label(APP_TITLE).classes("brand-title")
                    ui.label("Teach-in · weigh at the holder · bench test · sort into containers").classes("brand-sub")
            ui.space()
            with ui.row().classes("items-center gap-2 no-wrap hdr-pills"):
                self.h_backend = ui.label("").classes("pill")
                with ui.element("div").classes("pill"):
                    self.h_dot = ui.element("span").classes("dot")
                    self.h_conn = ui.label("Disconnected")
                with ui.element("div").classes("pill"):
                    ui.icon("smart_toy")
                    self.h_mode = ui.label("—")
                with ui.element("div").classes("pill") as self.h_job_pill:
                    ui.icon("autorenew")
                    self.h_job = ui.label("Idle")
            ui.space()
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.button("Worker view", icon="person", color=None).props("outline no-caps no-wrap") \
                    .classes("btn-outline").tooltip("Simple view for the worker: big digital twin, detected drive, "
                                                    "Restart and STOP").on("click", js_handler="() => setWorkerMode(true)")
                ui.button(icon="light_mode" if c.s.dark else "dark_mode", on_click=self._toggle_theme, color=None) \
                    .props("flat round dense").classes("icon-btn").tooltip("Switch light / dark theme")
                ui.button("STOP", icon="pan_tool", on_click=self._stop, color=None).props("unelevated no-caps no-wrap") \
                    .classes("btn-stop").tooltip("Software stop (Esc): stops motion, auto mode and teaching. "
                                                 "NOT a safety function — keep the Franka stop button in reach.")

    # ------------------------------------------------------------ left column: all parameters
    def _section(self, title: str, icon: str, caption: str = "", open_: bool = True):
        exp = ui.expansion(title, icon=icon, value=open_).classes("section w-full")
        if caption:
            exp.props(f'caption="{caption}"')
        return exp

    def _build_settings(self) -> None:
        c = self.c
        with self._section("Connection", "cable", "Panda via FCI — robot unlocked in Desk"):
            with ui.element("div").classes("fields"):
                self.w_ip = self._text("Robot FCI IP", "robot_ip", "IP of the robot's FCI interface (franka_sort.py host)")
                self.b_connect = self._button("Connect", "link", self._connect_toggle, "primary", classes="full btn-lg")
                self.l_conn = ui.label("").classes("full hint")
            ui.label("Robot actions").classes("sub-head")
            with ui.element("div").classes("fields action-grid"):
                self.b_home = self._button("Go to start", "home", self._go_home,
                                           tip="Move to the first waypoint of to_holder (franka_sort.py go_home)")
                self.b_open = self._button("Open gripper", "open_in_full", c.open_gripper,
                                           tip="gripper.move(max width, open speed)")
                self.b_tgrasp = self._button("Test grasp", "back_hand", c.test_grasp,
                                             tip="gripper.grasp() with the settings below — shows width and "
                                                 "is_grasped without moving the arm")
                self.b_ghome = self._button("Home gripper", "restart_alt", c.home_gripper,
                                            tip="gripper.homing(): calibrates the max opening (franka_sort.py --homing)")
                self.b_recover = self._button("Recover errors", "healing", c.recover, classes="full",
                                              tip="panda.recover(): automatic error recovery after a reflex")

        with self._section("Motion", "speed", "speed, lift, safety prompts"):
            with ui.element("div").classes("fields"):
                self._slider("Speed factor", "speed", 0.05, 1.0, 0.05,
                             tip="Speed factor 0..1 for all cycle motions (franka_sort.py --speed, default 0.20)")
                self._slider("Return speed", "home_speed", 0.05, 1.0, 0.05,
                             tip="'Go to start' and recovery moves (franka_sort.py go_home: 0.15)")
                self._num("Lift height", "lift_mm", 5, 200, 5, "mm",
                          "Vertical lift for tare and weighing (franka_sort.py --lift 0.05 m)")
                self._num("Goal tolerance", "goal_tol_rad", 0.005, 0.5, 0.005, "rad",
                          "panda_py returns False when a motion ends slightly off target. Deviations below this "
                          "count as reached; larger ones abort the cycle.")
                self._switch("Confirm moves away from start", "confirm_moves",
                             "Ask before a motion that starts away from the start point "
                             "(like franka_sort.py's 'Clear the workspace, ENTER')")
                self._switch("Auto-recover after errors", "auto_recover",
                             "After a failed cycle: clear the robot errors (panda.recover()). The arm does NOT drive "
                             "back by itself — use 'Go to start', which follows the taught paths. Never after STOP.")

        with self._section("Gripper", "back_hand", "grasp force, speed, tolerances", open_=False):
            with ui.element("div").classes("fields"):
                self._num("Grasp force", "grasp_force_n", 1, 70, 1, "N", "franka_sort.py --force (default 40 N)")
                self._num("Grasp speed", "grasp_speed_mms", 5, 30, 1, "mm/s",
                          "Closing speed (franka_sort.py uses 50 mm/s — above the manual's 30 mm/s)")
                self._num("Grasp width", "grasp_width_mm", 0, 80, 0.5, "mm", "Commanded grasp width (franka_sort.py: 0)")
                self._num("ε inner", "eps_inner_mm", 0, 80, 0.5, "mm", "epsilon_inner (franka_sort.py: 5 mm)")
                self._num("ε outer", "eps_outer_mm", 0, 80, 0.5, "mm",
                          "epsilon_outer; 0 = max opening width (franka_sort.py: max_w)")
                self._num("Open speed", "open_speed_mms", 5, 30, 1, "mm/s",
                          "gripper.move speed (franka_sort.py uses 100 mm/s — above the manual's 30 mm/s)")
                self._num("Empty below", "empty_width_mm", 0, 20, 0.5, "mm", full=True,
                          tip="With grasp width 0 and ε outer = max, grasp() also 'succeeds' on air. A finger width "
                              "below this value is therefore treated as an empty grasp.")
                self._switch("Home gripper on connect", "homing_on_connect", "franka_sort.py --homing")

        with self._section("Weighing", "scale", "tare, averaging, calibration", open_=False):
            with ui.element("div").classes("fields"):
                self._num("Settle time", "settle_s", 0, 5, 0.1, "s", "Wait after each lift before sampling "
                          "(franka_sort.py mean_force settle=0.5)")
                self._num("Averaging", "measure_s", 0.1, 10, 0.1, "s", "Sampling time per measurement "
                          "(franka_sort.py mean_force duration=1.0)")
                self._num("Gravity g", "gravity", 9.0, 10.0, 0.01, "m/s²", "franka_sort.py G = 9.81")
                ui.toggle({"norm": "|ΔF| 3-axis", "z": "ΔFz only"}).props("no-caps unelevated spread dense") \
                    .classes("seg").bind_value(c.s, "force_mode") \
                    .tooltip("|ΔF| = norm of the 3D force difference (franka_sort.py); ΔFz ignores lateral drift")
                self._num("Scale", "mass_scale", 0.5, 2.0, 0.01, "×", "Calibration: weight = raw × scale + offset")
                self._num("Offset", "mass_offset_g", -200, 200, 1, "g", "Calibration: weight = raw × scale + offset")
                self._switch("Tare at the weigh spot before every part", "tare_every_part",
                             "On (default): empty trip to the weigh spot and tare before every grasp — most accurate. "
                             "Off: reuse the stored tare and go straight to the holder (faster)")
                self._num("Re-tare after", "tare_max_age_min", 0, 480, 5, "min",
                          "Only when 'Tare … before every part' is off: renew the stored tare when it is older "
                          "than this (0 = never). Also renewed after every reconnect and when to_weigh changes")
                self._button("Re-tare next part", "restart_alt", c.retare, classes="full",
                             tip="Forget the stored tare: the next part starts with an empty trip to the weigh spot")

        with self._section("Testing bench", "fact_check", "force-guarded rest while holding the part"):
            with ui.element("div").classes("fields"):
                self._num("Contact at rest load ≤", "contact_g", 1, 500, 1, "g",
                          "Stop lowering when the load still hanging in the gripper drops to this value "
                          "(franka_sort.py --contact-g 40) …")
                self._num("… and ≤ % of the part", "contact_pct", 5, 90, 1, "%",
                          "V17: light parts need a lower threshold — the smaller value counts "
                          "(60 g SSD at 35 % → 21 g; a 40 g threshold fired after 0.1 mm)")
                self._num("Descent speed", "descent_speed_mms", 1, 50, 1, "mm/s",
                          "Lowering speed of the CartesianImpedance controller down to the seat (franka_sort.py "
                          "--descent-speed 0.01 m/s)")
                self._num("Hold reference", "bench_hold_s", 0.3, 3, 0.1, "s",
                          "Hold still before lowering to record the loaded reference force (franka_sort.py: 0.7 s)")
                self._num("Force averaging", "bench_avg_ms", 5, 500, 5, "ms",
                          "Moving average during the descent (franka_sort.py: 50 samples ≈ 50 ms)")
                self._num("Test time on bench", "bench_pause_s", 0, 120, 0.5, "s",
                          "How long the part stays in the bench for the (demo) test before it is re-grasped")
                self._num("Open wider than part", "bench_open_mm", 1, 80, 1, "mm",
                          "At the bench the gripper opens to the measured part width + this — keep it small "
                          "if the bench slot is narrow")
                self._switch("Open the gripper during the test", "bench_release",
                             "Lower → open → test → re-grasp → lift. Only opens when bench contact was detected "
                             "(otherwise the part would drop). Off = keep holding the part on the bench")
                self._switch("Open fully at the bench", "bench_open_full",
                             "Placed → open all the way → test → down to the seat (first point of the exit path) "
                             "→ grasp. Off = open only to part width + 'Open wider than part'")
                self._num("Defect rate (simulated test)", "defect_pct", 0, 100, 5, "%", full=True,
                          tip="V18: after the bench test the part counts as defective with this chance and goes to "
                              "the reject container (to_reject25 / to_reject35), otherwise to its own container. "
                              "0 = every part works")
                self._switch("Wait for ‘Continue’ on the bench", "bench_wait",
                             "Instead of the test time: keep the part on the bench until you press Continue "
                             "(franka_sort.py --bench-wait)")

        with self._section("Classification", "rule", "trained model, reject rules"):
            with ui.element("div").classes("fields"):                    # V18: trained model
                ui.label(DRIVE_MODEL_NOTE).classes("full hint")
                self._num("SSD only if the model is ≥", "ssd_min_conf_pct", 0, 99, 5, "% sure", full=True,
                          tip="Below this probability an SSD decision goes to the likelier HDD class instead, because "
                              "an HDD sorted as SSD is the data-security risk. 0 = always the most likely class")
                self._switch("Reject implausible parts", "reject_enabled",
                             "Parts outside the plausibility window of their class go to the reject bin "
                             "(to_reject) or back into the holder")
            with ui.element("div").classes("fields").bind_visibility_from(c.s, "reject_enabled"):
                ui.label("Plausibility windows").classes("full sub-head")
                self._num("SSD min weight", "ssd_mass_min_g", 0, 500, 1, "g")
                self._num("2.5″ HDD max weight", "hdd25_mass_max_g", 0, 1000, 1, "g")
                self._num("2.5″ min width", "w25_min_mm", 0, 60, 0.5, "mm")
                self._num("2.5″ max width", "w25_max_mm", 0, 60, 0.5, "mm")
                self._num("3.5″ min weight", "hdd35_mass_min_g", 0, 2000, 10, "g")
                self._num("3.5″ max weight", "hdd35_mass_max_g", 0, 3000, 10, "g")
                self._num("3.5″ max width", "hdd35_w_max_mm", 0, 80, 0.5, "mm", full=True)

        with self._section("Auto mode", "all_inclusive", "continuous sorting", open_=False):
            with ui.element("div").classes("fields"):
                self._num("Stop after empty", "auto_max_empty", 1, 50, 1, "×", integer=True,
                          tip="Auto mode stops after this many empty grasps in a row")
                self._num("Pause", "auto_pause_s", 0, 30, 0.5, "s", "Pause between auto cycles")

        with self._section("Data & files", "folder_open", "trial tag, CSV, waypoints", open_=False):
            with ui.element("div").classes("fields"):
                self._text("Trial tag (e.g. P3-T2-robot)", "trial_tag",
                           "Stored with every result row — participant / trial / condition")
                self._text("Log folder (.log, CSVs and summary per session)", "results_dir",
                           "Per session: .log (everything), .csv (parts), _steps.csv, _weighing.csv, _bench.csv, "
                           "_summary.md (KPIs for the evaluation)")
                self._text("Waypoints file", "waypoints_file", "Same file as franka_sort.py --file")
                self._button("Reload waypoints", "refresh", lambda: c.load_waypoints(), classes="full")
                self._button("Reset parameters", "settings_backup_restore", self._reset_settings, classes="full",
                             tip="Back to the defaults (files and IP are kept)")
            ui.label("Settings are saved automatically to dashboard_settings.json").classes("hint")

    # ------------------------------------------------------------ sorting cycle (centre column)
    def _build_cycle(self) -> None:
        c = self.c
        with card("Sorting cycle", "play_circle",
                  "start → holder (tare · grasp · weigh) → bench → container → start  (franka_sort.py run)") as (_, right):
            with right:
                self.l_elapsed = ui.label("").classes("elapsed mono")
            self.l_truth = ui.label("Part placed on the sliding holder (ground truth for the evaluation)") \
                .classes("field-label")
            ui.toggle({"": "Unlabeled", "ssd": "SSD", "hdd25": "HDD 2.5″", "hdd35": "HDD 3.5″", "other": "Other"}) \
                .props("no-caps unelevated spread").classes("w-full seg seg-lg").bind_value(c, "truth")
            with ui.row().classes("w-full items-center gap-2 cycle-actions q-mt-sm"):
                self.b_sort = self._button("Sort part", "play_arrow", lambda: self._start_cycle("sort"), "primary",
                                           "One part like ENTER in franka_sort.py run: weigh at the holder, rest on "
                                           "the bench, drop into the container, back to start", "btn-xl")
                self.b_measure = self._button("Measure only", "scale", lambda: self._start_cycle("measure"), "outline",
                                              "Weigh and classify at the holder, then put the part back — no bench, "
                                              "no container. For calibration data", "btn-xl")
                self.b_auto = self._button("Auto", "all_inclusive", self._auto_toggle, "outline",
                                           "Repeat sort cycles until STOP or N empty grasps in a row", "btn-xl")
                with self._button("Dry run", "route", lambda: None, "outline",
                                  "Drive holder → bench → container without a part — checks the taught paths",
                                  "btn-xl") as self.b_dry:
                    with ui.menu().props("auto-close"):
                        for cls in ["ssd", "hdd25", "hdd35", "reject"]:
                            ui.menu_item(f"To the {LABEL[cls]} container" + ("" if cls == "reject" else " via the bench"),
                                         on_click=lambda cls=cls: self._start_cycle("dry", cls))
            with ui.row().classes("w-full items-center gap-3 no-wrap bench-wait q-mt-sm") as self.bench_wait_row:
                ui.icon("hourglass_top")
                ui.label("The part rests on the testing bench — run your test, then continue.").classes("grow")
                self._button("Continue", "play_arrow", c.bench_continue, "accent",
                             "Re-grasp the part, lift it off the bench and drop it into its container")
            self.bench_wait_row.set_visibility(False)
            self.ready_box = ui.element("div").classes("ready q-mt-sm")
            ui.label("Cycle steps").classes("sub-head q-mt-sm")
            self.stepper = ui.element("div").classes("stepper")
            self.progress = ui.linear_progress(value=0, show_value=False).props("rounded size=5px color=primary") \
                .classes("progress q-mt-sm")

    # ------------------------------------------------------------ teach-in, test drive, event log (right column)
    def _build_teach(self) -> None:
        c = self.c
        with card("Teach-in", "pan_tool", "Hand-guide the arm and record waypoints per path") as (_, right):
            with right:
                self.b_teach = self._button("Enable", "pan_tool", self._teach_toggle, "primary",
                                            "Teaching mode = gravity compensation (panda.teaching_mode)")
            self._switch("Guide with the wrist buttons (no teaching mode)", "teach_buttons",
                         "franka_sort.py --teach-buttons: move the arm with the guiding buttons on the wrist instead "
                         "of panda-py teaching mode — W records the pose")
            with ui.element("div").classes("teach-banner") as self.teach_banner:
                ui.icon("warning_amber")
                self.teach_banner_txt = ui.label("")
            self.path_box = ui.column().classes("gap-2 w-full")
            with ui.element("div").classes("draft w-full"):
                self.l_draft = ui.label("").classes("draft-title")
                with ui.row().classes("gap-2 w-full items-center no-wrap"):
                    self.b_add = self._button("Add point (W)", "add_location_alt", c.teach_add, "primary")
                    ui.button(icon="undo", on_click=c.teach_undo, color=None).props("outline dense") \
                        .classes("btn-outline").tooltip("Remove the last waypoint")
                    ui.button(icon="clear_all", on_click=c.teach_clear, color=None).props("outline dense") \
                        .classes("btn-outline").tooltip("Discard the draft")
                    ui.space()
                    self.b_save = self._button("Save path", "save", c.teach_save, "accent",
                                               "Replace the stored path with the draft and write the file")
                self.l_draft_pts = ui.label("").classes("hint mono draft-pts")
                with ui.row().classes("w-full justify-end"):
                    self.b_delpath = ui.button("Delete stored path", icon="delete_outline", on_click=self._delete_path,
                                               color=None).props("flat dense no-caps").classes("icon-btn")
            self.b_target = self._button("Finish at the start point", "flag", c.teach_end_target, classes="full",
                                         tip="Adds the exact pose this path has to end at as last waypoint "
                                             "(home_* → start point, start_to_weigh → weigh spot) — no gap")
            self.l_link = ui.label("").classes("hint")
            self._button("Drive to end of previous path", "skip_previous", self._goto_path_start, classes="full",
                         tip="Drive along the taught paths to where the selected path starts (end of the previous "
                             "one), e.g. to re-teach or extend it. Teaching mode pauses while the arm moves")
            with ui.row().classes("gap-2 w-full items-center no-wrap"):     # V18: straight down/up for bench seats
                ui.label("Straight").classes("hint")
                for mm, icon in ((-10, "keyboard_double_arrow_down"), (-2, "keyboard_arrow_down"),
                                 (2, "keyboard_arrow_up"), (10, "keyboard_double_arrow_up")):
                    ui.button(f"{abs(mm)} mm", icon=icon, on_click=lambda _=None, mm=mm: c.jog_z(mm), color=None) \
                        .props("outline dense no-caps").classes("btn-outline") \
                        .tooltip(f"Gripper straight {'down' if mm < 0 else 'up'} {abs(mm)} mm (keeps x, y and "
                                 "orientation). For out_bench*: 'Drive to end of previous path', then ▼ until the "
                                 "part sits in the bench, then W")
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                ui.icon("description").classes("muted")
                self.l_wpfile = ui.label("").classes("hint mono ellipsis grow")

    def _build_drive(self) -> None:
        with card("Test drive", "route", "Drive taught paths one after another — no gripper actions"):
            with ui.row().classes("gap-2 w-full items-center"):
                ui.label("Route").classes("field-label")
                for key, lbl in (("ssd", "SSD"), ("hdd25", "HDD 2.5″"), ("hdd35", "HDD 3.5″"), ("reject", "Reject"),
                                 ("bad25", "Defect 2.5″"), ("bad35", "Defect 3.5″")):
                    ui.button(lbl, on_click=lambda key=key: self.w_drive.set_value(list(ROUTES[key])), color=None) \
                        .props("outline dense no-caps").classes("btn-outline") \
                        .tooltip("Select: " + " → ".join(PATH_LABEL[n] for n in ROUTES[key]))
                ui.button(icon="clear", on_click=lambda: self.w_drive.set_value([]), color=None) \
                    .props("flat round dense").classes("icon-btn").tooltip("Clear the selection")
            self.w_drive = ui.select({n: f"{PATH_LABEL[n]} · {n}" for n in ALL_PATHS}, multiple=True, value=[],
                                     label="Paths (driven in route order)") \
                .props("use-chips dense outlined").classes("w-full q-mt-sm")
            with ui.row().classes("gap-2 w-full q-mt-sm no-wrap"):
                self._button("Drive", "play_arrow", self._drive_fwd, "primary", "Drive the selected paths forwards",
                             classes="grow")
                self._button("Backwards", "undo", self._drive_back, "outline",
                             "Drive the selected paths backwards (last path first, each reversed)")

    def _build_log(self) -> None:
        with card("Event log", "receipt_long", "Everything the robot does — also in the session .log file") as (_, right):
            with right:
                self.w_verbose = ui.switch("verbose", value=False).props("dense") \
                    .tooltip("Also show panda_py library messages and debug details")
                ui.button(icon="delete_sweep", on_click=lambda: self.logview.clear(), color=None) \
                    .props("flat round dense").classes("icon-btn").tooltip("Clear this view")
            self.logview = ui.log(max_lines=400).classes("logview w-full")
            for _, ts, level, msg in list(self.c.log)[-150:]:
                if level != "debug":
                    self._push_log(ts, level, msg)
            self._seen["log"] = self.c.log_seq

    # ------------------------------------------------------------ actions
    def _stop(self) -> None:
        self.c.stop()
        ui.notify("STOP — motion halted", type="negative", position="top")

    def _toggle_theme(self) -> None:
        self.c.s.dark = not self.c.s.dark
        self.c.s.save(self.c.settings_path)
        ui.navigate.reload()

    def _connect_toggle(self) -> None:
        if self.c.connected:
            self.c.disconnect()
        else:
            self.c.connect()

    async def _confirm(self, title: str, text: str, ok_text: str = "Move", danger: bool = False) -> bool:
        with ui.dialog() as dlg, ui.element("div").classes("card").style("min-width:360px;max-width:460px"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("warning_amber" if not danger else "delete_forever", size="22px") \
                    .style(f"color:{STATUS['warning'] if not danger else STATUS['critical']}")
                ui.label(title).classes("card-title")
            ui.label(text).classes("reason q-mt-sm")
            with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                ui.button("Cancel", on_click=lambda: dlg.submit(False), color=None).props("flat no-caps") \
                    .classes("icon-btn")
                ui.button(ok_text, on_click=lambda: dlg.submit(True), color=None).props("unelevated no-caps") \
                    .classes("btn-primary" if not danger else "btn-stop")
        result = await dlg
        dlg.delete()
        return bool(result)

    async def _start_cycle(self, mode: str, target: Optional[str] = None) -> None:
        c = self.c
        blockers = c.cycle_blockers()
        if blockers:
            ui.notify("Cannot start: " + "; ".join(blockers), type="warning")
            return
        if c.s.confirm_moves and not c.near_home():
            if not await self._confirm("Robot moves to the start point first",
                                       "The arm is not at the start point. The first motion goes straight to it — "
                                       "clear the workspace."):
                return
        c.run_cycle(mode, target)

    async def _go_home(self) -> None:
        c = self.c
        if c.s.confirm_moves and not await self._confirm("Move to the start point?",
                                                         "The robot drives to the first waypoint of to_holder. "
                                                         "Clear the workspace."):
            return
        c.go_home()

    def _auto_toggle(self) -> None:
        if self.c.auto_on:
            self.c.stop_auto()
        else:
            self.c.start_auto()

    def _teach_toggle(self) -> None:
        self.c.set_teaching(not self.c.teaching)

    def _select_path(self, name: str) -> None:
        self.c.teach_select(name)

    async def _goto_path_start(self) -> None:
        c = self.c
        name = c.draft_path
        chain = [n for n in c.chain_before(name) if n != LIFT]
        how = ("start point → " + " → ".join(PATH_LABEL[n] for n in chain)) if chain else "the start point"
        if await self._confirm(f"Drive to where '{PATH_LABEL[name]}' starts?",
                               f"The arm drives along the taught paths ({how}), back to the start first if it is not "
                               "already on that way. Teaching mode pauses while it moves. Clear the workspace.",
                               ok_text="Drive"):
            c.goto_path_start(name)

    async def _drive_fwd(self) -> None:
        await self._drive(False)

    async def _drive_back(self) -> None:
        await self._drive(True)

    async def _drive(self, backwards: bool) -> None:
        c = self.c
        names = [n for n in ALL_PATHS if n in (self.w_drive.value or [])]
        if not names:
            ui.notify("Select paths first (or pick a route)", type="info")
            return
        if not c.connected:
            ui.notify("Connect first", type="warning")
            return
        try:
            segs = c._segments(names, backwards)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")
            return
        jumps = c.plan_drive(segs)
        order = " → ".join("lift" if n == LIFT else PATH_LABEL[n] + (" ↩" if backwards else "") for n, _, _ in segs)
        text = (order + ". " + ("Direct, NOT taught moves: " + "; ".join(jumps) + ". " if jumps else
                                "All moves follow taught waypoints. ") + "No gripper actions. Clear the workspace.")
        if await self._confirm("Test drive" + (" backwards" if backwards else ""), text, ok_text="Drive"):
            c.drive_paths(names, backwards)

    async def _delete_path(self) -> None:
        name = self.c.draft_path
        if not self.c.waypoints.get(name):
            ui.notify(f"{name} has no stored waypoints", type="info")
            return
        if await self._confirm(f"Delete path {name}?", f"Removes {len(self.c.waypoints[name])} stored waypoints "
                                                       f"from {self.c.wp_path().name} (a .bak backup exists).",
                               ok_text="Delete", danger=True):
            self.c.teach_delete(name)

    async def _reset_settings(self) -> None:
        if await self._confirm("Reset all parameters?", "Motion, gripper, weighing and classification values "
                                                        "go back to the franka_sort.py defaults.", ok_text="Reset"):
            self.c.reset_settings()

    # ------------------------------------------------------------ periodic updates (called by Page._tick_fast / _tick_slow)
    def _upd_header(self) -> None:
        c = self.c
        ip = f"FCI {c.s.robot_ip.strip()}"
        if self._changed("backend", ip):
            self.h_backend.set_text(ip)
            self.h_backend.classes(replace="pill chip-real")
        if c.connected:
            conn, dot = "Connected", "dot good"
        elif c.job == "connect":
            conn, dot = "Connecting…", "dot warn"
        else:
            conn, dot = "Disconnected", "dot bad"
        self.h_conn.set_text(conn)
        if self._changed("hdot", dot):
            self.h_dot.classes(replace=dot)
        mode = c.tele.get("mode", "—")
        self.h_mode.set_text("Teaching" if c.teaching else (NICE_MODE.get(mode, mode) if c.connected else "—"))
        if c.job:
            txt = f"{c.job} · {time.time() - c.job_t0:.0f} s" + (f" · {c.motion_label}" if c.motion_label else "")
        else:
            txt = "Idle"
        self.h_job.set_text(txt)
        if self._changed("jobpill", bool(c.job)):
            self.h_job_pill.classes(replace="pill busy" if c.job else "pill")
        info = c.lib_info if c.connected else "Robot must be unlocked in Desk with FCI active."
        self.l_conn.set_text(info)

    def _upd_buttons(self) -> None:
        c = self.c
        conn, busy, teach = c.connected, c.busy, c.teaching
        can_cycle = conn and not busy and not teach
        states = {
            "b_sort": can_cycle, "b_measure": can_cycle, "b_dry": can_cycle, "b_auto": can_cycle or c.auto_on,
            "b_home": can_cycle, "b_open": conn and not busy, "b_tgrasp": conn and not busy,
            "b_ghome": conn and not busy, "b_recover": conn and not busy, "b_connect": not busy,
            "w_ip": not conn and not busy, "b_teach": conn and not busy and (teach or not c.s.teach_buttons),
            "b_add": conn and not busy, "b_save": bool(c.draft) and not busy,
        }
        for name, en in states.items():
            if self._changed("en_" + name, en):
                getattr(self, name).set_enabled(en)
        waiting = bool(c.bench.get("waiting"))
        if self._changed("bench_wait", waiting):
            self.bench_wait_row.set_visibility(waiting)
        if self._changed("conn_btn", conn):
            self.b_connect.set_text("Disconnect" if conn else "Connect")
            self.b_connect.props(f"icon={'link_off' if conn else 'link'}")
        if self._changed("auto_btn", c.auto_on):
            self.b_auto.set_text("Stop auto" if c.auto_on else "Auto")
            self.b_auto.props(f"icon={'stop' if c.auto_on else 'all_inclusive'}")
        if self._changed("teach_btn", teach):
            self.b_teach.set_text("Disable" if teach else "Enable")
            self.b_teach.props(f"icon={'lock' if teach else 'pan_tool'}")

    def _upd_steps(self) -> None:
        c = self.c
        key = (c.cycle_n, len(c.steps))
        if key != self._steps_key:
            self._steps_key = key
            self.stepper.clear()
            self._step_els = []
            with self.stepper:
                if not c.steps:
                    ui.label("No cycle yet — press ‘Sort part’.").classes("hint")
                for st in c.steps:
                    with ui.element("div").classes("step pending") as el:
                        ic = ui.icon("radio_button_unchecked")
                        lb = ui.label(st["label"])
                        tl = ui.label("").classes("step-t")
                    self._step_els.append((el, ic, lb, tl))
        icons = {"pending": "radio_button_unchecked", "active": "autorenew", "done": "check_circle",
                 "failed": "cancel", "skipped": "remove_circle_outline"}
        done = 0.0
        now = time.time()
        for i, (st, (el, ic, lb, tl)) in enumerate(zip(c.steps, self._step_els)):
            status = st["status"]
            dur = None if st["t0"] is None else (st["t1"] or now) - st["t0"]
            if self._changed(f"step{i}", (status, st["label"])):
                el.classes(replace=f"step {status}")
                ic.name = icons.get(status, "radio_button_unchecked")
                lb.set_text(st["label"])
            tl.set_text("" if dur is None or status == "skipped" else f"{dur:.1f}s")
            done += 1.0 if status in ("done", "skipped", "failed") else 0.5 if status == "active" else 0.0
        if c.steps:
            self.progress.set_value(done / len(c.steps))
        if c.steps and c.cycle_t0:
            end = c.cycle_t1 or now
            self.l_elapsed.set_text(f"{end - c.cycle_t0:5.1f} s")
        else:
            self.l_elapsed.set_text("")

    def _push_log(self, ts: str, level: str, msg: str) -> None:
        tag = {"debug": "·", "info": "i", "ok": "✓", "warn": "!", "error": "✗"}[level]
        self.logview.push(f"{ts}  {tag}  {msg}", classes=f"lv-{level}")

    def _upd_log(self) -> None:
        c = self.c
        if c.log_seq == self._seen["log"]:
            return
        verbose = bool(self.w_verbose.value)
        for seq, ts, level, msg in list(c.log):
            if seq > self._seen["log"] and (verbose or level != "debug"):
                self._push_log(ts, level, msg)
        self._seen["log"] = c.log_seq

    def _upd_toasts(self) -> None:
        c = self.c
        if c.toast_seq == self._seen["toast"]:
            return
        for seq, msg, kind in list(c.toasts):
            if seq > self._seen["toast"]:
                ui.notify(msg, type=kind if kind in ("positive", "negative", "warning", "info") else "info",
                          position="top" if self._worker else "bottom-right", timeout=3200)
        self._seen["toast"] = c.toast_seq

    def _upd_teach(self) -> None:
        c = self.c
        buttons = bool(c.s.teach_buttons)
        if self._changed("banner", (c.teaching, buttons)):
            self.teach_banner.set_visibility(c.teaching or buttons)
            self.teach_banner_txt.set_text(
                "Teaching mode ON — the arm is free. Guide it by hand, press W or ‘Add point’ at each waypoint, "
                "then ‘Save path’. If the joint values in Telemetry do not change while you move the arm, use "
                "the wrist buttons instead." if c.teaching else
                "Wrist-button guiding — move the arm with the guiding buttons on the wrist, press W or "
                "‘Add point’ at each waypoint, then ‘Save path’.")
        stored = len(c.waypoints.get(c.draft_path, []))
        self.l_draft.set_text(f"Recording {c.draft_path} — draft {len(c.draft)} waypoint(s)"
                              + (f" · stored: {stored}" if stored else " · nothing stored yet"))
        if self._changed("draft_v", c.ver["draft"]):
            self.l_draft_pts.set_text("\n".join(f"#{i + 1}  [" + ", ".join(f"{x:+.3f}" for x in q) + "]"
                                                for i, q in enumerate(c.draft)) or "No waypoints recorded yet.")
        self.l_wpfile.set_text(str(c.wp_path().resolve()))
        st, end = c._link_targets(c.draft_path)
        if self._changed("target", (c.draft_path, end is not None)):
            self.b_target.set_visibility(end is not None)
            if end is not None:
                self.b_target.set_text(f"Finish at the {end[1]}")
        if self._changed("link_hint", (c.draft_path, c.ver["wp"], len(c.draft), round(float(c.tele["t"]) * 2, 0))):
            hint = ""
            if st is not None and not c.draft:
                d = float(np.max(np.abs(np.asarray(c.tele["q"], dtype=float) - st[0])))
                hint = (f"Starts at the {st[1]} — arm is {d:.2f} rad away"
                        + (" ✓" if d <= st[2] else " → 'Drive to end of previous path' first"))
            bench = next((b for b, o in BENCH_OUT.items() if o == c.draft_path), None)
            if bench and c.waypoints.get(bench) and not c.draft:      # V18: seat straight below the bench path
                dxy, ang, dz = c.seat_offset(c.waypoints[bench][-1], c.tele["q"])
                good = dxy <= SEAT_XY_TOL_MM and ang <= SEAT_ANG_TOL_DEG and dz >= -1.0
                hint = (f"First point = where the part sits, straight below the end of {bench}: arm is "
                        f"{dxy:.0f} mm beside, {ang:.0f}° turned, {dz:.0f} mm below"
                        + (" ✓" if good else " → 'Drive to end of previous path', then ▼"))
            self.l_link.set_text(hint)
            self.l_link.set_visibility(bool(hint))
        en = bool(stored) and not c.busy
        if self._changed("en_del", en):
            self.b_delpath.set_enabled(en)

    def _upd_ready(self) -> None:
        items = self.c.readiness()
        sig = tuple(items)
        if not self._changed("ready", sig):
            return
        self.ready_box.clear()
        with self.ready_box:
            for label, ok, detail in items:
                with ui.element("div").classes("rchip " + ("ok" if ok else "bad")):
                    ui.icon("check_circle" if ok else "error_outline")
                    ui.label(label)
                    ui.label(detail).classes("rdetail")

    def _upd_paths(self) -> None:
        """Path list in the teach-in card (the twin draws the same paths: Page._upd_twin_paths)."""
        c = self.c
        sig = (c.ver["wp"], c.ver["draft"], c.draft_path)
        if not self._changed("paths_v", sig):
            return
        self.path_box.clear()
        with self.path_box:
            for name in ALL_PATHS:
                n = len(c.waypoints.get(name, []))
                cls = PATH_CLASS.get(name) or HOME_CLASS.get(name)
                color = self.th["ink2"] if cls is None else self._col(cls)
                row = ui.element("div").classes("path-row" + (" sel" if name == c.draft_path else ""))
                row.on("click", lambda name=name: self._select_path(name))
                with row:
                    ui.element("span").classes("path-dot").style(f"background:{color}")
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label(name).classes("path-name")
                        ui.label(PATH_HINT[name]).classes("path-hint")
                    if n:
                        ui.label(f"{n} pts").classes("badge ok")
                    elif name in OPTIONAL_PATHS:
                        ui.label("optional").classes("badge")
                    else:
                        ui.label("missing").classes("badge miss")
