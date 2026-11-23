# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Git repo `loui-s-git/ASiM1` on GitHub. The user usually adds files through GitHub's web upload, so `origin` can be ahead of this clone: fetch before editing.

- `Franka_Dashboard_V18.py` + `Franka_Robot_V18.py`: current NiceGUI web dashboard, the main codebase, split in two files that must sit in the same folder. `Franka_Dashboard_V18.py` (~1,700 lines) is the digital twin and evaluation part and the one that is started; `Franka_Robot_V18.py` (~3,200 lines) is sorting and robot setup and is only imported.
- `drive_dimensions.csv`: datasheet values of 48 real drives (type, length, width, height = thickness, weight, shop link), the training data of the classifier.
- `train_drive_model.py`: reads `drive_dimensions.csv` with `pandas.read_csv` (only `type`, `height_mm`, `weight_g` via `usecols`), trains the classifier with scikit-learn and writes `drive_model.json`. Its packages are in `requirements-train.txt`, meant for a separate venv; the dashboard never imports pandas or scikit-learn. Rerun it and commit the new `drive_model.json` whenever the data changes.
- `drive_model.json`: the trained model the dashboard loads once at start, from next to `Franka_Robot_V18.py`. It has to be copied to the robot PC with the `.py` files.
- `requirements.txt`: numpy, nicegui and panda-python pinned to the versions on the robot PC.
- `README.md`: German terminal commands for the robot PC (including training), then the teach-in checklist for all 17 paths (formerly `Teach_Ablauf_V18.md`). Update it when path semantics or command-line options change.
- `logs/`: session files copied over from the robot PC for analysis. The `.gitignore` is GitHub's Visual Studio template, and its `[Ll]ogs/` rule keeps `logs/` out of git.

Referenced below but not in this repo (ask the user for a copy if you need one):

- `franka_sort.py`: original terminal tool (`teach` / `run`). The dashboard mirrors its defaults and the first 17 columns of its CSV (`CSV_FIELDS`); keep those compatible. Its `classify()` threshold rule is gone: the dashboard only classifies with the trained model.
- `franka_measure.py`: standalone grasp-and-weigh demo.
- `install_franka.sh`: Ubuntu setup (venv `~/franka_venv`, numpy + panda-python with bundled libfranka 0.9.2, realtime limits). It does not install nicegui.

## Where things run

The robot PC is Ubuntu (`/home/bloo52/Downloads`, Python 3.12, numpy, nicegui 3.19, panda-python 1.1.1, robot FCI at `172.16.0.2`). This Mac has Python 3.9 without numpy, nicegui or panda_py, so nothing robot-related runs here. Code is edited here and copied to the robot PC (both `.py` files); only the user runs it on the robot.

## Commands

On the robot PC:

```bash
source ~/franka_venv/bin/activate
pip install -r requirements.txt                 # once; install_franka.sh does not install nicegui
python3 train_drive_model.py                    # retrain drive_model.json (needs requirements-train.txt, own venv)
python3 Franka_Dashboard_V18.py                # opens http://localhost:8080 (worker view: /?view=worker)
```

There are no command-line options and no terminal output (on the user's request: bare bones). Robot IP and waypoints file are settings in the UI, everything else goes to the UI log and the session `.log`; `ui.run` binds to 127.0.0.1 on port 8080. Start it from the folder that holds `waypoints.json` and `drive_model.json`: the waypoints file, `dashboard_settings.json` and `logs/` are resolved relative to the working directory.

There are no tests, linters or build steps. The basic check on this Mac is a syntax check:

```bash
python3 -I -c "import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]" Franka_Dashboard_V18.py Franka_Robot_V18.py train_drive_model.py
```

Deeper checks are possible with packages installed into a scratch folder (`pip install --target <dir>`, then `PYTHONPATH=<dir>`): pyflakes for undefined names and unused imports (one known warning: an f-string without placeholders in the controller), and `nicegui==3.6.1` + numpy (the newest nicegui for Python 3.9) for a smoke test without robot: import the dashboard, construct `DashboardController`, build a `Page` inside a `ui.page` and call `_tick_fast()` / `_tick_slow()`. Failed updaters are collected in `Page._errors_reported`. That covers UI wiring only, never robot motion. `scikit-learn==1.6.1` (the newest for Python 3.9) and pandas install the same way, for running `train_drive_model.py` and for comparing `model_proba` with `predict_proba`. Note that `python3 -I` ignores `PYTHONPATH`, so run these without `-I`.

For offline geometry checks (e.g. where a taught joint pose puts the TCP), reimplement the DH forward kinematics from section 3 of `Franka_Robot_V18.py` in plain Python, or use the scratch numpy.

## Versioning

Each release is a full copy of both files, named `Franka_Dashboard_V<n>.py` and `Franka_Robot_V<n>.py`. A new version changes both filenames, the filenames in both module docstrings, the `from Franka_Robot_V<n> import (...)` line in the dashboard, the robot module's start hint at its end and `APP_TITLE` (in the robot file). The split into two files happened inside V18 without a version bump; the single-file V18 is in git history (commit `05881b0`). Session logs record the version in their header, so check which version produced a log before reasoning about it against the current code. Behaviour changes are annotated inline as `# V<n>: <reason>`, usually with what was observed on the robot. The git commits are bulk uploads ("Add files via upload") with no description of the changes, so these comments remain the real change history; keep the convention.

## Dashboard architecture

Two files with sections separated by `# ═══ <n>. <name>` banners. The import goes one way only: the dashboard imports from the robot module, never the reverse.

- `Franka_Robot_V18.py` (sorting and robot setup): 1 constants (path model, statuses, labels, `STEP_SETS`, `CSV_FIELDS`, `LIMITS`, `STATUS` colours), 2 settings, 3 kinematics, 4 classification (`classify`, `rest_load`), 5 controller, 6 robot panels (`card()` and the `RobotPanels` mixin: input helpers, header, parameter sections, sorting-cycle card, teach-in, test drive, event log and their `_upd_*` methods).
- `Franka_Dashboard_V18.py` (digital twin and evaluation): 1 display constants (`THEMES`, `FONT`, `BENCH_DRAW_DROP`), 2 evaluation (`compute_metrics`, `_stats`, `DashboardController`), 3 web UI (CSS, fullscreen JS, `Twin`, `Page`), 4 main.

When adding code, put it by topic: anything that moves, sorts or configures the robot goes into the robot module; twin, telemetry, charts, KPIs, confusion matrix, results table, summary and worker view go into the dashboard. A name the dashboard needs from the robot module must be added to its import list (pyflakes finds the missing ones).

- **Controller vs Page.** One controller (created in `main()`) owns the robot, jobs, results and logs, shared by all browser tabs. It is a `DashboardController`, i.e. the robot module's `Controller` plus `summary_md()` (the base class only has a stub that raises). Each tab builds a `Page(RobotPanels)`, which never talks to the robot. `Page.__init__` creates the attributes and timers the mixin relies on, and `_tick_fast` / `_tick_slow` list the updaters of both files. It polls the controller with two `ui.timer`s (0.1 s and 0.5 s) and redraws only when a `ctrl.ver[...]` counter changed, so new state shown in the UI needs a `ver` bump.
- **Threading.** `run_job` allows one job thread at a time; every robot and gripper command runs under `hw_lock`. The telemetry thread reads state only when it gets the lock without blocking, because with panda_py ≥ 1.0 `get_state()` may call `readOnce()`. STOP sets the `abort` event; long operations must go through `_check_abort()` / `_sleep()` and raise `Abort`. STOP is a software stop, not a safety function.
- **Settings.** Every tunable is a field of the `Settings` dataclass, clamped by `LIMITS` (several limits come from the Franka manual: payload, 70 N grasp force, 80 mm travel, 30 mm/s). A background loop writes `dashboard_settings.json` within a second of any change. A cycle works on a copy, so changes apply from the next part. `Settings.load` has no migrations (removed with the bare-bones cleanup): unknown keys in the file are ignored, and a changed default only reaches an existing install if its key is missing from `dashboard_settings.json`, so add a migration again (with a `settings_version` bump) when that matters. UI fields are built with `RobotPanels._num` / `_text` / `_switch`; number and text fields commit only on Enter or blur.
- **Paths.** Taught joint paths live in `waypoints.json` (name → list of 7-joint lists, same format as franka_sort.py), backed up to `waypoints.json.bak` on the first save after each dashboard start. `waypoints_lift.json` stores the measured pose above the holder for the `to_weigh` link check. The path model is spread over module constants that must stay consistent: `PATHS`, `OPTIONAL_PATHS`, `ALL_PATHS`, `CLASS_TO_PATH`, `BENCH_PATH`, `HOME_PATH`, `BENCH_OUT`, `BENCH_REJECT`, `CHAIN_BEFORE`, `ROUTES`, `PATH_LABEL`, `PATH_HINT`, `PATH_CLASS`, plus `Controller._link_targets`, `Controller.chain_before` and `Controller._route_home`. The start point is the first point of `to_holder`. franka_sort.py uses an older path set (a single `to_bench`), so both tools share the file format but not the path semantics.
- **Sort cycle** (`Controller._cycle`, steps in `STEP_SETS`): open, empty trip to the weigh spot and tare, `to_holder`, grasp (finger width = size), lift, `to_weigh`, weigh, classify, `to_bench25`/`to_bench35`, force-guarded descent, open, test, down to the seat, re-grasp, `out_bench*`, then the simulated bench test result (`random` < `defect_pct`, default 50 %): defective → `to_reject25`/`to_reject35` → `home_reject`, working → its container path → `home_*`. The result is `res['test']` (CSV `bench_test`); `pred` stays the drive type, so the confusion matrix is not affected. With `defect_pct` > 0, `path_issues` requires `to_reject25`, `to_reject35` and `home_reject`. All reject paths end at the same drop pose (end of `to_reject`, else of `to_reject25`), where `home_reject` starts. Mode `measure` puts the part back into the holder; mode `dry` drives the paths without a part and with no gripper action except the initial open.
- **Taught vs generated motion.** Taught paths are driven with `move_to_joint_position`, i.e. joint-space interpolation, not straight lines. The only generated moves are vertical Cartesian ones: the lift (`lift_mm`), the bench descent (CartesianImpedance controller, or the `steps` fallback), straight down to the bench seat, lowering for the put-back, and the teach-in jog buttons. V18 deliberately has no sideways generated move at the bench; V17's line-up move collided with the 2.5″ bench.
- **Link checks.** `path_gaps` flags paths that do not start or end where their neighbour does (0.15 rad tolerance); the cycle drives such gaps as untaught direct joint moves. `bench_geometry` requires `out_bench*` to start straight below the end of `to_bench*` (≤ 5 mm, ≤ 4°), and `bench_blockers` refuses to start cycles otherwise.
- **Classification** (section 4 of the robot module). Only the trained model, no fallback: `drive_model.json` must exist next to the `.py` files (the module reads it at import as `DRIVE_MODEL`).
  - **Model.** scikit-learn's `StandardScaler` + multinomial `LogisticRegression` on [gripper width, ln weight], exported as mean/std/coef/intercept; `model_proba()` computes the softmax itself, identical to `predict_proba`.
  - **Features.** The robot can only measure gripper width (= drive thickness = CSV `height_mm`) and weight, so length and form-factor width must never become features.
  - **SSD guard.** An SSD decision needs `ssd_min_conf_pct` (default 60 %), otherwise the likelier HDD wins, because HDD → SSD is the data-security KPI.
  - **Reject windows** apply after the model. `classify()` returns `(class, reason, confidence)`; results carry `conf`, the last CSV column.
  - **Scatter chart.** It shades the model's decision regions (`Page._model_regions`, `classify()` evaluated on a grid and cached) and shows the training points.
  - **Training.** `train_drive_model.py` is kept deliberately minimal at the user's request (no CLI options, no extra reports): pandas `read_csv`, stratified `train_test_split` 75/25 with seed 42, then `StandardScaler` + `LogisticRegressionCV` (C ∈ 0.1…100, shuffled stratified 5-fold, log loss) on the training part. It prints test accuracy and the confusion matrix, refits `LogisticRegression` with that C on all rows, and writes the JSON.
  - **Why not kNN.** kNN was tried and rejected: it forms "SSD islands", where a heavier drive of the same thickness becomes SSD again (1-NN at 7 mm from 98 g on). Logistic regression cannot do that, because at a fixed thickness the SSD/HDD boundary is linear in ln weight.
  - **Small data.** The choice of C changes with the split, and the 12 test drives cannot tell model and old rule apart. Do not over-read single numbers.
- **Worker feedback.** The worker view asks "correct / wrong" for `Controller.feedback_target()`, the last part with a decision, also while it is still `sorting`. `worker_feedback()` turns the answer into the 'Actual' label via `set_truth()`. A label given before the part is in `results` waits in `_pending_truth` and is applied in `_add_result()`. Counters `feedback_correct` / `feedback_wrong` appear in the summary.
- **Weighing.** `O_F_ext_hat_K` has a pose-dependent bias, so tare and load are measured at the same weigh spot (last point of `to_weigh`); mass = |ΔF| / g × scale + offset. The stored tare is dropped on reconnect and when the weigh spot moves.
- **Kinematics.** Own modified-DH forward kinematics (flange 0.107 m, −45° hand rotation, Hand TCP 0.1034 m) drive the 3D twin and the seat geometry. `_bench_seat_z` cross-checks it against the robot's `O_T_EE` and falls back to the plain force-guarded descent if they disagree. panda_py quaternions are xyzw.

## panda_py behaviour the code relies on

- `move_to_joint_position` / `move_to_pose` return `False` instead of raising. `_after_motion` reads the robot state and turns that into a `MotionError` unless the arm ended within tolerance.
- panda_py logs through the `panda` logger. `_PandaLogHandler` routes it into the dashboard log and watches for "Control loop interruped" (panda_py's spelling) to detect that the robot rejected or stopped teaching mode.
- Only panda_py ≥ 1.0 is supported (robot PC: 1.1.1): `get_state()` may call `readOnce()`, so telemetry reads only under `hw_lock`.

## Reading session logs

Each session writes `logs/dashboard_<stamp>` files: `.log` (header with versions, settings and waypoints, then every event including debug lines), `.csv` (one row per part), `_steps.csv`, `_weighing.csv`, `_bench.csv` and `_summary.md`. Useful signals:

- `Untaught moves between paths`: path gaps, largest first.
- `cartesian_reflex` together with panda_py's "Motion finished … from the goal": contact during that move. The `move '<label>'` debug line just before it names the motion.
- "User stopped": the user stop button was pressed; that is not a collision.
- `_steps.csv` shows which step failed. The same motion can be booked under different step keys in `sort` and `dry` mode.

## Working with the user

The user writes in German; code, UI text and logs are English. Changes that affect robot motion cannot be verified here, so say so and name the on-robot check (test drive at speed factor 0.1, then a dry run).
