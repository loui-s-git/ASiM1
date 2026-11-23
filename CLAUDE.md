# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this folder is

`~/Downloads` is not a dedicated repo and is not under git. It also holds unrelated personal files (3D-print files, PDFs, photos, other projects); leave those alone. The robot project is:

- `Franka_Dashboard_V18.py`: current NiceGUI web dashboard, the main codebase (single file, ~4,800 lines).
- `Franka_Dashboard_V17.py`, `franka_dashboard.py`: older dashboard versions kept for reference. Only the early `franka_dashboard.py` has a `--sim` simulator; the V-numbered files need a real robot to do anything beyond showing the UI.
- `franka_sort.py`: original terminal tool (`teach` / `run`). The dashboard mirrors its defaults, its `classify()` decision tree and the first 17 columns of its CSV (`CSV_FIELDS`); keep those compatible.
- `franka_measure.py`: standalone grasp-and-weigh demo.
- `install_franka.sh`: Ubuntu setup (venv `~/franka_venv`, numpy + panda-python with bundled libfranka 0.9.2, realtime limits). It does not install nicegui.
- `logs/`: session files copied over from the robot PC for analysis.
- `Teach_Ablauf_V18.md`: German teach-in checklist for every path. Update it when path semantics change.

## Where things run

The robot PC is Ubuntu (`/home/bloo52/Downloads`, Python 3.12, numpy, nicegui 3.19, panda-python 1.1.1, robot FCI at `172.16.0.2`). This Mac has Python 3.9 without numpy, nicegui or panda_py, so nothing robot-related runs here. Code is edited here and copied to the robot PC; only the user runs it on the robot.

## Commands

On the robot PC:

```bash
source ~/franka_venv/bin/activate
python3 Franka_Dashboard_V18.py 172.16.0.2     # opens http://localhost:8080 (worker view: /?view=worker)
python3 Franka_Dashboard_V18.py 172.16.0.2 --no-browser --port 8081 --lan --verbose --file other_waypoints.json
python3 franka_sort.py teach 172.16.0.2
python3 franka_sort.py run 172.16.0.2 --speed 0.2
```

There are no tests, linters or build steps. The only check possible on this Mac is a syntax check:

```bash
python3 -I -c "import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]" Franka_Dashboard_V18.py
```

For offline geometry checks (e.g. where a taught joint pose puts the TCP), reimplement the DH forward kinematics from section 3 in plain Python, since numpy is missing here.

## Versioning

Each release is a full copy named `Franka_Dashboard_V<n>.py`. A new version changes the filename, the filename in the module docstring and `APP_TITLE`. Session logs record the version in their header, so check which version produced a log before reasoning about it against the current code. Behaviour changes are annotated inline as `# V<n>: <reason>`, usually with what was observed on the robot. With no git, these comments are the change history; keep the convention.

## Dashboard architecture (`Franka_Dashboard_V18.py`)

Seven sections separated by `# ═══ <n>. <name>` banners: constants, settings, kinematics, classification & metrics, controller, web UI, main.

- **Controller vs Page.** One `Controller` (global `CTRL`) owns the robot, jobs, results and logs, shared by all browser tabs. Each tab builds a `Page`, which never talks to the robot. It polls the controller with two `ui.timer`s (0.1 s and 0.5 s) and redraws only when a `ctrl.ver[...]` counter changed, so new state shown in the UI needs a `ver` bump.
- **Threading.** `run_job` allows one job thread at a time; every robot and gripper command runs under `hw_lock`. The telemetry thread reads state only when it gets the lock without blocking, because with panda_py ≥ 1.0 `get_state()` may call `readOnce()`. STOP sets the `abort` event; long operations must go through `_check_abort()` / `_sleep()` and raise `Abort`. STOP is a software stop, not a safety function.
- **Settings.** Every tunable is a field of the `Settings` dataclass, clamped by `LIMITS` (several limits come from the Franka manual: payload, 70 N grasp force, 80 mm travel, 30 mm/s). A background loop writes `dashboard_settings.json` within a second of any change. A cycle works on a copy, so changes apply from the next part. Changing a default for existing installs needs a `settings_version` bump plus a migration in `Settings.load`. UI fields are built with `Page._num` / `_text` / `_switch`; number and text fields commit only on Enter or blur.
- **Paths.** Taught joint paths live in `waypoints.json` (name → list of 7-joint lists, same format as franka_sort.py), backed up to `waypoints.json.bak` on the first save after each dashboard start. `waypoints_lift.json` stores the measured pose above the holder for the `to_weigh` link check. The path model is spread over module constants that must stay consistent: `PATHS`, `OPTIONAL_PATHS`, `ALL_PATHS`, `CLASS_TO_PATH`, `BENCH_PATH`, `HOME_PATH`, `BENCH_OUT`, `CHAIN_BEFORE`, `ROUTES`, `PATH_LABEL`, `PATH_HINT`, plus `Controller._link_targets` and `Controller._route_home`. The start point is the first point of `to_holder`. franka_sort.py uses an older path set (a single `to_bench`), so both tools share the file format but not the path semantics.
- **Sort cycle** (`Controller._cycle`, steps in `STEP_SETS`): open, empty trip to the weigh spot and tare, `to_holder`, grasp (finger width = size), lift, `to_weigh`, weigh, classify, `to_bench25`/`to_bench35`, force-guarded descent, open, test, down to the seat, re-grasp, `out_bench*`, container path, `home_*`. Mode `measure` puts the part back into the holder; mode `dry` drives the paths without a part and with no gripper action except the initial open.
- **Taught vs generated motion.** Taught paths are driven with `move_to_joint_position`, i.e. joint-space interpolation, not straight lines. The only generated moves are vertical Cartesian ones: the lift (`lift_mm`), the bench descent (CartesianImpedance controller, or the `steps` fallback), straight down to the bench seat, lowering for the put-back, and the teach-in jog buttons. V18 deliberately has no sideways generated move at the bench; V17's line-up move collided with the 2.5″ bench.
- **Link checks.** `path_gaps` flags paths that do not start or end where their neighbour does (0.15 rad tolerance); the cycle drives such gaps as untaught direct joint moves. `bench_geometry` requires `out_bench*` to start straight below the end of `to_bench*` (≤ 5 mm, ≤ 4°), and `bench_blockers` refuses to start cycles otherwise.
- **Weighing.** `O_F_ext_hat_K` has a pose-dependent bias, so tare and load are measured at the same weigh spot (last point of `to_weigh`); mass = |ΔF| / g × scale + offset. The stored tare is dropped on reconnect and when the weigh spot moves.
- **Kinematics.** Own modified-DH forward kinematics (flange 0.107 m, −45° hand rotation, Hand TCP 0.1034 m) drive the 3D twin and the seat geometry. `_bench_seat_z` cross-checks it against the robot's `O_T_EE` and falls back to the plain force-guarded descent if they disagree. panda_py quaternions are xyzw.

## panda_py behaviour the code relies on

- `move_to_joint_position` / `move_to_pose` return `False` instead of raising. `_after_motion` reads the robot state and turns that into a `MotionError` unless the arm ended within tolerance.
- panda_py logs through the `panda` logger. `_PandaLogHandler` routes it into the dashboard log and watches for "Control loop interruped" (panda_py's spelling) to detect that the robot rejected or stopped teaching mode.
- `live_motion_state` is derived from whether the `Panda` object has `is_moving`, which distinguishes panda_py < 1.0 from ≥ 1.0.

## Reading session logs

Each session writes `logs/dashboard_<stamp>` files: `.log` (header with versions, settings and waypoints, then every event including debug lines), `.csv` (one row per part), `_steps.csv`, `_weighing.csv`, `_bench.csv` and `_summary.md`. Useful signals:

- `Untaught moves between paths`: path gaps, largest first.
- `cartesian_reflex` together with panda_py's "Motion finished … from the goal": contact during that move. The `move '<label>'` debug line just before it names the motion.
- "User stopped": the user stop button was pressed; that is not a collision.
- `_steps.csv` shows which step failed. The same motion can be booked under different step keys in `sort` and `dry` mode.

## Working with the user

The user writes in German; code, UI text and logs are English. Changes that affect robot motion cannot be verified here, so say so and name the on-robot check (test drive at speed factor 0.1, then a dry run).
