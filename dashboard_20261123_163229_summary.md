# Session dashboard_20261123_163229 — summary

Franka Dashboard V17 · session start 2026-11-23 16:32:29 · updated 2026-11-23 16:43:15 · robot 172.16.0.2

## Overview

| Metric | Value |
|---|---|
| Parts handled (sort + measure only) | 1 (sort 1, measure only 0) |
| Sorted into a container | 0 |
| … with bench contact | — |
| Holder empty (no grasp) | 0 |
| Lost at bench / error / stopped | 0 / 1 / 0 |
| Dry runs (not in the results) | 4 |

## Cycle time (sorted parts)

No sorted parts yet.

## Time per step (sorted parts)

No step times yet.

## Classification (parts with an 'Actual' label)

No labelled parts yet — pick the actual part in 'Actual' (results table) to get accuracy and the confusion matrix.

## Measurements

No measured parts yet.

## Interventions & events

| Event | Count |
|---|---|
| STOP pressed | 0 |
| Restart in the worker view | 0 |
| … part put back into the holder after a restart | 0 |
| Recover errors (manual) | 0 |
| Automatic recovery after an error | 4 |
| Bench 'Continue' (worker confirms the test) | 0 |
| Auto mode started | 0 |
| Tare measurements (empty gripper at the weigh spot) | 1 |
| Robot connections | 3 |
| Warnings / errors in the log | 11 / 6 |

Unplanned interventions (STOP + Restart + Recover): **0**

## Key settings (end of session — all settings are in the .log header)

speed 0.2 · home speed 0.15 · grasp force 70 N · weighing 1 s (settle 1 s, norm) at the weigh spot, tare before every part · bench contact ≤ 40 g and ≤ 35 % of the part, descent 10 mm/s (impedance), gripper opened during the test, test 10 s · thresholds: width ≥ 20 mm → HDD 3.5″, mass ≥ 90 g → HDD 2.5″ · taught exit path out of each bench · own way home from every container

## Files of this session

- `dashboard_20261123_163229.log` — full event log incl. debug details and settings header
- `dashboard_20261123_163229.csv` — one row per part
- `dashboard_20261123_163229_steps.csv` — start, end and duration of every step
- `dashboard_20261123_163229_weighing.csv` — raw force samples (tare / load)
- `dashboard_20261123_163229_bench.csv` — bench descent curve (descent vs. rest load)
