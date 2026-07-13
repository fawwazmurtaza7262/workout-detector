# AI Exercise Form Checker

Real-time exercise form and rep tracking using pose estimation — no wearables,
just a webcam. Built with MediaPipe's Tasks API (PoseLandmarker) and OpenCV.

## What it does

Tracks two exercises simultaneously from a single camera feed:

| Exercise | Counter logic | Form checks |
|---|---|---|
| Squat | knee-angle state machine (up ↔ down) | back rounding, knee valgus (caving in) |
| Bicep curl | per-arm state machines, shared rep counter | elbow drift, swinging/momentum |

For every completed rep, across both exercises, the app records:
- which exercise
- timestamp
- **tempo** (seconds spent in the working phase of the rep)
- a feedback tag (`"Good rep!"`, `"Squat deeper"`, `"Don't swing"`, etc.)

On quit, this is written out as both a `.json` (full rep-by-rep log + summary)
and a `.csv` (flat table) in `session_logs/`, and a summary prints to the
terminal, e.g.:

```
--- Session summary ---
Squat: 12 reps, 83% good, most common fault: Squat deeper
Curl (L): 10 reps, 90% good, most common fault: Extend fully
Curl (R): 9 reps, 100% good, most common fault: none
```

The on-screen overlay also shows a rolling list of the last 5 reps (any
exercise) so you can see recent form trends without waiting for the session
to end.

## Why this project

Off-the-shelf fitness apps rely on wearables or manual logging. This tracks
form geometrically, in real time, from joint angles alone — no extra
hardware. It's also a concrete demonstration of:
- working with a **live video pipeline** (frame capture → pose inference →
  geometric analysis → feedback), not a static dataset
- **independent per-limb state machines** (e.g. left/right arm curl
  detection tracked separately but reported through one shared counter)
- adapting to a **breaking API change** (MediaPipe deprecated the legacy
  `mp.solutions.pose` API; this uses the current Tasks API / PoseLandmarker)

## Requirements

```
pip install mediapipe opencv-python numpy
```

Python 3.10+ recommended. First run downloads `pose_landmarker_lite.task`
(~5MB) into the project folder automatically.

## Usage

```
python combined_form_checker.py
```

- **Camera setup:** a 3/4 angle, ~2m back, is the best single compromise for
  tracking both exercises. Side-on is better for squats alone; front-on is
  better for curls alone.
- **Controls:** `r` resets all rep counters and the on-screen history (does
  not affect the saved log), `q` quits and writes the session summary.

## Known limitations

- Single-person tracking only (`num_poses=1`).
- Thresholds (angle cutoffs, drift ratios) are tuned for a roughly front-
  facing adult body and may need adjustment for different camera distances
  or body proportions — see the constants at the top of the script.

## Possible extensions

- A third exercise (e.g. push-ups) — deliberately left out for now since it
  would just repeat the same up/down angle state machine pattern already
  shown by squats; more valuable once exercises are config-driven (below)
  so adding one is cheap and shows the abstraction actually generalizes
- Exercise auto-detection (classify which exercise is being performed
  instead of running both detectors at once)
- Config-driven exercise definitions (add new exercises via an angle-
  threshold dict rather than a new function)
- A Streamlit dashboard wrapping the webcam feed + live counters + a chart
  of form quality over time
- Voice feedback (`pyttsx3`) for form cues without needing to look at the
  screen mid-set