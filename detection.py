"""
Combined Workout Form Checker (Squat + Bicep Curl)
Pose estimation with MediaPipe Tasks API (PoseLandmarker) + OpenCV.

Runs BOTH detectors on the same camera feed simultaneously:
- SQUAT counter: knee angle state machine (up <-> down)
- CURL counter:  single shared counter, increments whenever EITHER arm
                 completes a curl rep (so alternating or simultaneous
                 curls both count)

Each completed rep, across both exercises, records:
    - exercise name
    - timestamp
    - tempo (seconds spent in the "down"/working phase of that rep)
    - feedback tag ("Good rep!", "Squat deeper", "Don't swing", etc.)

On quit ('q'), the full rep-by-rep log is printed to the terminal as CSV,
followed by a one-line-per-exercise summary. Redirect stdout if you want
it saved to a file, e.g.:
    python combined_form_checker.py > session.csv

Live feedback:
    Squat:
        "Back too rounded"        -> torso angle too small (excessive forward lean)
        "Knees collapsing inward" -> knees track inside the ankles (valgus)
    Curl (per arm, prefixed L/R):
        "Keep elbow pinned"       -> elbow drifting forward/away from torso
        "Don't swing"             -> shoulder moved a lot since the rep started

End-of-rep feedback:
    Squat:  "Squat deeper" / "Good rep!"
    Curl:   "Extend fully" / "Curl higher" / "Good rep!"

On-screen rep history: last 5 reps (any exercise) shown bottom-right
with exercise name + feedback tag, most recent last.

Controls: 'r' resets ALL counters + history (does not touch the log),
'q' quits and writes the session summary.

Camera setup: side-on view works ok for squats, front-on works better for
curls -- a 3/4 angle ~2m back is the best compromise for tracking both.

First run downloads pose_landmarker_lite.task (~5MB) into this folder.
"""

import csv
import os
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Model setup ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# --- Squat thresholds ---
UP_ANGLE = 160
DOWN_ANGLE = 110
GOOD_DEPTH_ANGLE = 100
BACK_LEAN_ANGLE = 45
VALGUS_RATIO = 0.7

# --- Curl thresholds ---
EXTENDED_ANGLE = 160
CURLED_ANGLE = 50
GOOD_EXTENSION_ANGLE = 155
GOOD_CURL_ANGLE = 60
ELBOW_DRIFT_RATIO = 0.35
SWING_RATIO = 0.12

FEEDBACK_FRAMES = 45  # ~1.5s at 30fps
VISIBILITY_THRESH = 0.3
HISTORY_LEN = 5

# --- 33-point body landmark indices ---
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def calculate_angle(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees (0-180)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def to_xy(lm):
    return [lm.x, lm.y]


def dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def draw_skeleton(image, landmarks, w, h):
    for a_idx, b_idx in POSE_CONNECTIONS:
        a, b = landmarks[a_idx], landmarks[b_idx]
        if a.visibility < VISIBILITY_THRESH or b.visibility < VISIBILITY_THRESH:
            continue
        pa = (int(a.x * w), int(a.y * h))
        pb = (int(b.x * w), int(b.y * h))
        cv2.line(image, pa, pb, (245, 117, 66), 2)

    for idx in set(i for pair in POSE_CONNECTIONS for i in pair):
        lm = landmarks[idx]
        if lm.visibility < VISIBILITY_THRESH:
            continue
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (245, 66, 230), -1)


# ---------------- Session log ----------------
class SessionLog:
    """Collects every completed rep across all exercises for export."""

    def __init__(self):
        self.reps = []  # list of dicts: exercise, timestamp, tempo_s, feedback

    def record(self, exercise, tempo_s, feedback):
        self.reps.append({
            "exercise": exercise,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tempo_s": round(tempo_s, 2),
            "feedback": feedback,
        })

    def summary(self):
        """Per-exercise: total reps, good-rep %, most common fault."""
        out = {}
        for r in self.reps:
            ex = r["exercise"]
            out.setdefault(ex, {"total": 0, "good": 0, "faults": {}})
            out[ex]["total"] += 1
            if r["feedback"] == "Good rep!":
                out[ex]["good"] += 1
            else:
                out[ex]["faults"][r["feedback"]] = out[ex]["faults"].get(r["feedback"], 0) + 1
        return out

    def save(self):
        """Print the rep log as CSV directly to the terminal instead of
        writing a file — pipe/redirect stdout if you want it saved,
        e.g. `python combined_form_checker.py > session.csv`."""
        if not self.reps:
            print("No reps recorded.")
            return

        print("\n--- Session log (CSV) ---")
        writer = csv.DictWriter(sys.stdout, fieldnames=["exercise", "timestamp", "tempo_s", "feedback"])
        writer.writeheader()
        writer.writerows(self.reps)

        print("\n--- Session summary ---")
        for ex, stats in self.summary().items():
            pct = 100 * stats["good"] / stats["total"] if stats["total"] else 0
            top_fault = max(stats["faults"], key=stats["faults"].get) if stats["faults"] else "none"
            print(f"{ex}: {stats['total']} reps, {pct:.0f}% good, most common fault: {top_fault}")


# ---------------- Squat state ----------------
def new_squat_state():
    return {
        "stage": "up",
        "min_knee_angle_this_rep": 180,
        "counter": 0,
        "rep_feedback": "",
        "rep_feedback_timer": 0,
        "rep_start_time": None,
    }


def update_squat(state, knee_angle, back_angle, knee_L, knee_R, ankle_L, ankle_R, log):
    live_warnings = []

    if state["stage"] == "up" and knee_angle < DOWN_ANGLE:
        state["stage"] = "down"
        state["min_knee_angle_this_rep"] = knee_angle
        state["rep_start_time"] = time.time()

    elif state["stage"] == "down":
        state["min_knee_angle_this_rep"] = min(state["min_knee_angle_this_rep"], knee_angle)
        if knee_angle > UP_ANGLE:
            state["stage"] = "up"
            state["counter"] += 1
            tempo = time.time() - state["rep_start_time"] if state["rep_start_time"] else 0.0
            if state["min_knee_angle_this_rep"] > GOOD_DEPTH_ANGLE:
                state["rep_feedback"] = "Squat deeper"
            else:
                state["rep_feedback"] = "Good rep!"
            state["rep_feedback_timer"] = FEEDBACK_FRAMES
            state["min_knee_angle_this_rep"] = 180
            log.record("Squat", tempo, state["rep_feedback"])

    if back_angle < BACK_LEAN_ANGLE:
        live_warnings.append("Back too rounded")

    knee_dist = abs(knee_L[0] - knee_R[0])
    ankle_dist = abs(ankle_L[0] - ankle_R[0])
    if ankle_dist > 0.01 and (knee_dist / ankle_dist) < VALGUS_RATIO:
        live_warnings.append("Knees collapsing inward")

    if state["rep_feedback_timer"] > 0:
        state["rep_feedback_timer"] -= 1

    return live_warnings


# ---------------- Curl state ----------------
def new_arm_state():
    return {
        "stage": "down",
        "min_angle_this_rep": 180,
        "max_angle_this_rep": 0,
        "rep_feedback": "",
        "rep_feedback_timer": 0,
        "swing_baseline": None,
        "rep_start_time": None,
    }


def update_arm(state, elbow_angle, shoulder_xy, elbow_xy, torso_width, bicep_counter_ref, side_label, log):
    """bicep_counter_ref is a single-element list acting as a shared mutable int
    so either arm can increment the ONE shared curl counter."""
    live_warnings = []

    if state["stage"] == "down" and elbow_angle < CURLED_ANGLE + 30:
        state["stage"] = "up"
        state["min_angle_this_rep"] = elbow_angle
        state["max_angle_this_rep"] = elbow_angle
        state["swing_baseline"] = shoulder_xy
        state["rep_start_time"] = time.time()

    elif state["stage"] == "up":
        state["min_angle_this_rep"] = min(state["min_angle_this_rep"], elbow_angle)
        state["max_angle_this_rep"] = max(state["max_angle_this_rep"], elbow_angle)

        if state["swing_baseline"] is not None and torso_width > 0.01:
            swing = dist(shoulder_xy, state["swing_baseline"]) / torso_width
            if swing > SWING_RATIO:
                live_warnings.append("Don't swing")

        if elbow_angle > EXTENDED_ANGLE:
            state["stage"] = "down"
            bicep_counter_ref[0] += 1  # shared counter, either arm increments it
            tempo = time.time() - state["rep_start_time"] if state["rep_start_time"] else 0.0

            if state["min_angle_this_rep"] > GOOD_CURL_ANGLE:
                state["rep_feedback"] = "Curl higher"
            elif state["max_angle_this_rep"] < GOOD_EXTENSION_ANGLE:
                state["rep_feedback"] = "Extend fully"
            else:
                state["rep_feedback"] = "Good rep!"
            state["rep_feedback_timer"] = FEEDBACK_FRAMES
            state["swing_baseline"] = None
            log.record(f"Curl ({side_label})", tempo, state["rep_feedback"])

    if torso_width > 0.01:
        elbow_drift = abs(elbow_xy[0] - shoulder_xy[0]) / torso_width
        if elbow_drift > ELBOW_DRIFT_RATIO:
            live_warnings.append("Keep elbow pinned")

    if state["rep_feedback_timer"] > 0:
        state["rep_feedback_timer"] -= 1

    return live_warnings


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Workout Form Checker', cv2.WINDOW_NORMAL)

    squat = new_squat_state()
    left_arm = new_arm_state()
    right_arm = new_arm_state()
    bicep_counter = [0]  # shared mutable counter, incremented by either arm

    log = SessionLog()
    rep_history = deque(maxlen=HISTORY_LEN)  # each item: "Exercise: feedback"

    start_time = time.time()
    last_timestamp_ms = -1

    def history_snapshot():
        """Call after each update_* to catch newly logged reps and mirror them into rep_history."""
        if log.reps:
            latest = log.reps[-1]
            tag = f'{latest["exercise"]}: {latest["feedback"]}'
            if not rep_history or rep_history[-1] != tag:
                rep_history.append(tag)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        image = frame
        all_warnings = []

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            shoulder_L, elbow_L, wrist_L = to_xy(lm[LEFT_SHOULDER]), to_xy(lm[LEFT_ELBOW]), to_xy(lm[LEFT_WRIST])
            shoulder_R, elbow_R, wrist_R = to_xy(lm[RIGHT_SHOULDER]), to_xy(lm[RIGHT_ELBOW]), to_xy(lm[RIGHT_WRIST])
            hip_L, hip_R = to_xy(lm[LEFT_HIP]), to_xy(lm[RIGHT_HIP])
            knee_L, knee_R = to_xy(lm[LEFT_KNEE]), to_xy(lm[RIGHT_KNEE])
            ankle_L, ankle_R = to_xy(lm[LEFT_ANKLE]), to_xy(lm[RIGHT_ANKLE])

            torso_width = dist(shoulder_L, shoulder_R) or dist(shoulder_L, hip_L)

            # --- Curl angles/updates ---
            angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
            angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
            warn_curl_L = update_arm(left_arm, angle_L, shoulder_L, elbow_L, torso_width, bicep_counter, "L", log)
            history_snapshot()
            warn_curl_R = update_arm(right_arm, angle_R, shoulder_R, elbow_R, torso_width, bicep_counter, "R", log)
            history_snapshot()

            # --- Squat angles/updates ---
            knee_angle_L = calculate_angle(hip_L, knee_L, ankle_L)
            knee_angle_R = calculate_angle(hip_R, knee_R, ankle_R)
            knee_angle = (knee_angle_L + knee_angle_R) / 2
            back_angle = calculate_angle(shoulder_L, hip_L, knee_L)
            warn_squat = update_squat(squat, knee_angle, back_angle, knee_L, knee_R, ankle_L, ankle_R, log)
            history_snapshot()

            all_warnings = (
                [f"L: {w_}" for w_ in warn_curl_L]
                + [f"R: {w_}" for w_ in warn_curl_R]
                + warn_squat
            )

            draw_skeleton(image, lm, w, h)

            cv2.putText(image, f'Knee: {int(knee_angle)}  Back: {int(back_angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'L elbow: {int(angle_L)}  R elbow: {int(angle_R)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f'Squat stage: {squat["stage"]}  L: {left_arm["stage"]}  R: {right_arm["stage"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Counters (always shown) ---
        cv2.putText(image, f'SQUATS: {squat["counter"]}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(image, f'CURLS: {bicep_counter[0]}', (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3)

        for i, warn in enumerate(all_warnings):
            cv2.putText(image, warn, (10, 430 - i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        y_off = 0
        for label, state in (("Squat", squat), ("L curl", left_arm), ("R curl", right_arm)):
            if state["rep_feedback_timer"] > 0:
                color = (0, 255, 0) if state["rep_feedback"] in ("Good rep!",) else (0, 255, 255)
                cv2.putText(image, f'{label}: {state["rep_feedback"]}', (10, 460 + y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_off += 28

        # --- Rep history, bottom-right ---
        hist_x = w - 320
        cv2.putText(image, "Recent reps:", (hist_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, tag in enumerate(rep_history):
            cv2.putText(image, tag, (hist_x, 55 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Workout Form Checker', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            squat["counter"] = 0
            bicep_counter[0] = 0
            rep_history.clear()

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    log.save()


if __name__ == "__main__":
    main()