"""
Squat Form Checker
Pose estimation with MediaPipe Tasks API (PoseLandmarker) + OpenCV.

- Counts reps via knee angle state machine (up <-> down)
- Live feedback:
    "Back too rounded"        -> torso angle too small (excessive forward lean)
    "Knees collapsing inward" -> knees track inside the ankles (valgus)
- End-of-rep feedback:
    "Squat deeper"            -> bottom of rep never reached target knee angle
    "Good rep!"                -> depth was sufficient

Controls: 'r' resets rep counter, 'q' quits.
Camera setup: side-on view works best (so the knee/hip/shoulder angle is
meaningful). Stand ~2m from the webcam, full body in frame.

First run downloads pose_landmarker_lite.task (~5MB) into this folder.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model setup 
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

#--- Squat thresholds ---
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

    cv2.namedWindow('Squat Form Checker', cv2.WINDOW_NORMAL)

    counter = 0
    stage = "up"
    min_knee_angle_this_rep = 180

    rep_feedback = ""
    rep_feedback_timer = 0

    start_time = time.time()
    last_timestamp_ms = -1

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
        live_warnings = []

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            hip_L, knee_L, ankle_L = to_xy(lm[LEFT_HIP]), to_xy(lm[LEFT_KNEE]), to_xy(lm[LEFT_ANKLE])
            hip_R, knee_R, ankle_R = to_xy(lm[RIGHT_HIP]), to_xy(lm[RIGHT_KNEE]), to_xy(lm[RIGHT_ANKLE])
            shoulder_L = to_xy(lm[LEFT_SHOULDER])

            # Average knee angle (both legs) for rep tracking
            knee_angle_L = calculate_angle(hip_L, knee_L, ankle_L)
            knee_angle_R = calculate_angle(hip_R, knee_R, ankle_R)
            knee_angle = (knee_angle_L + knee_angle_R) / 2

            # Torso/back angle (left side, since side-on camera is assumed)
            back_angle = calculate_angle(shoulder_L, hip_L, knee_L)

            # Rep counting state machine 
            if stage == "up" and knee_angle < DOWN_ANGLE:
                stage = "down"
                min_knee_angle_this_rep = knee_angle

            elif stage == "down":
                min_knee_angle_this_rep = min(min_knee_angle_this_rep, knee_angle)
                if knee_angle > UP_ANGLE:
                    stage = "up"
                    counter += 1
                    if min_knee_angle_this_rep > GOOD_DEPTH_ANGLE:
                        rep_feedback = "Squat deeper"
                    else:
                        rep_feedback = "Good rep!"
                    rep_feedback_timer = FEEDBACK_FRAMES
                    min_knee_angle_this_rep = 180

            # --- Live form checks ---
            if back_angle < BACK_LEAN_ANGLE:
                live_warnings.append("Back too rounded")

            knee_dist = abs(knee_L[0] - knee_R[0])
            ankle_dist = abs(ankle_L[0] - ankle_R[0])
            if ankle_dist > 0.01 and (knee_dist / ankle_dist) < VALGUS_RATIO:
                live_warnings.append("Knees collapsing inward")

            draw_skeleton(image, lm, w, h)

            # HUD text
            cv2.putText(image, f'Knee angle: {int(knee_angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Back angle: {int(back_angle)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Stage: {stage}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(image, f'REPS: {counter}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        for i, warn in enumerate(live_warnings):
            cv2.putText(image, warn, (10, 430 - i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if rep_feedback_timer > 0:
            color = (0, 255, 0) if rep_feedback == "Good rep!" else (0, 255, 255)
            cv2.putText(image, rep_feedback, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            rep_feedback_timer -= 1

        cv2.imshow('Squat Form Checker', image)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            print(f"key pressed: {key}")
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter = 0

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()