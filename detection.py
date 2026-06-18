import cv2
import mediapipe as mp

# --- Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Key landmark indices
KEYPOINTS = {
    "Nose": 0,
    "L Shoulder": 11, "R Shoulder": 12,
    "L Elbow": 13,    "R Elbow": 14,
    "L Wrist": 15,    "R Wrist": 16,
    "L Hip": 23,      "R Hip": 24,
    "L Knee": 25,     "R Knee": 26,
    "L Ankle": 27,    "R Ankle": 28,
}

print("Pose Detection running. Press 'q' to quit.")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        display = frame.copy()

        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                display,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            lm = results.pose_landmarks.landmark

            # --- Visibility overlay (top-left panel) ---
            panel_h = len(KEYPOINTS) * 18 + 10
            cv2.rectangle(display, (0, 0), (160, panel_h), (0, 0, 0), -1)
            for i, (name, idx) in enumerate(KEYPOINTS.items()):
                vis = lm[idx].visibility
                color = (0, 255, 0) if vis > 0.6 else (0, 165, 255) if vis > 0.3 else (0, 0, 255)
                cv2.putText(display, f"{name}: {vis:.2f}", (5, 15 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # --- Status banner ---
            cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(display, "POSE DETECTED", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.putText(display, "No Person Detected", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Pose Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()