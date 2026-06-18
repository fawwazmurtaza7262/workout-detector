import os
import time
import urllib.request
 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Model setup ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
cap = cv2.VideoCapture(0)

UP_ANGLE = 160
DOWN_ANGLE = 110
GOOD_ANGLE_DEPTH = 100
BACK_LEAN_ANGLE = 45

VALGUS_RATIO = 0.7      
FEEDBACK_FRAMES = 45    
VISIBILITY_THRESH = 0.3


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
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
        
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def to_xy(lm):
    return [lm.x, lm.y]

def draw_skeleton(image, landmarks, w, h):
    for a_idx , b_idx in POSE_CONNECTIONS:
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
        




cap.release()
cv2.destroyAllWindows()