# Importing all required Python packages, libraries and frameworks
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Path to the downloaded hand landmark model used by the Tasks API
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')

# Standard 21-point hand landmark topology, used for drawing connections
# since the legacy mp.solutions.hands.HAND_CONNECTIONS is no longer available
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky finger
    (0, 17),                                 # palm
]

# Creating a HandLandmarker for either static images or a live video stream
def create_hand_landmarker(running_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)

# Performing mediapipe detection for images
def mediapipe_detection(image, landmarker, timestamp_ms=None):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    if timestamp_ms is None:
        results = landmarker.detect(mp_image)
    else:
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

    return image, results

# Drawing the landmarks and the hand connections
def draw_styles_landmarks(image, results):
    if results.hand_landmarks:
        height, width = image.shape[:2]
        for hand_landmarks in results.hand_landmarks:
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

            for point in points:
                cv2.circle(image, point, 3, (0, 0, 255), -1)


# Extracting the keypoints from the detected landmarks
def extract_keypoints(results):
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()

    return np.zeros(21*3)

# Defining the paths and parameters for data detection
DATA_PATH = os.path.join('MP_Data')
actions = ['A', 'B', 'C']
no_sequences = 30
sequence_length = 30
