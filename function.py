# Importing all required Python packages, libraries and frameworks
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from config import *  # noqa: F401,F403 -- re-exported for `from function import *` call sites

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

# Connections between the 7-point pose subset, expressed as indices into
# POSE_LANDMARK_INDICES (i.e. local positions in the extracted subset, not
# the original 33-point MediaPipe Pose indices)
POSE_CONNECTIONS = [
    (1, 2),  # left_shoulder - right_shoulder
    (1, 3),  # left_shoulder - left_elbow
    (3, 5),  # left_elbow - left_wrist
    (2, 4),  # right_shoulder - right_elbow
    (4, 6),  # right_elbow - right_wrist
]


# Creating a HandLandmarker for either static images or a live video stream.
# num_hands=1 (default) suits one-handed fingerspelling; word signs need 2.
def create_hand_landmarker(running_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5, num_hands=1):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


# Creating a PoseLandmarker for either static images or a live video stream
def create_pose_landmarker(running_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


# Performing mediapipe detection for images (works for both hand and pose landmarkers)
def mediapipe_detection(image, landmarker, timestamp_ms=None):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    if timestamp_ms is None:
        results = landmarker.detect(mp_image)
    else:
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

    return image, results


# Drawing the hand landmarks/connections, and optionally the pose subset
def draw_styles_landmarks(image, hand_results, pose_results=None):
    height, width = image.shape[:2]

    if hand_results.hand_landmarks:
        for hand_landmarks in hand_results.hand_landmarks:
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

            for point in points:
                cv2.circle(image, point, 3, (0, 0, 255), -1)

    if pose_results is not None and pose_results.pose_landmarks:
        pose = pose_results.pose_landmarks[0]
        points = [(int(pose[i].x * width), int(pose[i].y * height)) for i in POSE_LANDMARK_INDICES]

        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

        for point in points:
            cv2.circle(image, point, 4, (255, 0, 0), -1)


# Extracting the keypoints from the detected hand and pose landmarks.
# Hand-missing and pose-missing are independent events, so each block is
# zero-filled independently rather than zeroing the whole vector if either
# detector misses.
def extract_keypoints(hand_results, pose_results):
    if hand_results.hand_landmarks:
        hand = hand_results.hand_landmarks[0]
        hand_keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    else:
        hand_keypoints = np.zeros(21 * 3)

    if pose_results.pose_landmarks:
        pose = pose_results.pose_landmarks[0]
        pose_keypoints = np.array([
            [pose[i].x, pose[i].y, pose[i].z, pose[i].visibility or 0.0]
            for i in POSE_LANDMARK_INDICES
        ]).flatten()
    else:
        pose_keypoints = np.zeros(len(POSE_LANDMARK_INDICES) * 4)

    return np.concatenate([hand_keypoints, pose_keypoints])


# Extracting keypoints for two-handed word signs. Each hand occupies a fixed
# slot by handedness label ("Left"/"Right") rather than detection order, so
# the same anatomical hand always lands in the same position in the vector
# regardless of the order MediaPipe happens to report hands in. Slots are
# independently zero-filled if that hand isn't present in a frame. If two
# hands somehow report the same handedness label in one frame (rare tracking
# glitch), keep the higher-confidence one rather than silently overwriting.
def extract_keypoints_two_hand(hand_results, pose_results):
    hand_slots = {'Left': None, 'Right': None}

    if hand_results.hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
            label = handedness[0].category_name
            score = handedness[0].score or 0.0

            if label not in hand_slots:
                continue

            if hand_slots[label] is not None and hand_slots[label][1] >= score:
                continue

            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            hand_slots[label] = (keypoints, score)

    hand_keypoints = np.concatenate([
        hand_slots['Left'][0] if hand_slots['Left'] is not None else np.zeros(21 * 3),
        hand_slots['Right'][0] if hand_slots['Right'] is not None else np.zeros(21 * 3),
    ])

    if pose_results.pose_landmarks:
        pose = pose_results.pose_landmarks[0]
        pose_keypoints = np.array([
            [pose[i].x, pose[i].y, pose[i].z, pose[i].visibility or 0.0]
            for i in POSE_LANDMARK_INDICES
        ]).flatten()
    else:
        pose_keypoints = np.zeros(len(POSE_LANDMARK_INDICES) * 4)

    return np.concatenate([hand_keypoints, pose_keypoints])
