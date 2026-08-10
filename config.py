import os
import string

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'MP_Data')
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Vocabulary — full fingerspelling alphabet. J and Z are the only motion-based
# letters; the rest are static handshapes.
actions = list(string.ascii_uppercase)

# Data collection
no_sequences = 60
sequence_length = 40

# Pose landmarks to keep: nose, shoulders, elbows, wrists (upper body only —
# hips/knees/ankles/feet are irrelevant to signing and would just add noise).
POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16]

# MediaPipe detection/tracking confidence
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Training
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Live inference
PREDICTION_CONFIDENCE_THRESHOLD = 0.8
PREDICTION_AGREEMENT_WINDOW = 10

# Set True (or export DEBUG=1) to print live per-class confidence and full
# tracebacks from the prediction loop instead of silently swallowing them.
DEBUG = os.environ.get('DEBUG', '') not in ('', '0', 'false', 'False')
