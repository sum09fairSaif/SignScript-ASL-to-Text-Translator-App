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
# 0.8 was a placeholder carried over from the old 3-letter model and was never
# calibrated against real 26-class behavior. Live debug output on the actual
# trained model showed correct, stable predictions peaking around 0.45-0.57
# confidence (softmax spread thin across 26 classes rather than sharply
# peaked), so 0.8 silently filtered out every real prediction. Revisit this
# once more training data narrows the model's confidence spread.
PREDICTION_CONFIDENCE_THRESHOLD = 0.4
PREDICTION_AGREEMENT_WINDOW = 10

# Set True (or export DEBUG=1) to print live per-class confidence and full
# tracebacks from the prediction loop instead of silently swallowing them.
DEBUG = os.environ.get('DEBUG', '') not in ('', '0', 'false', 'False')
