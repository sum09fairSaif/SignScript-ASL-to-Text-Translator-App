import os

from config import BASE_DIR, MODELS_DIR, DEBUG  # noqa: F401 -- re-exported
from config import HAND_MIN_DETECTION_CONFIDENCE, HAND_MIN_TRACKING_CONFIDENCE  # noqa: F401
from config import POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE  # noqa: F401

# Paths
GLOSS_DATA_PATH = os.path.join(BASE_DIR, 'GlossData')
GLOSS_MODELS_DIR = os.path.join(MODELS_DIR, 'gloss')
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
ASL_CITIZEN_DIR = os.path.join(EXTERNAL_DATA_DIR, 'asl_citizen')

# Core vocabulary recorded live via gloss_data.py -- common conversational
# signs. The full training vocabulary (`glosses` used by gloss_trainmodel.py)
# is NOT this fixed list: with a full ASL Citizen import (2,731 signs) on top
# of this core set, hardcoding every class name here would be unwieldy and
# would drift from whatever's actually present in GLOSS_DATA_PATH. Instead
# gloss_trainmodel.py/gloss_evaluate.py/app_gloss.py discover the real
# vocabulary by listing GLOSS_DATA_PATH's subdirectories at run time (see
# `discover_glosses()` below) -- this list is only the *recording target* for
# gloss_data.py's live-capture sessions.
CORE_GLOSSES = [
    'HELLO', 'THANK-YOU', 'PLEASE', 'YES', 'NO', 'HELP', 'WANT', 'NEED',
    'EAT', 'DRINK', 'MORE', 'FINISH', 'GOOD', 'BAD', 'LOVE', 'FAMILY',
    'FRIEND', 'WORK', 'SCHOOL', 'HOME', 'WHERE', 'WHAT', 'WHO', 'NAME',
    'LEARN',
]


def discover_glosses(data_path=GLOSS_DATA_PATH):
    """The real training vocabulary: whatever gloss subdirectories actually
    exist under data_path, sorted for a deterministic label mapping. Combines
    self-recorded (gloss_data.py) and externally-imported
    (gloss_import_external.py) classes transparently -- both write into the
    same directory structure, so this doesn't need to know which is which."""
    if not os.path.isdir(data_path):
        return []
    return sorted(
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    )


# Two-handed capture
GLOSS_NUM_HANDS = 2

# Data collection (live recording of the core vocabulary)
GLOSS_NO_SEQUENCES = 60
GLOSS_SEQUENCE_LENGTH = 40

# Training
GLOSS_BATCH_SIZE = 32
GLOSS_MAX_EPOCHS = 300
GLOSS_EARLY_STOPPING_PATIENCE = 25
GLOSS_LEARNING_RATE = 1e-3
GLOSS_VAL_SPLIT = 0.15
GLOSS_TEST_SPLIT = 0.15

# Live inference -- separate from the letter pipeline's thresholds since a
# much larger, differently-distributed vocabulary will need its own
# empirical calibration (the letter threshold needed retuning from 0.8 to
# 0.4 once tried against real 26-class behavior; expect the same here).
GLOSS_PREDICTION_CONFIDENCE_THRESHOLD = 0.4
GLOSS_PREDICTION_AGREEMENT_WINDOW = 10
