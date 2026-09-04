import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR  # noqa: E402

TRANSLATOR_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(TRANSLATOR_DIR, 'data', 'raw', 'aslg_pc12_train.parquet')
PREPARED_DATA_DIR = os.path.join(TRANSLATOR_DIR, 'data', 'prepared')
TRANSLATOR_MODELS_DIR = os.path.join(MODELS_DIR, 'translator')

# google/flan-t5-small chosen over plain T5-small or OPUS-MT/Marian: its
# instruction-tuning gives more robust fine-tuning convergence on a custom,
# non-standard-language-pair seq2seq task like gloss->English than a model
# built for a specific bilingual pair. ~80M params, runs on CPU-only torch.
MODEL_NAME = 'google/flan-t5-small'

# Prepended to every gloss input, matching T5's text-to-text task-prefix
# convention.
TASK_PREFIX = 'translate ASL gloss to English: '

# ASLG-PC12 gloss/text averages ~12-13 words, max ~54-59 -- 64 tokens covers
# the observed distribution with margin for subword tokenization overhead.
MAX_GLOSS_LEN = 64
MAX_EN_LEN = 64

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_STATE = 42

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4
