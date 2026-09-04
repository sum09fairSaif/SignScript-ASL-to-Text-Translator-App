# Loads the fine-tuned translator and exposes translate(gloss_tokens) -> str
# for app_gloss.py.
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from translator_config import MAX_EN_LEN, MAX_GLOSS_LEN, TASK_PREFIX, TRANSLATOR_MODELS_DIR

FINAL_MODEL_DIR = os.path.join(TRANSLATOR_MODELS_DIR, 'final')

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return

    if not os.path.isdir(FINAL_MODEL_DIR):
        raise RuntimeError(f"Missing {FINAL_MODEL_DIR} -- run translator/train.py first")

    _tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
    _model = AutoModelForSeq2SeqLM.from_pretrained(FINAL_MODEL_DIR)
    _model.eval()


def translate(gloss_tokens):
    """gloss_tokens: list[str] of recognized gloss words -> a grammatical
    English sentence (str)."""
    _load()

    if not gloss_tokens:
        return ''

    gloss_text = ' '.join(gloss_tokens)
    inputs = _tokenizer(TASK_PREFIX + gloss_text, return_tensors='pt', max_length=MAX_GLOSS_LEN, truncation=True)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_length=MAX_EN_LEN)

    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)
