import os

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import MODELS_DIR, actions

MODEL_JSON_PATH = os.path.join(MODELS_DIR, 'model_v2_26letter.json')
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'model_v2_26letter.h5')
TEST_SPLIT_PATH = os.path.join(MODELS_DIR, 'test_split.npz')


def load_model():
    with open(MODEL_JSON_PATH, 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


def main():
    if not os.path.exists(MODEL_JSON_PATH) or not os.path.exists(TEST_SPLIT_PATH):
        raise RuntimeError(
            f"Missing {MODEL_JSON_PATH} or {TEST_SPLIT_PATH} -- run trainmodel.py first"
        )

    model = load_model()

    test_split = np.load(TEST_SPLIT_PATH)
    X_test, y_test = test_split['X_test'], test_split['y_test']

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Only report on classes that actually appear in the test set -- with a
    # dataset still being built out, some letters may have too few sequences
    # to appear in the held-out split at all
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [actions[i] for i in present_labels]

    print(f"Test set size: {len(y_true)} sequences across {len(present_labels)} letters\n")
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    side = max(8, len(present_labels) * 0.4)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(
        ax=ax, xticks_rotation='vertical', colorbar=False
    )
    fig.tight_layout()

    cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    fig.savefig(cm_path)
    print(f"\nSaved confusion matrix to {cm_path}")


if __name__ == '__main__':
    main()
