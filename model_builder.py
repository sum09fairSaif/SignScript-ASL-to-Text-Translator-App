import os

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

from keras import regularizers
from keras.layers import (
    Bidirectional,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    LSTM,
)
from keras.models import Sequential


def build_model(input_shape, num_classes):
    # No Masking layer: Conv1D doesn't propagate masks, so a Masking layer here
    # would be silently destroyed and give a false sense of robustness to
    # zero-padded frames. Incomplete sequences are filtered out at data-loading
    # time instead (see trainmodel.py) rather than padded with synthetic zeros.
    return Sequential([
        Conv1D(64, kernel_size=3, padding='causal', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
