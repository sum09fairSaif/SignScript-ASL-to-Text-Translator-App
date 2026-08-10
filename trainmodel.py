# Importing necessary libraries, program files and modules
from function import *

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from model_builder import build_model

RANDOM_STATE = 42

# Mapping actions to numerical values (labels)
label_map = {label: num for num, label in enumerate(actions)}

# Loading complete sequences only. A sequence is skipped (not zero-padded)
# if any of its frame files are missing -- synthesizing fake frames for an
# interrupted recording session would reintroduce the same class of bug as
# the earlier static-photo-repeated-30x issue this project already hit.
sequences, labels = [], []
skipped = 0
for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_dir):
        continue

    for sequence_name in sorted(os.listdir(action_dir)):
        if not sequence_name.isdigit():
            continue

        sequence_dir = os.path.join(action_dir, sequence_name)
        frame_paths = [os.path.join(sequence_dir, f"{frame_num}.npy") for frame_num in range(sequence_length)]

        if not all(os.path.exists(p) for p in frame_paths):
            skipped += 1
            continue

        sequences.append([np.load(p) for p in frame_paths])
        labels.append(label_map[action])

if skipped:
    print(f"Skipped {skipped} incomplete sequence(s) (missing frame files)")

if not sequences:
    raise RuntimeError("No complete sequences found in MP_Data -- run data.py first")

counts = Counter(labels)
missing_actions = [action for action in actions if counts[label_map[action]] == 0]
if missing_actions:
    print(f"Warning: no data at all for: {', '.join(missing_actions)}")

sparse_actions = [action for action in actions if 0 < counts[label_map[action]] < 5]
if sparse_actions:
    print(f"Warning: fewer than 5 sequences for: {', '.join(sparse_actions)} -- consider recording more before trusting results for these letters")

# Preparing the train/val/test datasets. Test is split off first and touched
# only once, later, by evaluate.py -- val drives early stopping here.
X = np.array(sequences)
y_labels = np.array(labels)
y = to_categorical(y_labels, num_classes=len(actions)).astype(int)

X_temp, X_test, y_temp, y_test, labels_temp, _ = train_test_split(
    X, y, y_labels, test_size=TEST_SPLIT, stratify=y_labels, random_state=RANDOM_STATE
)
val_fraction = VAL_SPLIT / (1 - TEST_SPLIT)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_fraction, stratify=labels_temp, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

os.makedirs(MODELS_DIR, exist_ok=True)
np.savez(os.path.join(MODELS_DIR, 'test_split.npz'), X_test=X_test, y_test=y_test)

# Defining and compiling the model
model = build_model((sequence_length, X.shape[2]), len(actions))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

checkpoint_path = os.path.join(MODELS_DIR, 'checkpoint.weights.h5')
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

# Training the model
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint],
)

# Belt-and-suspenders: explicitly reload the best checkpoint rather than
# relying solely on EarlyStopping's restore_best_weights semantics
model.load_weights(checkpoint_path)

# Saving the model
model_json = model.to_json()
model_json_path = os.path.join(MODELS_DIR, 'model_v2_26letter.json')
model_h5_path = os.path.join(MODELS_DIR, 'model_v2_26letter.h5')

with open(model_json_path, 'w') as json_file:
    json_file.write(model_json)

model.save(model_h5_path)
print(f"Saved model to {model_json_path} and {model_h5_path}")

# Saving a loss/accuracy-vs-epoch plot for a quick convergence sanity check
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['categorical_accuracy'], label='train')
axes[1].plot(history.history['val_categorical_accuracy'], label='val')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

fig.tight_layout()
history_path = os.path.join(MODELS_DIR, 'training_history.png')
fig.savefig(history_path)
print(f"Saved training history plot to {history_path}")
print("Run evaluate.py next before trusting these results -- training accuracy alone is not sufficient evidence with 26 classes.")
