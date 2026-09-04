# Trains the word-level gloss recognizer, parallel to trainmodel.py but for
# a dynamically-discovered, much larger vocabulary spanning self-recorded
# and externally-imported data.
from function import *
from gloss_config import GLOSS_DATA_PATH, GLOSS_MODELS_DIR, GLOSS_SEQUENCE_LENGTH
from gloss_config import GLOSS_BATCH_SIZE, GLOSS_MAX_EPOCHS, GLOSS_EARLY_STOPPING_PATIENCE
from gloss_config import GLOSS_LEARNING_RATE, GLOSS_VAL_SPLIT, GLOSS_TEST_SPLIT
from gloss_config import discover_glosses

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

import json

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

# A class needs enough sequences to survive a stratified 3-way split (at
# least one example in each of train/val/test, ideally more). Unlike the
# fixed 26-letter vocabulary (uniformly 60 sequences/letter), the gloss
# vocabulary mixes a self-recorded core with externally-imported classes of
# widely varying sample counts, so under-sampled classes are excluded
# outright rather than just warned about -- a handful of 1-2-sample classes
# would otherwise crash train_test_split's stratification entirely.
MIN_SEQUENCES_PER_CLASS = 5

candidate_glosses = discover_glosses(GLOSS_DATA_PATH)
if not candidate_glosses:
    raise RuntimeError(f"No gloss data found in {GLOSS_DATA_PATH} -- run gloss_data.py or gloss_import_external.py first")

# Loading complete sequences only, same reasoning as trainmodel.py: a
# sequence is skipped (not zero-padded) if any frame file is missing rather
# than synthesizing fake temporal data.
sequences_by_gloss = {gloss: [] for gloss in candidate_glosses}
skipped = 0
for gloss in candidate_glosses:
    gloss_dir = os.path.join(GLOSS_DATA_PATH, gloss)
    for sequence_name in sorted(os.listdir(gloss_dir)):
        if not sequence_name.isdigit():
            continue

        sequence_dir = os.path.join(gloss_dir, sequence_name)
        frame_paths = [os.path.join(sequence_dir, f"{frame_num}.npy") for frame_num in range(GLOSS_SEQUENCE_LENGTH)]

        if not all(os.path.exists(p) for p in frame_paths):
            skipped += 1
            continue

        sequences_by_gloss[gloss].append([np.load(p) for p in frame_paths])

if skipped:
    print(f"Skipped {skipped} incomplete sequence(s) (missing frame files)")

excluded = sorted(g for g, seqs in sequences_by_gloss.items() if len(seqs) < MIN_SEQUENCES_PER_CLASS)
if excluded:
    preview = ', '.join(excluded[:20]) + ('...' if len(excluded) > 20 else '')
    print(f"Excluding {len(excluded)} gloss(es) with fewer than {MIN_SEQUENCES_PER_CLASS} sequences: {preview}")

glosses = sorted(g for g, seqs in sequences_by_gloss.items() if len(seqs) >= MIN_SEQUENCES_PER_CLASS)
if not glosses:
    raise RuntimeError(f"No gloss has at least {MIN_SEQUENCES_PER_CLASS} complete sequences -- record/import more data first")

label_map = {label: num for num, label in enumerate(glosses)}

sparse = sorted(g for g in glosses if MIN_SEQUENCES_PER_CLASS <= len(sequences_by_gloss[g]) < 10)
if sparse:
    preview = ', '.join(sparse[:20]) + ('...' if len(sparse) > 20 else '')
    print(f"Warning: fewer than 10 sequences for {len(sparse)} gloss(es): {preview}")

print(f"Training vocabulary: {len(glosses)} glosses")

sequences, labels = [], []
for gloss in glosses:
    for seq in sequences_by_gloss[gloss]:
        sequences.append(seq)
        labels.append(label_map[gloss])

# Preparing the train/val/test datasets. Test is split off first and touched
# only once, later, by gloss_evaluate.py -- val drives early stopping here.
X = np.array(sequences)
y_labels = np.array(labels)
y = to_categorical(y_labels, num_classes=len(glosses)).astype(int)

X_temp, X_test, y_temp, y_test, labels_temp, _ = train_test_split(
    X, y, y_labels, test_size=GLOSS_TEST_SPLIT, stratify=y_labels, random_state=RANDOM_STATE
)
val_fraction = GLOSS_VAL_SPLIT / (1 - GLOSS_TEST_SPLIT)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_fraction, stratify=labels_temp, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

os.makedirs(GLOSS_MODELS_DIR, exist_ok=True)
np.savez(os.path.join(GLOSS_MODELS_DIR, 'test_split.npz'), X_test=X_test, y_test=y_test)

# Persist the exact vocabulary/label ordering used for training. gloss_evaluate.py
# and app_gloss.py load this rather than re-discovering from GlossData/, since
# the directory contents could change (more recordings/imports) between
# training and evaluation/inference and silently shift the label mapping.
label_map_path = os.path.join(GLOSS_MODELS_DIR, 'label_map.json')
with open(label_map_path, 'w') as f:
    json.dump(glosses, f)
print(f"Saved label map ({len(glosses)} glosses) to {label_map_path}")

# Defining and compiling the model (same architecture as the letter model,
# just a different input/output shape -- model_builder.build_model is fully
# generic)
model = build_model((GLOSS_SEQUENCE_LENGTH, X.shape[2]), len(glosses))
model.compile(optimizer=Adam(learning_rate=GLOSS_LEARNING_RATE), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

checkpoint_path = os.path.join(GLOSS_MODELS_DIR, 'checkpoint.weights.h5')
early_stopping = EarlyStopping(monitor='val_loss', patience=GLOSS_EARLY_STOPPING_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

# Training the model
history = model.fit(
    X_train, y_train,
    epochs=GLOSS_MAX_EPOCHS,
    batch_size=GLOSS_BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint],
)

# Belt-and-suspenders: explicitly reload the best checkpoint rather than
# relying solely on EarlyStopping's restore_best_weights semantics
model.load_weights(checkpoint_path)

# Saving the model
model_json = model.to_json()
model_json_path = os.path.join(GLOSS_MODELS_DIR, 'model_gloss.json')
model_h5_path = os.path.join(GLOSS_MODELS_DIR, 'model_gloss.h5')

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
history_path = os.path.join(GLOSS_MODELS_DIR, 'training_history.png')
fig.savefig(history_path)
print(f"Saved training history plot to {history_path}")
print(f"Run gloss_evaluate.py next before trusting these results -- training accuracy alone is not sufficient evidence with {len(glosses)} classes.")
