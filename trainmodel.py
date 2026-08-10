# Importing necessary libraries, program files and modules
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

import numpy as np
import os

# Mapping actions to numerical values (labels)
label_map = {label: num for num, label in enumerate(actions)}

# Loading and padding the sequences
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            
            if os.path.exists(npy_path):
                res = np.load(npy_path, allow_pickle=True)
                window.append(res)
            else:
                window.append(np.zeros(21*3))
                
        sequences.append(window)
        labels.append(label_map[action])
        

# Preparing the training and validation datasets
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

# Saving the logs for tensorboard
logs_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=logs_dir)

# Defining a model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 21*3)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=True, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'), 
    Dense(len(actions), activation='softmax')
])

# Compiling the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.summary()

# Saving the model
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Saved model successfully")


