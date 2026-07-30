# Importing all required Python packages, libraries and frameworks
import cv2
import numpy as np
import os
import mediapipe as mp

# Initializing the mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hand

# Performing mediapipe detection for images
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

# Drawing the landmarks and the hand connections
def draw_styles_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


# Extracting the keypoints from the detected landmarks
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
        return rh
    
    return np.zeros(21*3)

# Defining the paths and parameters for data detection
DATA_PATH = os.path.join('MP_Data')
actions = ['A', 'B', 'C']
no_sequences = 30
sequence_length = 30