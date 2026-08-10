# Importing the necessary libraries, modules, packages, and functions/methods
from function import *

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

from keras.models import model_from_json
import time
import traceback

# Loading the trained model
MODEL_JSON_PATH = os.path.join(MODELS_DIR, 'model_v2_26letter.json')
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'model_v2_26letter.h5')

with open(MODEL_JSON_PATH, 'r') as json_file:
    model = model_from_json(json_file.read())
model.load_weights(MODEL_WEIGHTS_PATH)

# Setting colors for different actions
colors = [(245, 117, 16) for _ in actions]

# Creating a function to visualize the probabilities of different actions
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


# Detecting and displaying the variables
sequence = []
sentence = []
accuracy = []
predictions = []

cap = cv2.VideoCapture(0)

# Initializing mediapipe for hand and pose tracking
with create_hand_landmarker(
    vision.RunningMode.VIDEO,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE) as hands, create_pose_landmarker(
    vision.RunningMode.VIDEO,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE) as pose:

    # Looping through every frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Warning: failed to read from webcam')
            continue

        # Full, uncropped frame -- must match data.py's capture pipeline
        # exactly, since any divergence here is silent train/serve skew
        timestamp_ms = int(time.time() * 1000)
        image, hand_results = mediapipe_detection(frame, hands, timestamp_ms=timestamp_ms)
        _, pose_results = mediapipe_detection(frame, pose, timestamp_ms=timestamp_ms)

        keypoints = extract_keypoints(hand_results, pose_results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        res = None
        try:
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        except Exception:
            if DEBUG:
                traceback.print_exc()
            res = None

        if res is not None:
            predictions.append(np.argmax(res))

            if DEBUG:
                print(f"predicted={actions[np.argmax(res)]} confidence={res[np.argmax(res)]:.3f} "
                      f"all={dict(zip(actions, res.round(3)))}")

            if np.unique(predictions[-PREDICTION_AGREEMENT_WINDOW:])[0] == np.argmax(res):
                if res[np.argmax(res)] > PREDICTION_CONFIDENCE_THRESHOLD:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        accuracy.append(str(res[np.argmax(res)] * 100))

            if len(sentence) > 1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]

        draw_styles_landmarks(frame, hand_results, pose_results)
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, 'Output: ' + ' '.join(sentence) + ''.join(accuracy), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
