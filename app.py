# Importing the necessary libraries, modules, packages, and functions/methods
from function import *

# tensorflow has no build for this Python version, so use the torch backend instead
os.environ.setdefault('KERAS_BACKEND', 'torch')

from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import time

# Loading the trained model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('model.h5')

# Setting colors for different actions
colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))
    
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
threshold = 0.8

cap = cv2.VideoCapture(0)

# Initializing mediapipe for hand tracking
with create_hand_landmarker(
    vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        
    # Looping through every frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Processing the frame and the region 
        cropframe = frame[40:400, 0:300]
        
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands, timestamp_ms=int(time.time() * 1000))
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))
                print(f"predicted={actions[np.argmax(res)]} confidence={res[np.argmax(res)]:.3f} all={dict(zip(actions, res.round(3)))}")
            
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
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
                
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, 'Output: ' + ' '.join(sentence) + ''.join(accuracy), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

        