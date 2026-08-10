# Importing necessary libraries, program files and modules
from function import *
import cv2
import time

# Creating directories for each action and storing the frames corresponding to each action
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)

# Initializing mediapipe for hand detections
with create_hand_landmarker(
    vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # Cropping to the same region used for live inference
                cropframe = frame[40:400, 0:300]

                image, results = mediapipe_detection(cropframe.copy(), hands, timestamp_ms=int(time.time() * 1000))
                draw_styles_landmarks(image, results)

                message = f'Collecting frames for {action} Video {sequence}'
                cv2.putText(image, message, (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                if frame_num == 0:
                    # Pause at the start of each sequence so you can reset your hand position
                    cv2.putText(image, 'STARTING COLLECTION', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.imshow('OpenCV Feed', image)

                # Extracting and saving the keypoints
                keypoints = extract_keypoints(results)

                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()  # closing the OpenCV window after the data collection is complete
