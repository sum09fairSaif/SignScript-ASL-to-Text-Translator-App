# Importing necessary libraries, program files and modules
from function import *
import argparse
import cv2
import time


def next_free_sequence_index(action):
    """Scan existing sequence subfolders so re-running this script appends
    new sequences instead of overwriting earlier recording sessions."""
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_dir):
        return 0
    existing = [int(name) for name in os.listdir(action_dir) if name.isdigit()]
    return max(existing) + 1 if existing else 0


def collect(target_actions, sequences_per_action):
    cap = cv2.VideoCapture(0)
    stopped = False

    with create_hand_landmarker(
        vision.RunningMode.VIDEO,
        HAND_MIN_DETECTION_CONFIDENCE,
        HAND_MIN_TRACKING_CONFIDENCE) as hands, create_pose_landmarker(
        vision.RunningMode.VIDEO,
        POSE_MIN_DETECTION_CONFIDENCE,
        POSE_MIN_TRACKING_CONFIDENCE) as pose:

        for action in target_actions:
            if stopped:
                break

            start_index = next_free_sequence_index(action)

            for offset in range(sequences_per_action):
                if stopped:
                    break

                sequence = start_index + offset
                sequence_dir = os.path.join(DATA_PATH, action, str(sequence))
                os.makedirs(sequence_dir, exist_ok=True)

                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print('Warning: failed to read from webcam')
                        continue

                    # Full, uncropped frame -- both landmarkers do their own
                    # localization, and a shared coordinate frame lets the
                    # model learn hand-relative-to-body spatial relationships
                    timestamp_ms = int(time.time() * 1000)
                    image, hand_results = mediapipe_detection(frame, hands, timestamp_ms=timestamp_ms)
                    _, pose_results = mediapipe_detection(frame, pose, timestamp_ms=timestamp_ms)
                    draw_styles_landmarks(image, hand_results, pose_results)

                    message = f'Collecting frames for {action} sequence {sequence}'
                    cv2.putText(image, message, (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    if frame_num == 0:
                        # Pause at the start of each sequence so you can reset your hand position
                        cv2.putText(image, 'STARTING COLLECTION', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.imshow('OpenCV Feed', image)

                    # Extracting and saving the keypoints
                    keypoints = extract_keypoints(hand_results, pose_results)

                    npy_path = os.path.join(sequence_dir, str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        stopped = True
                        break

    cap.release()
    cv2.destroyAllWindows()  # closing the OpenCV window after the data collection is complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record ASL fingerspelling training sequences from the webcam.')
    parser.add_argument('--actions', type=str, default=None,
                         help='Comma-separated letters to record, e.g. "J,Z" (default: all configured actions)')
    parser.add_argument('--sequences', type=int, default=no_sequences,
                         help=f'Number of sequences to record per action this session (default: {no_sequences})')
    args = parser.parse_args()

    target_actions = [a.strip().upper() for a in args.actions.split(',')] if args.actions else actions
    collect(target_actions, args.sequences)
