# Live webcam recording for word-level ASL glosses (two-handed), parallel to
# data.py's fingerspelling (one-handed) recording script.
from function import *
from gloss_config import *
import argparse
import cv2
import time


def next_free_sequence_index(gloss):
    """Scan existing sequence subfolders so re-running this script appends
    new sequences instead of overwriting earlier recording sessions."""
    gloss_dir = os.path.join(GLOSS_DATA_PATH, gloss)
    if not os.path.isdir(gloss_dir):
        return 0
    existing = [int(name) for name in os.listdir(gloss_dir) if name.isdigit()]
    return max(existing) + 1 if existing else 0


def collect(target_glosses, sequences_per_gloss):
    cap = cv2.VideoCapture(0)
    stopped = False

    with create_hand_landmarker(
        vision.RunningMode.VIDEO,
        HAND_MIN_DETECTION_CONFIDENCE,
        HAND_MIN_TRACKING_CONFIDENCE,
        num_hands=GLOSS_NUM_HANDS) as hands, create_pose_landmarker(
        vision.RunningMode.VIDEO,
        POSE_MIN_DETECTION_CONFIDENCE,
        POSE_MIN_TRACKING_CONFIDENCE) as pose:

        for gloss in target_glosses:
            if stopped:
                break

            start_index = next_free_sequence_index(gloss)

            for offset in range(sequences_per_gloss):
                if stopped:
                    break

                sequence = start_index + offset
                sequence_dir = os.path.join(GLOSS_DATA_PATH, gloss, str(sequence))
                os.makedirs(sequence_dir, exist_ok=True)

                for frame_num in range(GLOSS_SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        print('Warning: failed to read from webcam')
                        continue

                    # Full, uncropped frame -- must match gloss_import_external.py's
                    # and app_gloss.py's capture exactly (same reasoning as the
                    # letter pipeline's data.py/app.py lockstep requirement)
                    timestamp_ms = int(time.time() * 1000)
                    image, hand_results = mediapipe_detection(frame, hands, timestamp_ms=timestamp_ms)
                    _, pose_results = mediapipe_detection(frame, pose, timestamp_ms=timestamp_ms)
                    draw_styles_landmarks(image, hand_results, pose_results)

                    message = f'Collecting frames for {gloss} sequence {sequence}'
                    cv2.putText(image, message, (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    if frame_num == 0:
                        # Pause at the start of each sequence so you can reset your hand position
                        cv2.putText(image, 'STARTING COLLECTION', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.imshow('OpenCV Feed', image)

                    # Extracting and saving the keypoints (two-handed, 154-dim)
                    keypoints = extract_keypoints_two_hand(hand_results, pose_results)

                    npy_path = os.path.join(sequence_dir, str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        stopped = True
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record ASL word-sign (gloss) training sequences from the webcam.')
    parser.add_argument('--glosses', type=str, default=None,
                         help='Comma-separated glosses to record, e.g. "HELLO,THANK-YOU" (default: the core vocabulary)')
    parser.add_argument('--sequences', type=int, default=GLOSS_NO_SEQUENCES,
                         help=f'Number of sequences to record per gloss this session (default: {GLOSS_NO_SEQUENCES})')
    args = parser.parse_args()

    target_glosses = [g.strip().upper() for g in args.glosses.split(',')] if args.glosses else CORE_GLOSSES
    collect(target_glosses, args.sequences)
