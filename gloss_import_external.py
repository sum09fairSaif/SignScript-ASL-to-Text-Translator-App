# Imports word-sign videos from the ASL Citizen dataset into GlossData/,
# running our own two-handed landmark extraction so imported and
# self-recorded (gloss_data.py) sequences share an identical feature format.
#
# ASL Citizen: https://www.microsoft.com/en-us/download/details.aspx?id=105253
# Archive layout (verified against the real downloaded zip):
#   ASL_Citizen/videos/<numeric_id>-<GLOSS>.mp4
#   ASL_Citizen/splits/{train,val,test}.csv  (columns: Participant ID, Video file, Gloss, ASL-LEX Code)
from function import *
from gloss_config import ASL_CITIZEN_DIR, GLOSS_DATA_PATH, GLOSS_NUM_HANDS, GLOSS_SEQUENCE_LENGTH
from gloss_config import HAND_MIN_DETECTION_CONFIDENCE, HAND_MIN_TRACKING_CONFIDENCE
from gloss_config import POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE

import argparse
import csv
import io
import json
import tempfile
import time
import zipfile

import cv2

ARCHIVE_PATH = os.path.join(ASL_CITIZEN_DIR, 'ASL_Citizen.zip')
PROGRESS_PATH = os.path.join(ASL_CITIZEN_DIR, 'import_progress.json')

DEFAULT_VIDEOS_PER_GLOSS = 10


def load_manifest(zf):
    """Combine the three official split CSVs into one (gloss -> sorted video
    filename list) mapping. Sorted for a deterministic, reproducible
    selection when capping videos/gloss."""
    by_gloss = {}
    for split in ('train', 'val', 'test'):
        with zf.open(f'ASL_Citizen/splits/{split}.csv') as f:
            content = f.read().decode('utf-8')
        for row in csv.DictReader(io.StringIO(content)):
            by_gloss.setdefault(row['Gloss'], []).append(row['Video file'])

    for gloss in by_gloss:
        by_gloss[gloss] = sorted(set(by_gloss[gloss]))

    return by_gloss


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return set(json.load(f))
    return set()


def save_progress(done):
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(sorted(done), f)


def next_free_sequence_index(gloss):
    gloss_dir = os.path.join(GLOSS_DATA_PATH, gloss)
    if not os.path.isdir(gloss_dir):
        return 0
    existing = [int(name) for name in os.listdir(gloss_dir) if name.isdigit()]
    return max(existing) + 1 if existing else 0


def extract_video_frames(zf, video_path):
    with zf.open(video_path) as src:
        data = src.read()

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames


def sample_frames(frames, count):
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, count).astype(int)
    return [frames[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description='Import ASL Citizen word-sign videos into GlossData/.')
    parser.add_argument('--videos-per-gloss', type=int, default=DEFAULT_VIDEOS_PER_GLOSS,
                         help=f'Max videos to import per gloss (default: {DEFAULT_VIDEOS_PER_GLOSS})')
    parser.add_argument('--glosses', type=str, default=None,
                         help='Comma-separated glosses to import (default: all 2731)')
    parser.add_argument('--limit', type=int, default=None,
                         help='Stop after importing this many videos total (for testing)')
    args = parser.parse_args()

    if not os.path.exists(ARCHIVE_PATH):
        raise RuntimeError(f"Missing {ARCHIVE_PATH} -- download ASL Citizen first (see requirements.txt)")

    zf = zipfile.ZipFile(ARCHIVE_PATH)
    by_gloss = load_manifest(zf)
    print(f"Archive contains {len(by_gloss)} glosses, {sum(len(v) for v in by_gloss.values())} videos total")

    target_glosses = sorted(by_gloss.keys())
    if args.glosses:
        wanted = {g.strip().upper() for g in args.glosses.split(',')}
        target_glosses = [g for g in target_glosses if g in wanted]

    done = load_progress()
    imported_this_run = 0
    skipped_already_done = 0
    t_start = time.time()

    with create_hand_landmarker(
        vision.RunningMode.IMAGE,
        HAND_MIN_DETECTION_CONFIDENCE,
        HAND_MIN_TRACKING_CONFIDENCE,
        num_hands=GLOSS_NUM_HANDS) as hands, create_pose_landmarker(
        vision.RunningMode.IMAGE,
        POSE_MIN_DETECTION_CONFIDENCE,
        POSE_MIN_TRACKING_CONFIDENCE) as pose:

        for gloss in target_glosses:
            video_files = by_gloss[gloss][:args.videos_per_gloss]

            for video_file in video_files:
                video_path = f'ASL_Citizen/videos/{video_file}'

                if video_file in done:
                    skipped_already_done += 1
                    continue

                frames = extract_video_frames(zf, video_path)
                sampled = sample_frames(frames, GLOSS_SEQUENCE_LENGTH)

                if len(sampled) < GLOSS_SEQUENCE_LENGTH:
                    print(f"Warning: {video_file} has only {len(frames)} frame(s), skipping")
                    done.add(video_file)
                    continue

                sequence = next_free_sequence_index(gloss)
                sequence_dir = os.path.join(GLOSS_DATA_PATH, gloss, str(sequence))
                os.makedirs(sequence_dir, exist_ok=True)

                for frame_num, frame in enumerate(sampled):
                    _, hand_results = mediapipe_detection(frame, hands)
                    _, pose_results = mediapipe_detection(frame, pose)
                    keypoints = extract_keypoints_two_hand(hand_results, pose_results)
                    np.save(os.path.join(sequence_dir, str(frame_num)), keypoints)

                done.add(video_file)
                imported_this_run += 1

                if imported_this_run % 20 == 0:
                    save_progress(done)
                    elapsed = time.time() - t_start
                    rate = imported_this_run / elapsed
                    print(f"Imported {imported_this_run} video(s) this run "
                          f"({skipped_already_done} already done, resumed) -- "
                          f"{rate:.3f} videos/s, {elapsed/60:.1f} min elapsed")

                if args.limit and imported_this_run >= args.limit:
                    save_progress(done)
                    print(f"Reached --limit {args.limit}, stopping")
                    return

    save_progress(done)
    print(f"Done. Imported {imported_this_run} video(s) this run, "
          f"{skipped_already_done} already done from a prior run.")


if __name__ == '__main__':
    main()
