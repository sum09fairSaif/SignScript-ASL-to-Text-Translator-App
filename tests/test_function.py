import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from function import extract_keypoints, POSE_LANDMARK_INDICES


class MockLandmark:
    def __init__(self, x=0.1, y=0.2, z=0.3, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockHandResults:
    def __init__(self, hand_landmarks=None):
        self.hand_landmarks = hand_landmarks or []


class MockPoseResults:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = pose_landmarks or []


def make_hand_results(detected=True):
    if not detected:
        return MockHandResults([])
    return MockHandResults([[MockLandmark() for _ in range(21)]])


def make_pose_results(detected=True):
    if not detected:
        return MockPoseResults([])
    # 33 landmarks total, matching MediaPipe Pose's full topology --
    # extract_keypoints only reads the indices in POSE_LANDMARK_INDICES
    return MockPoseResults([[MockLandmark() for _ in range(33)]])


def test_extract_keypoints_shape_both_present():
    kp = extract_keypoints(make_hand_results(True), make_pose_results(True))
    assert kp.shape == (91,)


def test_extract_keypoints_shape_both_missing():
    kp = extract_keypoints(make_hand_results(False), make_pose_results(False))
    assert kp.shape == (91,)
    assert not kp.any()


def test_extract_keypoints_hand_missing_zero_fills_independently():
    kp = extract_keypoints(make_hand_results(False), make_pose_results(True))
    assert kp.shape == (91,)
    hand_block, pose_block = kp[:63], kp[63:]
    assert not hand_block.any()
    assert pose_block.any()


def test_extract_keypoints_pose_missing_zero_fills_independently():
    kp = extract_keypoints(make_hand_results(True), make_pose_results(False))
    assert kp.shape == (91,)
    hand_block, pose_block = kp[:63], kp[63:]
    assert hand_block.any()
    assert not pose_block.any()


def test_pose_landmark_indices_length_matches_feature_vector():
    # 7 pose points x 4 values (x, y, z, visibility) = 28; + 63 hand values = 91
    assert len(POSE_LANDMARK_INDICES) == 7
    assert len(POSE_LANDMARK_INDICES) * 4 + 21 * 3 == 91


def test_config_actions_is_full_alphabet_no_duplicates():
    assert len(config.actions) == 26
    assert len(set(config.actions)) == 26
    assert all(len(letter) == 1 and letter.isupper() for letter in config.actions)
