import cv2
import os
import shutil

import numpy as np

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, processOneImage
from step01_annotate_image_mmpose.configs import mmpose_config as mcfg
from utils.opencv_utils import cropFrame
from PIL import Image

def video_name2properties(video_name:str) -> int:
    """
    extract properties written in video's name.
    :param video_name:
    :return:
    """
    properties = tuple(video_name.split("_"))
    hand = properties[-1][0]

    hand_idx = {"L": 0, "R": 1, "B": 2}[hand]

    return hand_idx

def video2dataset(video_path, dataset_save_dir, step_size=10) -> None:
    """
    save effective frames as input data from a video with mmpose and crop-frame methods.
    :param video_path:
    :param dataset_save_dir:
    :param step_size:
    :return:
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} does not exist!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot configure cv2. Video might be corrupted, "
                      "or you may not have read permission of this file.")

    os.makedirs(dataset_save_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    dataset_dir = os.path.join(dataset_save_dir, video_name)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True) # prepare the output folder

    which_hand = video_name2properties(video_name)
    """
    one single char, only can be 'L', 'R', 'B'. (left, right, both)
    """
    if which_hand not in ["L", "R", "B"]:
        raise ValueError(f"{which_hand} is not in ['L', 'R', 'B']")

    # mmpose
    bbox_detector, pose_estimator, _ = getMMPoseEssentials(
        det_config=mcfg.det_config_train,
        det_chkpt=mcfg.det_checkpoint_train,
        pose_config=mcfg.pose_config_train,
        pose_chkpt=mcfg.pose_checkpoint_train)

    frame_count = 0
    stored_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

        keypoints_list, xyxy_list, _ = processOneImage(frame, bbox_detector, pose_estimator)

        keypoints, xyxy = keypoints_list[0], xyxy_list[0]

        bbox_w, bbox_h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        hand_hw = (int(bbox_w * 0.7), int(bbox_w * 0.7))
        """Height and width (sequence matter) of the bounding box."""

        lhand_center, rhand_center = keypoints[9][:2], keypoints[10][:2]

        l_arm_vect, r_arm_vect = keypoints[9][:2] - keypoints[7][:2], keypoints[10][:2] - keypoints[8][:2]
        lhand_center += l_arm_vect * 0.8
        rhand_center += r_arm_vect * 0.8

        hand_frames_xyxy = [None, None, None]

        if np.linalg.norm(lhand_center - rhand_center) > 0.21 * bbox_w:
            hand_frames_xyxy[0] = cropFrame(frame, lhand_center, hand_hw)
            hand_frames_xyxy[1] = cropFrame(frame, rhand_center, hand_hw)
        else:
            hand_frames_xyxy[2] = cropFrame(frame, (lhand_center + rhand_center) // 2, hand_hw)

        target_frame = hand_frames_xyxy[which_hand]
        if target_frame is None:
            continue

        frame_count += 1
        if frame_count % step_size == 0:
            stored_frames.append(target_frame)

    # save
    cap.release()
    for i, (frame, _) in enumerate(stored_frames):
        image = Image.fromarray(frame)
        resized_image = image.resize((64, 64))
        resized_image.save(os.path.join(dataset_dir, f"_fig{i}.jpg"))

if __name__ == "__main__":
    video2dataset(video_path="./tmp_videos/tmp000.mp4",
                  dataset_save_dir="./datasets",
                  step_size=10)