# Basics
import cv2
import os
import shutil
from typing import Union, Dict

# Utilities
import numpy as np
from PIL import Image
from tqdm import tqdm

# Local
from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, processOneImage
from step01_annotate_image_mmpose.configs import mmpose_config as mcfg
from dvalue import yolo_input_size
from utils.opencv_utils import cropFrame


def video_name2properties(video_name: str) -> int:
    # TODO: I expanded this function's usages in getVideoProperties below.
    #   The hashing performance has no need to worry.
    #   The function will only be run once for each video.
    """
    Extract properties written in video's name.
    :param video_name: Video file name without extension.
    :return: Hand index. L:0, R:1, B:2.
    """
    properties = tuple(video_name.split("_"))
    hand = properties[-1][0]

    hand_idx = {"L": 0, "R": 1, "B": 2}[hand]

    return hand_idx


def getVideoProperties(video_name: str, item=None) -> Union[Dict, int, str]:
    """
    Extract properties from a video's name. Available properties:
    date, time, model name, position, green info, hand index, hand item, and hex id.
    The hand index and its corresponding chars are: L:0, R:1, B:2.
    :param video_name: Name of the video file.
    :param item: Which item of the properties to extract.
    If not specified, the entire properties object will be given.
    :return: Properties object or the required properties item.
    """
    if video_name.endswith(".mp4"):
        raise ValueError("File name should not contain extension.")

    # Form properties object.
    prop_dict_values = tuple(video_name.split("_"))
    prop_dict_keys = ["date", "time", "model name", "position", "color info", "hand info"]
    properties = {k: v for k, v in zip(prop_dict_keys, prop_dict_values)}

    # Expand hand information.
    properties["hand index"] = {"L": 0, "R": 1, "B": 2}[properties["hand info"][0]]
    properties["hand item"] = properties["hand info"][1:]
    del properties["hand info"]

    # "item" isn't specified => Directly give out all stuff.
    if item is None:
        return properties

    # "item" is specified.
    if item not in list(properties.keys()) + ["hex id"]:
        raise KeyError(f"Invalid item {item}. Can't extract item from properties.")
    elif item == "hex id":
        hex_id = f"{properties['date'][:8]}_{properties['time']}"
        return hex_id
    else:
        return properties[item]


def video2dataset(video_path: str,
                  dataset_save_dir: str,
                  mmpose_essentials,
                  step_size: int = 10) -> None:
    """
    Save effective frames as input data from a video with mmpose and crop-frame methods.
    :param video_path: Path to the video .mp4 file.
    :param dataset_save_dir: Path to the directory where the dataset folders are saved.
    :param mmpose_essentials: A tuple of bounding box detector and a pose estimator.
    :param step_size: Step size of frame sampling.
    :return: Nothing.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} does not exist!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot configure cv2. Video might be corrupted, "
                      "or you may not have read permission of this file.")

    # Set up datasets storage dir if necessary.
    os.makedirs(dataset_save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # No file ext.
    dataset_dir = os.path.join(dataset_save_dir, video_name)

    # Set up dir for this dataset.
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)  # Overwrite previous
    os.makedirs(dataset_dir, exist_ok=True)

    # Hand index.
    which_hand = getVideoProperties(video_name, "hand index")
    """
    Index within range [0, 1, 2]. Corresponding to char as ['L', 'R', 'B'].
    """
    if which_hand < 0 or which_hand > 2:
        raise ValueError(f"{which_hand} is not in ['L', 'R', 'B']")

    # mmpose
    bbox_detector, pose_estimator = mmpose_essentials

    # Extract frames.
    frame_count = 0     # Num. of frames the loop has covered.
    total_frame_amount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stored_frames = []  # Num of frames recorded.

    with tqdm(total=total_frame_amount, desc=f"Processing {video_path}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cv2.waitKey(5) & 0xFF == 27:
                break

            pbar.update()

            # Extract the first person.
            keypoints_list, xyxy_list, _ = processOneImage(frame, bbox_detector, pose_estimator)
            keypoints, xyxy = keypoints_list[0], xyxy_list[0]

            # Hand frame shape.
            bbox_w, bbox_h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
            hand_hw = (int(bbox_w * 0.7), int(bbox_w * 0.7))
            """Height and width (sequence matter) of the bounding box."""

            # Hand frame centers.
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
    video_hex_id = getVideoProperties(video_name, "hex id")
    for i, (frame, _) in tqdm(enumerate(stored_frames), desc=f"Saving {video_path}"):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized_image = image.resize((yolo_input_size, yolo_input_size))
        resized_image.save(os.path.join(str(dataset_dir), f"{video_hex_id}_fig{i}.jpg"))


def videos2datasets(videos_save_dir: str, dataset_save_dir: str, sample_step_size=10):
    """
    Batch converting videos in a directory into datasets.
    :param videos_save_dir: Directory where videos are stored.
    :param dataset_save_dir: Directory where dataset folder is stored.
    :param sample_step_size: Step size for frame extraction.
    :return: Nothing.
    """

    # Initialize MMPose essentials.
    bbox_detector, pose_estimator, _ = getMMPoseEssentials(
        det_config=mcfg.det_config_train,
        det_chkpt=mcfg.det_checkpoint_train,
        pose_config=mcfg.pose_config_train,
        pose_chkpt=mcfg.pose_checkpoint_train)

    # Extract all videos.
    used_videos = []
    for root, dirs, files in os.walk(videos_save_dir):
        if "used" in dirs:      # Skip used videos
            dirs.remove("used")

        for file in files:
            if not file.endswith('.mp4'):
                print(f"Ignoring file {file}: Not an acceptable video file.")
                continue
            file_path = os.path.join(root, file)
            print(f"\nReading file {file}: Formulating dataset.")
            video2dataset(video_path=file_path,
                          dataset_save_dir=dataset_save_dir,
                          mmpose_essentials=(bbox_detector, pose_estimator),
                          step_size=sample_step_size)

            print(f"Moving file {file} to {os.path.join(root, 'used')}.")
            shutil.move(file_path, os.path.join(root, "used"))


if __name__ == "__main__":
    step_size = input("Enter step size > ")
    step_size = 5 if not step_size.isdigit() else int(step_size)

    videos2datasets(videos_save_dir="../data/blob/yolo_videos/",
                    dataset_save_dir="../data/yolo_dataset",
                    sample_step_size=step_size)
