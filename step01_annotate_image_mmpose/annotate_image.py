# Built-in
import copy
import itertools
import os
import time
from typing import List, Union, Tuple, Dict, Any

# Packages
import cv2
import mmpose.structures
import numpy as np
from mmdet.models import RTMDet
from mmpose.models import TopdownPoseEstimator
from mmpose.visualization import PoseLocalVisualizer
from numpy import ndarray
from tqdm import tqdm

# MMPose
import mmcv
from mmpose.apis import (
    inference_topdown,
    init_model as init_pose_estimator)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Local
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.calculations import calc_keypoint_angle
from utils.parse_file_name import parseFileName

register_all_modules()


def getMMPoseEssentials(det_config: str, det_chkpt: str,
                        pose_config: str, pose_chkpt: str) -> Tuple[RTMDet, TopdownPoseEstimator, PoseLocalVisualizer]:
    """
    Get essential detectors and visualizers of MMPose.getMMPose.

    - bbox_detector: Detector that gives xyxy of the boundary boxes.

    - pose_estimator: Estimator that gives the landmarks of a person.

    - visualizer: MMPose's built-in visualizer to visualize the bbox and pose skeleton.

    :return: bbox_detector, pose_estimator, visualizer

    """

    # 1. Boundary Box detector
    bbox_detector = init_detector(det_config, det_chkpt, device=mcfg.device)
    bbox_detector.cfg = adapt_mmdet_pipeline(bbox_detector.cfg)

    # 2. Pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_chkpt,
        device=mcfg.device,
        cfg_options=dict(
            model=dict(
                test_cfg=dict(output_heatmaps=mcfg.draw_heatmap)))
    )

    # 3. Visualizer
    pose_estimator.cfg.visualizer.radius = mcfg.radius
    pose_estimator.cfg.visualizer.alpha = mcfg.alpha
    pose_estimator.cfg.visualizer.line_width = mcfg.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=mcfg.skeleton_style)

    return bbox_detector, pose_estimator, visualizer


def processVideosInDir(video_dir: str,
                       bbox_detector_model: RTMDet,
                       pose_estimator_model: TopdownPoseEstimator,
                       detection_target_list: Union[List[List[Union[Tuple[str, str], str]]], ndarray],
                       skip_interval: int = 10,
                       mode: str = None) -> List[Dict[str, Union[str, np.ndarray]]]:
    """
    Convert all the videos gathered from a given directory.
    "kas" = key-angle-score
    :param video_dir: Directory to the video.
    :param bbox_detector_model: MMPose boundary box detector model.
    :param pose_estimator_model: MMPose pose estimation model.
    :param detection_target_list: List of detection targets.
    :param skip_interval: Interval between sampled frames.
    :param mode: A single result to be an array or a cube.
    :return:
    """
    named_feature_matrices = []

    for file_name in os.listdir(video_dir):

        # Settle file sources
        if not file_name.endswith('.mp4'):  # Skip all non-videos
            continue
        video_path = os.path.join(video_dir, file_name)
        file_name_with_extension = os.path.basename(video_path)
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

        # Converting Process

        feature_matrix = processOneVideo(video_path,
                                         bbox_detector_model,
                                         pose_estimator_model,
                                         detection_target_list,
                                         skip_interval,
                                         mode)
        named_feature_matrices.append({"name": file_name_without_extension,
                                       "feature_matrix": feature_matrix})

    return named_feature_matrices


def processOneVideo(video_path: str,
                    bbox_detector_model: RTMDet,
                    pose_estimator_model: TopdownPoseEstimator,
                    detection_targets: Union[List[List[Union[Tuple[str, str], str]]], ndarray],
                    skip_interval: int = 10,
                    mode: str = None) -> ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot find {video_path}")
        raise FileNotFoundError

    cur_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    key_angels_scores = []
    with tqdm(total=total_frames, desc=f"Processing {video_path}") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                pbar.update(pbar.total - pbar.n)
                break

            cur_frame += 1
            if cur_frame % skip_interval != 0:
                continue
            pbar.update(skip_interval)

            landmarks, _, data_samples = processOneImage(frame, bbox_detector_model, pose_estimator_model)
            one_person = translateOneLandmarks(detection_targets, landmarks[0], mode)

            key_angels_scores.append(one_person)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    key_angels_scores = np.array(key_angels_scores)

    if mode == 'mjj':
        key_angels_scores = np.transpose(key_angels_scores, (0, 1, 4, 2, 3))
        # transpose from shape (n_frame, input_channels, height, width, depth)

    # when hyz -> ndarray: (n, 1716) | (n_frame, length)
    # when mjj -> ndarray: (n, 2, 7, 12, 11) | (n_frame, input_channels, depth, height, width)
    return key_angels_scores


def processOneImage(img: Union[str, np.ndarray],
                    bbox_detector_model,
                    pose_estimator_model,
                    bbox_threshold=mcfg.bbox_thr_single) -> Tuple[ndarray, ndarray, PoseDataSample]:
    """
    This is the 1st information layer. This function gets the raw data from the image.

    data_samples --> pred_instances --> [keypoints, keypoint_scores, bboxes]

    :param img: The image.
    :param bbox_detector_model: The model to retrieve boundary boxes.
    :param pose_estimator_model: The model to estimate a person's pose, i.e., retrieve key points.
    :param bbox_threshold: The threshold of IoU where the boundary boxes will be recorded into the list.

    Returns:

    - keypoints_list: 3-layered list. First layer: Num of people. Second layer: Num of keypoints. Third Layer: Num of
    values in a keypoint, i.e., x, y and score.

    - xyxy_list: 2-layered list. First layer: Num of people. Second layer: xmin, xmax, ymin, ymax of one bbox.

    - data_samples: Raw data samples from mmpose for visualization.
    """

    # List of boundary boxes xyxy coordinates.
    # Only keep bboxes above the bbox threshold.
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == mcfg.det_cat_id, pred_instance.scores > bbox_threshold)]
    bboxes = bboxes[nms(bboxes, mcfg.nms_thr), :4]

    # A list of keypoint sets.
    # A keypoint set is a list of keypoints.
    pose_results = inference_topdown(pose_estimator_model, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    raw_predictions = data_samples.get('pred_instances', None)

    # Keypoint coordinates --> (num_people, 17, 3)
    _keypoints_coords = raw_predictions.keypoints  # (num_people, 17, 2)
    _keypoints_scores = np.expand_dims(raw_predictions.keypoint_scores, axis=-1)  # (num_people, 17, 1)
    keypoints_list = np.concatenate((_keypoints_coords, _keypoints_scores), axis=-1)  # (num_people, 17, 3)

    # Boundary Boxes coordinates --> (num_people, 4)
    xyxy_list = raw_predictions.bboxes

    """Example formats of returned objects.
       1. keypoints_list: keypoints coordinates of landmarks and boundary.
           - Shape: (num_people, 17, 3) => (num_people, num_targets, (x,y,score))

           - Format:
           [
             [                             # First Person
                 [x_0, y_0, score_0],
                 [x_1, y_1, score_1],
                 ...
                 [x_16, y_16, score_16]
             ],
             [                             # Second Person
                 [x_0, y_0, score_0],
                 [x_1, y_1, score_1],
                 ...
                 [x_16, y_16, score_16]
             ],
             ...                           # ... and so on
           ]

       2. xyxy_list: Coordinates of boundary boxes, i.e. individual person.
           - Shape: (num_people, 4) => (xmin, ymin, xmax, ymax)
           - Format:
           [
               [xmin, ymin, xmax, ymax],       # First Person
               [xmin, ymin, xmax, ymax],       # Second Person
               [xmin, ymin, xmax, ymax],       # Third Person
               ...                             # ... and so on
           ]
       """

    return keypoints_list, xyxy_list, data_samples


def translateOneLandmarks(targets: List,
                          landmarks: ndarray,
                          mode: str = None) -> List[Union[float, Any]]:
    """
    This is the 2nd information layer. This function translates the raw landmarks data from the first information layer
    into a composited set of data in the form of an array or a cube.

    From the keypoints list, extract the most confident person in the image. Then, convert the
    keypoints list of this person into a flattened feature vector.

    :param mode: What type of data. "hyz" -> angles and scores in lines, "mjj" -> angles and scores in cubes.
    :param landmarks: A list of keypoints set of multiple people, gathered from the image.
    :param targets: The list/cube of detection targets.
    :return: A flattened array or cube of feature values.
    """
    if mode == 'hyz':
        # Only get the person with the highest detection confidence.
        kas_one_person = []

        # From keypoints list, get angle-score vector.
        for target in targets:
            angle_value, angle_score = calc_keypoint_angle(landmarks, kcfg.keypoint_indexes, target[0], target[1])
            kas_one_person.append(angle_value)
            kas_one_person.append(angle_score)

        # Shape=(2m)
        return kas_one_person

    elif mode == 'mjj':
        angles = copy.deepcopy(targets)
        scores = copy.deepcopy(targets)
        for i, j, k in itertools.product(range(len(angles)), range(len(angles[0])), range(len(angles[0][0]))):
            angles[i][j][k], scores[i][j][k] = calc_keypoint_angle(
                landmarks,
                kcfg.keypoint_indexes,
                targets[i][j][k][0],
                targets[i][j][k][1],
            )
        return [angles, scores]


# def saveFeatureMatToNPY(mat: np.ndarray, _save_path: str) -> None:
#     """
#     Save the feature matrix into a npy file, under the given path.
#     :param mat:
#     :param _save_path:
#     :return:
#     """
#     # Shape: (num_people, num_features)
#     feature_matrix = np.array(mat)
#     np.save(_save_path, feature_matrix)


def renderTheResults(img: Union[str, np.ndarray],
                     data_samples,
                     estim_results_visualizer=None,
                     show_interval=0.001):
    """
    Render the results of inferences into the mmcv canvas.
    :param img: The path of the image, or a video frame.
    :param data_samples: The inference result from that current image.
    :param estim_results_visualizer: The mmpose result visualizer object.
    :param show_interval: Display interval.
    :return:
    """
    # Render the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if estim_results_visualizer is not None:
        estim_results_visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=mcfg.draw_heatmap,
            draw_bbox=mcfg.draw_bbox,
            show_kpt_idx=mcfg.show_kpt_idx,
            skeleton_style=mcfg.skeleton_style,
            show=mcfg.show,
            wait_time=show_interval,
            kpt_thr=mcfg.kpt_thr)


if __name__ == "__main__":
    solution_mode = 'hyz'
    # solution_mode = 'mjj'
    video_folder = "../data/blob/videos"

    # Initialize MMPose essentials
    detector, pose_estimator, visualizer = getMMPoseEssentials(
        det_config=mcfg.det_config_train,
        det_chkpt=mcfg.det_checkpoint_train,
        pose_config=mcfg.pose_config_train,
        pose_chkpt=mcfg.pose_checkpoint_train
    )

    # Save the feature matrices.
    named_feature_mats = processVideosInDir(video_dir=video_folder,
                                            bbox_detector_model=detector,
                                            pose_estimator_model=pose_estimator,
                                            detection_target_list=kcfg.get_targets(solution_mode),
                                            skip_interval=10,
                                            mode=solution_mode)

    for name_mat in named_feature_mats:
        save_path = "../data/train/" + name_mat['name'] + ".npy"
        matrix = name_mat['feature_matrix']
        np.save(save_path, matrix)
