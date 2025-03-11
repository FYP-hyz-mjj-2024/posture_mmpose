# Basic
import os
import copy
import itertools
from typing import List, Union, Tuple, Dict, Any

# Utilities
import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm

# MMPose
from mmdet.models import RTMDet
from mmpose.models import TopdownPoseEstimator
from mmpose.visualization import PoseLocalVisualizer
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


def getMMPoseEssentials(det_config: str = mcfg.det_config,
                        det_chkpt: str = mcfg.det_checkpoint,
                        pose_config: str = mcfg.pose_config,
                        pose_chkpt: str = mcfg.pose_checkpoint) -> Tuple[RTMDet, TopdownPoseEstimator, PoseLocalVisualizer]:
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
                       skip_interval: int = 10) -> List[Dict[str, Union[str, np.ndarray]]]:
    """
    Convert all the .mp4 video files into lists of 2-channel 3-d features.

    "kas" = key-angle-score

    :param video_dir: Directory where the videos are stored.
    :param bbox_detector_model: MMPose boundary box detector model.
    :param pose_estimator_model: MMPose pose estimation model.
    :param detection_target_list: List of detection targets.
    :param skip_interval: Interval between sampled frames.
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
        file_props = parseFileName(file_name_without_extension)
        if 'frame_number' in file_props:
            skip_interval = file_props['frame_number']

        feature_matrix = processOneVideo(video_path,
                                         bbox_detector_model,
                                         pose_estimator_model,
                                         detection_target_list,
                                         skip_interval) # (n, 2, 7, 12, 11) | (n_frame, n_channels, depth, height, width)
        named_feature_matrices.append({"name": file_name_without_extension,
                                       "feature_matrix": feature_matrix})

    return named_feature_matrices


def processOneVideo(video_path: str,
                    bbox_detector_model: RTMDet,
                    pose_estimator_model: TopdownPoseEstimator,
                    detection_targets: Union[List[List[Union[Tuple[str, str], str]]], ndarray],
                    skip_interval: int = 10) -> ndarray:
    """
    Use RTMPose pose estimation to convert a video into a list of features.
    A feature is defaulted to a 2-channel 3-d structure (4d). This means that
    the output of this function would be 5-dimensional.

    :param video_path: Path to a specific .mp4 video file.
    :param bbox_detector_model: MMPose boundary box detector model.
    :param pose_estimator_model: MMPose pose estimation model.
    :param detection_targets: List of detection targets.
    :param skip_interval: Skip interval of frame sampling.
    :return: Default to a list 2-channel 3-d structures. (n_frame, input_channels=2, depth=7, height=12, width=11)
    """
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

            # Here, each person will be translated into 2-channel 3-d structure.
            one_person = translateOneLandmarks(detection_targets, landmarks[0])

            key_angels_scores.append(one_person)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    key_angels_scores = np.array(key_angels_scores)

    # transpose from shape (n_frame, n_channels, height, width, depth)
    key_angels_scores = np.transpose(key_angels_scores, (0, 1, 4, 2, 3))

    # ndarray: (n, 2, 7, 12, 11) | (n_frame, n_channels, depth, height, width)
    return key_angels_scores


def processOneImage(img: Union[str, np.ndarray],
                    bbox_detector_model,
                    pose_estimator_model,
                    bbox_threshold=mcfg.bbox_thr_single) -> Tuple[ndarray, ndarray, PoseDataSample]:
    """
    This is the 1st information layer. This function gets the raw data from the image.

    data_samples --> pred_instances --> [keypoints, keypoint_scores, bboxes]

    :param img: The image.
    :param bbox_detector_model: MMPose boundary box detector model.
    :param pose_estimator_model: MMPose pose estimation model, the model to estimate a person's pose,
                                 i.e., retrieve key points.
    :param bbox_threshold: The threshold of IoU where the boundary boxes will be recorded into the list.

    :returns:

    - keypoints_list: 3-layered list. (num_people, num_keypoints, num_values=3), values: [x, y, score]

    - xyxy_list: 2-layered list. (num_people, xyxy), xyxy: [xmin, xmax, ymin,ymax] of bbox.

    - data_samples: Raw data samples from mmpose for visualization.
    """

    # List of boundary boxes xyxy coordinates.
    # Only keep bboxes above the bbox threshold.
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1) # [n, 4 + 1], xyxy + score
    bboxes = bboxes[np.logical_and(pred_instance.labels == mcfg.det_cat_id, pred_instance.scores > bbox_threshold)] # TODO: simplify this operation
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
                          landmarks: ndarray) -> List[Union[float, Any]]:
    """
    This is the 2nd information layer.
    This function translates the raw landmarks data from the first information layer
    into a composited set of data in the form of a 4-d structure.

    The cube is pre-structured with the "targets" parameter and is filled with actual
    value with assistance of the "landmarks" parameter. For each angle in "targets",
    there are two kinds of values to be calculated: Angle Value and Angle Score.
    These two set of values will be respectively "filled" into two structures suggested
    by "targets", forming a 2-channel structure, where each channel is a 3-d cube in
    the structure of "targets".

    :param targets: Targets of detection angles, i.e., the "structure" of one channel.
    :param landmarks: A list of keypoints [x, y, conf] gathered with RTMPose.
    :return: A 2-channel 3-d structure. Two channels: Angle Value, Angle Score.
    """
    # Fold the angles into the targeted 2-channel 3-d struct.
    # angles = copy.deepcopy(targets)
    # scores = copy.deepcopy(targets)
    # WARNING: don't use copy.deepcopy, it is SLOW!!
    angles = [[[None for _ in sublist] for sublist in inner_list] for inner_list in targets]
    scores = [[[None for _ in sublist] for sublist in inner_list] for inner_list in targets]
    for i, j, k in itertools.product(range(len(angles)), range(len(angles[0])), range(len(angles[0][0]))):
        angles[i][j][k], scores[i][j][k] = calc_keypoint_angle(
            landmarks,
            kcfg.keypoint_indexes,
            targets[i][j][k][0],
            targets[i][j][k][1],
        )
    return [angles, scores]


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
    video_folder = "../data/blob/videos"

    # Initialize MMPose essentials
    detector, pose_estimator, visualizer = getMMPoseEssentials(det_config=mcfg.det_config_train,
                                                               det_chkpt=mcfg.det_checkpoint_train,
                                                               pose_config=mcfg.pose_config_train,
                                                               pose_chkpt=mcfg.pose_checkpoint_train)

    # Save the feature matrices.
    named_feature_mats = processVideosInDir(video_dir=video_folder,
                                            bbox_detector_model=detector,
                                            pose_estimator_model=pose_estimator,
                                            detection_target_list=kcfg.get_targets(),
                                            skip_interval=10)

    save_rt = "../data/train/3dnpy/"
    for name_mat in named_feature_mats:
        save_path = save_rt + name_mat['name'] + ".npy"
        matrix = name_mat['feature_matrix']
        np.save(save_path, matrix)
