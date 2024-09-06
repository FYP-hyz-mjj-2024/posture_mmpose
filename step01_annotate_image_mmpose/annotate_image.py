import numpy as np
import math
import cv2
import json_tricks as json
import mmcv
import config as cfg
import mmengine
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from mmpose.apis import (
    inference_topdown,
    init_model as init_pose_estimator)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Local
from keypoint_info import keypoint_indexes, keypoint_names

register_all_modules()


def process_one_image(img,
                      bbox_detector_model,
                      pose_estimator_model,
                      estim_results_visualizer=None,
                      show_interval=0):
    """
    Given an image, first use bbox detection model to retrieve object boundary boxes.
    Then, feed the sub images defined by the bbox into the pose estimation model to get key points.
    Lastly, visualize predicted key points (and heatmaps) of one image.
    :param img: The image.
    :param bbox_detector_model: The model to retrieve boundary boxes.
    :param pose_estimator_model: The model to estimate a person's pose, i.e., retrieve key points.
    :param estim_results_visualizer: The results visualizer.
    :return: Raw Results of prediction.
    """

    if not isinstance(img, str) and not isinstance(img, np.ndarray):
        raise ValueError(f"Image should either be the path to the image file or the nd.aray instance."
                         f"Current image type is {type(img)}")

    # Get boundary boxes
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == cfg.det_cat_id,
                                   pred_instance.scores > cfg.bbox_thr_single)]    # Single box detection
    bboxes = bboxes[nms(bboxes, cfg.nms_thr), :4]

    # Get key points
    pose_results = inference_topdown(pose_estimator_model, img, bboxes)
    data_samples = merge_data_samples(pose_results)

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
            draw_heatmap=cfg.draw_heatmap,
            draw_bbox=cfg.draw_bbox,
            show_kpt_idx=cfg.show_kpt_idx,
            skeleton_style=cfg.skeleton_style,
            show=cfg.show,
            wait_time=show_interval,
            kpt_thr=cfg.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def get_keypoints_and_xyxy(raw_predictions):
    """
    From the raw prediction data of process_one_image, retrieve key coordinates of landmarks and boundary box frames.
    :param raw_predictions: The prediction data from process_one_image.
    :returns: keypoints_list (n, 91, 3); xyxy_list: boundary box coordinates (n, 4).
    """

    """Example formats of returned objects.
    1. keypoints_list: keypoints coordinates of landmarks and boundary.
        - Shape: (num_people, 91, 3) => (num_people, num_targets, (x,y,score))
    
        - Format:
        [
          [                             # First Person
              [x_0, y_0, score_0],
              [x_1, y_1, score_1],
              ...
              [x_90, y_n, score_90]
          ],
          [                             # Second Person
              [x_0, y_0, score_0],
              [x_1, y_1, score_1],
              ...
              [x_90, y_n, score_90]
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

    # Keypoint coordinates --> (num_people, 91, 3)
    _keypoints_coords = raw_predictions.keypoints   # Shape: (num_people, 133, 2)
    _keypoints_scores = np.expand_dims(             # Shape: (num_people, 133, 1)
        raw_predictions.keypoint_scores,
        axis=-1)
    _keypoints_scores_coords = np.concatenate(      # Shape: (num_people, 133, 3)
        (_keypoints_coords, _keypoints_scores),
        axis=-1)
    keypoints_list = _keypoints_scores_coords[:, list(range(0, 91)), :]     # Shape: (num_people, 91, 3)

    # Boundary Boxes coordinates --> (num_people, 4)
    xyxy_list = raw_predictions.bboxes              # Shape: (num_people, 4)

    return keypoints_list, xyxy_list


def _calc_angle(
        edge_points: [[float, float], [float, float]],
        mid_point: [float, float]) -> float:
    """
    Calculate the angle based on the given edge points and middle point.
    :param edge_points: A tuple of two coordinates of the edge points of the angle.
    :param mid_point: The coordinate of the middle point of the angle.
    :return: The degree value of the angle.
    """
    # Left, Right
    p1, p2 = [np.array(pt) for pt in edge_points]

    # Mid
    m = np.array(mid_point)

    # Angle
    radians = np.arctan2(p2[1] - m[1], p2[0] - m[0]) - np.arctan2(p1[1] - m[1], p1[0] - m[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


def calc_keypoint_angle(
        keypoints_one_person,
        edge_keypoints_names: [str, str],
        mid_keypoint_name: str) -> [float, float]:
    """
    Calculate the angle using the given edge pionts and middle point by their names.
    :param keypoints_one_person: The set of keypoints of a single person. (91, 3)
    :param edge_keypoints_names: A tuple of the names of the two edge keypoints.
    :param mid_keypoint_name: The name of the middle keypoint.
    :return: The targeted angle.
    """

    # Names
    n1, n2 = edge_keypoints_names
    nm = mid_keypoint_name

    # Coordinates
    # Name --> [keypoint_indexes] --> index_number --> [keypoints_one_person] --> (x,y,score) --> [:2] --> (x,y)
    coord1, coord2 = keypoints_one_person[keypoint_indexes[n1]][:2], keypoints_one_person[keypoint_indexes[n2]][:2]
    coordm = keypoints_one_person[keypoint_indexes[nm]][:2]

    # Score of the angle
    s1, s2 = keypoints_one_person[keypoint_indexes[n1]][2], keypoints_one_person[keypoint_indexes[n2]][2]
    sm = keypoints_one_person[keypoint_indexes[nm]][2]

    # Angle Score: Geometric Mean
    angle_score = math.exp((1/3) * (math.log(s1) + math.log(s2) + math.log(sm)))

    return _calc_angle([coord1, coord2], coordm), angle_score


if __name__ == "__main__":
    """
    1. Build bbox detector
    """
    detector = init_detector(cfg.det_config, cfg.det_checkpoint, device=cfg.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    """
    2. Build pose estimator
    """
    pose_estimator = init_pose_estimator(
        cfg.pose_config,
        cfg.pose_checkpoint,
        device=cfg.device,
        cfg_options=dict(
            model=dict(
                test_cfg=dict(output_heatmaps=cfg.draw_heatmap)))
    )

    """
    3. Build Visualizer
    """
    pose_estimator.cfg.visualizer.radius = cfg.radius
    pose_estimator.cfg.visualizer.alpha = cfg.alpha
    pose_estimator.cfg.visualizer.line_width = cfg.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=cfg.skeleton_style)

    """
    4. Initialize Detection Targets
    """
    target_list = [
        [("Body-Left_shoulder", "Body-Left_wrist"), "Body-Left_elbow"],
        [("Body-Right_shoulder", "Body-Right_wrist"), "Body-Right_elbow"],
        [("Body-Left_hip", "Body-Left_elbow"), "Body-Left_shoulder"],
        [("Body-Right_hip", "Body-Right_elbow"), "Body-Right_shoulder"],
    ]

    """
    5. Image Processing
    """
    input_type = 'image'

    if input_type == 'image':
        # 1. Raw predictions
        _pred_instances = process_one_image(cfg.input, detector, pose_estimator, visualizer)

        # 2. Keypoints List (n, 91, 3), Boundary Boxes (n, 4)
        keypoints_list, xyxy_list = get_keypoints_and_xyxy(_pred_instances)

        # 3. For each detected person.
        _title, title_ = "\033[1;34m", "\033[00m"
        for idx, (keypoints, xyxy) in enumerate(zip(keypoints_list, xyxy_list)):
            # 3.1 Keypoints
            print(f"{_title}No.{idx+1}.\n"
                  f"Key points:{title_}")
            for keypoint_idx, keypoint in enumerate(keypoints):
                print(f"({round(keypoint[0],3):.3f}, {round(keypoint[1],3):.3f}), score={round(keypoint[2],4):.4f} - "
                      f"{keypoint_names[keypoint_idx]}")

            # 3.2 Boundaries
            print(f"{_title}xyxy:{title_}\n"
                  f"{xyxy}")

            # 3.3 Key Angles
            print(f"{_title}Key Angles (Display only to 4 digits after .):{title_}")
            for target in target_list:
                target_angle, target_angle_score = calc_keypoint_angle(keypoints, target[0], target[1])
                angle_name = f"{target[0][0]}_|_{target[1]}_|_{target[0][1]}"
                print(f"value={target_angle:4f} deg - score={target_angle_score} - {angle_name}")
            print("\n")
