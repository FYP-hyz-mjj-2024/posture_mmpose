import numpy as np
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
    init_model as init_pose_estimator
)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

register_all_modules()

keypoint_names = {
    0: 'Body-Chin',
    1: 'Body-Left_eye',
    2: 'Body-Right_eye',
    3: 'Body-Left_ear',
    4: 'Body-Right_ear',
    5: 'Body-Left_shoulder',
    6: 'Body-Right_shoulder',
    7: 'Body-Left_elbow',
    8: 'Body-Right_elbow',
    9: 'Body-Left_wrist',
    10: 'Body-Right_wrist',
    11: 'Body-Left_hip',
    12: 'Body-Right_hip',
    13: 'Body-Left_knee',
    14: 'Body-Right_knee',
    15: 'Body-Left_ankle',
    16: 'Body-Right_ankle',
    17: 'Foot-Left_toe',
    18: 'Foot-Left_pinky',
    19: 'Foot-Left_heel',
    20: 'Foot-Right_toe',
    21: 'Foot-Right_pinky',
    22: 'Foot-Right_heel',
    23: 'Face-Right_hairroot',
    24: 'Face-Right_zyngo',
    25: 'Face-Right_face_top',
    26: 'Face-Right_face_mid',
    27: 'Face-Right_face_bottom',
    28: 'Face-Right_chin_top',
    29: 'Face-Right_chin_mid',
    30: 'Face-Right_chin_bottom',
    31: 'Face-Chin',
    32: 'Face-Left_chin_bottom',
    33: 'Face-Left_chin_mid',
    34: 'Face-Left_chin_top',
    35: 'Face-Left_face_bottom',
    36: 'Face-Left_face_mid',
    37: 'Face-Left_face_top',
    38: 'Face-Left_zyngo',
    39: 'Face-Left_hairroot',
    40: 'Face-Right_eyebrow_out',
    41: 'Face-Right_eyebrow_out_mid',
    42: 'Face-Right_eyebrow_mid',
    43: 'Face-Right_eyebrow_mid_in',
    44: 'Face-Right_eyebrow_in',
    45: 'Face-Left_eyebrow_in',
    46: 'Face-Left_eyebrow_mid_in',
    47: 'Face-Left_eyebrow_mid',
    48: 'Face-Left_eyebrow_out_mid',
    49: 'Face-Left_eyebrow_out',
    50: 'Face-Nose_top',
    51: 'Face-Nose_top_mid',
    52: 'Face-Nose_bottom_mid',
    53: 'Face-Nose_bottom',
    54: 'Face-Right_nostril_out',
    55: 'Face-Right_nostril_mid',
    56: 'Face-Nostril',
    57: 'Face-Left_nostril_mid',
    58: 'Face-Left_nostril_out',
    59: 'Face-Right_eye_out',
    60: 'Face-Right_eye_up_out',
    61: 'Face-Right_eye_up_in',
    62: 'Face-Right_eye_in',
    63: 'Face-Right_eye_down_in',
    64: 'Face-Right_eye_down_out',
    65: 'Face-Left_eye_in',
    66: 'Face-Left_eye_up_in',
    67: 'Face-Left_eye_up_out',
    68: 'Face-Left_eye_out',
    69: 'Face-Left_eye_down_out',
    70: 'Face-Left_eye_down_in',
    71: 'Face-Lips_l1_right_out',
    72: 'Face-Lips_l1_right_mid',
    73: 'Face-Lips_l1_right_in',
    74: 'Face-Lips_l1_mid',
    75: 'Face-Lips_l1_left_in',
    76: 'Face-Lips_l1_left_mid',
    77: 'Face-Lips_l1_left_out',
    78: 'Face-Lips_l4_left_out',
    79: 'Face-Lips_l4_left_in',
    80: 'Face-Lips_l4_mid',
    81: 'Face-Lips_l4_right_in',
    82: 'Face-Lips_l4_right_out',
    83: 'Face-Lips_l2_right_out',
    84: 'Face-Lips_l2_right_in',
    85: 'Face-Lips_l2_mid',
    86: 'Face-Lips_l2_left_in',
    87: 'Face-Lips_l2_left_out',
    88: 'Face-Lips_l3_left',
    89: 'Face-Lips_l3_mid',
    90: 'Face-Lips_l3_right',
}


def process_one_image(img,
                      bbox_detector_model,
                      pose_estimator_model,
                      estim_results_visualizer=None,
                      show_interval=0):
    """
    Visualize predicted key points (and heatmaps) of one image.
    Given an image, first use bbox detection model to retrieve object boundary boxes.
    Then, feed the sub images defined by the bbox into the pose estimation model to get key points.
    Lastly, render and output the key points.
    :param img: The image.
    :param bbox_detector_model: The model to retrieve boundary boxes.
    :param pose_estimator_model: The model to estimate a person's pose, i.e., retrieve key points.
    :param estim_results_visualizer: The results visualizer.
    """

    if not isinstance(img, str) and not isinstance(img, np.ndarray):
        raise ValueError(f"Image should either be the path to the image file or the nd.aray instance."
                         f"Current image type is {type(img)}")

    # Get boundary boxes
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]),axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == cfg.det_cat_id,
                                   pred_instance.scores > cfg.bbox_thr)]
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


if __name__ == "__main__":

    """
    Build detector
    """
    detector = init_detector(cfg.det_config, cfg.det_checkpoint, device=cfg.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    """
    Build pose estimator
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
    Build Visualizer
    """
    pose_estimator.cfg.visualizer.radius = cfg.radius
    pose_estimator.cfg.visualizer.alpha = cfg.alpha
    pose_estimator.cfg.visualizer.line_width = cfg.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=cfg.skeleton_style)

    """
    Image Processing
    """
    input_type = 'image'

    if input_type == 'image':
        # Raw predictions, need to extract futher data
        _pred_instances = process_one_image(cfg.input, detector, pose_estimator, visualizer)

        # Keypoint coordinates
        _keypoints_coords = _pred_instances.keypoints
        _keypoints_scores = np.expand_dims(_pred_instances.keypoint_scores, axis=-1)
        _combined_data = np.concatenate((_keypoints_coords, _keypoints_scores), axis=-1)
        keypoints_list = _combined_data[:, list(range(0, 91)), :]    #(n, 91, 3) => (num_people, num_targets, (x, y, confidence))

        # Boundary Boxes coordinates
        xyxy_list = _pred_instances.bboxes  #(n,4) => (num_people, (xmin, ymin, xmax, ymax))

        for idx, (keypoints, xyxy) in enumerate(zip(keypoints_list, xyxy_list)):
            print("Key points:")
            for keypoint_idx, keypoint in enumerate(keypoints):
                print(f"({round(keypoint[0],3):.3f}, {round(keypoint[1],3):.3f}), score={round(keypoint[2],4):.4f} - "
                      f"{keypoint_names[keypoint_idx]}")
            print(f"xyxy:\n{xyxy}\n\n")
