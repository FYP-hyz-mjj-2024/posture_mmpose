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


def _calc_angle(edge_points, mid_point):
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


def calc_keypoint_angle(keypoints_one_person, edge_keypoints, mid_keypoint):
    """
    Calculate the angle using the given edge pionts and middle point by their names.
    :param keypoints_one_person: The set of keypoints of a single person.
    :param edge_keypoints: A tuple of the names of the two edge keypoints.
    :param mid_keypoint: The name of the middle keypoint.
    :return: The targeted angle.
    """

    # Names
    n1, n2 = edge_keypoints
    nm = mid_keypoint

    # Coordinates
    # Name --> [keypoint_indexes] --> index_number --> [keypoints_one_person] --> (x,y,score) --> [:2] --> (x,y)
    coord1, coord2 = keypoints_one_person[keypoint_indexes[n1]][:2], keypoints_one_person[keypoint_indexes[n2]][:2]
    coordm = keypoints_one_person[keypoint_indexes[nm]][:2]

    return _calc_angle([coord1, coord2], coordm)


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

        # TODO: Test angle calculation.
        target = [("Body-Left_shoulder", "Body-Left_wrist"), "Body-Left_elbow"]

        for idx, (keypoints, xyxy) in enumerate(zip(keypoints_list, xyxy_list)):
            print("Key points:")
            for keypoint_idx, keypoint in enumerate(keypoints):
                print(f"({round(keypoint[0],3):.3f}, {round(keypoint[1],3):.3f}), score={round(keypoint[2],4):.4f} - "
                      f"{keypoint_names[keypoint_idx]}")
            print(f"xyxy:\n{xyxy}")

            target_angle = calc_keypoint_angle(keypoints, target[0], target[1])
            print(f"target_angle: {target_angle}\n\n")


