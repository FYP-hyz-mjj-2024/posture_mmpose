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

    # predict bbox
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]),
        axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == cfg.det_cat_id,
                                   pred_instance.scores > cfg.bbox_thr)]
    bboxes = bboxes[nms(bboxes, cfg.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator_model, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
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

    # build pose estimator
    pose_estimator = init_pose_estimator(
        cfg.pose_config,
        cfg.pose_checkpoint,
        device=cfg.device,
        cfg_options=dict(
            model=dict(
                test_cfg=dict(
                    output_heatmaps=cfg.draw_heatmap
                )
            )
        )
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
        _pred_instances = process_one_image(cfg.input, detector, pose_estimator, visualizer)

        _keypoints_coords = _pred_instances.keypoints
        _keypoints_scores = np.expand_dims(_pred_instances.keypoint_scores, axis=-1)
        _combined_data = np.concatenate((_keypoints_coords, _keypoints_scores), axis=-1)

        keypoints = _combined_data[:, list(range(0, 91)), :]
        xyxy = _pred_instances.bboxes

        # for idx, instance in enumerate(split_instances(pred_instances)):
        #     print(f"\n{idx}.\nkeypoints:\n")
        #     for keypoint in instance['keypoints']:
        #         print(f"\t{keypoint}")
        #     print(f"bbox:\n{instance['bbox']}\nscore:\n{instance['bbox_score']}\n")


    pause = 0
