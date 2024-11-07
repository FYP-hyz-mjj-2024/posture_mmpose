# Built-in
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


def renderTheResults(img: Union[str, np.ndarray],
                     data_samples,
                     estim_results_visualizer=None,
                     show_interval=0.001):
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


def processOneImage(img: Union[str, np.ndarray],
                    bbox_detector_model,
                    pose_estimator_model,
                    bbox_threshold=mcfg.bbox_thr_single) -> Tuple[ndarray, ndarray, PoseDataSample]:
    """
    Given an image, first use bbox detection model to retrieve object boundary boxes.
    Then, feed the sub images defined by the bbox into the pose estimation model to get key points.
    Lastly, visualize predicted key points (and heatmaps) of one image.

    data_samples --> pred_instances --> [keypoints, keypoint_scores, bboxes]

    :param img: The image.
    :param bbox_detector_model: The model to retrieve boundary boxes.
    :param pose_estimator_model: The model to estimate a person's pose, i.e., retrieve key points.
    :param estim_results_visualizer: The result visualizer.
    :param bbox_threshold: The threshold of IoU where the boundary boxes will be recorded into the list.
    :param show_interval: The wait time of the visualizer.
    :return: Raw Results of prediction.
    """

    # Get boundary boxes xyxy list, only keep bboxes with high confidence.
    det_result = inference_detector(bbox_detector_model, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == mcfg.det_cat_id, pred_instance.scores > bbox_threshold)]
    bboxes = bboxes[nms(bboxes, mcfg.nms_thr), :4]

    # Get key points list.
    pose_results = inference_topdown(pose_estimator_model, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # if there is no instance detected, return None
    # return data_samples.get('pred_instances', None)
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


def getOneFeatureRow(keypoints_list: ndarray,
                     detection_targets: Union[List, ndarray],
                     mode: str = 'hyz') -> Union[List, Tuple[ndarray, ndarray]]:
    """
    Post process the raw features received from process_one_image.
    From the keypoints list, extract the most confident person in the image. Then, convert the
    keypoints list of this person into a flattened feature vector.

    :param mode: What type of data. "hyz" -> angles and scores in lines, "mjj" -> angles and scores in cubes.
    :param keypoints_list: A list of keypoints set of multiple people, gathered from the image.
    :param detection_targets: The list/cube of detection targets.
    :return: A flattened array or cube of feature values.
    """
    if mode == 'hyz':
        # Only get the person with the highest detection confidence.
        keypoints = keypoints_list
        kas_one_person = []

        # From keypoints list, get angle-score vector.
        for target in detection_targets:
            angle_value, angle_score = calc_keypoint_angle(keypoints, kcfg.keypoint_indexes, target[0], target[1])
            kas_one_person.append(angle_value)
            kas_one_person.append(angle_score)

        # Shape=(2m)
        return kas_one_person
    elif mode == 'mjj':
        _shape = detection_targets.shape
        angles = np.empty(shape=_shape)
        scores = np.empty(shape=_shape)
        for k in range(_shape[2]):
            for i in range(_shape[0]):
                for j in range(_shape[1]):
                    angles[i, j, k], scores[i, j, k] = calc_keypoint_angle(keypoints_list[i],
                                                                           kcfg.keypoint_indexes,
                                                                           detection_targets[i, j, k, 0],
                                                                           detection_targets[i, j, k, 1])


def processImagesInDir(img_dir: str,
                       bbox_detector_model,
                       pose_estimator_model,
                       estim_results_visualizer=None,
                       show_interval=0,
                       detection_target_list=None,
                       use_weight=False) -> np.ndarray:
    """
    Batch annotate multiple images within a directory.
    :param img_dir: Directory where multiple images are stored.
    :param bbox_detector_model: MMPose boundary box detection model.
    :param pose_estimator_model: MMPose estimator model.
    :param estim_results_visualizer: MMPose estimation results visualizer.
    :param show_interval: Interval among each image.
    :param detection_target_list: List of detection target.
    :param use_weight: Whether to use manual-given weight as label.
    :return: Feature Matrix.
    """

    # Shape: (num_file=num_people, num_features)
    # num_features = num_angles + num_scores = 2 * num_angles = 2 * num_scores
    kas_multiple_images = []

    for root, dirs, files in os.walk(img_dir):
        # For multiple images, do:
        for file in files:
            if not file.endswith(".jpg"):
                continue
            file_path = os.path.join(root, file)

            # Single Image, may be multiple person
            keypoints_list, xyxy_list, data_samples = processOneImage(
                file_path,
                bbox_detector_model,
                pose_estimator_model)

            renderTheResults(file_path, data_samples, estim_results_visualizer, 0.001)

            # A flattened angle-score vector of a single person.
            one_row = getOneFeatureRow(keypoints_list[0], detection_target_list)

            # Information of the image
            img_info = parseFileName(file, ".jpg")
            if use_weight:
                if 'weight' not in img_info:
                    raise Exception("You need to specify weight in the file name!")
                one_row.append(img_info['weight'])

            # Collect this person.
            kas_multiple_images.append(one_row)

    # Shape: (num_people, num_features)
    feature_matrix = np.array(kas_multiple_images)

    return feature_matrix


def processVideosInDir(video_dir: str,
                       bbox_detector_model: RTMDet,
                       pose_estimator_model: TopdownPoseEstimator,
                       detection_target_list: Union[List[List[Union[Tuple[str, str], str]]], ndarray],
                       skip_interval: int = 10,
                       mode: str = 'hyz') -> List[Dict[str, Union[str, np.ndarray]]]:
    named_feature_matrices = []

    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue

        kas_video = []
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        file_name_with_extension = os.path.basename(video_path)
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

        if not cap.isOpened():
            print(f"Cannot find {video_file}")
            continue

        cur_frame = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing {video_file}") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                cur_frame += 1
                pbar.update(1)
                if cur_frame % skip_interval != 0:
                    continue

                landmarks, _, data_samples = processOneImage(frame, bbox_detector_model, pose_estimator_model)
                # renderTheResults(frame, data_samples, estim_results_visualizer=visualizer, show_interval=.001)
                one_row = getOneFeatureRow(landmarks[0], detection_target_list) # TODO

                angle_cube, score_cube = getOneFeatureRow(landmarks[0], detection_target_list, mode)

                img_info = parseFileName(file_name_without_extension + f"_{cur_frame}", ".mp4")
                # if 'weight' not in img_info:
                #     raise Exception("You need to specify weight in the file name!")
                # one_row.append(img_info['weight'])

                kas_video.append(one_row)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            feature_matrix = np.array(kas_video)
            named_feature_matrices.append({"name": file_name_without_extension,
                                           "feature_matrix": feature_matrix})

            # saveFeatureMatToNPY(feature_matrix, save_path="../data/train/" + file_name_without_extension + ".npy")
    return named_feature_matrices


def saveFeatureMatToNPY(mat: np.ndarray, save_path: str) -> None:
    """
    Save the feature matrix into a npy file, under the given path.
    :param mat:
    :param save_path:
    :return:
    """
    # Shape: (num_people, num_features)
    feature_matrix = np.array(mat)
    np.save(save_path, feature_matrix)


if __name__ == "__main__":

    # Initialize MMPose essentials
    detector, pose_estimator, visualizer = getMMPoseEssentials(
        det_config=mcfg.det_config_train,
        det_chkpt=mcfg.det_checkpoint_train,
        pose_config=mcfg.pose_config_train,
        pose_chkpt=mcfg.pose_checkpoint_train
    )

    # List of detection targets

    """
    5. Image Processing
    """
    # input_type = 'video2hyz_npy'  # Alter this between 'image' and 'video'
    input_type = 'video2mjj_npy'  #
    overwrite = False

    if input_type == 'image':

        for root, dirs, files in os.walk("../data/train/img_from_video"):
            for sub_dir in dirs:
                print(f"Processing Images in: {sub_dir}...", end=" ")
                if not overwrite and os.path.exists(f"../data/train/{sub_dir}.npy"):
                    print(".npy file exists, deported.")
                    continue
                print("Saved.")

                kas_multiple_images = processImagesInDir(
                    img_dir=os.path.join(root, sub_dir),
                    bbox_detector_model=detector,
                    pose_estimator_model=pose_estimator,
                    estim_results_visualizer=visualizer,
                    detection_target_list=kcfg.get_target_list('hyz')
                )
                saveFeatureMatToNPY(kas_multiple_images, save_path=f"../data/train/{sub_dir}.npy")
    elif input_type == 'video2hyz_npy':

        video_folder = "../data/blob/videos"
        named_feature_mats = processVideosInDir(video_dir=video_folder,
                                                bbox_detector_model=detector,
                                                pose_estimator_model=pose_estimator,
                                                detection_target_list=kcfg.get_target_list('hyz'),
                                                skip_interval=10)

        [
            saveFeatureMatToNPY(named_feature_mat['feature_matrix'],
                                save_path="../data/train/" + named_feature_mat['name'] + ".npy")
            for named_feature_mat in named_feature_mats
        ]
    elif input_type == 'video2mjj_npy':

        video_folder = "../data/blob/videos"
        named_feature_mats = processVideosInDir(video_dir=video_folder,
                                                bbox_detector_model=detector,
                                                pose_estimator_model=pose_estimator,
                                                detection_target_list=kcfg.get_target_list('mjj'),
                                                skip_interval=10)

        for named_feature_mat in named_feature_mats:
            saveFeatureMatToNPY(named_feature_mat['feature_matrix'],
                                save_path="../data/train/" + named_feature_mat['name'] + ".npy")
