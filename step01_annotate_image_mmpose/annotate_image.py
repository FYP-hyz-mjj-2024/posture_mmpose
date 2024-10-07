import time
import cv2
import os
import numpy as np
from tqdm import tqdm

import mmcv
from typing import List
from mmpose.apis import (
    inference_topdown,
    init_model as init_pose_estimator)
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline, register_all_modules

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Local
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from utils.calculations import calc_keypoint_angle
from utils.opencv_utils import init_websocket, render_detection_rectangle, yield_video_feed
from utils.parse_file_name import parseFileName

register_all_modules()


def getMMPoseEssentials():
    """
    Get essential detectors and visualizers of MMPose.getMMPose.

    - bbox_detector: Detector that gives xyxy of the boundary boxes.

    - pose_estimator: Estimator that gives the landmarks of a person.

    - visualizer: MMPose's built-in visualizer to visualize the bbox and pose skeleton.

    :return: bbox_detector, pose_estimator, visualizer

    """

    # 1. Boundary Box detector
    bbox_detector = init_detector(mcfg.det_config_train, mcfg.det_checkpoint_train, device=mcfg.device)
    bbox_detector.cfg = adapt_mmdet_pipeline(bbox_detector.cfg)

    # 2. Pose estimator
    pose_estimator = init_pose_estimator(
        mcfg.pose_config_train,
        mcfg.pose_checkpoint_train,
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


def processOneImage(img,
                    bbox_detector_model,
                    pose_estimator_model,
                    estim_results_visualizer=None,
                    bbox_threshold = mcfg.bbox_thr_single,
                    show_interval=0.001):
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
    bboxes = bboxes[np.logical_and(pred_instance.labels == mcfg.det_cat_id,
                                   pred_instance.scores > bbox_threshold)]    # Single box detection
    bboxes = bboxes[nms(bboxes, mcfg.nms_thr), :4]

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
            draw_heatmap=mcfg.draw_heatmap,
            draw_bbox=mcfg.draw_bbox,
            show_kpt_idx=mcfg.show_kpt_idx,
            skeleton_style=mcfg.skeleton_style,
            show=mcfg.show,
            wait_time=show_interval,
            kpt_thr=mcfg.kpt_thr)

    # if there is no instance detected, return None
    # return data_samples.get('pred_instances', None)
    raw_predictions = data_samples.get('pred_instances', None)

    # Keypoint coordinates --> (num_people, 91, 3)
    _keypoints_coords = raw_predictions.keypoints  # Shape: (num_people, 133, 2)
    _keypoints_scores = np.expand_dims(  # Shape: (num_people, 133, 1)
        raw_predictions.keypoint_scores,
        axis=-1)
    _keypoints_scores_coords = np.concatenate(  # Shape: (num_people, 133, 3)
        (_keypoints_coords, _keypoints_scores),
        axis=-1)

    # keypoints_list = _keypoints_scores_coords[:, list(range(0, 91)), :]  # Shape: (num_people, 91, 3)
    keypoints_list = _keypoints_scores_coords   # Shape: (num_people, 17, 3)

    # Boundary Boxes coordinates --> (num_people, 4)
    xyxy_list = raw_predictions.bboxes  # Shape: (num_people, 4)

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

    return keypoints_list, xyxy_list


def getOneFeatureRow(keypoints_list: np.ndarray,
                     detection_target_list: List) -> List:
    """
    Post process the raw features received from process_one_image.
    From the keypoints list, extract the most confident person in the image. Then, convert the
    keypoints list of this person into a flattened feature vector.
    :param keypoints_list: A list of keypoints set of multiple people, gathered from the image.
    :param detection_target_list: The list of detection targets.
    :return: A flattened array of feature values.
    """

    # Only get the person with the highest detection confidence.
    keypoints = keypoints_list[0]
    kas_one_person = []

    # From keypoints list, get angle-score vector.
    for target in detection_target_list:
        angle_value, angle_score = calc_keypoint_angle(keypoints, kcfg.keypoint_indexes, target[0], target[1])
        kas_one_person.append(angle_value)
        kas_one_person.append(angle_score)

    # Shape=(2m)
    return kas_one_person


def processMultipleImages(img_dir: str,
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
            keypoints_list, xyxy_list = processOneImage(
                file_path,
                bbox_detector_model,
                pose_estimator_model,
                estim_results_visualizer)

            # A flattened angle-score vector of a single person.
            one_row = getOneFeatureRow(keypoints_list, detection_target_list)

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


def saveFeatureMatToNPY(mat: np.ndarray, save_path: str):
    """
    Save the feature matrix into a npy file, under the given path.
    :param mat:
    :param save_path:
    :return:
    """
    # Shape: (num_people, num_features)
    feature_matrix = np.array(mat)
    np.save(save_path, feature_matrix)


def videoDemo(bbox_detector_model,
              pose_estimator_model,
              estim_results_visualizer=None,
              classifier_model=None,
              ws=None):

    # cap = cv2.VideoCapture("../data/demo/demo_video.mp4")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

        keypoints_list, xyxy_list = processOneImage(frame,
                                                    bbox_detector_model,
                                                    pose_estimator_model,
                                                    estim_results_visualizer=estim_results_visualizer,
                                                    bbox_threshold=mcfg.bbox_thr)
        # TODO: Model Prediction

        # Render using opencv instead of the built-in renderer of mmpose.
        if estim_results_visualizer is not None:
            continue
        [render_detection_rectangle(frame, "label", xyxy, is_ok=True) for xyxy in xyxy_list]

        yield_video_feed(frame, mode='local', title="Smart Device Usage Detection", ws=ws)

        time.sleep(0.085)

    cap.release()
    pass


if __name__ == "__main__":

    # Initialize MMPose essentials
    detector, pose_estimator, visualizer = getMMPoseEssentials()

    # List of detection targets
    target_list = kcfg.target_list

    """
    5. Image Processing
    """
    input_type = 'video'    # Alter this between 'image' and 'video'
    overwrite = False

    if input_type == 'image':

        for root, dirs, files in os.walk("../data/train/img_from_video"):
            for sub_dir in dirs:
                print(f"Processing Images in: {sub_dir}...", end=" ")
                if not overwrite and os.path.exists(f"../data/train/{sub_dir}.npy"):
                    print(".npy file exists, deported.")
                    continue
                print("Saved.")

                kas_multiple_images = processMultipleImages(
                    img_dir=os.path.join(root, sub_dir),
                    bbox_detector_model=detector,
                    pose_estimator_model=pose_estimator,
                    estim_results_visualizer=visualizer,
                    detection_target_list=target_list
                )
                saveFeatureMatToNPY(kas_multiple_images, save_path=f"../data/train/{sub_dir}.npy")

    elif input_type == 'video':
        # nn_model = MLP(input_size=len(target_list), hidden_size=100, output_size=2)
        # nn_model.load_state_dict(torch.load(""))
        # nn_model.eval()
        ws = init_websocket(server_url="ws://152.42.198.96:8976")
        # ws = init_websocket(server_url="ws://localhost:8976")
        videoDemo(bbox_detector_model=detector,
                  pose_estimator_model=pose_estimator,
                  # estim_results_visualizer=visualizer,
                  classifier_model=None,
                  ws=ws)
    elif input_type == 'video2npy':

        # video_folder = "/Users/maijiajun/Desktop/pose/Processed_videos"
        video_folder = "../data/blob/videos"

        for video_file in os.listdir(video_folder):
            if video_file.endswith('.mp4'):
                kas_multiple_images = []
                video_path = os.path.join(video_folder, video_file)
                cap = cv2.VideoCapture(video_path)

                file_name_with_extension = os.path.basename(video_path)
                file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

                if not cap.isOpened():
                    print(f"Cannot find {video_file}")
                    continue

                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                with tqdm(total=total_frames, desc=f"Processing {video_file}") as pbar:
                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame_count += 1
                        pbar.update(1)
                        if frame_count % 10 != 0:
                            continue

                        landmarks, _ = processOneImage(frame, detector, pose_estimator)
                        one_row = getOneFeatureRow(landmarks, target_list)

                        img_info = parseFileName(file_name_without_extension + f"_{frame_count}", ".mp4")
                        if 'weight' not in img_info:
                            raise Exception("You need to specify weight in the file name!")
                        one_row.append(img_info['weight'])

                        kas_multiple_images.append(one_row)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                cap.release()

                feature_matrix = np.array(kas_multiple_images)

                saveFeatureMatToNPY(feature_matrix, save_path="../data/train/" + file_name_without_extension + ".npy")