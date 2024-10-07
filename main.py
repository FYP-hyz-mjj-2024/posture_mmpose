import cv2
import time
from typing import List

import numpy as np
import torch

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, getOneFeatureRow
from step01_annotate_image_mmpose.calculations import calc_keypoint_angle
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.annotate_image import processOneImage, renderTheResults
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed

from step02_train_model_cnn.train_model_hyz import MLP


def videoDemo(bbox_detector_model,
              pose_estimator_model,
              detection_target_list,
              estim_results_visualizer=None,
              classifier_model=None,
              classifier_func=None,
              ws=None):
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

        keypoints_list, xyxy_list, data_samples = processOneImage(frame,
                                                                  bbox_detector_model,
                                                                  pose_estimator_model,
                                                                  bbox_threshold=mcfg.bbox_thr)
        renderTheResults(frame, data_samples, estim_results_visualizer, show_interval=.001)

        [processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func)
         for keypoints, xyxy in zip(keypoints_list, xyxy_list)]

        yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=ws)

        time.sleep(0.085) if (ws is not None) else None

    cap.release()


def processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func, ):
    kas_one_person = getOneKeyAngleScore(keypoints, detection_target_list)
    classifier_result_str, classify_is_ok = classifier_func(classifier_model, kas_one_person)
    render_detection_rectangle(frame, classifier_result_str, xyxy, is_ok=classify_is_ok)


def getOneKeyAngleScore(keypoints: List,
                        detection_target_list: List) -> List:
    """
    Post process the raw features received from process_one_image.
    From the keypoints list, extract the most confident person in the image. Then, convert the
    keypoints list of this person into a flattened feature vector.

    :param keypoints: A list of keypoints set of multiple people, gathered from the image.
    :param detection_target_list: The list of detection targets.
    :return: A flattened array of feature values.
    """
    # Only get the person with the highest detection confidence.
    kas_one_person = []

    # From keypoints list, get angle-score vector.
    for target in detection_target_list:
        angle_value, angle_score = calc_keypoint_angle(keypoints, kcfg.keypoint_indexes, target[0], target[1])
        kas_one_person.append(angle_value)
        kas_one_person.append(angle_score)

    # Shape=(2m)
    return kas_one_person


def classify(classifier_model, numeric_data):
    # TODO: This is an interface maintained to further inject model usage.

    # Normalize Data
    input_data = np.array(numeric_data).reshape(1, -1)
    input_data[:, ::2] /= 180  # 角度字段归一化到 [0, 1]
    mean_X = np.mean(input_data)
    std_dev_X = np.std(input_data)
    input_data = (input_data - mean_X) / std_dev_X
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()

    # for i in range(0, 9999):
    #     continue
    return ("Using" if prediction == 1 else "Not Using"), (prediction != 1)


if __name__ == '__main__':
    # Initialize MMPose essentials
    detector, pose_estimator, visualizer = getMMPoseEssentials(
        det_config=mcfg.det_config,
        det_chkpt=mcfg.det_checkpoint,
        pose_config=mcfg.pose_config,
        pose_chkpt=mcfg.pose_checkpoint
    )

    # List of detection targets
    target_list = kcfg.target_list

    # Classifier Model
    classifier = MLP(input_size=18, hidden_size=100, output_size=2)
    classifier.load_state_dict(torch.load("./data/models/posture_mmpose_nn.pth"))
    classifier.eval()

    videoDemo(bbox_detector_model=detector,
              pose_estimator_model=pose_estimator,
              detection_target_list=target_list,
              # estim_results_visualizer=visualizer,
              classifier_model=classifier,
              classifier_func=classify,
              ws=None)
