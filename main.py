import cv2
import time
from typing import List

import numpy as np

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, getOneFeatureRow
from step01_annotate_image_mmpose.calculations import calc_keypoint_angle
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.annotate_image import processOneImage
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed


def videoDemo(bbox_detector_model,
              pose_estimator_model,
              detection_target_list,
              estim_results_visualizer=None,
              classifier_model=None,
              classifier_func=None,
              ws=None):

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

        [processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func)
         for keypoints, xyxy in zip(keypoints_list, xyxy_list)]

        yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=ws)

        time.sleep(0.085) if (ws is not None) else None

    cap.release()


def processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func,):
    kas_one_person = getOneKeyAngleScore(keypoints, detection_target_list)
    classifier_result_str = classifier_func(classifier_model, kas_one_person)
    render_detection_rectangle(frame, classifier_result_str, xyxy, is_ok=True)


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


def classify(classifier_model, numeric_data) -> str:
    # TODO: This is an interface maintained to further inject model usage.
    for i in range(0, 9999):
        continue
    return "Label"


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

    videoDemo(bbox_detector_model=detector,
              pose_estimator_model=pose_estimator,
              detection_target_list=target_list,
              # estim_results_visualizer=visualizer,
              classifier_model=None,
              classifier_func=classify,
              ws=None)
