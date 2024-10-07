import cv2
import time

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials
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

        for keypoints, xyxy in zip(keypoints_list, xyxy_list):
            kas_one_person = []
            for target in detection_target_list:
                angle_value, angle_score = calc_keypoint_angle(keypoints, kcfg.keypoint_indexes, target[0], target[1])
                kas_one_person.append(angle_value)
                kas_one_person.append(angle_score)

            # Model Prediction
            classifier_result_str = classifier_func(classifier_model, kas_one_person)
            render_detection_rectangle(frame, classifier_result_str, xyxy, is_ok=True)

        yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=ws)

        time.sleep(0.085) if (ws is not None) else None

    cap.release()


def classify(classifier_model, numeric_data) -> str:
    # TODO: This is an interface maintained to further inject model usage.
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
