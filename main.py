import cv2
import time

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.annotate_image import processOneImage
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed


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

        yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=ws)

        time.sleep(0.085) if (ws is not None) else None

    cap.release()


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
              # estim_results_visualizer=visualizer,
              classifier_model=None,
              ws=None)