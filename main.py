import cv2
import time
from typing import List, Union

import numpy as np
import torch

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, getOneFeatureRow
from step01_annotate_image_mmpose.calculations import calc_keypoint_angle
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.annotate_image import processOneImage, renderTheResults
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed, init_websocket, getUserConsoleConfig

from step02_train_model_cnn.train_model_hyz import MLP


def videoDemo(src: Union[str, int],
              bbox_detector_model,
              pose_estimator_model,
              detection_target_list,
              estim_results_visualizer=None,
              classifier_model=None,
              classifier_func=None,
              websocket_obj=None):

    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

        keypoints_list, xyxy_list, data_samples = processOneImage(frame,
                                                                  bbox_detector_model,
                                                                  pose_estimator_model,
                                                                  bbox_threshold=mcfg.bbox_thr)
        renderTheResults(frame, data_samples, estim_results_visualizer, show_interval=.001)

        if estim_results_visualizer is not None:
            # MMPose Logic
            estim_results_visualizer.add_datasample(
                'result',
                frame,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=mcfg.draw_heatmap,
                draw_bbox=mcfg.draw_bbox,
                show_kpt_idx=mcfg.show_kpt_idx,
                skeleton_style=mcfg.skeleton_style,
                show=mcfg.show,
                wait_time=0.01,
                kpt_thr=mcfg.kpt_thr)
        else:
            # Classification Model Logic
            [processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func)
             for keypoints, xyxy in zip(keypoints_list, xyxy_list)]
            yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=websocket_obj)

        time.sleep(0.085) if (websocket_obj is not None) else None

    cap.release()


def processOnePerson(frame, keypoints, xyxy, detection_target_list, classifier_model, classifier_func, ):
    kas_one_person = getOneFeatureRow(keypoints, detection_target_list)
    classifier_result_str, classify_is_ok = classifier_func(classifier_model, kas_one_person)
    render_detection_rectangle(frame, classifier_result_str, xyxy, is_ok=classify_is_ok)


def classify(classifier_model, numeric_data):
    # TODO: This is an interface maintained to further inject model usage.
    model, params = classifier_model

    # Normalize Data
    input_data = np.array(numeric_data).reshape(1, -1)
    input_data[:, ::2] /= 180
    mean_X = params['mean_X']
    std_dev_X = params['std_dev_X']
    input_data = (input_data - mean_X) / std_dev_X
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()

    # for i in range(0, 9999):
    #     continue
    out0 = 100 * outputs[0][0]/(outputs[0][0]+outputs[0][1])
    out1 = 100 * outputs[0][1]/(outputs[0][0]+outputs[0][1])

    return (f"Using {out1:.2f}" if prediction == 1 else f"Not Using {out0:.2f}"), (prediction != 1)


if __name__ == '__main__':
    # Configuration
    is_remote, video_source, use_mmpose_visualizer = getUserConsoleConfig(max_required_num=3)
else:
    is_remote, video_source, use_mmpose_visualizer = False, 0, False

# Initialize MMPose essentials
detector, pose_estimator, visualizer = getMMPoseEssentials(
    det_config=mcfg.det_config,
    det_chkpt=mcfg.det_checkpoint,
    pose_config=mcfg.pose_config,
    pose_chkpt=mcfg.pose_checkpoint
)

# List of detection targets
target_list = kcfg.get_target_list()

# Classifier Model
model_state = torch.load("./data/models/posture_mmpose_nn.pth")
classifier = MLP(input_size=2*len(target_list), hidden_size=100, output_size=2)
classifier.load_state_dict(model_state['model_state_dict'])
classifier.eval()

classifier_params = {
    'mean_X': model_state['mean_X'].item(),
    'std_dev_X': model_state['std_dev_X'].item()
}


# WebSocket Object
ws = init_websocket("ws://152.42.198.96:8976") if is_remote else None

videoDemo(src=int(video_source) if video_source is not None else 0,
          bbox_detector_model=detector,
          pose_estimator_model=pose_estimator,
          detection_target_list=target_list,
          estim_results_visualizer=visualizer if use_mmpose_visualizer else None,
          classifier_model=[classifier, classifier_params],
          classifier_func=classify,
          websocket_obj=ws)
