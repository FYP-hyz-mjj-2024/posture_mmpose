import cv2
import time
from typing import List, Union, Tuple, Dict

import numpy as np
import torch

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, translateOneLandmarks
from step01_annotate_image_mmpose.calculations import calc_keypoint_angle
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step01_annotate_image_mmpose.annotate_image import processOneImage, renderTheResults
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed, init_websocket, getUserConsoleConfig

from step02_train_model_cnn.train_model_hyz import MLP

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)


def videoDemo(src: Union[str, int],
              bbox_detector_model,
              pose_estimator_model,
              detection_target_list,
              estim_results_visualizer=None,
              classifier_model=None,
              classifier_func=None,
              websocket_obj=None,
              mode: str = None) -> None:
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

        keypoints_list, xyxy_list, data_samples = processOneImage(frame,
                                                                  bbox_detector_model,
                                                                  pose_estimator_model,
                                                                  bbox_threshold=mcfg.bbox_thr)

        if estim_results_visualizer is not None:
            renderTheResults(frame, data_samples, estim_results_visualizer, show_interval=.001)
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
            for keypoints, xyxy in zip(keypoints_list, xyxy_list):
                processOnePerson(frame,
                                 keypoints,
                                 xyxy,
                                 detection_target_list,
                                 classifier_model,
                                 classifier_func,
                                 mode)

            yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=websocket_obj)

        time.sleep(0.085) if (websocket_obj is not None) else None

    cap.release()


def processOnePerson(frame: np.ndarray,  # shape: (H, W, 3)
                     keypoints: np.ndarray,  # shape: (17, 3)
                     xyxy: np.ndarray,  # shape: (4,)
                     detection_target_list: List[List[Union[Tuple[str, str], str]]],  # {list: 858}
                     classifier_model: List[Union[MLP, Dict[str, float]]],
                     classifier_func,
                     mode: str = None) -> None:
    # If detected backside, don't do inference.
    l_shoulder_x, r_shoulder_x = keypoints[5][0], keypoints[6][0]
    l_shoulder_s, r_shoulder_s = keypoints[5][2], keypoints[6][2]  # score
    backside_ratio = (l_shoulder_x - r_shoulder_x) / (xyxy[2] - xyxy[0])  # shoulder_x_diff / width_diff

    if r_shoulder_s > 0.3 and l_shoulder_s > 0.3 and backside_ratio < -0.2:  # backside_threshold = -0.2
        classifier_result_str = f"Backside {((r_shoulder_s + l_shoulder_s) / 2.0 + 1.0) / 2.0:.2f}"
        classify_signal = -1
    else:
        kas_one_person = translateOneLandmarks(detection_target_list, keypoints, mode)
        classifier_result_str, classify_signal = classifier_func(classifier_model, kas_one_person)

    render_detection_rectangle(frame, classifier_result_str, xyxy, ok_signal=classify_signal)


def classify(classifier_model: List[Union[MLP, Dict[str, float]]],
             numeric_data: List[Union[float, np.float32]]) -> Tuple[str, int]:
    model, params = classifier_model

    # Normalize Data
    input_data = np.array(numeric_data).reshape(1, -1)  # TODO: compatible with mode 'mjj'
    input_data[:, ::2] /= 180
    mean_X = params['mean_X']
    std_dev_X = params['std_dev_X']
    input_data = (input_data - mean_X) / std_dev_X
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.view(input_data.shape[0], 6, -1)

    with torch.no_grad():
        outputs = model(input_tensor)
        sg = torch.sigmoid(outputs[0])
        prediction = int(sg[0] < sg[1] or sg[1] > 0.48)
        # prediction = torch.argmax(sg, dim=0).item()

    out0, out1 = sg
    classify_signal = 1 if prediction != 1 else 0
    classifier_result_str = f"Using {out1:.2f}" if (prediction == 1) else f"Not Using {out0:.2f}"

    return classifier_result_str, classify_signal


if __name__ == '__main__':
    # Configuration
    is_remote, video_source, use_mmpose_visualizer = getUserConsoleConfig(max_required_num=3)
else:
    is_remote, video_source, use_mmpose_visualizer = False, 0, False

# Decision on mode
solution_mode = 'hyz'
# solution_mode = 'mjj'

# Initialize MMPose essentials
detector, pose_estimator, visualizer = getMMPoseEssentials(
    det_config=mcfg.det_config,
    det_chkpt=mcfg.det_checkpoint,
    pose_config=mcfg.pose_config,
    pose_chkpt=mcfg.pose_checkpoint
)

# List of detection targets
target_list = kcfg.get_targets(solution_mode)

# Classifier Model
model_state = torch.load('./data/models/posture_mmpose_vgg.pth', map_location=device)
classifier = MLP(input_channel_num=6, output_class_num=2)
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
          websocket_obj=ws,
          mode=solution_mode)
