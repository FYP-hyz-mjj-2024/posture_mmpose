# Basic
import os
from tqdm import tqdm
import time
from typing import Union

# Utilities
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Locals
from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials
from step01_annotate_image_mmpose.annotate_image import processOneImage, renderTheResults
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step02_train_model_cnn.train_model_hyz import MLP
from step02_train_model_cnn.train_model_mjj import MLP3d
from utils.opencv_utils import yieldVideoFeed, init_websocket, getUserConsoleConfig
from utils.plot_report import plot_report
from processing import processOnePerson, classify, classify3D, detectPhone, global_device_name, global_device


def videoDemo(src: Union[str, int],
              pkg_mmpose,
              pkg_classifier,
              pkg_phone_detector,
              runtime_save_handframes_path: str,
              device_name: str = global_device_name,
              mode: str = None,
              websocket_obj=None):
    """
    Overall demonstration function of this project. Uses live video.
    :param src: Video Source. Int: Live; Str: Path to pre-recorded video.
    :param pkg_mmpose: Tool package of mmpose.
    :param pkg_classifier: Tool package of mlp posture classifier.
    :param pkg_phone_detector: Tool package of phone detector.
    :param runtime_save_handframes_path: Path to save runtime hand frames.
    :param device_name: Name of hardware, cpu or cuda.
    :param mode: Mode of convolution: hyz or mjj.
    :param websocket_obj: Websocket object.
    :return: None.
    """

    # Extract mmpose tools from package.
    bbox_detector_model = pkg_mmpose["bbox_detector_model"]
    pose_estimator_model = pkg_mmpose["pose_estimator_model"]
    detection_target_list = pkg_mmpose["detection_target_list"]
    estim_results_visualizer = pkg_mmpose["estim_results_visualizer"]

    cap = cv2.VideoCapture(src)

    # Determine video size according to out source.
    _set_video_w, _set_video_h = [384, 288] if websocket_obj else [640, 480]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _set_video_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _set_video_h)

    # Record frame rate
    last_time = time.time()

    # Record Performance
    performance = {
        "mmpose": [],
        "mlp": [],
        "yolo": []
    }

    # Clear all runtime yolo dataset images
    if runtime_save_handframes_path is not None:
        print(f"Clearing all previous runtime hand frames in dir {runtime_save_handframes_path}...")
        files = [f for f in os.listdir(runtime_save_handframes_path)
                 if os.path.isfile(os.path.join(runtime_save_handframes_path, f))]
        for file in tqdm(files):
            file_path = os.path.join(runtime_save_handframes_path, file)
            os.remove(file_path)
        print(f"Done!\n")
    else:
        print(f"Path to save runtime hand frames is not defined.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Key op detection
        key = cv2.waitKey(5) & 0xFF
        runtime_save_handframes_path_cur = None

        if key == 27:
            break
        elif websocket_obj is None and (key == 82 or key == 114):
            # Press R or r to save hand frames at run time.
            runtime_save_handframes_path_cur = runtime_save_handframes_path

        t_start_frame = time.time()
        keypoints_list, xyxy_list, data_samples = processOneImage(frame,
                                                                  bbox_detector_model,
                                                                  pose_estimator_model,
                                                                  bbox_threshold=mcfg.bbox_thr)

        performance["mmpose"].append(time.time() - t_start_frame)

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
            mlp_yolo_times = [
                processOnePerson(frame=frame,
                                 keypoints=keypoints,
                                 xyxy=xyxy,
                                 detection_target_list=detection_target_list,
                                 pkg_classifier=pkg_classifier,
                                 pkg_phone_detector=pkg_phone_detector,
                                 runtime_options={
                                     "runtime_save_handframes_path": runtime_save_handframes_path_cur
                                 },
                                 device_name=device_name,
                                 mode=mode)
                for keypoints, xyxy in zip(keypoints_list, xyxy_list)
            ]

            if websocket_obj is None:
                mlp_yolo_times = np.array(mlp_yolo_times)
                performance["mlp"].append(np.sum(mlp_yolo_times[:, 0]))
                performance["yolo"].append(np.sum(mlp_yolo_times[:, 1]))

            # Calculate and display frame rate
            this_time = time.time()
            frame_rate = 1 / (this_time - last_time + np.finfo(np.float32).eps)     # Handle divide-0 error
            last_time = this_time

            cv2.putText(
                frame,
                str(f"FPS: {frame_rate:.3f}"),
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2
            )

            yieldVideoFeed(frame, title="Pedestrian Cell Phone Usage Detection", ws=websocket_obj)

        if websocket_obj is not None:
            time.sleep(0.005)

    cap.release()

    return performance


if __name__ == '__main__':
    # Configuration
    is_remote, video_source, use_mmpose_visualizer, use_trained_yolo = getUserConsoleConfig()
else:
    is_remote, video_source, use_mmpose_visualizer, use_trained_yolo = False, 0, False, False

# Decision on mode
solution_mode = 'mjj'   # or 'hyz'

# Initialize MMPose essentials
bbox_detector, pose_estimator, visualizer = getMMPoseEssentials()

# List of detection targets
target_list = kcfg.get_targets(solution_mode)

# Classifier Model
if solution_mode == 'hyz':
    model_state = torch.load('./data/models/posture_mmpose_vgg1d_17315770488631685.pth', map_location=global_device)
    classifier = MLP(input_channel_num=6, output_class_num=2)
else:
    model_state = torch.load('./data/models/posture_mmpose_vgg3d_17349570075562594.pth', map_location=global_device)
    classifier = MLP3d(input_channel_num=2, output_class_num=2)

classifier.load_state_dict(model_state['model_state_dict'])
classifier.eval()
classifier.to(global_device)

norm_params = {
    'mean_X': model_state['mean_X'].item(),
    'std_dev_X': model_state['std_dev_X'].item()
}

# Classifier Function
classifier_function = classify if solution_mode == 'hyz' else classify3D

# YOLO object detection model
if use_trained_yolo:
    yolo_path = "step03_yolo_phone_detection/archived_onnx/best.pt"
else:
    yolo_path = "step03_yolo_phone_detection/non_tuned/yolo11n.pt"


phone_detector = YOLO(yolo_path)

# WebSocket Object
ws = init_websocket("ws://localhost:8976") if is_remote else None

# Package up essential binding tools into dictionaries.
package_mmpose = {
    "bbox_detector_model": bbox_detector,
    "pose_estimator_model": pose_estimator,
    "detection_target_list": target_list,
    "estim_results_visualizer": visualizer if use_mmpose_visualizer else None,
}

package_classifier = {
    "classifier_model": classifier,
    "classifier_func": classifier_function,
    "norm_params": norm_params,
}

package_phone_detector = {
    "phone_detector_model": phone_detector,
    "phone_detector_func": detectPhone,
    "self_trained": use_trained_yolo
}

runtime_save_hf_path = "data/yolo_dataset_runtime/"

# Start the loop
demo_performance = videoDemo(src=int(video_source) if video_source is not None else 0,
                             # Model Task packages
                             pkg_mmpose=package_mmpose,
                             pkg_classifier=package_classifier,
                             pkg_phone_detector=package_phone_detector,
                             # Runtime configs
                             runtime_save_handframes_path=runtime_save_hf_path,
                             # Configs
                             device_name=global_device_name,
                             mode=solution_mode,
                             websocket_obj=ws)

# Performance Report
plot_report(
    arrays=np.array(list(demo_performance.values()))[:, 1:],
    labels=["RTMPose", "posture", "yolo"],
    config={"title": "Frame Computation Time", "x_name": "Frame Number", "y_name": "Time (s)"},
    plot_mean=True
)
