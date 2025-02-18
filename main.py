# Author Copyright
# Copyright (c) 2024-2025 Huang Yanzhen, Mai Jiajun, Bob Zhang. All rights reserved.

# Third-party Library Usage
# This project uses the ultralytics library for YOLO11 object detection.
# ultralytics is licensed under the AGPL-3.0 License. Source: https://github.com/ultralytics/ultralytics.
# This project uses the ultralytics library as-is, without any modifications to its source code.
# The license text can be found in the following locations:
# - Local Copy: LICENSES/AGPL_ultralytics/LICENSE.txt.
# - Source Repository: https://github.com/ultralytics/ultralytics/blob/main/LICENSE.
#
# You should have received a copy of the AGPL-3.0 License along with this project. If not, see:
# https://www.gnu.org/licenses/agpl-3.0.html

# Basic
import os
import copy
from tqdm import tqdm
import time
from datetime import datetime
from typing import Union, Dict

# Utilities
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pynput import mouse

# Locals
from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, processOneImage, renderTheResults
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step02_train_model_cnn.train_model_hyz import MLP
from step02_train_model_cnn.train_model_mjj import MLP3d
from utils.opencv_utils import yieldVideoFeed, init_websocket, getUserConsoleConfig, render_ui_text, announceFaceFrame
from utils.plot_report import plot_report
from processing import processOnePerson, classify, classify3D, detectPhone, global_device_name, global_device
from utils.decorations import BANNER, CONSOLE_COLORS as CC


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

    def toggle_runtime_handframes_save(x, y, button, pressed):
        """
        (Callback Registration)
        
        The callback function for mouse-clicking that toggles the saving path
        of the runtime hand frames. The targeted path value will be set to None
        at the beginning of each frame. This function, if invoked, will assign
        this none-valued variable the provided saving path, which will toggle
        further process to save the hand frames at this frame.
        :param x: x position of mouse.
        :param y: y position of mouse.
        :param button: Which button is pressed on the mouse.
        :param pressed: Whether is pressed on the mouse.
        :return: None.
        """
        if not pressed or button != mouse.Button.x2:
            return
        nonlocal runtime_params
        runtime_params["path_runtime_handframes"] = runtime_save_handframes_path
        print(f"{CC['green']}Saved runtime handframes at {time.strftime('%Y-%m-%d %H:%M:%S')}.{CC['reset']}")

    # Extract mmpose tools from package.
    bbox_detector_model = pkg_mmpose["bbox_detector_model"]
    pose_estimator_model = pkg_mmpose["pose_estimator_model"]
    detection_target_list = pkg_mmpose["detection_target_list"]
    estim_results_visualizer = pkg_mmpose["estim_results_visualizer"]

    # Determine video size and UI margins according to output source.
    _set_video_w, _set_video_h = (384, 288) if websocket_obj else (640, 480)
    _margin_w, _margin_h = (10, 20) if websocket_obj else (20, 40)

    # Initialize video source.
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _set_video_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _set_video_h)

    # Runtime parameters.
    # These parameters will be constantly changing.
    t_program_start = time.time()
    runtime_params: Dict[str, Union[float, str, None]] = {
        "time_last_record_framerate": t_program_start,
        "time_last_announce_face": t_program_start,
        "path_runtime_handframes": None,
    }

    # Record Performance
    performance = {
        "mmpose": [],
        "mlp": [],
        "yolo": []
    }

    # If local, listen to mouse click event.
    if websocket_obj is None:
        listener = mouse.Listener(on_click=toggle_runtime_handframes_save)
        listener.start()

    # Clear all runtime yolo dataset images
    if runtime_save_handframes_path is not None:
        print(f"Clearing all previous runtime hand frames in dir {runtime_save_handframes_path}...")
        files = [f for f in os.listdir(runtime_save_handframes_path)
                 if os.path.isfile(os.path.join(runtime_save_handframes_path, f))]
        if len(files) > 0:
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
        if key == 27:
            break

        # Reset the hand frame path.
        # If mouse toggled below this, this var will be set to a str value.
        if runtime_params["path_runtime_handframes"] is not None:
            runtime_params["path_runtime_handframes"] = None

        t_start_frame = time.time()
        keypoints_list, xyxy_list, data_samples = processOneImage(frame,
                                                                  bbox_detector_model,
                                                                  pose_estimator_model,
                                                                  bbox_threshold=mcfg.bbox_thr)

        performance["mmpose"].append(time.time() - t_start_frame)

        if estim_results_visualizer is not None:
            # MMPose Logic
            renderTheResults(frame, data_samples, estim_results_visualizer, show_interval=.001)
        else:
            # Copy content of the un-rendered frame.
            # Prevent intervention with object detection.
            ori_frame = copy.deepcopy(frame)

            '''
            response_list: A list of responses from a series of processOnePerson functions.
            The length of the list is the number of person inferred.
            
            Structure of a single response:
            {
                "performance": (t_mlp, t_yolo),
                "time_last_announce_face": time_last_announce_face      # float, response from the same runtime param
                "announced_face_frame": announced_face_frame            # announced face frame of this person or None
            }
            '''

            response_list = [
                processOnePerson(frame=frame,
                                 original_frame=ori_frame,
                                 keypoints=keypoints,
                                 xyxy=xyxy,
                                 detection_target_list=detection_target_list,
                                 pkg_classifier=pkg_classifier,
                                 pkg_phone_detector=pkg_phone_detector,
                                 runtime_parameters=runtime_params,
                                 device_name=device_name,
                                 mode=mode)
                for keypoints, xyxy in zip(keypoints_list, xyxy_list)
            ]

            # Performance Record
            if websocket_obj is None:
                mlp_yolo_times = np.array([res["performance"] for res in response_list])
                performance["mlp"].append(np.sum(mlp_yolo_times[:, 0]))
                performance["yolo"].append(np.sum(mlp_yolo_times[:, 1]))

            # Update framerate
            now: float = time.time()
            frame_rate = 1 / (now - runtime_params["time_last_record_framerate"] + np.finfo(np.float32).eps)
            runtime_params["time_last_record_framerate"] = now

            # Update frame announcing time
            # "time_last_announce_face" of the last inference person of this frame.
            # May not be changed if no face in this frame is announced.
            runtime_params["time_last_announce_face"] = response_list[-1]["time_last_announce_face"]

            # Frame rate
            render_ui_text(frame=frame,
                           text=str(f"FPS: {frame_rate:.3f}"),
                           frame_wh=(_set_video_w, _set_video_h),
                           margin_wh=(_margin_w, _margin_h),
                           align="left",
                           order=0)

            # Current Time
            render_ui_text(frame=frame,
                           text=f"Time: {time.strftime('%Y%m%d %H:%M:%S')}",
                           frame_wh=(_set_video_w, _set_video_h),
                           margin_wh=(_margin_w, _margin_h),
                           align="left",
                           order=1)

            # Last announce face time (LAFT).
            render_ui_text(frame=frame,
                           text=f"LAFT: "
                                f"{datetime.fromtimestamp(runtime_params['time_last_announce_face']).strftime('%Y%m%d %H:%M:%S')}",
                           frame_wh=(_set_video_w, _set_video_h),
                           margin_wh=(_margin_w, _margin_h),
                           align="left",
                           order=2)

            yieldVideoFeed(frame, title="Pedestrian Cell Phone Usage Detection", ws=websocket_obj)

            # Announce face frames
            announced_face_frames = [
                response["announced_face_frame"] for response in response_list
                if response["announced_face_frame"] is not None]
            if websocket_obj is not None and len(announced_face_frames) > 0:
                announceFaceFrame(announced_face_frames, ws=websocket_obj)

            del ori_frame

        if websocket_obj is not None:
            time.sleep(0.005)

    cap.release()

    return performance


if __name__ == '__main__':
    # Display the nice banner
    print(BANNER)
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
    "self_trained": use_trained_yolo,
    "face_announce_interval": 2
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
