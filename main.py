import time
from typing import List, Union, Tuple, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from step01_annotate_image_mmpose.annotate_image import getMMPoseEssentials, translateOneLandmarks
from step01_annotate_image_mmpose.annotate_image import processOneImage, renderTheResults
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg, mmpose_config as mcfg
from step02_train_model_cnn.train_model_hyz import MLP
from step02_train_model_cnn.train_model_mjj import MLP3d
from utils.opencv_utils import render_detection_rectangle, yieldVideoFeed, init_websocket, getUserConsoleConfig, \
    cropFrame
from utils.plot_report import plot_report

global_device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
global_device = torch.device(global_device_name)

def videoDemo(src: Union[str, int],
              pkg_mmpose,
              pkg_classifier,
              pkg_phone_detector,
              device_name: str = global_device_name,
              mode: str = None,
              websocket_obj=None):
    """
    Overall demonstration function of this project. Uses live video.
    :param src: Video Source. Int: Live; Str: Path to pre-recorded video.
    :param pkg_mmpose: Tool package of mmpose.
    :param pkg_classifier: Tool package of mlp posture classifier.
    :param pkg_phone_detector: Tool package of phone detector.
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

    if websocket_obj:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 384)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Record frame rate
    last_time = time.time()

    # Record Performance
    performance = {
        "mmpose": [],
        "mlp": [],
        "yolo": []
    }

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or cv2.waitKey(5) & 0xFF == 27:
            break

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
                processOnePerson(frame,
                                 keypoints,
                                 xyxy,
                                 detection_target_list,
                                 pkg_classifier,
                                 pkg_phone_detector,
                                 device_name,
                                 mode)
                for keypoints, xyxy in zip(keypoints_list, xyxy_list)
            ]

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

            yieldVideoFeed(frame, title="Smart Device Usage Detection", ws=websocket_obj)

        time.sleep(0.005) if (websocket_obj is not None) else None

    cap.release()

    return performance


def processOnePerson(frame: np.ndarray,  # shape: (H, W, 3)
                     keypoints: np.ndarray,  # shape: (17, 3)
                     xyxy: np.ndarray,  # shape: (4,)
                     detection_target_list: List[List[Union[Tuple[str, str], str]]],  # {list: 858}
                     pkg_classifier,
                     pkg_phone_detector,
                     device_name: str = "cpu",
                     mode: str = None) -> Union[None, List[float]]:

    # Extract posture classifier and phone detector tools from respective packages.
    classifier_model = pkg_classifier["classifier_model"]
    classifier_func = pkg_classifier["classifier_func"]
    normalize_parameters = pkg_classifier["norm_params"]
    phone_detector_model = pkg_phone_detector["phone_detector_model"]
    phone_detector_func = pkg_phone_detector["phone_detector_func"]

    # Performance
    t_mlp = 0
    t_yolo = 0

    # Global variables:
    _num_value = 0.0                # Arbitrary numeric value slot
    display_str = ""                # String that displays on the screen
    classifier_result_str = ""      # Classification result (in numeric percentage)
    posture_signal = 0              # Default: Not Using. Used to control bbox color.

    # Entry state of the state machine
    classify_state = kcfg.TO_BE_CLASSIFIED

    # Person is out of frame.
    if np.sum(keypoints[:13, 2] < 0.3) >= 5:
        classify_state = kcfg.OUT_OF_FRAME

    # Person not out of frame, but show back.
    if not (classify_state == kcfg.OUT_OF_FRAME):
        l_shoulder_x, r_shoulder_x = keypoints[5][0], keypoints[6][0]
        l_shoulder_s, r_shoulder_s = keypoints[5][2], keypoints[6][2]  # score
        backside_ratio = (l_shoulder_x - r_shoulder_x) / (xyxy[2] - xyxy[0])  # shoulder_x_diff / width_diff
        if r_shoulder_s > 0.3 and l_shoulder_s > 0.3 and backside_ratio < -0.2:  # backside_threshold = -0.2
            _num_value = ((r_shoulder_s + l_shoulder_s) / 2.0 + 1.0) / 2.0
            classify_state = kcfg.BACKSIDE

    # Filter with accordance to STATE.
    # If any of the filtering condition is fulfilled, make the signal to -1.
    # Otherwise, keep the original signal.
    if classify_state == kcfg.TO_BE_CLASSIFIED:
        kas_one_person = translateOneLandmarks(detection_target_list, keypoints, mode)
        # Here, if the person is sus for using phone, signal will be assigned to 1.
        # Otherwise, keep the original 0, i.e., not using.
        start_mlp = time.time()
        classifier_result_str, posture_signal = classifier_func(classifier_model, normalize_parameters, kas_one_person)
        t_mlp = time.time() - start_mlp
    elif classify_state == kcfg.BACKSIDE:
        display_str = f"Back {_num_value:.2f}"
        posture_signal = -1
    elif classify_state == kcfg.OUT_OF_FRAME:
        display_str = f"Out Of Frame"
        posture_signal = -1

    # Real classification logic starts here.
    # Posture model finds the posture suspicious.
    # Invokes YOLO for further detection.
    if posture_signal == 0:
        classify_state = kcfg.NOT_USING
        display_str = "- " + classifier_result_str
    elif posture_signal == 1 and phone_detector_model is not None:
        # Set suspicious
        classify_state = kcfg.SUSPICIOUS
        display_str = "? " + classifier_result_str

        phone_detect_signal = 0

        # TODO: If two hand frame is too close, merge to one.
        # TODO: Size logic still not optimized.
        bbox_w, bbox_h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]

        """ Crop two hands """
        hand_hw = (int(bbox_w * 0.7), int(bbox_w * 0.7))      # Only relate to width of bbox
        """Height and width (sequence matter) of the bounding box."""

        # Landmark index of left & right hand: 9, 10
        lhand_center, rhand_center = keypoints[9][:2], keypoints[10][:2]

        # Landmark of left & right elbow: 7 & 8
        # Vectors for left & right arm.
        l_arm_vect, r_arm_vect = keypoints[9][:2] - keypoints[7][:2], keypoints[10][:2] - keypoints[8][:2]
        lhand_center += l_arm_vect * 0.8
        rhand_center += r_arm_vect * 0.8

        # Coordinate of left & right hand's cropped frame
        lh_frame_xyxy = cropFrame(frame, lhand_center, hand_hw)
        rh_frame_xyxy = cropFrame(frame, rhand_center, hand_hw)

        # TODO: Use something else than for-loop...
        hand_frames_xyxy = [f for f in [lh_frame_xyxy, rh_frame_xyxy] if f is not None]

        for subframe, subframe_xyxy in hand_frames_xyxy:
            start_yolo = time.time()
            phone_detect_signal = phone_detector_func(phone_detector_model, subframe, device=device_name, threshold=0.3)
            t_yolo = time.time() - start_yolo

            phone_detect_str = "phone" if phone_detect_signal == 2 else "-"
            render_detection_rectangle(frame, phone_detect_str, subframe_xyxy, signal=phone_detect_signal)

            if phone_detect_signal == 2:
                break

        if phone_detect_signal == 2:  # TODO: face_detection model
            # Set UI to display using logic
            classify_state = kcfg.USING
            posture_signal = 2
            display_str = "+ " + classifier_result_str

            """ Crop Face """
            face_len = int((keypoints[4][0] - keypoints[3][0]) * 1.1)
            face_hw = (face_len, face_len)
            face_center = keypoints[0][:2]

            face_frame, face_xyxy = cropFrame(frame, face_center, face_hw)
            face_detect_str = "="

            render_detection_rectangle(frame, face_detect_str, face_xyxy, signal=2)

    render_detection_rectangle(frame, display_str, xyxy, signal=posture_signal)
    return [t_mlp, t_yolo]


def classify(classifier_model: List[Union[MLP, Dict[str, float]]],
             # TODO: lazy tag
             numeric_data: List[Union[float, np.float32]]) -> Tuple[str, int]:
    model, params = classifier_model

    # Normalize Data
    input_data = np.array(numeric_data).reshape(1, -1)
    input_data[:, ::2] /= 180
    mean_X = params['mean_X']
    std_dev_X = params['std_dev_X']
    input_data = (input_data - mean_X) / std_dev_X
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.view(input_data.shape[0], 6, -1)

    # a = input_data.shape

    with torch.no_grad():
        outputs = model(input_tensor)
        sg = torch.sigmoid(outputs[0])
        prediction = int(sg[0] < sg[1] or sg[1] > 0.48)
        # prediction = torch.argmax(sg, dim=0).item()

    out0, out1 = sg
    classify_signal = 1 if prediction != 1 else 0
    classifier_result_str = f"+ {out1:.2f}" if (prediction == 1) else f"- {out0:.2f}"

    return classifier_result_str, classify_signal


def classify3D(classifier_model: MLP,
               normalize_parameters: Dict[str, float],
               numeric_data: List[Union[float, np.float32]]) -> Tuple[str, int]:
    # Normalize
    input_data = np.array(numeric_data)
    input_data[0, :, :, :] /= 180
    mean_X = normalize_parameters['mean_X']
    std_dev_X = normalize_parameters['std_dev_X']
    input_data = (input_data - mean_X) / std_dev_X

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.permute(0, 3, 1, 2)       # Convert from (Cin, H, W, D) to (Cin, D, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(global_device)  # Add a "batch" dimension for the model: (N, C, D, H, W)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        sg = torch.sigmoid(outputs[0])
        prediction = int(sg[0] < sg[1] or sg[1] > 0.42)
        # prediction = torch.argmax(sg, dim=0).item()

    # out0: Conf for "using"; out1: conf for "not using".
    out0, out1 = sg
    # Note: prediction=0 => classify_signal=1 (Using); prediction=1 => classify_signal=0 (Not using).
    classify_signal = 0 if prediction != 1 else 1
    classifier_result_str = f"{out1:.2f}" if (prediction == 1) else f"{out0:.2f}"

    return classifier_result_str, classify_signal


def detectPhone(model: YOLO, frame: np.ndarray, device: str = 'cpu', threshold: float = 0.2):
    cv2.imwrite("./logs/inital_frame.png", frame)
    empty_frame = np.zeros([640, 640, 3])
    h, w, _ = frame.shape

    if h > 640:
        start_clip_h = (640 - h) // 2
        h = 640
        frame = frame[start_clip_h:start_clip_h + h, :, :]

    if w > 640:
        start_clip_w = (640 - w) // 2
        w = 640
        frame = frame[:, start_clip_w:start_clip_w + w, :]

    start_put_h, start_put_w = (640 - h) // 2, (640 - w) // 2
    empty_frame[start_put_h:start_put_h + h, start_put_w:start_put_w + w] = frame

    # cv2.imwrite("./logs/frame.png", frame)
    # cv2.imwrite("./logs/empty_frame.png", empty_frame)

    # resized_frame = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    resized_frame = empty_frame
    tensor_frame = torch.from_numpy(resized_frame).float() / 255.0
    tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0).to(device)

    results_tmp = model(tensor_frame)
    results_tensor = results_tmp[0]
    results_cls = results_tensor.boxes.cls.cpu().numpy().astype(np.int32)

    if not any(results_cls == 67):
        return 0    # Not using phone

    results_conf = results_tensor.boxes.conf.cpu().numpy().astype(np.float32)[results_cls == 67]

    # 2 stands for positive now
    return 2 if any(results_conf > threshold) else 0


if __name__ == '__main__':
    # Configuration
    is_remote, video_source, use_mmpose_visualizer = getUserConsoleConfig(max_required_num=3)
else:
    is_remote, video_source, use_mmpose_visualizer = False, 0, False

# Decision on mode
# solution_mode = 'hyz'
solution_mode = 'mjj'

# Initialize MMPose essentials
bbox_detector, pose_estimator, visualizer = getMMPoseEssentials()

# List of detection targets
target_list = kcfg.get_targets(solution_mode)

# Classifier Model
if solution_mode == 'hyz':
    model_state = torch.load('./data/models/posture_mmpose_vgg1d_17315770488631685.pth', map_location=global_device)
    classifier = MLP(input_channel_num=6, output_class_num=2)
else:   # elif solution_mode == 'mjj':
    model_state = torch.load('./data/models/posture_mmpose_vgg3d_17349570075562594.pth', map_location=global_device)
    classifier = MLP3d(input_channel_num=2, output_class_num=2)

classifier.load_state_dict(model_state['model_state_dict'])
classifier.eval()
classifier.to(global_device)

norm_params = {
    'mean_X': model_state['mean_X'].item(),
    'std_dev_X': model_state['std_dev_X'].item()
}

classifier_function = classify if solution_mode == 'hyz' else classify3D

# YOLO object detection model
# best_pt_path_main = "step03_yolo_phone_detection/archived onnx/best.pt"
# phone_detector = YOLO(best_pt_path_main)
phone_detector = YOLO("step03_yolo_phone_detection/non_tuned/yolo11m.pt")

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
    "phone_detector_func": detectPhone
}

# Start the loop
demo_performance = videoDemo(src=int(video_source) if video_source is not None else 0,

                             pkg_mmpose=package_mmpose,
                             pkg_classifier=package_classifier,
                             pkg_phone_detector=package_phone_detector,

                             device_name=global_device_name,
                             mode=solution_mode,
                             websocket_obj=ws)


plot_report(
    arrays=np.array(list(demo_performance.values()))[:, 1:],
    labels=["mmpose", "mlp", "yolo"],
    config={"title": "Performance Report", "x_name": "Frame", "y_name": "Time"},
    plot_mean=True
)
