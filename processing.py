# Basic
import os
import time
from typing import List, Union, Tuple, Dict

# Simple package
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Local packages
from step01_annotate_image_mmpose.annotate_image import translateOneLandmarks
from step01_annotate_image_mmpose.configs import keypoint_config as kcfg
from step02_train_model_cnn.train_model import normalize
from step03_yolo_phone_detection.dvalue import yolo_input_size
from utils.opencv_utils import render_detection_rectangle, cropFrame, resizeFrameToSquare, relativeToAbsolute
from utils.decorations import CONSOLE_COLORS as CC

# Hardware devices
global_device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
global_device = torch.device(global_device_name)


def processOnePerson(frame: np.ndarray,             # shape: (H, W, 3)
                     original_frame: np.ndarray,    # shape: same as frame
                     keypoints: np.ndarray,         # shape: (13, 3)
                     xyxy: np.ndarray,              # shape: (4,)
                     detection_target_list: List[List[Union[Tuple[str, str], str]]],  # {list: 858}
                     pkg_classifier,
                     pkg_phone_detector,
                     runtime_parameters,  # Save runtime handframes, crop face frame, etc.
                     device_name: str = "cpu") -> Dict[str, Union[Tuple[float, float], float, np.ndarray]]:
    """
    Infer one pedestrian. Use a state machine to perform two-layer detection, and yield the rendered result
    to the destination source.
    :param frame: One video frame in array. Shape (height, weight, channels=3). BGR format.
    :param original_frame: The original copy of frame that's never rendered anything on. BGR format.
    :param keypoints: A list of key upper-body key points, each being a list of x, y and confidence
                      score. Shape: (13, 3).
    :param xyxy: The list of bounding box diagonal coordinates. Shape: (4,).
    :param detection_target_list: List of detection targets.
    :param pkg_classifier: Package object for posture recognition.
    :param pkg_phone_detector: Package object for cell-phone detection.
    :param runtime_parameters: Runtime parameters that records various running states of the system.
    :param device_name: Name of the hardware device.
    :return: Evaluated time for posture recognition and object detection at this pedestrian at this frame.
    """

    # Posture Recognition.
    classifier_model = pkg_classifier["classifier_model"]
    classifier_func = pkg_classifier["classifier_func"]
    classifier_conf = pkg_classifier["pose_conf"]
    # normalize_parameters = pkg_classifier["norm_params"]

    # Cell Phone Detection
    phone_detector_model = pkg_phone_detector["phone_detector_model"]
    phone_detector_func = pkg_phone_detector["phone_detector_func"]
    self_trained = pkg_phone_detector["self_trained"]
    face_announce_interval = pkg_phone_detector["face_announce_interval"]
    phone_conf = pkg_phone_detector["phone_conf"]
    spareness = pkg_phone_detector["spare"]

    # Runtime options
    runtime_save_handframes_path = runtime_parameters["path_runtime_handframes"]
    time_last_announce_face = runtime_parameters["time_last_announce_face"]
    time_frame_start = runtime_parameters["time_last_record_framerate"]     # Constant for all people in this frame

    # Performance
    t_mlp = 0
    t_yolo = 0

    # Global variables:
    # Runtime: Change along with state machine.
    classifier_result_str = ""      # Classification result (in numeric percentage)
    # Constant
    face_center = keypoints[0][:2]
    phone_frame_size = yolo_input_size if self_trained else 640     # Yolo input frame size
    phone_index = 0 if self_trained else 67     # Phone index of model

    # Entry state of the state machine
    _state = kcfg.TO_BE_CLASSIFIED  # Private param within the state machine.

    # Announce Faces.
    announced_face_frame = None

    # Bounding box size.
    bbox_w, bbox_h = abs(xyxy[2] - xyxy[0]), abs(xyxy[3] - xyxy[1])

    # Person is out of frame.
    if _state == kcfg.TO_BE_CLASSIFIED:
        if np.sum(keypoints[:13, 2] < 0.3) >= 5:
            _state = kcfg.OUT_OF_FRAME

    # Person shows backside.
    if _state == kcfg.TO_BE_CLASSIFIED:
        l_shoulder_x, r_shoulder_x = keypoints[5][0], keypoints[6][0]
        l_shoulder_s, r_shoulder_s = keypoints[5][2], keypoints[6][2]  # score
        backside_ratio = (l_shoulder_x - r_shoulder_x) / (xyxy[2] - xyxy[0])  # shoulder_x_diff / width_diff
        if r_shoulder_s > 0.3 and l_shoulder_s > 0.3 and backside_ratio < -0.2:  # backside_threshold = -0.2
            _num_value = ((r_shoulder_s + l_shoulder_s) / 2.0 + 1.0) / 2.0
            classifier_result_str = f"{_num_value:.2f}"
            _state = kcfg.BACKSIDE

    # Still in starting state after filtering.
    if _state == kcfg.TO_BE_CLASSIFIED:
        # Translation of one person's landmarks to targeted key points.
        kas_one_person = translateOneLandmarks(detection_target_list, keypoints)

        # Posture recognition model inference.  0: Not using, 1: Suspicious.
        start_mlp = time.time()
        classifier_result_str, posture_signal = classifier_func(classifier_model, kas_one_person, conf=classifier_conf)

        # Adjust states according to posture recognition results.
        if posture_signal == 0:
            _state = kcfg.NOT_USING
        elif posture_signal == 1:
            _state = kcfg.SUSPICIOUS
        else:
            raise ValueError(f"Invalid posture signal {posture_signal}.")
        t_mlp = time.time() - start_mlp

    # Object detection.
    if _state == kcfg.NOT_USING:
        pass
    elif _state == kcfg.SUSPICIOUS:
        '''
        Phase 1: Retrieve hand centers and their distances to the face center.
        '''
        # 1.1 Crop two hands.
        if bbox_w / (frame.shape[1] + np.finfo(np.float32).eps) < 0.6:
            # 1.1.2 Body is far, make hand frame width & height relate to bbox width.
            hand_hw = (int(bbox_w * 0.7), int(bbox_w * 0.7))
            """Height and width (sequence matter) of the bounding box."""
        else:
            # 1.1.3 Body takes up too much space, restrict hand frame size to 0.45 * frame width.
            hand_hw = (int(frame.shape[1] * 0.45), int(frame.shape[1] * 0.45))
            """Height and width (sequence matter) of the bounding box."""

        # 1.2 L & R wrists: 9, 10; L & R elbows: 7, 8.
        lwrist_coord, rwrist_coord = keypoints[9][:2], keypoints[10][:2]
        lelbow_coord, relbow_coord = keypoints[7][:2], keypoints[8][:2]

        # 1.3 L & R arm vectors.
        l_arm_vect, r_arm_vect = lwrist_coord - lelbow_coord, rwrist_coord - relbow_coord

        # 1.4 L & R hand center coordinates.
        lhand_center = lwrist_coord + l_arm_vect * 0.8
        rhand_center = rwrist_coord + r_arm_vect * 0.8

        # 1.5 L & R wrist to face center distances.
        lhand_face_dist = np.linalg.norm(lwrist_coord - face_center)
        rhand_face_dist = np.linalg.norm(rwrist_coord - face_center)
        spare_dist_ratio = 1    # A ratio of two distance.
        """
        Spare distance ratio is the ratio of the two distances between the two hands to the face center.
        Under non-strict mode, the smaller the spare ratio, the less likely the other hand is under
        engagement.
        """

        '''
        Phase 2: Decide the primary hand with respect to distances to face.
        '''
        if np.linalg.norm(lhand_center - rhand_center) > 0.21 * bbox_w:
            # Data structure: [frame: np.ndarray, frame_xyxy:List[int]]
            lhand_frame_xyxy = cropFrame(original_frame, lhand_center, hand_hw)
            rhand_frame_xyxy = cropFrame(original_frame, rhand_center, hand_hw)

            # 2.1.1 Calculate the spare ratio. Don't bother to do this if not in strict mode.
            try:
                spare_dist_ratio = (lhand_face_dist / (rhand_face_dist + np.finfo(np.float32).eps)
                                    if lhand_face_dist <= rhand_face_dist
                                    else rhand_face_dist / (lhand_face_dist + np.finfo(np.float32).eps))
            except (TypeError, ZeroDivisionError) as e:
                # No matter for what reason the above calculation failed, just allow pass.....
                # TypeError: A None type is minus-ed or multiplied. (e.g., None + 1, 1 + None, None - None, etc.)
                # ZeroDivisionError: I sincerely don't know whether this will happen, but just handle it. ;)
                pass

            print(f"{CC['yellow']}{spare_dist_ratio}{CC['reset']}")

            # 2.1.2 If the wrist of one hand is closer to face, that hand is the primary. The other is secondary.
            prmhand_frame_xyxy, sndhand_frame_xyxy = (
                (lhand_frame_xyxy, rhand_frame_xyxy)
                if lhand_face_dist < rhand_face_dist
                else (rhand_frame_xyxy, lhand_frame_xyxy)
            )
        else:
            # 2.2 If two hands are close enough, merge together to be the primary. The secondary is None.
            prmhand_frame_xyxy = cropFrame(original_frame, (lhand_center + rhand_center) // 2, hand_hw)
            sndhand_frame_xyxy = None

        '''
        Phase 3: YOLO inference primary first. If not detected, inference secondary.
        '''
        start_yolo = time.time()
        for idx, hand_frame_xyxy in enumerate([prmhand_frame_xyxy, sndhand_frame_xyxy]):
            # This for-loop will only run at most two iterations.

            # 3.0 Filter out undetectable cases.
            if hand_frame_xyxy is None or not isinstance(hand_frame_xyxy, Tuple):
                # 3.0.1 Guard 1: Make subframe_xyxy expandable to frame & xyxy.
                continue

            hand_frame, hand_xyxy = hand_frame_xyxy   # hand_frame: BGR

            if len(hand_frame_xyxy) <= 0 or hand_frame is None or hand_xyxy is None:
                # 3.0.2 Guard 2: hand_frame_xyxy is expandable, and any of its content shouldn't be None.
                print(f"{CC['yellow']}Pedestrian too close to detect.{CC['reset']}")
                continue

            # 3.1 Let YOLO infer and get result.
            # Default result.
            phone_detect_str, phone_relative_xyxy, subframe_wh, hand_display_color = "", None, (0, 0), "green"
            try:
                # Record subframe size (this value won't change even if subframe is resized).
                subframe_wh = abs(hand_xyxy[2] - hand_xyxy[0]), abs(hand_xyxy[3] - hand_xyxy[1])

                # Resize subframe to YOLO required size.
                # Below ratio: Direct stretch. After ratio: crop.
                hand_frame = resizeFrameToSquare(frame=hand_frame,
                                                 edge_length=phone_frame_size,
                                                 ratio_threshold=0.5625)     # 9 / 16

                # Convert BGR subframe to RGB for YOLO inference.
                hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)

                # Detection string and the relative bbox xyxy of cell phone.
                phone_detect_str, phone_relative_xyxy = phone_detector_func(phone_detector_model, hand_frame,
                                                                            device=device_name, threshold=phone_conf,
                                                                            cell_phone_index=phone_index)

            except (cv2.error, IOError, TypeError) as e:     
                # hand_frame and hand_xyxy could possibly be mal-shaped, which is undetectable.
                # Simply fall-out to undetected state.
                print(f"{CC['yellow']}"
                      f"Error in detectPhone: Failed inferring hand frame at this point, skipping to the next frame.\n"
                      f"{e}"
                      f"{CC['reset']}")

            # 3.2 Render hand frames and YOLO inference frame based on the signal.
            if phone_relative_xyxy is not None:
                _state = kcfg.USING     # Trigger state change.
                hand_display_color = kcfg.state_display_type[_state]["color"]

                # Convert the relative cell phone xyxy to the absolute one.
                phone_absolute_xyxy = relativeToAbsolute(
                    from_mother_wh=(phone_frame_size, phone_frame_size),
                    to_mother_wh=subframe_wh,
                    from_child_xyxy=phone_relative_xyxy,
                    to_mother_xy=hand_xyxy[:2]
                )

                # Render the YOLO inference of cell phone.
                render_detection_rectangle(frame, phone_detect_str, phone_absolute_xyxy, color=hand_display_color)

            # Render the hand frame.
            render_detection_rectangle(frame, f"Hand {idx}", hand_xyxy, color=hand_display_color)

            # 3.3 Press mouse to save supplementary dataset on the go.
            if runtime_save_handframes_path is not None:
                try:
                    image_file_name = f"{time.strftime('%Y%m%d-%H%M%S')}_runtime.png"
                    save_path = os.path.join(runtime_save_handframes_path, image_file_name)
                    cv2.imwrite(save_path, cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB))
                    print(f"{CC['green']}"
                          f"Saved runtime handframes at {time.strftime('%Y-%m-%d %H:%M:%S')}."
                          f"{CC['reset']}")
                except Exception as e:
                    print(f"{CC['yellow']}"
                          f"Failed to save current hand frame. Exception: {e}"
                          f"{CC['reset']}")

            # Stopping criteria.
            if ((phone_relative_xyxy is not None)
                    or (spare_dist_ratio < spareness)):
                # A. If the primary hand is already holding a phone, don't detect another.
                # B. Not in strict mode. If the primary hand is not holding a phone, and it
                # is far away from the secondary hand, then spare the secondary hand.
                break

        t_yolo = time.time() - start_yolo

    if _state == kcfg.USING:
        # Edge length of the face sub-frame
        face_len = max(abs(int((keypoints[4][0] - keypoints[3][0]) * 1.1)), 0.3 * bbox_w)
        face_hw = (face_len, face_len)  # Dimensions of the face sub-frame.

        # Face subframe and xyxy.
        face_frame, face_xyxy = cropFrame(original_frame, face_center, face_hw)
        face_detect_str = "Face"

        if face_frame is not None and face_xyxy is not None:    # In case pedestrian is out of frame.
            # Diff between time of this frame and last announce face time is longer than the interval.
            # Note: For all person in a single frame (i.e., each call of this function), time_frame_start is all same.
            if time_frame_start - time_last_announce_face > face_announce_interval:
                announced_face_frame = face_frame

            # Render face frame.
            render_detection_rectangle(frame, face_detect_str, face_xyxy, color="white")

    # End of state machine. The _state is finalized.
    # Get display color and string according to _state.
    color = kcfg.state_display_type[_state]["color"]
    display_str = f"{kcfg.state_display_type[_state]['str']} {classifier_result_str}"

    # Render inference results for this specific person onto the frame.
    render_detection_rectangle(frame, display_str, xyxy, color=color)

    return {
        "performance": (t_mlp, t_yolo),
        "announced_face_frame": announced_face_frame,   # np.ndarray or None, depending on the interval.
    }


def classify3D(classifier_model,
               numeric_data: List[Union[float, np.float32]],
               conf=0.75) -> Tuple[str, int]:
    """
    Use the posture recognition model to classify a pedestrian's pose.
    :param classifier_model: The trained posture recognition model instance.
    :param numeric_data: 2-channel 3-d structured posture angle-score data.
    :param conf: The threshold confidence.
    :return: Classification result string and the binary classification signal.
    """
    # Normalize
    input_data = np.array([numeric_data])   # Add a "batch" dimension for the model: (N, C, D, H, W)
    input_data = normalize(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.permute(0, 1, 4, 2, 3)       # Convert from (N, C, H, W, D) to (N, C, D, H, W)
    input_tensor = input_tensor.to(global_device)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        sg = torch.sigmoid(outputs[0])
        prediction = sg[1] > sg[0] and sg[1] > conf

    out0, out1 = sg
    classify_signal = 0 if prediction != 1 else 1
    classifier_result_str = f"{out1:.2f}" if (prediction == 1) else f"{out0:.2f}"

    return classifier_result_str, classify_signal


def detectPhone(model: YOLO, frame: np.ndarray,
                device: str = 'cpu', threshold: float = 0.2,
                cell_phone_index: int = 0):
    """
    Infers the cropped hand frame of a pedestrian and use a YOLO model to detect the existence of a cell-phone.
    :param model: YOLO model of arbitrary variant.
    :param frame: Frame array in shape [height, width, channels], in RGB format.
    :param device: Device string to use for inference.
    :param threshold: Minimum confidence of phone detection to output a positive result.
    :param cell_phone_index: Index of cell phone in YOLO inference result.
    :return: Detection result string and cell phone's relative xyxy.
    """
    # Move resized frame to tensor
    tensor_frame = torch.from_numpy(frame).float() / 255.0
    tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0).to(device)    # Add a "batch" dimension

    # Model inference result.
    results_tmp = model(tensor_frame)
    results_tensor = results_tmp[0]
    results_cls = results_tensor.boxes.cls.cpu().numpy().astype(np.int32)

    if not any(results_cls == cell_phone_index):
        # No bbox with class of cell phone.
        return "", None

    # If the code run into here, there must be a cell phone.
    # P.S. 67 is the index of "cell phone" in the non-tuned model
    result_confs = results_tensor.boxes.conf.cpu().numpy().astype(np.float32)[results_cls == cell_phone_index]
    relative_xyxys = results_tensor.boxes.data.cpu().numpy().astype(np.float32)[results_cls == cell_phone_index]

    # Find the most confident detection of cell phone.
    max_conf_index = np.argmax(result_confs)
    result_conf = result_confs[max_conf_index]

    # If even the most confident one is lower than the threshold,
    # regard it as "no phone" as well.
    if result_conf < threshold:
        return "", None

    # The detection is confident enough, report.
    detection_result_str = f"Phone: {max(result_confs):.3f}"
    relative_xyxy = relative_xyxys[max_conf_index][:4]

    return detection_result_str, relative_xyxy
