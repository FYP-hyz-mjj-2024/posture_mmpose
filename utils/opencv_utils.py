# Basic
from typing import Union, Tuple, List
import time

# Utilities
import numpy as np
import cv2

# Stream Pushing
import base64
import json
import websocket

# Local
from .decorations import CONSOLE_COLORS as CC

color_bgr = {
    "green": (0, 255, 0),
    "orange": (51, 140, 232),
    "red": (0, 0, 255),
    "gray": (155, 155, 155),
    "white": (255, 255, 255)
}

color_thickness = {
    "green": 2,
    "orange": 2,
    "red": 2,
    "gray": 2,
    "white": 2
}

ui_text_styles = {
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale": 0.5,
    "color": color_bgr["white"],
    "thickness": 2
}

detection_rect_styles = {
    "thickness": 2
}


def render_detection_rectangle(frame, text: str, xyxy: List[float], color: str):
    """
    Render a common YOLO detection rectangle onto a frame with opencv.

    :param frame: The video/stream frame to render onto in BGR format.
    :param text: The description of the detection target, e.g. detection label.
    :param xyxy: The coordinates of the rectangle (x1, y1, x2, y2).
    :param color: The color string.
    :returns: None.
    """

    if xyxy is None:
        print(f"{CC['yellow']}"
              f"Error in render_detection_rectangle: "
              f"Coordinates of two vertex xyxy is None."
              f"Skipping to the next frame."
              f"{CC['reset']}")
        return

    # Label Text
    cv2.putText(
        frame,
        text,
        org=(int(xyxy[0]), int(xyxy[1])-5),
        fontFace=ui_text_styles["fontFace"],
        fontScale=0.6,
        color=color_bgr[color],
        thickness=ui_text_styles["thickness"]     # TODO:
    )

    # Rectangle
    cv2.rectangle(
        frame,
        pt1=(int(xyxy[0]), int(xyxy[1])),
        pt2=(int(xyxy[2]), int(xyxy[3])),
        color=color_bgr[color],
        thickness=detection_rect_styles["thickness"]
    )


def render_ui_text(frame, text: str,
                   frame_wh: Tuple[int, int], margin_wh: Tuple[int, int],
                   align: str, order: int):
    """
    Render text on image frame as UI.
    :param frame: The video/stream frame to render onto in BGR format.
    :param text: Text content.
    :param frame_wh: Frame size.
    :param margin_wh: Frame margin size.
    :param align: Align method: left or right.
    :param order: Align order that starts with 0.
    :return:
    """
    frame_w, frame_h = frame_wh
    margin_w, margin_h = margin_wh

    (text_width, text_height), _ = cv2.getTextSize(text=text,
                                                   fontFace=ui_text_styles["fontFace"],
                                                   fontScale=ui_text_styles["fontScale"],
                                                   thickness=ui_text_styles["thickness"])

    if align == "left":
        org = (margin_w, margin_h + order * (text_height + 5))
    elif align == "right":
        org = (frame_w - margin_w - int((0.01 if frame_w <= 600 else 1) * text_width),
               margin_h + order * (text_height + 5))
    else:
        raise ValueError("Unrecognized alignment: align parameter should be either 'left' or 'right'.")

    cv2.putText(
        frame,
        text,
        org=org,
        fontFace=ui_text_styles["fontFace"],
        fontScale=ui_text_styles["fontScale"],
        color=ui_text_styles["color"],
        thickness=ui_text_styles["thickness"]
    )


def getUserConsoleConfig():
    """
    Get user selections as configuration details for this runtime.
    :return:
    """
    # Video Source Selection
    video_source = input(f"Which camera would you like to use? > ")

    # Remote stream pushing / Local running
    is_remote = input("Push video frame to remote? [y/n] > ") == 'y'

    # Use mmpose visualizer
    use_mmpose_visualizer = input("Use MMPose visualizer? [y/n] > ") == 'y'

    # Use trained YOLO11 or un-tuned.
    use_trained_yolo = input("Use self-trained YOLO model? [y/n] > ") == 'y'

    return is_remote, video_source, use_mmpose_visualizer, use_trained_yolo


def yieldVideoFeed(frame_to_yield, title="", ws=None) -> None:
    """
    Yield the video frame. Either using local mode, which will invoke an
    opencv imshow window, or use the HTTP Streaming to the server.
    :param frame_to_yield: The video frame in BGR format.
    :param title: The title of the local window.
    :param ws: The websocket object initialized with server_url.
    """
    if ws is not None:
        cv2.imshow(title, frame_to_yield)
        # JPEG encode, convert to bytes
        _, jpeg_encoded = cv2.imencode('.jpg', frame_to_yield)
        jpeg_bytes = jpeg_encoded.tobytes()
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        # Send request
        ws.send(json.dumps({'frameBase64': jpeg_base64, 'timestamp': str("{:.3f}".format(float(time.time())))}))

    # No matter send to remote or not, leave a local copy.
    cv2.imshow(title, frame_to_yield)


def announceFaceFrame(face_frames, ws) -> None:
    encoded_frames = []
    for frame in face_frames:
        try:
            _, jpeg_encoded = cv2.imencode('.jpg', frame)
            jpeg_bytes = jpeg_encoded.tobytes()
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            encoded_frames.append(jpeg_base64)
        except Exception as e:
            print(f"Failed to announce face. Error: {e}")

    if len(encoded_frames) <= 0:
        print("Failed to announce face. Error: The list of encoded frames is empty.")
        return

    print(f"Announcing {len(encoded_frames)} face(s).")
    ws.send(json.dumps({'announced_face_frames': encoded_frames, 'timestamp': str("{:.3f}".format(float(time.time())))}))


def init_websocket(server_url) -> Union[websocket.WebSocket, None]:
    """
    Initialize a websocket object using the url of the server.
    :param server_url: The url of the server.
    """
    try:
        ws = websocket.WebSocket()
        ws.connect(server_url)
        return ws
    except ConnectionRefusedError as e:
        print(f"{CC['yellow']}"
              f"Warning: Failed to connect to WebSocket server as it might be closed.\n"
              f"Error: {e}\n"
              f"Fallback: Use local mode instead.\n"
              f"{CC['reset']}")
        return None


def cropFrame(frame: np.ndarray,
              ct_xy: np.ndarray,
              crop_hw: Tuple[int, int]) -> Union[Tuple[np.ndarray, List], Tuple[None, None]]:
    """
    Crop out sub-frames from a large frame.
    :param frame: The original frame to be cropped.
    :param ct_xy: Center (x,y) coordinates of the sub-frame.
    :param crop_hw: Sub-frame size of (height, width).
    :return:
    """
    fh, fw, _ = frame.shape
    x, y = ct_xy
    ch, cw = crop_hw

    if not (0 <= x <= fw and 0 <= y <= fh):
        return None, None

    xs = np.array([x - (cw // 2), x + (cw // 2)]).astype(np.int32)
    ys = np.array([y - (ch // 2), y + (ch // 2)]).astype(np.int32)

    np.clip(xs, 0, fw, out=xs)
    np.clip(ys, 0, fh, out=ys)

    xyxy = [xs[0], ys[0], xs[1], ys[1]]

    return frame[ys[0]:ys[1], xs[0]:xs[1], :], xyxy


def resizeFrameToSquare(frame: np.ndarray,
                        edge_length: int,
                        ratio_threshold: float = 9 / 16) -> np.ndarray:
    """
    Resize a captured frame into a square shape, according to the requirements
    of a YOLO11 input. If the frame height and width exceeds the ratio threshold,
    then use the crop method. Otherwise, use the stretch-to-resize method.
    :param frame: The original frame to be resized.
    :param edge_length: Length of the square edge in pixels.
    :param ratio_threshold: Ratio threshold where the cropping method is decided.
    :return:
    """
    # Parameter constraints.
    if not (0 < ratio_threshold <= 1):
        raise ValueError("resizeFrameToSquare Failed: Ratio threshold should be in (0, 1].")

    if edge_length <= 0:
        raise ValueError(f"resizeFrameToSquare Failed: "
                         f"Edge length should be larger than 0. Inputted value:{edge_length}.")

    # IO constraints.
    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        raise IOError(f"resizeFrameToSquare Failed: "
                      f"Original size of frame is invalid. size: (h={h},w={w}).")

    if ratio_threshold < (h / w) < 1 / ratio_threshold:
        # Aspect ratio is in a reasonable range.
        resized_frame = cv2.resize(frame, (edge_length, edge_length))
    else:
        # Resize original frame to let longer edge to be frame_size.
        scale = edge_length / max(w, h)
        new_h, new_w = int(h * scale), int(w * scale)
        _resized_frame = cv2.resize(frame, (new_w, new_h))

        # Initialize a black frame.
        resized_frame = np.zeros((edge_length, edge_length, 3), dtype=_resized_frame.dtype)

        # Put the scaled original frame into center.
        start_h = round(edge_length / 2 - new_h / 2)
        end_h = start_h + new_h
        start_w = round(edge_length / 2 - new_w / 2)
        end_w = start_w + new_w

        resized_frame[start_h:end_h, start_w:end_w, :] = _resized_frame

    return resized_frame


def relativeToAbsolute(from_mother_wh, to_mother_wh, from_child_xyxy, to_mother_xy=(0,0)):
    """
    Calculate the absolute xyxy from a relative xyxy. Principle:

    `from_x / from_mother_w = to_x / to_mother_w`

    `=> to_x = (to_mother_w / from_mother_w) * from_x`

    :param from_mother_wh: Width and height of the "from" mother frame.
    :param to_mother_wh: Width and height of the "to" mother frame.
    :param from_child_xyxy: Xyxy of the "from" child frame w.r.t. to the "from" mother frame.
    :param to_mother_xy: Top-left corner of the "to" mother frame.
    :return:
    """

    """
    See this to comprehend what it does.
    ┌─m────────────────────┐           ┌─m──────┐   \n
    │      ┌─c───────┐     │           │        │   \n
    │      │         │     │           │   ┌c┐  │   \n
    │      └─────────┘     │           │   │ │  │   \n
    │                      │    ───>   │   └─┘  │   \n
    │                      │           │        │   \n
    └──────────────────────┘           │        │   \n
                                       └────────┘   \n
    """

    # Width & height of the "from" child.
    from_child_wh = abs(from_child_xyxy[2] - from_child_xyxy[0]), abs(from_child_xyxy[3] - from_child_xyxy[1])

    # Top-left corner of the "to" child.
    to_child_xy = [(to_size / (from_size + np.finfo(np.float32).eps)) * from_child_xyxy[i]
                   for i, (from_size, to_size) in enumerate(zip(from_mother_wh, to_mother_wh))]

    # Width & height of the "to" child.
    to_child_wh = [(to_size / (from_size + np.finfo(np.float32).eps)) * from_child_wh[i]
                   for i, (from_size, to_size) in enumerate(zip(from_mother_wh, to_mother_wh))]

    # "from" child's top-left corner should bias along with the "to" mother's top-left corner.
    to_child_xy = [v1 + v2 for v1, v2 in zip(to_child_xy, to_mother_xy)]

    # Xyxy of the "to" child.
    to_child_xyxy = to_child_xy + [v1 + v2 for v1, v2 in zip(to_child_xy, to_child_wh)]

    return to_child_xyxy

