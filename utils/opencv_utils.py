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

    :param frame: The video/stream frame to render onto.
    :param text: The description of the detection target, e.g. detection label.
    :param xyxy: The coordinates of the rectangle (x1, y1, x2, y2).
    :param color: The color string.
    :returns: None.
    """

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
    :param frame: Image frame.
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
    :param frame_to_yield: The video frame.
    :param title: The title of the local window.
    :param ws: The websocket object initialized with server_url.
    """
    if ws is None:
        cv2.imshow(title, frame_to_yield)
    else:
        # JPEG encode, convert to bytes
        _, jpeg_encoded = cv2.imencode('.jpg', frame_to_yield)
        jpeg_bytes = jpeg_encoded.tobytes()
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        # Send request
        ws.send(json.dumps({'frameBase64': jpeg_base64, 'timestamp': str("{:.3f}".format(float(time.time())))}))


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
        print("Failed to announce face as the list of encoded frames is empty.")
        return

    print("Announcing Face.")
    ws.send(json.dumps({'announced_frames': encoded_frames, 'timestamp': str("{:.3f}".format(float(time.time())))}))


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
        print(f"Connection to WebSocked Failed. The server might be closed. Error: {e}\n"
              f"If you are using local mode, you can ignore this error.")
        return None


def cropFrame(frame: np.ndarray,
              ct_xy: np.ndarray,
              crop_hw: Tuple[int, int]) -> Union[Tuple[np.ndarray, List], None]:
    """
    Crop out sub-frames from a large frame.
    :param frame: The frame.
    :param ct_xy: Center (x,y) coordinates.
    :param crop_hw: Sub-frame size of (height, width).
    :return:
    """
    fh, fw, _ = frame.shape
    x, y = ct_xy
    ch, cw = crop_hw

    if not (0 <= x <= fw and 0 <= y <= fh):
        return None

    xs = np.array([x - (cw // 2), x + (cw // 2)]).astype(np.int32)
    ys = np.array([y - (ch // 2), y + (ch // 2)]).astype(np.int32)

    np.clip(xs, 0, fw, out=xs)
    np.clip(ys, 0, fh, out=ys)

    xyxy = [xs[0], ys[0], xs[1], ys[1]]

    return frame[ys[0]:ys[1], xs[0]:xs[1], :], xyxy
