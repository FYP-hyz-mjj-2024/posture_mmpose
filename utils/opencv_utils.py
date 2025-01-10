import cv2
import base64
import json
import websocket
from typing import Union, Tuple, List
import time
import numpy as np


def render_detection_rectangle(frame, text, xyxy, signal: int = 1):
    """
    Render a common YOLO detection rectangle onto a frame with opencv.

    :param frame: The video/stream frame to render onto.
    :param text: The description of the detection target.
    :param xyxy: The coordinates of the rectangle (x1, y1, x2, y2).
    :param signal: The integer signal that helps to choose the color of rectangle:
                      1: green (not_using)
                      0: red (using)
                     -1: gray (backside)
    :returns: None.
    """
    color_dict = {0: (0, 255, 0),  # green: not_using
                  1: (51, 140, 232),  # orange: suspicious
                  2: (0, 0, 255),  # red: using
                  -1: (155, 155, 155),  # gray: do not classify
                  }  # BGR form
    rec_thickness_dict = {0: 2,  # green: not_using
                          1: 2,  # red: using
                          2: 2,  # orange: suspicious
                          -1: 2,  # gray: don't classify
                          }

    cv2.putText(
        frame,
        text,
        org=(int(xyxy[0]), int(xyxy[1])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color_dict[signal],
        thickness=rec_thickness_dict[signal]
    )
    cv2.rectangle(
        frame,
        pt1=(int(xyxy[0]), int(xyxy[1])),
        pt2=(int(xyxy[2]), int(xyxy[3])),
        color=color_dict[signal],
        thickness=rec_thickness_dict[signal]
    )


def getUserConsoleConfig(max_required_num=3):
    # Video Source Selection
    video_source = input(f"Which camera would you like to use? > ")

    # Remote stream pushing / Local running
    is_remote = input("Push video frame to remote? [y/n] > ") == 'y'

    # Use mmpose visualizer
    use_mmpose_visualizer = input("Use MMPose visualizer? [y/n] > ") == 'y'

    return is_remote, video_source, use_mmpose_visualizer


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