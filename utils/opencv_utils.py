import cv2
import base64
import json
import websocket
from typing import Union
import time


def render_detection_rectangle(frame, text, xyxy, is_ok=True):
    """
    Render a common YOLO detection rectangle onto a frame with opencv.
    :param frame: The video/stream frame to render onto.
    :param text: The description of the detection target.
    :param xyxy: The coordinates of the rectangle.
    :returns: None.
    """
    cv2.putText(
        frame,
        text,
        org=(int(xyxy[0]), int(xyxy[1])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0) if is_ok else (0, 0, 255),
        thickness=2
    )
    cv2.rectangle(
        frame,
        pt1=(int(xyxy[0]), int(xyxy[1])),
        pt2=(int(xyxy[2]), int(xyxy[3])),
        color=(0, 255, 0) if is_ok else (0, 0, 255),
        thickness=2
    )


def yield_video_feed(frame_to_yield, mode='local', title="", ws=None) -> None:
    """
    Yield the video frame. Either using local mode, which will invoke an
    opencv imshow window, or use the HTTP Streaming to the server.
    :param frame_to_yield: The video frame.
    :param mode: Yielding mode, either be 'local' or 'remote'.
    :param title: The title of the local window.
    :param ws: The websocket object initialized with server_url.
    """
    if mode == 'local':
        cv2.imshow(title, frame_to_yield)
    elif mode == 'remote':
        if ws is None:
            raise ValueError("WebSocket object is not initialized.")
        # JPEG encode, convert to bytes
        _, jpeg_encoded = cv2.imencode('.jpg', frame_to_yield)
        jpeg_bytes = jpeg_encoded.tobytes()
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        # Send request
        ws.send(json.dumps({'frameBase64': jpeg_base64, 'timestamp': str("{:.3f}".format(float(time.time())))}))
    else:
        raise ValueError("Video yielding mode should be either 'local' or 'remote'.")


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