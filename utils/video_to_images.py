import cv2
import os
import time


def video_to_images(video_path: str, images_dir: str, max_frames: int) -> None:
    """
    Converts a video file into a series of .jpg images.
    :param video_path: Path to the video file.
    :param images_dir: The directory that stores the output images.
    :param max_frames: Maximum amount of frames allowed to store.
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(images_dir, str(frame_idx) + "_" + str(time.time()) + ".jpg"), frame)
            frame_idx += 1
        else:
            continue

    cap.release()


if __name__ == '__main__':
    video_to_images(video_path="../data/demo/demo_video.mp4",
                    images_dir="../data/images_from_video",
                    max_frames=100)
