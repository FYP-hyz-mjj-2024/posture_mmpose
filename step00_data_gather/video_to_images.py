import cv2
import os
import time


def disassembleOneVideo(video_path: str, images_dir: str = None, max_frames: int = None) -> None:
    """
    Converts a video file into a series of .jpg images.
    :param video_path: Path to the video file.
    :param images_dir: The directory that stores the output images.
    :param max_frames: Maximum amount of frames allowed to store.
    :return:
    """
    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened() and (max_frames is None or len(frames) < max_frames):
        ret, frame = cap.read()
        if cv2.waitKey(1) == 27:
            break

        if not ret or (images_dir is None):
            continue

        frames.append(frame)

        cv2.imshow("Disassembling Video", frame)
    cap.release()

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(images_dir, str(idx) + "_" + str(time.time()) + ".jpg"), frame)


def disassembleMultipleVideos(video_dir: str, images_dir: str, max_frames_each=None) -> None:
    """
    Disassemble multiple images.
    :param video_dir: Directory that stores multiple videos.
    :param images_dir: Directory to store multiple image frames.
    :param max_frames_each: For each video, the maximum frame that can be stored.
    """
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if not file.endswith(".mp4"):
                continue
            disassembleOneVideo(video_path=os.path.join(root, file),
                                images_dir=images_dir,
                                max_frames=max_frames_each)


if __name__ == '__main__':
    disassembleOneVideo(video_path="../data/blob/videos/240916_1616_mjj.mp4",
                        images_dir="../data/train/img_from_video/using",
                        max_frames=1000)
