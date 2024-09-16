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
    frame_idx = 0
    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if cv2.waitKey(1) == 27:
            break

        if not ret:
            continue

        if images_dir:
            cv2.imwrite(os.path.join(images_dir, str(frame_idx) + "_" + str(time.time()) + ".jpg"), frame)

        cv2.imshow("Disassembling Video", frame)
        frame_idx += 1

    cap.release()


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
    disassembleOneVideo(video_path="../data/demo/demo_video.mp4",
                        images_dir="../data/demo/images_from_video",
                        max_frames=1000)
