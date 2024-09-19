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
        if not ret or cv2.waitKey(1) == 27:
            break

        if images_dir is None:
            continue

        cv2.imshow("Disassembling Video", frame)
        file_path = os.path.join(images_dir, os.path.basename(video_path)) + "_" + str(frame_idx) + ".jpg"
        cv2.imwrite(file_path, frame)
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
    disassembleOneVideo(video_path="../data/blob/videos/using/20240919_1517_xyl_U_A.mp4",
                        images_dir="../data/train/img_from_video/using")
    disassembleOneVideo(video_path="../data/blob/videos/using/20240919_1523_xyl_U_A.mp4",
                        images_dir="../data/train/img_from_video/using")
    disassembleOneVideo(video_path="../data/blob/videos/not_using/20240919_1527_xyl_N_A.mp4",
                        images_dir="../data/train/img_from_video/not_using")
