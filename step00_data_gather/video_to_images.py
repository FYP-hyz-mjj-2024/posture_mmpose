import cv2
import os
import time


def disassembleOneVideo(video_path: str, output_images_dir: str, max_frames: int = None) -> None:
    """
    Converts a video file into a series of .jpg images.
    :param video_path: Path to the video file.
    :param output_images_dir: The directory that stores the output images.
    :param max_frames: Maximum amount of frames allowed to store.
    :return:
    """

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).replace(".mp4", "")

    dedicated_dir = os.path.join(output_images_dir, video_name)
    if not os.path.exists(dedicated_dir):
        os.makedirs(dedicated_dir)

    frame_idx = 0
    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) == 27:
            break

        frame_idx += 1

        if frame_idx % 10 != 0:
            continue

        file_path = os.path.join(dedicated_dir, video_name) + "_" + str(frame_idx) + ".jpg"
        cv2.imwrite(file_path, frame)

        cv2.imshow(f"Disassembling Video:{video_name}", frame)

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
                                output_images_dir=images_dir,
                                max_frames=max_frames_each)


if __name__ == '__main__':
    # disassembleOneVideo(video_path="../data/blob/videos/using/20240926_1509_mjj_UN-Vert_WN-Wiggle_100.mp4",
    #                     output_images_dir="../data/train/img_from_video",
    #                     max_frames=100)
    disassembleMultipleVideos(video_dir="../data/blob/videos/using",
                              images_dir="../data/train/img_from_video")