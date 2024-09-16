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

# def disassemble_multiple_videos(video_dir:str, images_dir:str, max_frames_each:List[int]) -> None:
#     pass


if __name__ == '__main__':
    disassembleOneVideo(video_path="../data/demo/demo_video.mp4",
                        images_dir="../data/demo/images_from_video",
                        max_frames=1000)
