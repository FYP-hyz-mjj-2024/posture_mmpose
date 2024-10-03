import re

def parseFileName(video_file_name: str, extension: str):
    """
    Parse file name to get the information about the file.
    Information:

    - capture_date: Date of capture.
    - capture_time: Time of capture.
    - human_model_name: Name of the human model.
    - label: Label of the video. For example: "UN-Horiz-L" means "using phone horizontally with left hand".
    - extensions: Extending postures. For example: "TC-Face-L" means "touching face with left hand".
    - weight: The "Label" for regression, i.e. the confidence that the person is using a cellphone.

    :param video_file_name: Name of the video file.
    :return: An information dictionary of the video.
    """
    parsed = re.split(r'_', video_file_name.replace(extension, ""))

    if len(parsed) not in range(6,8):
        raise Exception("File name is not valid.")

    info = {
        'capture_date': int(parsed[0]),
        'capture_time': int(parsed[1]),
        'human_model_name': str(parsed[2]),
        'label': str(parsed[3]),
        'extensions': str(parsed[4]),
        'weight': float(parsed[5][:1] + "." + parsed[5][1:]),
        'frame_number': int(parsed[6]) if len(parsed) > 6 else None
    }
    return info