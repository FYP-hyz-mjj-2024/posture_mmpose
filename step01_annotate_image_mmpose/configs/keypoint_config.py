import itertools
from typing import List, Tuple, Union, Any

keypoint_names = {
    0: 'Body-Chin',
    1: 'Body-Left_eye',
    2: 'Body-Right_eye',
    3: 'Body-Left_ear',
    4: 'Body-Right_ear',
    5: 'Body-Left_shoulder',
    6: 'Body-Right_shoulder',
    7: 'Body-Left_elbow',
    8: 'Body-Right_elbow',
    9: 'Body-Left_wrist',
    10: 'Body-Right_wrist',
    11: 'Body-Left_hip',
    12: 'Body-Right_hip',
    13: 'Body-Left_knee',
    14: 'Body-Right_knee',
    15: 'Body-Left_ankle',
    16: 'Body-Right_ankle',
    17: 'Foot-Left_toe',
    18: 'Foot-Left_pinky',
    19: 'Foot-Left_heel',
    20: 'Foot-Right_toe',
    21: 'Foot-Right_pinky',
    22: 'Foot-Right_heel',
    23: 'Face-Right_hairroot',
    24: 'Face-Right_zyngo',
    25: 'Face-Right_face_top',
    26: 'Face-Right_face_mid',
    27: 'Face-Right_face_bottom',
    28: 'Face-Right_chin_top',
    29: 'Face-Right_chin_mid',
    30: 'Face-Right_chin_bottom',
    31: 'Face-Chin',
    32: 'Face-Left_chin_bottom',
    33: 'Face-Left_chin_mid',
    34: 'Face-Left_chin_top',
    35: 'Face-Left_face_bottom',
    36: 'Face-Left_face_mid',
    37: 'Face-Left_face_top',
    38: 'Face-Left_zyngo',
    39: 'Face-Left_hairroot',
    40: 'Face-Right_eyebrow_out',
    41: 'Face-Right_eyebrow_out_mid',
    42: 'Face-Right_eyebrow_mid',
    43: 'Face-Right_eyebrow_mid_in',
    44: 'Face-Right_eyebrow_in',
    45: 'Face-Left_eyebrow_in',
    46: 'Face-Left_eyebrow_mid_in',
    47: 'Face-Left_eyebrow_mid',
    48: 'Face-Left_eyebrow_out_mid',
    49: 'Face-Left_eyebrow_out',
    50: 'Face-Nose_top',
    51: 'Face-Nose_top_mid',
    52: 'Face-Nose_bottom_mid',
    53: 'Face-Nose_bottom',
    54: 'Face-Right_nostril_out',
    55: 'Face-Right_nostril_mid',
    56: 'Face-Nostril',
    57: 'Face-Left_nostril_mid',
    58: 'Face-Left_nostril_out',
    59: 'Face-Right_eye_out',
    60: 'Face-Right_eye_up_out',
    61: 'Face-Right_eye_up_in',
    62: 'Face-Right_eye_in',
    63: 'Face-Right_eye_down_in',
    64: 'Face-Right_eye_down_out',
    65: 'Face-Left_eye_in',
    66: 'Face-Left_eye_up_in',
    67: 'Face-Left_eye_up_out',
    68: 'Face-Left_eye_out',
    69: 'Face-Left_eye_down_out',
    70: 'Face-Left_eye_down_in',
    71: 'Face-Lips_l1_right_out',
    72: 'Face-Lips_l1_right_mid',
    73: 'Face-Lips_l1_right_in',
    74: 'Face-Lips_l1_mid',
    75: 'Face-Lips_l1_left_in',
    76: 'Face-Lips_l1_left_mid',
    77: 'Face-Lips_l1_left_out',
    78: 'Face-Lips_l4_left_out',
    79: 'Face-Lips_l4_left_in',
    80: 'Face-Lips_l4_mid',
    81: 'Face-Lips_l4_right_in',
    82: 'Face-Lips_l4_right_out',
    83: 'Face-Lips_l2_right_out',
    84: 'Face-Lips_l2_right_in',
    85: 'Face-Lips_l2_mid',
    86: 'Face-Lips_l2_left_in',
    87: 'Face-Lips_l2_left_out',
    88: 'Face-Lips_l3_left',
    89: 'Face-Lips_l3_mid',
    90: 'Face-Lips_l3_right',
}

keypoint_indexes = {
    'Body-Chin': 0,
    'Body-Left_eye': 1,
    'Body-Right_eye': 2,
    'Body-Left_ear': 3,
    'Body-Right_ear': 4,
    'Body-Left_shoulder': 5,
    'Body-Right_shoulder': 6,
    'Body-Left_elbow': 7,
    'Body-Right_elbow': 8,
    'Body-Left_wrist': 9,
    'Body-Right_wrist': 10,
    'Body-Left_hip': 11,
    'Body-Right_hip': 12,
    'Body-Left_knee': 13,
    'Body-Right_knee': 14,
    'Body-Left_ankle': 15,
    'Body-Right_ankle': 16,
    'Foot-Left_toe': 17,
    'Foot-Left_pinky': 18,
    'Foot-Left_heel': 19,
    'Foot-Right_toe': 20,
    'Foot-Right_pinky': 21,
    'Foot-Right_heel': 22,
    'Face-Right_hairroot': 23,
    'Face-Right_zyngo': 24,
    'Face-Right_face_top': 25,
    'Face-Right_face_mid': 26,
    'Face-Right_face_bottom': 27,
    'Face-Right_chin_top': 28,
    'Face-Right_chin_mid': 29,
    'Face-Right_chin_bottom': 30,
    'Face-Chin': 31,
    'Face-Left_chin_bottom': 32,
    'Face-Left_chin_mid': 33,
    'Face-Left_chin_top': 34,
    'Face-Left_face_bottom': 35,
    'Face-Left_face_mid': 36,
    'Face-Left_face_top': 37,
    'Face-Left_zyngo': 38,
    'Face-Left_hairroot': 39,
    'Face-Right_eyebrow_out': 40,
    'Face-Right_eyebrow_out_mid': 41,
    'Face-Right_eyebrow_mid': 42,
    'Face-Right_eyebrow_mid_in': 43,
    'Face-Right_eyebrow_in': 44,
    'Face-Left_eyebrow_in': 45,
    'Face-Left_eyebrow_mid_in': 46,
    'Face-Left_eyebrow_mid': 47,
    'Face-Left_eyebrow_out_mid': 48,
    'Face-Left_eyebrow_out': 49,
    'Face-Nose_top': 50,
    'Face-Nose_top_mid': 51,
    'Face-Nose_bottom_mid': 52,
    'Face-Nose_bottom': 53,
    'Face-Right_nostril_out': 54,
    'Face-Right_nostril_mid': 55,
    'Face-Nostril': 56,
    'Face-Left_nostril_mid': 57,
    'Face-Left_nostril_out': 58,
    'Face-Right_eye_out': 59,
    'Face-Right_eye_up_out': 60,
    'Face-Right_eye_up_in': 61,
    'Face-Right_eye_in': 62,
    'Face-Right_eye_down_in': 63,
    'Face-Right_eye_down_out': 64,
    'Face-Left_eye_in': 65,
    'Face-Left_eye_up_in': 66,
    'Face-Left_eye_up_out': 67,
    'Face-Left_eye_out': 68,
    'Face-Left_eye_down_out': 69,
    'Face-Left_eye_down_in': 70,
    'Face-Lips_l1_right_out': 71,
    'Face-Lips_l1_right_mid': 72,
    'Face-Lips_l1_right_in': 73,
    'Face-Lips_l1_mid': 74,
    'Face-Lips_l1_left_in': 75,
    'Face-Lips_l1_left_mid': 76,
    'Face-Lips_l1_left_out': 77,
    'Face-Lips_l4_left_out': 78,
    'Face-Lips_l4_left_in': 79,
    'Face-Lips_l4_mid': 80,
    'Face-Lips_l4_right_in': 81,
    'Face-Lips_l4_right_out': 82,
    'Face-Lips_l2_right_out': 83,
    'Face-Lips_l2_right_in': 84,
    'Face-Lips_l2_mid': 85,
    'Face-Lips_l2_left_in': 86,
    'Face-Lips_l2_left_out': 87,
    'Face-Lips_l3_left': 88,
    'Face-Lips_l3_mid': 89,
    'Face-Lips_l3_right': 90
}

TO_BE_CLASSIFIED = 0
OUT_OF_FRAME = 1
BACKSIDE = 2
SUSPICIOUS = 4
USING = 32
NOT_USING = 128

state_display_type = {
    TO_BE_CLASSIFIED: {
        "color": "gray",
        "str": "-"
    },
    OUT_OF_FRAME: {
        "color": "gray",
        "str": "Out of frame"
    },
    BACKSIDE: {
        "color": "gray",
        "str": "Back"
    },
    SUSPICIOUS: {
        "color": "orange",
        "str": "?"
    },
    USING: {
        "color": "pink",
        "str": "+"
    },
    NOT_USING: {
        "color": "green",
        "str": "-"
    },
}


# ===================================================== #

def get_targets() -> List:
    _target_list = get_cube_angles(use_str=True)
    return _target_list


def get_cube_angles(use_str: bool = True, num: int = 13) -> List[List[List[Any]]]:
    """
    Get the "structure" of one angle channel, using all the non-trivial angles
    calculated from the given keypoint indexes.
    :param use_str:
    :param num: Number of key points.
    :return:
    """
    ls = keypoint_indexes.keys() if use_str else keypoint_indexes.values()

    keys = list(ls)[:num]
    edge_combinations = list(itertools.combinations(keys, 2))
    sorted_angles = [[edge, corner] for corner in keys for edge in edge_combinations if corner not in edge]

    row = num - 1   # i
    col = num - 2   # j
    depth = (num + 1) // 2  # k

    init = use_str and [('', ''), ''] or [(0, 0), 0]
    o_indices: List[List[List[Any]]] = [[[init for _ in range(depth)] for _ in range(col)] for _ in range(row)]

    i_j_order = [(i, j) for i in range(row) for j in range(i, col)]         # Upper-right
    i_j_order += [(i, j) for j in range(col) for i in range(j + 1, row)]    # Lower-left

    idx = 0  # idx of sorted_angles
    for k in range(depth):
        for i, j in i_j_order:
            o_indices[i][j][k] = sorted_angles[idx]
            idx = (idx + 1) % len(sorted_angles)

        # for i in range(row):
        #     for j in range(col):
        #         print(f"{str(o_indices[i][j][k]):<15}", end='')
        #     print('')
        # print('')

    return o_indices
