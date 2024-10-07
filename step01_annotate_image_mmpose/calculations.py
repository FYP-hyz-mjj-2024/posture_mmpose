import numpy as np
import math


def _calc_angle(
        edge_points: [[float, float], [float, float]],
        mid_point: [float, float]) -> float:
    """
    Calculate the angle based on the given edge points and middle point.
    :param edge_points: A tuple of two coordinates of the edge points of the angle.
    :param mid_point: The coordinate of the middle point of the angle.
    :return: The degree value of the angle.
    """
    # Left, Right
    p1, p2 = [np.array(pt) for pt in edge_points]

    # Mid
    m = np.array(mid_point)

    # Angle
    radians = np.arctan2(p2[1] - m[1], p2[0] - m[0]) - np.arctan2(p1[1] - m[1], p1[0] - m[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


def calc_keypoint_angle(
        keypoints_one_person,
        keypoint_indexes,
        edge_keypoints_names: [str, str],
        mid_keypoint_name: str) -> [float, float]:
    """
    Calculate the angle using the given edge pionts and middle point by their names.
    :param keypoints_one_person: The set of keypoints of a single person. (91, 3)
    :param keypoint_indexes: A mapping dictionary from keypoint names to its indexes.
    :param edge_keypoints_names: A tuple of the names of the two edge keypoints.
    :param mid_keypoint_name: The name of the middle keypoint.
    :return: The targeted angle.
    """

    # Names
    n1, n2 = edge_keypoints_names
    nm = mid_keypoint_name

    # Coordinates
    # Name --> [keypoint_indexes] --> index_number --> [keypoints_one_person] --> (x,y,score) --> [:2] --> (x,y)
    coord1, coord2 = keypoints_one_person[keypoint_indexes[n1]][:2], keypoints_one_person[keypoint_indexes[n2]][:2]
    coordm = keypoints_one_person[keypoint_indexes[nm]][:2]

    # Score of the angle
    s1, s2 = keypoints_one_person[keypoint_indexes[n1]][2], keypoints_one_person[keypoint_indexes[n2]][2]
    sm = keypoints_one_person[keypoint_indexes[nm]][2]

    # Angle Score: Geometric Mean
    # angle_score = math.exp((1/3) * (math.log(s1) + math.log(s2) + math.log(sm)))  # Don't use, potential domain error.
    angle_score = np.cbrt(s1 * s2 * sm)

    return _calc_angle([coord1, coord2], coordm), angle_score
