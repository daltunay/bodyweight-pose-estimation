"""Functions for general calculations (angles, distances)"""

from typing import Iterable

import numpy as np


def calculate_angle(
    a: Iterable[float],
    b: Iterable[float],
    c: Iterable[float],
) -> float:
    """Function to calculate and angle between three points

    Args:
        a (Iterable[float]): 1st point
        b (Iterable[float]): 2nd point
        c (Iterable[float]): 3rd point

    Returns:
        float: angle in degrees
    """

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos)

    return np.degrees(angle)
