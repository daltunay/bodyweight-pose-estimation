"""Longest Common Subsequence (LCSS) calculation"""

from typing import Iterable, Union

import pandas as pd
import tslearn.metrics


def lcss_dist(
    frame_1: Union[Iterable[float], pd.DataFrame],
    frame_2: Union[Iterable[float], pd.DataFrame],
    epsilon: float = 1_000,
) -> float:
    """Function to compute the distance between two one or multi-dimensional\
        time series

    Args:
        * frame_1 (Union[Iterable[float], pd.DataFrame]): 1st time series
        * frame_2 (Union[Iterable[float], pd.DataFrame]): 2nd time series
        * epsilon (float): Maximum matching distance threshold

    Returns:
        * float: distance between frame_1 and frame_2, bewteen 0 and 1
    """

    return 1 - min(frame_1.shape[1], frame_2.shape[1]) * tslearn.metrics.lcss(
        frame_1, frame_2, epsilon
    )
