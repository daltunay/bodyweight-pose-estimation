"""Script for the angle time series"""
import sys

sys.path.append("src")

from collections import defaultdict
from typing import Iterable, Literal, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from data.coordinate_series import CoordinateSeries
from resources.joints import AngleJoints
from resources.smoothers import SMOOTHERS
from utils.trigonometry import calculate_angle


class AngleSeries:
    """Class for the angle time series (angle values over time)"""

    def __init__(
        self, coordinates: CoordinateSeries, angle_list: Iterable[AngleJoints]
    ):
        """Initialization

        Args:
            * coordinates (CoordinateSeries): Coordinates series from which the angles will be extracted
            * angle_list (Iterable[AngleJoints]): List of the angles to extract from the series
        """
        self.coordinates: CoordinateSeries = coordinates
        self.angle_list: Iterable[AngleJoints] = angle_list
        self.frame: pd.DataFrame = self.extract_angles()

    def extract_angles(self) -> pd.DataFrame:
        """Method to extract the angle values from the (x, y, z) coordinates of each joint

        Returns:
            pd.DataFrame: Angles time series
        """
        angles_dict = defaultdict(lambda: [])

        # computing each video frame
        for frame_coords in self.coordinates.frame.T.values:

            # computing each set of three joints (angle)
            for angle_joints in self.angle_list:
                a = frame_coords[angle_joints.joints_idx[0]]
                b = frame_coords[angle_joints.joints_idx[1]]
                c = frame_coords[angle_joints.joints_idx[2]]

                # add angle value to time series
                angles_dict[str(angle_joints)].append(
                    calculate_angle(
                        a=(a["x"], a["y"] + a["z"]),
                        b=(b["x"], b["y"] + b["z"]),
                        c=(c["x"], c["y"] + c["z"]),
                    )
                )
        frame = pd.DataFrame.from_dict(angles_dict)
        frame.index = self.coordinates.frame.columns

        return frame.T

    def scale(
        self, mu: float = 0.0, std: float = 1.0, inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """Method to standardize the angle time series, along each angle

        Args:
            * mu (float, optional): Mean of the scaled series. Defaults to 0.0
            * std (float, optional): Standard deviation of the scaled series. Defaults to 1.0
            * inplace (bool, optional): Whether to assign a new series (True), or return a new object (False). Defaults to False

        Returns:
            * Optional[pd.DataFrame]: Scaled angle time series
        """

        # initialize scaler
        scaler = TimeSeriesScalerMeanVariance(mu, std)

        # scale data
        scaled_values = scaler.fit_transform(self.frame)[:, :, 0]
        scaled_frame = pd.DataFrame(
            scaled_values, index=self.frame.index, columns=self.frame.columns
        )

        if inplace:
            # update frame
            self.frame = scaled_frame
        else:
            return scaled_frame

    def smooth(
        self,
        method: Literal[
            "kalman",
            "spline",
            "binner",
            "lowess",
            "convolution",
            "decompose",
        ],
        smooth_fraction: float = 0.1,
        batch_size: Optional[int] = None,
        inplace: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Method to smooth the angle time series, along each angle

        Args:
            * method (Literal["kalman", "spline", "binner", "lowess", "convolution", "decompose"]): Method to use for the smoothing
            * smooth_fraction (float, optional): Smoothing intensity, between 0 and 1. Defaults to 0.1
            * batch_size (Optional[int], optional): Parallelization batch size. Defaults to None
            * inplace (bool, optional): Whether to assign a new series (True), or return a new object (False). Defaults to False

        Returns:
            * Optional[pd.DataFrame]: Smoothed angle time series
        """

        # initialize smoother
        smoother = SMOOTHERS[method](
            smooth_fraction=smooth_fraction, batch_size=batch_size
        )

        # smooth data
        smoothed_data = smoother.smooth(self.frame).smooth_data
        smoothed_frame = pd.DataFrame(
            smoothed_data,
            index=self.frame.index,
            columns=self.frame.columns,
        )
        if inplace:
            # update frame
            self.frame = smoothed_frame
        else:
            return smoothed_frame

    def plot(
        self,
        angles: Iterable[AngleJoints] = None,
        figsize: Tuple = (24, 10),
        title: str = "Angle values evolution over time",
    ) -> Axes:
        """Method to plot the angle time series

        Args:
            * angles (Iterable[AngleJoints], optional): List of angles for which to plot the angle value evolution. Defaults to None.
            * figsize (Tuple, optional): Figure size. Defaults to (24, 10).
            * title (str, optional): Plot title. Defaults to "Angle coordinates evolution over time"

        Returns:
            * Axes: Ax on which the time series was plotted
        """

        plot_df = self.frame.loc[angles, :] if angles else self.frame

        _, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=plot_df.T, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Angle value")

        return ax
