"""Script for the coordinate time series"""

from typing import Iterable, Literal, Optional, Tuple

import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tsmoothie.smoother import LowessSmoother

from utils.miscellanous import merge_dict


class CoordinateSeries:
    """Class for the coordinates time series (x, y, z position values of\
        joints over time)"""

    def __init__(self, landmarks: NormalizedLandmarkList, fps: float = 30.0):
        """Initialization

        Args:
            * landmarks (NormalizedLandmarkList): Landmarks time series

            * fps (float, optional): Number of frames per second of the video. Defaults\
                to 30.0
        """
        self.landmarks: NormalizedLandmarkList = landmarks
        self.fps: float = fps
        self.frame: pd.DataFrame = self.extract_coordinates()
        self.frame.loc["MID_ANKLE"] = self.mean_coords(
            row_1="LEFT_ANKLE", row_2="RIGHT_ANKLE"
        )

    def extract_coordinates(self) -> pd.DataFrame:
        """Method to extract the (x, y, z) coordinates of each joint, from\
            the landmarks

        Returns:
            * pd.DataFrame: Time series dataframe of the joint coordinates
        """
        # access landmarks of each frame
        coordinates = [
            [
                {
                    "x": landmarks.landmark[i].x,
                    "y": landmarks.landmark[i].y,
                    "z": landmarks.landmark[i].z,
                }
                for i in range(33)  # all 33 joints
            ]
            for landmarks in self.landmarks
        ]

        frame = pd.DataFrame(
            coordinates,
            index=(
                pd.to_timedelta(range(len(coordinates)), unit="s") / self.fps
            ).total_seconds(),
            columns=[landmark._name_ for landmark in mp.solutions.pose.PoseLandmark],
        )

        return frame.T

    def mean_coords(self, row_1, row_2, out_name=None):
        if out_name is None:
            out_name = f"MID_{row_1.split('_')[-1]}_{row_2.split('_')[-1]}"
        s1, s2 = self.frame.loc[row_1], self.frame.loc[row_2]
        return [
            {k: dict_1[k] / 2 + dict_2[k] / 2 for k in dict_1}
            for dict_1, dict_2 in zip(s1.values, s2.values)
        ]

    def scale(
        self, mu: float = 0.0, std: float = 1.0, inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """Method to standardize the coordinate time series, along\
            each axis (x, y, z)

        Args:
            * mu (float, optional): Mean of the scaled series. Defaults to 0.0

            * std (float, optional): Standard deviation of the scaled series. Defaults\
                to 1.0

            * inplace (bool, optional): Whether to assign a new series (True) or return\
                a new object (False). Defaults to False

        Returns:
            * Optional[pd.DataFrame]: Scaled coordinate time series
        """

        data_x = self.frame.applymap(lambda vector: vector["x"])
        data_y = self.frame.applymap(lambda vector: vector["y"])
        data_z = self.frame.applymap(lambda vector: vector["z"])

        # initialize scaler
        scaler = TimeSeriesScalerMeanVariance(mu, std)

        # scale data
        scaled_data_x = scaler.fit_transform(data_x)[:, :, 0]
        scaled_data_y = scaler.fit_transform(data_y)[:, :, 0]
        scaled_data_z = scaler.fit_transform(data_z)[:, :, 0]

        scaled_frame = pd.DataFrame(
            data=np.vectorize(merge_dict)(
                pd.DataFrame(scaled_data_x).applymap(lambda x: {"x": x}),
                pd.DataFrame(scaled_data_y).applymap(lambda y: {"y": y}),
                pd.DataFrame(scaled_data_z).applymap(lambda z: {"z": z}),
            ),
            index=self.frame.index,
            columns=self.frame.columns,
        )

        if inplace:
            # update frame
            self.frame = scaled_frame
        else:
            return scaled_frame

    def smooth(
        self,
        smooth_fraction: float = 0.1,
        batch_size: Optional[int] = None,
        inplace: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Method to smooth the coordinate time series, along\
            each axis (x, y, z) along each joint

        Args:
            * smooth_fraction (float, optional): Smoothing intensity, between 0 and 1.\
                Defaults to 0.1

            * batch_size (Optional[int], optional): Parallelization batch size.\
                Defaults to None

            * inplace (bool, optional): Whether to assign a new series (True) or return\
                a new object (False). Defaults to False

        Returns:
            * Optional[pd.DataFrame]: Smoothed coordinate time series
        """

        # initialize smoother
        smoother = LowessSmoother(
            smooth_fraction=smooth_fraction, batch_size=batch_size
        )

        # parse data
        data_x = self.frame.applymap(lambda vector: vector["x"])
        data_y = self.frame.applymap(lambda vector: vector["y"])
        data_z = self.frame.applymap(lambda vector: vector["z"])

        # smooth data
        smoothed_data_x = smoother.smooth(data_x).smooth_data
        smoothed_data_y = smoother.smooth(data_y).smooth_data
        smoothed_data_z = smoother.smooth(data_z).smooth_data

        # rebuild frame
        smoothed_frame = pd.DataFrame(
            np.vectorize(merge_dict)(
                pd.DataFrame(smoothed_data_x).applymap(lambda x: {"x": x}),
                pd.DataFrame(smoothed_data_y).applymap(lambda y: {"y": y}),
                pd.DataFrame(smoothed_data_z).applymap(lambda z: {"z": z}),
            ),
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
        axis: Literal["x", "y", "z"],
        joints: Iterable[str] = None,
        figsize: Tuple = (24, 10),
        title: str = "Joint coordinates evolution over time",
    ) -> Axes:
        """Method to plot the coordinates time series, over one chosen axis

        Args:
            * axis (Literal["x", "y", "z"]): Chosen axis to consider

            * joints (Iterable[str], optional): List of joints for which to plot the\
                 coordinates evolution. Defaults to None

            * figsize (Tuple, optional): Figure size. Defaults to (24, 10).

            * title (str, optional): Plot title. Defaults to "Joint coordinates\
                 evolution over time"

        Returns:
            * Axes: Ax on which the time series was plotted
        """

        plot_df = (
            self.frame.loc[[joint.upper() for joint in joints], :]
            if joints
            else self.frame
        )

        _, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=plot_df.applymap(lambda vec: vec[axis]).T, ax=ax)
        ax.set_title(f"{title}\naxis = {axis}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{axis} value\n(world coordinate system)")

        return ax
