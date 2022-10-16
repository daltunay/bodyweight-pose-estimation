"""Functions related to video processing and pose estimation"""

import sys
from typing import Iterable, List, Literal, Optional, Union

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tqdm
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

sys.path.append("src")


from data.angle_series import AngleSeries
from data.coordinate_series import CoordinateSeries
from resources.joints import AngleJoints

# Parameters
_MIN_DETECTION_CONFIDENCE = 0.5
_MIN_TRACKING_CONFIDENCE = 0.5


class Video:
    """Class for a video to process"""

    def __init__(self, path: Union[str, int], label: Optional[str] = None) -> None:
        """Initialization

        Args:
            * path (Union[str, int]): Path to a local video, or integer (webcam input)
        """
        self.path: Union[str, int] = path
        self.label: Optional[str] = label
        self.frame_count: int = None
        self.fps: float = None
        self.landmarks_series: Optional[List[NormalizedLandmarkList]] = None

        self._coordinates: Optional[CoordinateSeries] = None  # CoordinateSeries object
        self.coordinates: pd.DataFrame = None  # corresponding time series frame

        self._angles: Optional[AngleSeries] = None  # AngleSeries object
        self.angles: pd.DataFrame = None  # corresponding time series dataframe

    def get_landmarks(
        self,
        model_complexity: Literal[0, 1, 2] = 1,
        show: bool = False,
        resize: float = 1,
    ) -> None:
        """Method to extract the landmarks of each frame of the video

        Args:
            * model_complexity (Literal[0, 1, 2]): Complexity of the Mediapipe pose\
                estimation model

            * show (bool): Whether to show the video and landmarks in an output window\
                or not

        Returns:
            * List[NormalizedLandmarkList]: Landmarks time series
        """
        if isinstance(self.path, int):
            show = True

        self.landmarks_series = []

        # video capture (local or webcam)
        cap = cv2.VideoCapture(self.path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        # instantiate pose processer
        with mp.solutions.pose.Pose(
            min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=_MIN_TRACKING_CONFIDENCE,
            model_complexity=model_complexity,
            smooth_landmarks=True,
        ) as pose:

            # progress bar
            pbar = tqdm.tqdm(
                total=self.frame_count if isinstance(self.path, str) else None,
                desc=self.path if isinstance(self.path, str) else "webcam",
                position=0,
                leave=True,
            )

            success, frame = cap.read()  # read first frame

            # loop over all frames
            while success:
                pbar.update(1)

                # process image
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # update landmarks series
                self.landmarks_series.append(results.pose_world_landmarks)

                # show frame and landmarks
                if show is True:
                    Video.show_landmarks(image=frame, landmarks=results.pose_landmarks)
                    Video.show_time(
                        image=frame, time=len(self.landmarks_series) / self.fps
                    )
                    cv2.imshow(
                        "Mediapipe Feed",
                        cv2.resize(frame, dsize=None, fx=resize, fy=resize),
                    )

                    # manual exit
                    if cv2.waitKey(1) & 0xFF == ord("\x1b"):  # press Esc
                        if isinstance(self.path, int):
                            break
                        show = False
                        cv2.destroyAllWindows()

                success, frame = cap.read()  # read next frame

            # auto exit
            cap.release()
            cv2.destroyAllWindows()

        self.landmarks_series = [
            landmarks for landmarks in self.landmarks_series if landmarks
        ]

    @staticmethod
    def show_landmarks(
        image: np.ndarray,
        landmarks: NormalizedLandmarkList,
    ) -> None:
        """Function to draw landmarks on a given input image

        Args:
            * image (np.ndarray): Image array with RGB format, shape (width, height, 3)

            * landmarks (NormalizedLandmarkList): Pose estimation result landmarks
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    @staticmethod
    def show_time(image: np.ndarray, time: float) -> None:
        """Function to show a time value on a given input image

        Args:
            * image (np.ndarray): Image array with RGB format, shape (width, height, 3)
            * time (float): Time in seconds
        """
        cv2.putText(
            img=image,
            text=f"{time:.3f}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def extract_coordinates(self) -> None:
        """Method to extract the coordinates of the different joints and save\
            the time series"""
        self._coordinates = CoordinateSeries(
            landmarks=self.landmarks_series, fps=self.fps
        )
        self.coordinates = self._coordinates.frame

    def extract_angles(self, angle_list: Iterable[AngleJoints]) -> None:
        """Function to exract the angle time series from a video, using the coordinates

        Args:
            * angle_list (Iterable[AngleJoints]): List of angles to consider
        """
        self._angles = AngleSeries(coordinates=self._coordinates, angle_list=angle_list)
        self.angles = self._angles.frame
