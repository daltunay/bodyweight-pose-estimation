"""Script to generate the dataset"""

import os
import sys

from video import Video

sys.path.append("src")
from resources.joints import JOINTS

DATA_RAW_FILEPATH = "data\\raw"
DATA_PROCESSED_FILEPATH = "data\\processed"


def main() -> None:
    """Function to generate the angle time series from the given input files"""
    for (dirpath, _, filenames) in os.walk(DATA_RAW_FILEPATH):
        for i, video_name in enumerate(filenames):
            label = dirpath.split("\\")[-1]
            if ".mp4" not in video_name:
                continue
            vid = Video(path=os.path.join(dirpath, video_name))
            vid.get_landmarks(model_complexity=2, show=False)
            vid.extract_coordinates()
            vid.extract_angles(angle_list=JOINTS)

            vid._angles.smooth(inplace=True)
            # vid._angles.scale(inplace=True)

            out = vid._angles.frame
            out.to_csv(
                os.path.join(
                    DATA_PROCESSED_FILEPATH,
                    label,
                    f"{video_name.replace('.mp4', '.csv')}",
                )
            )


if __name__ == "__main__":
    main()
