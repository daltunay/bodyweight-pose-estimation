"""Main script to try the classifier on a live input video"""
import pickle
import sys

from src.data import Video
from src.resources.joints import JOINTS

sys.path.append("src/models")

MODEL_COMPLEXITY = 1  # 0, 1 or 2


def main() -> None:
    model = pickle.load(open("models/model.pkl", "rb"))

    vid = Video(path=1)
    vid.get_landmarks(model_complexity=MODEL_COMPLEXITY, show=True)
    vid.extract_coordinates()
    vid.extract_angles(JOINTS)
    vid._angles.smooth(inplace=True)
    X = vid._angles.frame

    pred = model._predict(X)
    label = model.predict(X)

    print(pred)
    return label


if __name__ == "__main__":
    main()
