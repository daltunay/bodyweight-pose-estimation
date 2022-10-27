"""Script to train the k-Nearest-Neighbors classifier model"""
import os
import pickle

import pandas as pd
from model import kNNClassifier

DATA_PROCESSED_FILEPATH = "data\\processed"
K = 4
EPSILON = 150


def main() -> None:
    X_train, y_train = [], []

    for (dirpath, _, filenames) in os.walk(DATA_PROCESSED_FILEPATH):
        for frame_name in filenames:

            x = pd.read_csv(os.path.join(dirpath, frame_name), index_col=0)
            y = dirpath.split("\\")[-1]

            X_train.append(x)
            y_train.append(y)

    knn = kNNClassifier(k=K, epsilon=EPSILON)
    knn.fit(X=X_train, y=y_train)

    pickle.dump(knn, open("models/model.pkl", "wb"))


if __name__ == "__main__":
    main()
