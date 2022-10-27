"""Script for the k-Nearest-Neighbors classifier model"""
import operator
import sys
from collections import Counter
from typing import Dict, Iterable, List, Union

import pandas as pd

sys.path.append("src")


from distances import lcss_dist


class kNNClassifier:
    """Class for k-Nearest-Neighbors classification"""

    def __init__(self, k: int, epsilon: float = 1000) -> None:
        """Constructor for kNNClassifier

        Args:
            * k (int): Number of neighbors to consider for a prediction
        """
        self.k: int = k
        self.train_data = None
        self.train_labels = None
        self.epsilon = epsilon

    def fit(self, X: Iterable[pd.DataFrame], y: Iterable[str]) -> None:
        """Fit model to train data

        Args:
            * X (Iterable[pd.DataFrame]): List of dataframes on which to fit the model

            * y (Iterable[str]): List of the labels corresponding to each dataframe
        """
        self.train_data = X
        self.train_labels = y

    def _get_neighbors(self, x: pd.DataFrame) -> List[str]:
        """Method to get neighbors of the given dataframe and the corresponding\
            distances

        Args:
            * x (pd.DataFrame): Input dataframe of which to get neighbors

        Returns:
            * List[str]: List of the k nearest neighbors and the corresponding distances
        """
        distances = []
        for neighbor, neighbor_label in zip(self.train_data, self.train_labels):
            distances.append(
                [
                    lcss_dist(x, neighbor, epsilon=self.epsilon),
                    neighbor_label,
                ]
            )
            distances.sort(key=operator.itemgetter(0))

        return [distances[index] for index in range(self.k)]

    def _predict(
        self, X: Union[pd.DataFrame, Iterable[pd.DataFrame]]
    ) -> List[Dict[str, float]]:
        """Method to get the k nearest neighbors and the corresponding labels of each\
            dataframe of a given input

        Args:
            X (Union[pd.DataFrame, Iterable[pd.DataFrame]]): Input dataframe or list of\
                dataframes to classify

        Returns:
            * List[Tuple[str, float]]: List of the (label, proportion of label among\
                k nearest neighbors) values for each input frame
        """
        X = [X] if isinstance(X, pd.DataFrame) else X
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            labels = [row[1] for row in neighbors]
            counts = dict(Counter(labels))
            predictions.append({k: v / self.k for k, v in counts.items()})
        return predictions

    def predict(self, X: Union[pd.DataFrame, Iterable[pd.DataFrame]]) -> List[str]:
        """Method to predict the label of each input dataframe

        Args:
            * X (Union[pd.DataFrame, Iterable[pd.DataFrame]]): Input dataframe or list\
                of dataframes to classify

        Returns:
            * List[str]: List of the labels for each input frame
        """
        predictions = self._predict(X)
        labels = [max(pred, key=pred.get) for pred in predictions]
        return labels

    def predict_proba(
        self, X: Union[pd.DataFrame, Iterable[pd.DataFrame]]
    ) -> List[float]:
        """Method to predict the label of each input dataframe

        Args:
            * X (Union[pd.DataFrame, Iterable[pd.DataFrame]]): Input dataframe or list\
                of dataframes to classify

        Returns:
            * List[float]: List of the probabilities to belong to the predicted class,\
                for eachinput frame
        """
        predictions = self._predict(X)
        return [max(pred.values()) for pred in predictions]
