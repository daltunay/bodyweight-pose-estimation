import operator
import sys
from collections import Counter

sys.path.append("src")


from distances import lcss_dist


class kNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self._predictions = None

    def fit(self, X, y):
        self.train_data = X
        self.train_labels = y

    def get_neighbors(self, test_row):

        distances = []
        for i, train_row in enumerate(self.train_data):
            distances.append(
                [
                    train_row,
                    lcss_dist(test_row, train_row),
                    self.train_labels[i],
                ]
            )
            distances.sort(key=operator.itemgetter(1))

        return [distances[index] for index in range(self.k)]

    def _predict(self, X):
        self.test_data = X
        predictions = []
        for test_case in self.test_data:
            neighbors = self.get_neighbors(test_case)
            labels = [row[2] for row in neighbors]
            counts = Counter(labels)
            label, count = counts.most_common(1)
            predictions.append((label, count / self.k))

        return predictions

    def predict(self, X):
        predictions = self._predict(X)
        return [pred[0] for pred in predictions]

    def predict_proba(self, X):
        predictions = self._predict(X)
        return [pred[1] for pred in predictions]
