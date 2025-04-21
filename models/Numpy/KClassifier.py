import numpy as np
import numpy.typing as npt
from collections import Counter
from utils.Descriptors import BiggerThanZero
from typing import Dict, List, Any


class KClassifier:
    n_neighbors: int = BiggerThanZero()

    def __init__(self, n_neighbors: int = 1, weights: str = 'uniform', metric: str = 'minkowski', p: int = 2) -> None:
        """KClassifier is an implementation of the Knn algorithm for education purposes.\n
            Args:
                n_neighbors: int : This argument can not be smaller than 1 or an attribute exception will be raised.
                weights: str : 'uniform' or 'distance'.
                metric: str : 'minkowski', 'Euclidean' or 'manhattan'
                p: int
        """
        if weights.lower() not in {'uniform', 'distance'}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if metric.lower() not in {'minkowski', 'euclidean', 'manhattan'}:
            raise ValueError("Metrics must be 'minkowski', 'euclidean', or 'manhattan'")

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p

        self.train_data: npt.NDArray[np.float64] | None = None
        self.train_target: npt.NDArray[np.str_] | None = None

    def fit(self, X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.str_]) -> 'KClassifier':
        """Store the training data points and labels"""
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError('Missing train data or target argument.')

        self.train_data = X_train
        self.train_target = y_train
        return self

    def predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.str_]:
        """Predict the label of each data point in the test dataset
            Return: A string numpy array contains the predicted labels
        """
        if X_test.size == 0:
            raise ValueError('Test dataset is empty or missing.')
        if self.train_data is None or self.train_target is None:
            raise ValueError('The model has not been fitted yet.')

        # Reshaping the dataset for convenient calculation of distances
        X_test_reshaped = X_test[:, np.newaxis, :]  # (m, 1, n)
        train_data_reshaped = self.train_data[np.newaxis, :, :] # (1, k, n)

        # Calculating the distances between each test sample and their neighbors
        distances: npt.NDArray[np.float64] = self._distance(X_test_reshaped, train_data_reshaped, axis=2)
        # distances.shape: (n_test_samples, n_train_samples)

        # Find the k nearest neighbors indices for each test sample
        # 2D array each row represent a test sample and k columns represent the k nearest neighbors indices
        n_neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Get the labels of each test sample's neighbors
        labels_of_neighbors = self.train_target[n_neighbors_indices] # (n_sample, n_neighbors)

        # Predict the test sample's label when the weights argument is 'distance'
        if self.weights == 'distance':
            epsilon = np.finfo(float).eps

            row_indices = np.arange(distances.shape[0])[:, None]
            neighbors_distances = distances[row_indices, n_neighbors_indices]

            # Weight is 1 divided by the distance from a data point to the test sample.
            # plus (+) a small value (epsilon) to prevent divide by 0
            weights = 1 / (neighbors_distances + epsilon)

            # Normalize the data
            weights /= np.sum(weights, axis = 1)[:, None]

            # Store the predicted class labels for each test sample.
            weighted_votes = []

            for i, label_set in enumerate(labels_of_neighbors):
                # Dictionary store the labels of the nearest neighbors and the labels corresponding combined weights
                vote_count: Dict[str, float] = {}
                for j, label in enumerate(label_set):
                    weight = weights[i, j]
                    vote_count[label] = vote_count.get(label, 0.0) + weight

                # Choose the label with the max weight as the label for the test sample
                weighted_votes.append(max(vote_count, key = vote_count.get))

            return np.array(weighted_votes)

        else:
            predictions: List[str] = [Counter(label_set).most_common(1)[0][0] for label_set in labels_of_neighbors]
            return np.array(predictions)

    def _distance(self, point1: npt.NDArray[np.float64] = None, point2: npt.NDArray[np.float64] = None, axis: int = None)\
    -> npt.NDArray[np.float64]:
        """Calculate the distance between two points or vectors
            Return: floating number
        """
        if point1 is None or point2 is None:
            raise ValueError('Missing vector argument.')

        if self.metric == 'euclidean':
            p = 2
        elif self.metric == 'manhattan':
            p = 1
        elif self.metric == 'minkowski':
            p = self.p

        return np.sum(np.abs(point1 - point2) ** p, axis=axis) ** (1/p)

    def score(self, X_test: npt.NDArray[np.float64], y_test: npt.NDArray[np.str_]) -> float:
        """Calculate the accuracy of the model by dividing the total correct predictions by total test samples
            Return: floating number
        """
        if X_test.size == 0 or y_test.size == 0:
            raise ValueError('Missing X_test or y_test argument.')

        predictions = self.predict(X_test)
        correct_predictions = np.sum(predictions == y_test)
        return correct_predictions / y_test.size

    def get_params(self, deep = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metrics': self.metric
        }

    def set_param(self, **param) -> 'KClassifier':
        """Set parameters for this estimator."""
        for key, value in param.items():
            setattr(self, key, value)
        return self