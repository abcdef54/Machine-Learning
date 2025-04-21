import numpy as np
import numpy.typing as npt
from utils.Descriptors import BiggerThanZero
from typing import Dict, Any

# For testing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes


class KRegressor:
    n_neighbors: int = BiggerThanZero()

    def __init__(self, n_neighbors: int = 1, weights: str = 'uniform', metric: str = 'minkowski', p: int = 2) -> None:
        """KClassifier is an implementation of the Knn algorithm for education purposes.\n
            Args:
                n_neighbors: int : This argument can not be smaller than 1 or an attribute exception will be raised.
                weights: str : 'uniform' or 'distance'.
                metric: str : 'minkowski', 'euclidean' or 'manhattan'
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
        self.train_target: npt.NDArray[np.float64] | None = None

    def fit(self, X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.float64]) -> 'KRegressor':
        """Store the train data and target into the model"""
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError('Train data or target is empty.')

        self.train_data = X_train
        self.train_target = y_train
        return self

    def predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict the target of each data point in the test dataset
            Return: A floating numpy array contains the predicted targets
        """
        if X_test.size == 0:
            raise ValueError('Test dataset is empty.')
        if self.train_data is None or self.train_target is None:
            raise ValueError('Model has not been fitted yet.')

        # Reshaping the dataset for convenient calculation of distances
        X_test_reshaped = X_test[:, np.newaxis, :]
        train_data_reshaped = self.train_data[np.newaxis, :, :]

        # Calculating the distances between each test sample and their neighbors
        distances = self._distance(X_test_reshaped, train_data_reshaped, axis = 2)

        # Get the k nearest models indices
        n_neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Calculating the mean of k nearest models targets
        targets_of_neighbors = self.train_target[n_neighbors_indices]

        # Calculating the weighted mean if the given weight is 'distance'
        if self.weights == 'distance':
            epsilon = np.finfo(float).eps

            row_indices = np.arange(distances.shape[0])[:, None]
            neighbors_distances = distances[row_indices, n_neighbors_indices]
            weights = 1 / (neighbors_distances + epsilon)

            # Normalize the data
            weights /= np.sum(weights, axis = 1)[:, None]

            # Return a weighted sum along axis 1 (one prediction per test sample)
            return np.sum(weights * targets_of_neighbors, axis = 1)
        else:
            # Return the mean of the k nearest models to each test sample
            return np.mean(targets_of_neighbors, axis = 1)


    def _distance(self, point1: npt.NDArray[np.float64] = None, point2: npt.NDArray[np.float64] = None, axis: int = 1) \
        -> npt.NDArray[np.float64] | None:
        """Calculate the distance between two points or vectors
            Return: a floating value of a float numpy array
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

    def score(self, X_test: npt.NDArray[np.float64] = None, y_test: npt.NDArray[np.float64] = None) -> float:
        """Calculate the R-squared value of the model
            Return: floating number
        """
        if X_test is None or y_test is None:
            raise ValueError('Missing X_test or y_test argument.')

        predictions = self.predict(X_test)
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        if ss_tot == 0:
            return 1.0
        return 1.0 - ss_res / ss_tot

    def get_params(self, deep = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'n_neighbors' : self.n_neighbors,
            'weights' : self.weights,
            'metric' : self.metric
        }

    def set_params(self, **params) -> 'KRegressor':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == '__main__':
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    param_grid = {'n_neighbors' : range(1, 50, 2), 'weights' : ['uniform', 'distance']}
    grid_real = GridSearchCV(estimator=KNeighborsRegressor(), param_grid= param_grid)
    grid_real.fit(X_train, y_train)

    my_grid = GridSearchCV(estimator=KRegressor(), param_grid=param_grid)
    my_grid.fit(X_train, y_train)

    my_best = my_grid.best_params_
    their_best = grid_real.best_params_

    print(their_best)
    print(my_best)

    my_knn = KRegressor(n_neighbors = my_best['n_neighbors'], weights = my_best['weights'])
    real_knn = KNeighborsRegressor(n_neighbors=their_best['n_neighbors'], weights = their_best['weights'])

    my_knn.fit(X_train, y_train)
    real_knn.fit(X_train, y_train)

    my_score = my_knn.score(X_test, y_test)
    their_score = real_knn.score(X_test, y_test)
    print(f'My Knn: {my_score}')
    print(f'Real Knn: {their_score}')
    print(f'Difference: {np.abs(my_score - their_score)}')