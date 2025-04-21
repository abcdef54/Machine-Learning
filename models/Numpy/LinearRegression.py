import numpy as np
import numpy.typing as npt
from typing import Dict

# For testing
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class LinRegression:
    def __init__(self):
        """
        LinRegression is an implementation of the linear regression algorithm in machine learning for education purposes.
        """
        self.coefs_: npt.NDArray[np.float64] | None = None  # Coefficient values vector of the model
        self.Xhat: npt.NDArray[np.float64] | None = None    # Augmented feature matrix with intercept term
        self.y: npt.NDArray[np.float64] | None = None       # Vector of actual target values from the training data
        self.intercept_: float = 0.0                        # Intercept (bias) value of the model

    @property
    def full_coef_(self) -> npt.NDArray[np.float64]:
        """Get the full coefficient array including the intercept value
            Return:
                npt.NDArray[np.float64]
            Raises:
                ValueError: If the model has not yet been fitted.
        """
        if self.coefs_ is None:
            raise ValueError('Model has not been fitted yet.')
        return np.concatenate(([self.intercept_], self.coefs_))

    def fit(self, X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.float64]) -> 'LinRegression':
        """Fit the linear regression model to the training data"""
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError('Missing training data or target.')

        self.y = y_train

        # Augment X with a column of ones for the bias term
        ones = np.ones((X_train.shape[0], 1))
        self.Xhat = np.concatenate((ones, X_train), axis = 1)

        # Find optimal w (coefficient value) for all features
        Xhat_dot = np.dot(self.Xhat.T, self.Xhat)
        Xhat_pinv = np.linalg.pinv(Xhat_dot)
        yXhat_dot = np.dot(self.Xhat.T, y_train)

        # Store the optimal w values
        self.coefs_ = np.dot(Xhat_pinv, yXhat_dot)

        self.intercept_ = self.coefs_[0]
        self.coefs_ = self.coefs_[1:]
        return self

    def predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict the target values of the test dataset
            Return: Floating numpy array containing the predicted target values for the test dataset.
        """
        if X_test.size == 0:
            raise ValueError('Test dataset is empty or missing.')

        ones = np.ones((X_test.shape[0], 1))
        X_hat = np.concatenate((ones, X_test), axis = 1)
        return np.dot(X_hat, self.full_coef_)

    def score(self, X_test: npt.NDArray[np.float64], y_test: npt.NDArray[np.float64]) -> float:
        """Compute the R-squared score of the model.
            Return: Floating number
        """
        if self.coefs_ is None:
            raise ValueError('Model has not been fitted yet.')

        y_pred = self.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    def _loss_function(self) -> float:
        """Compute the mean squared error loss.
            Return: Floating number
        """
        y_pred = np.dot(self.Xhat, self.full_coef_)
        residuals = self.y - y_pred
        return 0.5 * np.sum(residuals ** 2)

if __name__ == '__main__':
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)

    real_lin = LinearRegression()
    real_lin.fit(X_train, y_train)

    my_lin = LinRegression()
    my_lin.fit(X_train, y_train)

    print(real_lin.score(X_test, y_test))
    print(my_lin.score(X_test, y_test))



