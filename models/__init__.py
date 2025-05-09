"""Machine learning models implemented in NumPy and PyTorch."""

# Import key classes from subpackages
from .Numpy import KClassifier, KRegressor, LinearRegression
from .Pytorch import TinyVGG, TinyVGGv2

# Define what's available for import with 'from models import *'
__all__ = [
    # NumPy models
    'KClassifier',
    'KRegressor',
    'LinearRegression',
    
    # PyTorch models
    'TinyVGG',
    'TinyVGGv2'
]