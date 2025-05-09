"""Machine Learning project with implementations in NumPy and PyTorch."""

# Import main components for easy access
from .models import KClassifier, KRegressor, LinearRegression, TinyVGG, TinyVGGv2
from .utils import (
    BiggerThanZero,
    download_20percent_data,
    plot_loss_curves,
    train_step,
    test_step,
    train,
    make_image_dataset,
    make_image_dataloader,
    to_tensors,
    make_train_test_dataloader
)

# Define what's available for import with 'from Machine_Learning import *'
__all__ = [
    # Models
    'KClassifier',
    'KRegressor',
    'LinearRegression',
    'TinyVGG',
    'TinyVGGv2',
    
    # Utilities
    'BiggerThanZero',
    'download_20percent_data',
    'plot_loss_curves',
    'train_step',
    'test_step',
    'train',
    'make_image_dataset',
    'make_image_dataloader',
    'to_tensors',
    'make_train_test_dataloader'
]