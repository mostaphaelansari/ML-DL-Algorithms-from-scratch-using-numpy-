"""
Utility functions for machine learning implementations.
This module provides evaluation metrics and helper functions.
"""


from .metrics import (
                        root_mean_squared_error ,
                        mean_absolute_error ,
                        mean_squared_error ,
                         r2_score 
                        )

# Define what should be imported with "from utils import *"

__all__ = [
    'root_mean_squared_error',
    'mean_squared_error',
    'r2_score',
    'mean_absolute_error'
]

# Package metadata
__version__ = '0.1.0'

