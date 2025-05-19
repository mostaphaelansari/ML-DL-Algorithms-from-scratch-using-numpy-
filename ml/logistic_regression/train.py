"""Train a logistic regression model with early stopping and data validation.
This script includes functions for loading data, training the model,
plotting cost history, and saving the model.
It also includes error handling and logging for better debugging."""

import argparse
import logging
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression.model import LogisticRegression


