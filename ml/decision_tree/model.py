"""
Here we will define the Decision Tree model class.
This class will implement the Decision Tree algorithm for classification tasks.
"""
import numpy as np
from collections import Counter

class Node:
    """
    A node in the decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Class prediction for leaf nodes

class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementation.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int, str, default=None
        Number of features to consider when looking for the best split.
    criterion : str, default='gini'
        The function to measure the quality of a split ('gini' or 'entropy').
    """
    def __init__(self,max_depth=None ,min_samples_split =2 ,min_samples_leaf =1 ,max_feature = None ,criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_max_features = features 
        self.criterion = criterion 
        self.root = None 
        self.n_classes = None 
        self.n_features = None 

    def fit(self, X ,y ) :
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """

        self.n_classes = len(np.unique(y)) 
        self.n_features = X.shape[1]

        # Determine max_features if not specified 
        if self.max_features is None:
            self.max_features = self.n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(self.n_features))
