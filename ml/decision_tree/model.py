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
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
        self.n_classes = None
        self.n_features = None
    
    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        X = np.array(X)
        y = np.array(y)
        
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
        
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        
        # Check minimum samples per leaf
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Create child nodes
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        """
        best_gain = -1
        best_feature, best_threshold = None, None
        
        # Randomly select features to consider
        feature_idxs = np.random.choice(self.n_features, self.max_features, replace=False)
        
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """
        Calculate the information gain from a split.
        """
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted average impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Information gain
        information_gain = parent_impurity - child_impurity
        return information_gain
    
    def _split(self, X_column, split_thresh):
        """
        Split the data based on a threshold.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _impurity(self, y):
        """
        Calculate the impurity (gini or entropy) of a node.
        """
        if len(y) == 0:
            return 0
        
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        
        if self.criterion == 'gini':
            return 1.0 - np.sum(probabilities**2)
        elif self.criterion == 'entropy':
            # Avoid log(0) by adding small epsilon
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(probabilities))
    
    def _most_common_label(self, y):
        """
        Return the most common class label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        X = np.array(X)
        return np.array([self._predict_single(sample) for sample in X])
    
    def _predict_single(self, x):
        """
        Predict the class label for a single sample.
        """
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X = np.array(X)
        return np.array([self._predict_proba_single(sample) for sample in X])
    
    def _predict_proba_single(self, x):
        """
        Predict class probabilities for a single sample.
        """
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # For leaf nodes, return probability distribution
        # This is a simplified version - in practice, you'd want to store
        # class distributions in leaf nodes during training
        proba = np.zeros(self.n_classes)
        proba[node.value] = 1.0
        return proba
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_depth(self):
        """
        Return the depth of the decision tree.
        """
        return self._get_depth(self.root)
    
    def _get_depth(self, node):
        """
        Recursively calculate the depth of the tree.
        """
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
    
    def get_n_leaves(self):
        """
        Return the number of leaves in the decision tree.
        """
        return self._get_n_leaves(self.root)
    
    def _get_n_leaves(self, node):
        """
        Recursively count the number of leaves.
        """
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return self._get_n_leaves(node.left) + self._get_n_leaves(node.right)
