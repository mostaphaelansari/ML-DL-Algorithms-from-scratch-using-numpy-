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

    def most_common_label(self,y) :
        """
            Return the most common class label.
        """
        counter = Counter(y)
        return Counter.most_common(1)[0][0]
    
    def Split(self,X_columnj, split_thresh) :
        """
            Split the data based on a thershold
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        
        return left_idxs , right_idxs

    def impurity(self, y) :
        """
            Calculate the impurity (gini or entropy) of a node.
        """
        if len(y) == 0 :
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)

        if self.criterion == 'gini' :
            return 1.0 - np.sum(probabilities**2)

        elif self.criterion == 'entropy' :
            # Avoid log(0) by adding small epsilon

            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(pprobabilities))
        
    def information_gain(self, y, X_column, threshold) :
        """
        Calculate the information gain from a split
        """
        # Parent_impurity
        parent_impurity = self.impurity(y)

        # Create children 
        left_idxs , right_idxs = self.spli(X_column , thershold)

        if len(left_idxs) == 0 or len(right_idxs) ==0 :
            return 0
        # Calculate the weighted average impurity of children 
        n = len(y)
        n_l , n_r = len(left_idxs) , len(right_idxs)
        e_l , e_r = self.impurity(y[left_idxs]) , self.impurity(y[right_idxs]) 
        child_impurity = (n_l /n) * e_l +(n_r /n) *e_r

        # Information gain 
        information_gain = parent_impurity - child_impurity
        return information_gain
        
    def best_split(self, X ,y) :
        """
            Find the best feature and threshold to split on.
        """
        best_gain = -1
        best_feature , best_thershold = None , None

        # Randomly select features to consider
        feature_idxs = np.random.choice(self.n_feature , self.max_features, replace = False)

        for feature_idx in feature_idxs :
            x_column = X[:, feature_idx]
            threshold = np.unique(X_column)

            for threshold in thresholds:
                gain = self.information_gain(y , X_column, threshold)

                if gain > best_gain :
                    best_gain = gain 
                    best_feature = feature_idx
                    best_threshold = threshold 

        return best_feature , best_threshold


    def build_tree(self, X , y , depth) :
        """
            Recursively build the decision tree.
        """
        n_samples , n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
            n_labels == 1 or \
            n_samples < self.min_samples_split :
                leaf_value = self.most_common_label(y)
                return Node(value = leaf_value)
        
        # FInd the best split 
        best_feature , best_thresold = self.best_split(X, y)

        if best_feature is  None :
            leaf_value = self.most_cimmon_label(y)
            return Node(value=leaf_value)

        # Split the data 
        left_idxs , right_idxs = self.split(X[:, best_feature], best_threshold)

        # Check minimum samples per leaf 
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf :
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # creat child Nodes 
        left = self.build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs],depth + 1)

        return Node(best_feature , best_threshold,right)


   

