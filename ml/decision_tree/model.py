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
    def __init__(self,feature =None, threshold=None, left =None, right = None, value=None) :
        self.feature = feature # Index of feature to split on 
