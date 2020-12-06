from abc import ABC, abstractmethod


class DecisionNode(ABC):
    '''Abstract class representing internal or leaf node in a decision tree.'''
    @abstractmethod
    def __init__(self):
        pass


class DecisionInternalNode(DecisionNode):
    '''Represents an internal node in a decision tree.'''
    def __init__(self, split_feature, children):
        '''Internal node splits on the feature at index split_feature.
        :param split_feature (int): the index of the feature to split on.
        :param children (list): a list of DecisionNodes, one for each feature
               value. len(children) == feature_class_sizes[split_feature].'''
        self.split_feature = split_feature
        self.children = children
        
    def __call__(self, example: list) -> int:
        """Inspired by AIMA-python implementation of the algorithm in AIMA-python

        Args:
            example (list): a feature vector of discrete feature values.

        Returns:
            int: the class label
        """
        return self.children[example[self.split_feature]](example)
            
        


class DecisionLeafNode(DecisionNode):
    '''Represents a leaf node, i.e. constant value, in a decision tree.'''
    def __init__(self, value):
        '''Creates a leaf node.
        :param value (int): the constant value, in the set {0,...,num_labels-1}.'''
        self.value = value
        
    def __call__(self, example: list) -> int:
        """Inspired by AIMA-python implementation of the algorithm in AIMA-python.
           Calling a Leaf node will just return its result/value.

        Args:
            example (list): a feature vector of discrete feature values.

        Returns:
            int: the class label
        """
        return self.value


class DecisionTree(object):
    '''Represents a discrete decision tree.'''
    def __init__(self, feature_class_sizes, num_labels, root):
        '''Create a discrete decision tree.
        :param feature_class_sizes (list): a list of integers, specifying the
               number of discrete values each input feature may take on.
               For a feature vector x, len(x) == len(feature_class_sizes) and the
               i'th feature x[i] must lie in {0, ..., feature_class_sizes[i] - 1}.
        :param num_labels (int): an integer specifying the number
               of discrete values the output label y may take on.
               That is, y must lie in {0, ..., num_labels - 1}.
        :param root (DecisionNode): the DecisionNode at the root of the tree.'''
        self.feature_class_sizes = feature_class_sizes
        self.num_labels = num_labels
        self.root = root


    def lookup(self, x):
        '''Returns the class label predicted by the tree given a feature vector.
        :param x (list): a feature vector of discrete feature values.
               len(x) == len(dtree.feature_class_sizes), and
               x[i] in {0, ..., dtree.feature_class_sizes[i] - 1}.
        :return: The class label predicted by the decision tree,
                 contained in {0, ..., dtree.num_labels - 1}.'''
        
        predicted_class = self.root(x)
        
        if predicted_class < 0 or predicted_class >= self.num_labels:
            raise Exception("Predicted class not valid.")
        
        return predicted_class
        