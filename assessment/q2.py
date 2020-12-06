from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
import math

class DecisionNode(ABC):
    '''Abstract class representing internal or leaf node in a decision tree.'''

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def lookup(self, x):
        '''Prediction of the decision tree rooted at this node for feature x.'''
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
        
    def add(self, value, node):
        
        self.children[value] = node    
        
    def lookup(self, example: list) -> int:
        """Inspired by AIMA-python implementation of the algorithm in AIMA-python

        Args:
            example (list): a feature vector of discrete feature values.

        Returns:
            int: the class label
        """
        return self.children[example[self.split_feature]](example)
        
    
    lookup = din_lookup  # ignore, only used during testing


class DecisionLeafNode(DecisionNode):
    '''Represents a leaf node, i.e. constant value, in a decision tree.'''

    def __init__(self, value):
        '''Creates a leaf node.
            :param value (int): the constant value, in the set {0,...,num_labels-1}.'''
        self.value = value
        
    def lookup(self, example: list) -> int:
        """Inspired by AIMA-python implementation of the algorithm in AIMA-python.
            Calling a Leaf node will just return its result/value.

        Args:
            example (list): a feature vector of discrete feature values.

        Returns:
            int: the class label
        """
        return self.value

    lookup = dln_lookup  # ignore, only used during testing


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
    
    lookup = dt_lookup  # ignore, only used during testing

    @classmethod
    def create_old(cls, xs, num_features, ys):
        '''Creates a DecisionTree to predict the output
           labels ys from the input feature vectors xs.
        :param xs (list of list): list of feature vectors, i.e. a 2D list of integers.
                   xs[i][j] represents the j'th feature of the i'th data point.
                   len(xs[i]) == num_features and xs[i][j] in {0, 1, 2}.
        :param num_features: the number of features each input consists of.
        :param ys: output, a list of integers.
                   ys[i] is the class label for feature vector xs[i].
                   ys[i] in {0, 1, 2}.
        :return: A DecisionTree instance.'''
        def helper(xs, ys, used_features):
            # Pick features to split on.
            features = [[xs[i][j] for i in range(len(xs))]
                          for j in range(num_features)]
            class_sum = [sum(x) for x in features]
            for j in used_features:
                class_sum[j] = -1
            j = class_sum.index(max(class_sum))

            # Split on j
            used_features = used_features.union([j])
            # Three features ought to be enough for anybody
            subtrees = [([], []) for i in range(3)]
            for x, y in zip(xs, ys):
                split_feature = x[j]
                sub_xs, sub_ys = subtrees[split_feature]
                sub_xs.append(x)
                sub_ys.append(y)

            children = []
            for i in range(3):
                sub_xs, sub_ys = subtrees[i]
                vals = set(sub_ys)
                if len(vals) == 0:  # empty
                    c = Counter()
                    c.update(ys)
                    c = OrderedDict(sorted(c.items(), key=lambda t: t[0]))
                    most_common = max(c.keys(), key=lambda k: c[k])
                    child = DecisionLeafNode(most_common)
                elif len(vals) == 1:  # constant
                    child = DecisionLeafNode(vals.pop())
                else:  # > 1 label
                    # still features left to split on
                    if len(used_features) < num_features:
                        child = helper(sub_xs, sub_ys, used_features)
                    else:
                        c = Counter()
                        c.update(sub_ys)
                        c = OrderedDict(sorted(c.items(), key=lambda t: t[0]))
                        most_common = max(c.keys(), key=lambda k: c[k])
                        child = DecisionLeafNode(most_common)
                children.append(child)

            return DecisionInternalNode(j, children)

        root = helper(xs, ys, set())
        return DecisionTree([3] * num_features, 3, root)


    @classmethod
    def entropy(cls, xs: list, ys: list) -> float:

        entropy = 0.0
        total_examples = len(xs)

        for unique_output in set(ys):
            probability_of_output = (
                len(
                    DecisionTree.examples_with_output(
                        xs, ys, unique_output
                    )
                )
                / total_examples
            )
            entropy -= probability_of_output * math.log2(probability_of_output)

        return entropy

    @classmethod
    def examples_with_output(cls, xs:list, ys:list, output: int) -> list:
        
        remaining_examples = []
        
        for i in range(len(xs)):
            if ys[i] == output:
                remaining_examples.append(xs[i])    
                
        return remaining_examples

    @classmethod
    def information_gain(
        cls, xs: list, ys: int, attr: int, current_measure: float) -> float:
        
        total_examples = len(xs)
        average_child_measure = 0.0

        for val in DecisionTree.unique_values(xs, attr):
            examples_with_val, corresponding_labels = DecisionTree.examples_with_value(xs, ys, attr, val)
            child_measure = DecisionTree.entropy(examples_with_val, corresponding_labels)
            average_child_measure += len(examples_with_val) * child_measure

        average_child_measure /= total_examples

        return current_measure - average_child_measure
        
    @classmethod        
    def most_common_output(cls, xs: list, ys:list, attribute: int) -> DecisionLeafNode:
        # FIXME: adapt for ys
        highest_count, highest_value = 0, 0
        for val in DecisionTree.unique_values(xs, ys, attribute):
            current_count = len(DecisionTree.examples_with_value(xs, ys, attribute, val))
            if current_count > highest_count:
                highest_count, highest_value = current_count, val

        return DecisionLeafNode(highest_value)

    @classmethod
    def unique_values(cls, xs: list, attribute: int) -> list:

        return list(set(example[attribute] for example in xs))
    
    
    @classmethod
    def examples_with_value(cls, xs: list,  ys: list, attribute: int, value: str) -> list:

        remaining_examples, remaining_labels = [], []
        for i in range(len(xs)):
            if xs[i][attribute] == value:
                remaining_examples.append(xs[i])
                remaining_labels.append(ys[i])
                
        return (remaining_examples, remaining_labels)
    
    @classmethod
    def find_best_attribute(
        cls, xs: list, ys: list, attributes: list) -> int:
        """Finds the best attribute to test for a set of examples and attributes.

        Args:
            examples (list): the list of examples, a list of lists
            attributes (list): a list of integers representing attribute indices

        Returns:
            int: [description]
        """
        highest_gain, best_attr = 0, 0

        # calculate current measure of information: entropy
        current_measure = DecisionTree.entropy(xs, ys)
        for attr in attributes:

            # calculate information gain for attribute
            current_gain = DecisionTree.information_gain(xs, ys, attr, current_measure)

            # find the highest information gain, store it
            # TODO: break ties with smaller label
            if current_gain > highest_gain:
                highest_gain, best_attr = current_gain, attr

        return best_attr    
        
    @classmethod
    def create(cls, xs, feature_class_sizes, ys, num_labels):
        '''Creates a DecisionTree to predict the output
           labels ys from the input feature vectors xs.
        :param xs (list of list): list of feature vectors, i.e. a 2D list of integers.
               xs[i][j] is the j'th feature of the i'th feature vector.
               len(xs[i]) == len(feature_class_sizes).
               xs[i][j] in {0, ..., feature_class_sizes[j] - 1}.
        :param feature_class_sizes (list): a list of integers, specifying the
               number of discrete values each input feature may take on.
        :param ys (list): a list of integers, representing class labels.
               ys[i] is the class label for corresponding to feature vector xs[i].
               ys[i] in {0, ..., num_labels - 1}.
        :param num_labels (int): an integer specifying the number of discrete
               values the output label may take on.
        :return: A DecisionTree instance, constructed using the information gain heuristic.
                 Throws ValueError if xs or ys values are out of range.'''
                 
        # test to ensure all xs and ys are in range, throw ValueError otherwise
        for i in range(len(xs)):
            for j in range(len(feature_class_sizes)):
                if xs[i][j] < 0 or xs[i][j] >= feature_class_sizes[j]:
                    raise ValueError("xs[{}][{}] = {} is out of range".format(i, j, xs[i][j]))
        for i in range(len(ys)):
            if ys[i] < 0 or ys[i] >= num_labels:
                raise ValueError("ys[{}] = {} is out of range".format(i, ys[i]))
        
        # test for stoppage
        # all same class, return Leaf node with that class
        if len(set(ys)) == 1:
            return DecisionLeafNode(ys[0])
        
        # examples empty
        if len(xs) == 0:
            
            # return most common output in all sibling nodes
            # find a way to do so
            pass
        
        # no attributes, return Leaf node with most common output in examples
        if len(feature_class_sizes) == 0:
            
            return DecisionLeafNode(DecisionTree.most_common_output(xs, ys))
            
        # pick attribute which maximises information gain
        attr = DecisionTree.find_best_attribute(xs, ys, list(range(len(feature_class_sizes))))
        
        # remove it from the feature list
        feature_class_sizes.pop(attr)
        
        # create a DecisionInternalNode with this attribute
        tree = DecisionInternalNode(attr, [])
        
        # and expand its children (populate branches with subtrees)
        for value in range(feature_class_sizes[attr]):
            examples_with_value, corresponding_labels = DecisionTree.examples_with_value(xs, attr, value)
            
            subtree = DecisionTree.create(examples_with_value, feature_class_sizes, corresponding_labels, num_labels)
            
            tree.add(value, subtree)
            
        # return the current tree
        return tree