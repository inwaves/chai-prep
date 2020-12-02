import numpy as np


# input is an (m x n) numpy matrix of the form
#       a1  a2  a3  ... an  b
# x1    v11 v12 v13 ... v1n b1
# x2    v21 v22 v23 ... v2n b2
# x3    v31 v32 v33 ... v3n b3
# ...   ... ... ... ... ... ...
# xm    vm1 vm2 vm3 ... vmn bm
# where x are examples, a are attributes and b is the output for that example

class DecisionTree:
    def __init__(self, training_set, attributes):
        """Constructor for the DecisionTree class. Splits labels from features in training set.
        
        Args:
            training_set (np.ndarray): the training set, an (m x n) matrix of attribute values for all examples, last column is the outputs/labels
            attributes (list): a list of attributes to structure this decision tree around
        """
        
        self.tree = self.decision_tree_learning(training_set, training_set, attributes)
        
    def decision_tree_learning(self, training_set, parent_training_set, attributes):
        
        # if no examples for this tree, return the most common output for parent examples
        if training_set == None:
            return np.bincount(parent_training_set[:, -1]).argmax()
        
        # if all examples have the same output, return it
        if len(np.unique(training_set[:, -1])) == 1:
            return training_set[0, -1]
        
        # if there are no more attributes, return the most common output
        if len(attributes) == 0:
            return np.bincount(training_set[:, -1]).argmax()
        
        # find the most important attribute using information gain
        best_gain, best_attr = 0, attributes[0]
        for attr in attributes:
            current_gain = importance(attr, training_set) 
            if current_gain > best_gain:
                best_gain = best_gain
                best_attr = attr
                
        # start a new tree whose root is the test node for that attribute
        remaining_attributes = [attr for attr in attributes if attr != best_attr]
        tree = self.decision_tree_learning(training_set, remaining_attributes)
        
        # get each value of the attributes
        unique_attribute_values = np.unique(training_set[:, attributes.index(best_attr)])
        
        # for each value of the attribute
        for value in unique_attribute_values:
            # get all the examples whose attribute value is that value: exs
            exs = [x for x in training_set if x[attributes.index(best_attr)] == value]
            
            # generate a subtree: decision_tree_learning(exs, all attributes except the one picked, current examples)
            subtree = self.decision_tree_learning(exs, training_set, remaining_attributes)
            
            # add a new branch for the current tree for the current value of the attribute; its subtree is the one generated above
            # TODO: is this a good way to represent branches?
            
            branch = best_attr + " = " + value
            tree[branch] = subtree

        return tree
        
    def importance(self, attr, training_set):
        # calculate the information gain of the attribute as the reduction in entropy
        
        # calculate total entropy for the output
        # TODO: implement the information gain
        pass