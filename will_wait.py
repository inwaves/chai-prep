import numpy as np
from components import Decision, DecisionNode, Leaf
########
# input: a numpy array with shape (12, 10)
# output: a boolean value, yes or no
########


def read_data():
    with open("examples.csv", 'r') as f:
        raw_data = f.readlines()
    
    raw_data = [line.strip("\n") for line in raw_data]
    
    # get attributes and remove the target variable
    attributes = raw_data[0].split(',')
    attributes = attributes[:-1] 
    
    # get training set, cast to numpy array for useful built-in functions
    training_set = [raw_data[i].split(",") for i in range(1, len(raw_data)-1)]
    training_set = np.array(training_set)
    
    print(attributes)

def entropy(examples):
    probability_true = len(examples[examples[-1]=="yes"])/len(examples)
    entropy = -probability_true * np.log2(probability_true) - (1-probability_true) * np.log2(1-probability_true)
    return entropy

def average_entropy_over_examples(children_examples, children_entropies):
    total_examples = np.sum([len(partition) for partition in children_examples.values()])
    s = 0.0
    for attribute_value, partition in children_examples.iteritems():
        s += len(partition) * children_entropies[attribute_value]
        
    return s / total_examples
  
def find_best_attribute_and_children(attributes, examples):
    
    best_gain, best_attr, best_examples = 0, attributes[0], []
    
    # calculate the current entropy, use it to measure information gain
    current_entropy = entropy(examples)
    
    for attr in attributes:
        # partition the example set according to values for this attribute
        examples_per_value = {}
        for val in np.unique(examples[attributes.index(attr)]):
            examples_per_value[val] = examples[examples[attributes.index(attr)]==val]
            
        # calculate the entropy for all these resulting partitions
        children_entropies = {}
        for child_value, child_examples in examples_per_value.iteritems():
            children_entropies[child_value] = entropy(child_examples)
        
        # the information gain is the expected reduction in entropy from attribute attr
        current_gain = current_entropy - average_entropy_over_examples(examples_per_value, children_entropies)
        
        # store the gain and the resulting child nodes, so we don't duplicate work later
        if current_gain > best_gain:
            best_gain, best_attr, best_examples = current_gain, attr, examples_per_value
    
    return best_attr, best_examples
              
def decision_tree_learning(examples, attributes, parent_examples):
    
    # stop when your current examples are classified the same (only one output in your leaf node)
    if len(np.unique(examples[:, -1])) == 1:
        return examples[0, -1]
    
    # or when you have no examples
    if examples == None:
        return np.bincount(parent_examples[:, -1]).argmax()
    
    # or when you have no more attributes to test
    if len(attributes) == 0:
        return np.bincount(examples[:, -1]).argmax()
    
    # at each step, pick the best attribute
    attr, child_examples = find_best_attribute_and_children(examples, attributes)
    
    # and recursively generate a subtree with the remaining examples and attributes
    remaining_attributes = attributes
    remaining_attributes.pop(attr)
    for  value_examples in child_examples.values():
        decision_tree_learning(value_examples, remaining_attributes, examples)
        
    return examples
    
if __name__ == '__main__':
    read_data()
    