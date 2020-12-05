import math

        
class Leaf:
    def __init__(self, result) -> None:
        self.result = result
        
    def __call__(self, example) -> str:
        return self.result
    
    def __repr__(self) -> str:
        return ("RESULT: {}".format(self.result))
    
class DataSet:
    def __init__(self, examples: list, attribute_names: list, target_index: int) -> None:
        self.examples = examples
        
        # assuming list of attributes contains target
        attribute_names.pop(-1)
        self.attribute_names = attribute_names
        self.attributes = list(range(len(attribute_names)))
        self.target_index = target_index
        
        
    def most_common_value(self, examples: list, attribute: int) -> Leaf:
        # FIXME: is this ever used for other attributes, not the class?
        highest_count, highest_value = 0, 0
        for val in self.unique_values(examples, attribute):
            current_count = len(self.examples_with_value(examples, attribute, val))
            if current_count > highest_count:
                highest_count, highest_value = current_count, val
                
        return Leaf(highest_value)
    
    
    def unique_values(self, examples: list, attribute: int) -> list:

        return list(set(example[attribute] for example in examples))
    
    
    def examples_with_value(self, examples: list, attribute: int, value: str) -> list:
        
        return [example for example in examples if example[attribute] == value]
        
        

class DecisionNode:
    def __init__(self, attribute: int, attribute_name:str, default_child, branches: dict={}) -> None:
        self.attribute = attribute
        self.attribute_name = attribute_name
        self.branches = branches
        self.default_child = default_child

    def __call__(self, example: list) -> str:
        
        # if I have a branch corresponding to the value of 
        # my attribute that this example contains, call that
        # subtree
        pass
        
        # otherwise, return my default value
        return self.default_child        
    
    def add_branch(self, value: str, subtree) -> None:
        
        self.branches[value] = subtree
        
    def __repr__(self):
        return ("DecisionNode: {} {} {}".format(self.attribute, 
                                              self.attribute_name,
                                              self.branches))

    
class DecisionTreeLearner:
    def __init__(self, dataset: DataSet) -> None:
        self.dataset = dataset
        self.tree = self.decision_tree_learner(dataset.examples, dataset.attributes)
        
    def decision_tree_learner(self, examples, attributes, parent_examples=()) -> DecisionNode:
        
        # check for stoppage, return Leafs
        # no more attributes
        if len(attributes) == 0:
            return self.dataset.most_common_value(examples, self.dataset.target_index)
        
        # no more examples
        if len(examples) == 0:
            return self.dataset.most_common_value(parent_examples, self.dataset.target_index)
        
        # all examples same class, return the class
        if len(self.dataset.unique_values(examples, self.dataset.target_index)) == 1:
            return examples[0][self.dataset.target_index]
        
        # find the best attribute
        attr = self.find_best_attribute(examples, attributes)
        
        tree = DecisionNode(attr, 
                            self.dataset.attribute_names[attr],
                            self.dataset.most_common_value(examples, self.dataset.target_index))
        
        attributes.pop(attr)
        # create DecisionNode with attribute, populate branches recursively
        for val in self.dataset.unique_values(examples, attr):
            
            examples_with_val = self.dataset.examples_with_value(examples, attr, val)
            subtree = self.decision_tree_learner(examples_with_val, attributes, examples)
            
            tree.add_branch(val, subtree)
        
        return tree
    
    def entropy(self, examples):
        
        entropy = 0.0
        total_examples = len(examples)
        
        for unique_output in self.dataset.unique_values(examples, self.dataset.target_index):
            probability_of_output = len(self.dataset.examples_with_value(examples, self.dataset.target_index, unique_output))/total_examples
            entropy -= probability_of_output * math.log2(probability_of_output)
            
        return entropy
            
    
    def information_gain(self, examples: list, attr: int, current_entropy: float) -> float:
        
        total_examples = len(examples)
        average_child_entropy = 0.0
        
        for val in self.dataset.unique_values(examples, attr):
            examples_with_val = self.dataset.examples_with_value(examples, attr, val)
            average_child_entropy += len(examples_with_val) * self.entropy(examples_with_val)
        
        average_child_entropy /= total_examples
        
        return current_entropy - average_child_entropy
    
    def find_best_attribute(self, examples: list, attributes: list) -> int:
        
        highest_gain, best_attr = 0, 0
        
        # calculate current entropy
        current_entropy = self.entropy(examples)
        
        for attr in attributes:
            
            # calculate information gain for attribute
            current_gain = self.information_gain(examples, attr, current_entropy)
            
            # find the highest information gain, store it
            if current_gain > highest_gain:
                highest_gain, best_attr = current_gain, attr 
                
        return best_attr
    
def read_data():
    with open("examples.csv", "r") as f:
        raw_data = f.readlines()

    raw_data = [line.strip("\n") for line in raw_data]

    # get attributes and remove the target variable
    attributes = raw_data[0].split(",")

    # get training set, cast to numpy array for useful built-in functions
    training_set = [raw_data[i].split(",") for i in range(1, len(raw_data) - 1)]

    return training_set, attributes


if __name__ == "__main__":
    # run_tests()
    examples, attributes = read_data()
    aima_dataset = DataSet(examples, attributes, -1)
    tree = DecisionTreeLearner(aima_dataset)
    print(tree.tree)

    # classify one example
    print(
        tree.tree(
            ["yes", "yes", "yes", "yes", "full", "$", "no", "no", "burger", "30-60"]
        )
    )
