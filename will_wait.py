import math


class DataSet:
    def __init__(self, examples: list, attr_names: list, target_index: int) -> None:
        self.examples = examples
        self.attr_names = attr_names
        self.attr = range(len(attr_names))
        self.target_index = target_index  # which column of the example is the output

    def examples_with_value(self, examples: list, attribute: int, value: str) -> list:
        return [example for example in examples if example[attribute] == value]

    def unique_values(self, examples: list, attr: int) -> set:
        return set(example[attr] for example in examples)

    def most_common_value_of_attribute(self, examples: list, attr: int) -> str:
        unique_vals = self.unique_values(examples, attr)
        max_count, max_val = 0, 0

        for val in unique_vals:
            current_count = self.examples_with_value(examples, attr, val)
            if current_count > max_count:
                max_count, max_val = current_count, val

        return max_val


class Leaf:
    def __init__(self, result) -> None:
        self.examples = result


class DecisionNode:
    def __init__(self, attr: str, branches: dict = None) -> None:
        self.attr = attr
        self.branches = branches


class DecisionTreeLearner:
    def __init__(self, dataset: DataSet) -> None:
        self.dataset = dataset

    def decision_tree_learner(
        self, examples: list, attributes: list, parent_examples: list
    ) -> DecisionNode:

        # check for stoppage first
        # no more attributes
        if len(attributes) == 0:
            return self.dataset.most_common_value_of_attribute(
                examples, self.dataset.target_index
            )

        # all examples have the same output
        unique_outputs = self.dataset.unique_values(examples, self.dataset.target_index)
        if len(unique_outputs) == 1:
            return unique_outputs[0]

        # there are no more examples
        if len(examples) == 0:
            return self.dataset.most_common_value_of_attribute(
                parent_examples, self.dataset.target_index
            )

        # no stoppage? continue with algorithm
        # find the best attribute
        attr = self.find_best_attribute(examples, attributes)
        tree = DecisionNode(attr)

        attributes.pop(attr)  # remove the attribute we've just used

        # use the current node to expand values for that attribute
        for val in self.dataset.unique_values(examples, attr):
            subtree = self.decision_tree_learner(
                self.dataset.examples_with_value(examples, attr, val),
                attributes,
                examples,
            )
            tree.branches[val] = subtree

        return tree

    def find_best_attribute(self, examples: list, attributes: list) -> int:

        # go through the list of attributes,
        # calculate its information gain
        max_gain, best_attr = 0, 0
        for attr in attributes:
            current_gain = self.information_gain(examples, attr)
            if current_gain > max_gain:
                max_gain, best_attr = current_gain, attr

        return best_attr

    def information_gain(self, examples: list, attr: int) -> float:

        # calculate the current entropy
        current_entropy = self.entropy(examples)
        total_examples = len(examples)
        average_child_entropy = 0.0

        # calculate the average entropy of child nodes
        for val in self.dataset.unique_values(examples, attr):
            examples_with_val = self.dataset.examples_with_value(examples, attr, val)
            average_child_entropy += len(
                examples_with_val + self.entropy(examples_with_val)
            )

        average_child_entropy /= total_examples

        # return difference, i.e. information gain
        return current_entropy - average_child_entropy

    def entropy(self, examples: list) -> float:
        total_examples = len(examples)
        unique_outputs = self.dataset.unique_values(examples, self.dataset.target_index)
        entropy = 0.0
        for output in unique_outputs:
            output_probability = (
                self.dataset.examples_with_value(
                    examples, self.dataset.target_index, output
                )
                / total_examples
            )
            entropy -= output_probability * math.log2(output_probability)
            
        return entropy

if __name__ == "__main__":
    