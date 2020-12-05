import math


class DataSet:
    def __init__(self, examples: list, attr_names: list, target_index: int) -> None:
        self.examples = examples

        attr_names.pop(target_index)
        self.attr_names = attr_names
        self.attr = list(range(len(attr_names)))
        self.target_index = target_index  # which column of the example is the output

    def examples_with_value(self, examples: list, attribute: int, value: str) -> list:
        return [example for example in examples if example[attribute] == value]

    def unique_values(self, examples: list, attr: int) -> list:
        return list(set(example[attr] for example in examples))

    def most_common_value_of_attribute(self, examples: list, attr: int) -> str:
        unique_vals = self.unique_values(examples, attr)
        max_count, max_val = 0, 0

        for val in unique_vals:
            current_count = len(self.examples_with_value(examples, attr, val))
            if current_count > max_count:
                max_count, max_val = current_count, val

        return Leaf(max_val)


class Leaf:
    def __init__(self, result) -> None:
        self.result = result

    def display(self):
        print("RESULT =", self.result)

    def __repr__(self):
        return repr(self.result)


class DecisionNode:
    def __init__(
        self, attr: int, attr_name: str, branches=None, default_child=None
    ) -> None:
        self.attr = attr
        self.attr_name = attr_name
        self.branches = branches or {}
        self.default_child = default_child

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""

        # takes a subtree and a value, and adds that tree as a branch with that value
        print(
            "Just added branch {}={} as subtree {} {}".format(
                self.attr, val, type(subtree), subtree
            )
        )
        self.branches[val] = subtree

    def __repr__(self) -> str:
        print("Attr: {}".format(self.attr))

    def display(self, indent):
        name = self.attr_name
        print("Test", name)
        for (val, subtree) in self.branches.items():
            print(" " * 4 * indent, name, "=", val, "==>", end=" ")
            print(subtree)
            # subtree.display(indent+1)

    def __repr__(self):
        return "DecisionNode({0!r}, {1!r}, {2!r})".format(
            self.attr, self.attr_name, self.branches
        )


class DecisionTreeLearner:
    def __init__(self, dataset: DataSet) -> None:
        self.dataset = dataset
        self.tree = self.decision_tree_learner(dataset.examples, dataset.attr)

    def decision_tree_learner(
        self, examples: list, attributes: list, parent_examples: list = []
    ) -> DecisionNode:

        # check for stoppage first
        # no more attributes
        if len(attributes) == 0:
            # print(
            #     "No more attributes, returning this Leaf: {}".format(
            #         self.dataset.most_common_value_of_attribute(
            #             examples, self.dataset.target_index
            #         )
            #     )
            # )
            return self.dataset.most_common_value_of_attribute(
                examples, self.dataset.target_index
            )

        # all examples have the same output
        unique_outputs = self.dataset.unique_values(examples, self.dataset.target_index)
        if len(unique_outputs) == 1:
            # print(
            #     "All examples have the same output, returning this Leaf: {}".format(
            #         Leaf(unique_outputs[0])
            #     )
            # )
            return Leaf(unique_outputs[0])

        # there are no more examples
        if len(examples) == 0:
            # print(
            #     "No more examples, returning this Leaf: {}".format(
            #         self.dataset.most_common_value_of_attribute(
            #             parent_examples, self.dataset.target_index
            #         )
            #     )
            # )
            return self.dataset.most_common_value_of_attribute(
                parent_examples, self.dataset.target_index
            )

        # no stoppage? continue with algorithm
        # find the best attribute
        attr = self.find_best_attribute(examples, attributes)

        tree = DecisionNode(
            attr,
            self.dataset.attr_names[attr],
            default_child=self.dataset.most_common_value_of_attribute(
                examples, self.dataset.target_index
            ),
        )

        attributes.pop(attr)  # remove the attribute we've just used

        # use the current node to expand values for that attribute
        for val in self.dataset.unique_values(examples, attr):
            # print(
            #     "Checking attr {} = {}\nGoing into subtree:".format(
            #         self.dataset.attr_names[attr], val
            #     )
            # )
            subtree = self.decision_tree_learner(
                self.dataset.examples_with_value(examples, attr, val),
                attributes,
                examples,
            )
            # print("Just created: {}".format(subtree))
            tree.add(val, subtree)
        return tree

    def find_best_attribute(self, examples: list, attributes: list) -> int:

        # go through the list of attributes,
        # calculate its information gain
        max_gain, best_attr = 0, 0
        for attr in attributes:
            current_gain = self.information_gain(examples, attr)
            if current_gain > max_gain:
                max_gain, best_attr = current_gain, attr

        # print("The best attribute currently is: {}, {}".format(best_attr, self.dataset.attr_names[best_attr]))
        return best_attr

    def information_gain(self, examples: list, attr: int) -> float:

        # calculate the current entropy
        current_entropy = self.entropy(examples)
        total_examples = len(examples)
        average_child_entropy = 0.0

        # calculate the average entropy of child nodes
        for val in self.dataset.unique_values(examples, attr):
            examples_with_val = self.dataset.examples_with_value(examples, attr, val)
            average_child_entropy += len(examples_with_val) * self.entropy(
                examples_with_val
            )

        average_child_entropy /= total_examples

        # return difference, i.e. information gain
        return (
            current_entropy - average_child_entropy
            if current_entropy > average_child_entropy
            else 0.0
        )

    def entropy(self, examples: list) -> float:
        total_examples = len(examples)
        unique_outputs = self.dataset.unique_values(examples, self.dataset.target_index)
        entropy = 0.0
        for output in unique_outputs:
            output_probability = (
                len(
                    self.dataset.examples_with_value(
                        examples, self.dataset.target_index, output
                    )
                )
                / total_examples
            )
            entropy -= output_probability * math.log2(output_probability)

        return entropy


def read_data():
    with open("examples.csv", "r") as f:
        raw_data = f.readlines()

    raw_data = [line.strip("\n") for line in raw_data]

    # get attributes and remove the target variable
    attributes = raw_data[0].split(",")

    # get training set, cast to numpy array for useful built-in functions
    training_set = [raw_data[i].split(",") for i in range(1, len(raw_data) - 1)]

    return training_set, attributes


def run_tests():
    training_data = [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
        ["Red", 1, "Grape"],
        ["Red", 1, "Grape"],
        ["Yellow", 3, "Lemon"],
    ]
    attributes = ["Colour", "Diameter"]

    dataset = DataSet(training_data, attributes, -1)
    assert dataset.examples_with_value(training_data, 0, "Green") == [
        ["Green", 3, "Apple"]
    ]
    assert dataset.examples_with_value(training_data, 1, 3) == [
        ["Green", 3, "Apple"],
        ["Yellow", 3, "Apple"],
        ["Yellow", 3, "Lemon"],
    ]
    assert (
        dataset.unique_values(training_data, 0).sort()
        == ["Yellow", "Green", "Red"].sort()
    )
    assert dataset.unique_values(training_data, 1).sort() == [1, 3].sort()
    assert (
        dataset.unique_values(training_data, -1).sort()
        == ["Apple", "Grape", "Lemon"].sort()
    )
    assert dataset.most_common_value_of_attribute(training_data, 0) in ["Yellow", "Red"]
    assert dataset.most_common_value_of_attribute(training_data, 1) == 3
    assert dataset.most_common_value_of_attribute(training_data, -1) in [
        "Grape",
        "Apple",
    ]

    tree = DecisionTreeLearner(dataset)
    assert tree.entropy(training_data) == 1.5219280948873621
    assert tree.information_gain(training_data, 0) == 1.1219280948873621
    assert tree.information_gain(training_data, 1) == 0.9709505944546684
    assert tree.find_best_attribute(training_data, range(len(attributes))) == 0


if __name__ == "__main__":
    # run_tests()
    examples, attributes = read_data()
    aima_dataset = DataSet(examples, attributes, -1)
    tree = DecisionTreeLearner(aima_dataset)
    print(tree.tree)
