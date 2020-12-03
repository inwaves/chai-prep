# need these in this file, will sort out a better way later
attributes = ['alternate', 'bar', 'frisat', 'hungry', 'patrons', 'price', 'raining', 'reservation', 'type', 'waitestimate']

class Decision:
    """Inspired by decision_tree.py Question class
    """
    
    def __init__(self, attribute_index, value) -> None:
        self.attributes = attributes
        self.attribute_index = attribute_index
        self.value = value
        
    def match(self, example) -> bool:
        # FIXME: could make this more robust
        return example[self.column] == self.value
    
    def __repr__(self) -> str:
        return "Is {} == {}?".format(attributes[self.attribute_index], self.value)

            
class Node:
    
    def __init__(self, examples) -> None:
        self.examples = examples

class DecisionNode(Node):
    def __init__(self, decision, examples):
        self.decision = decision
        self.examples = examples
        self.children = []
        
    def __repr__(self) -> str:
        # TODO: printing; I need to see this tree
        pass
        
class Leaf(Node):
    def __init__(self, examples) -> None:
        super().__init__(self, examples)
        

