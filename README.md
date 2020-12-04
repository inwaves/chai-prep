chai-prep
---

What's in a decision tree?
- structured data
  - examples consisting of features + an output label
  - a list of feature names
  - the index of the output label/target
- nodes
  - decision/fork nodes: contain an attribute, a dictionary of branches, {val: subtree} pairs where val is each value of the attribute
  - leaves: contain just a result, an output
- the learning algorithm

What does the learning algorithm do?
- it finds the best attribute according to some measure
- it uses the current node to expand values for that attribute
- it stores the possible values as branches leading to other nodes
- it does this until:
  [x] there are no more attributes to check——it returns the most common class in the current node
  [x] all the examples in this node have the same output——it returns the result
  [x] there are no more examples (no examples had this value of the previous attribute)——it returns the result of its parent

What auxiliary functions does it need?
- something to calculate the best attribute
- something to calculate the information gain
[x] something to find examples with a given value
[x] something to find unique values for an attribute
[x] something to calculate the frequency of each value of an attribute among a set of examples