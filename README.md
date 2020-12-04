chai-prep
---

### What's in a decision tree?
- structured data
  [x] examples consisting of features + an output label
  [x] a list of feature names
  [x] the index of the output label/target
- nodes
  [x] decision/fork nodes: contain an attribute, a dictionary of branches, {val: subtree} pairs where val is each value of the attribute
  [x] leaves: contain just a result, an output
- the learning algorithm

### What does the learning algorithm do?
[x] it finds the best attribute according to some measure
[x] it uses the current node to expand values for that attribute
[x] it stores the possible values as branches leading to other nodes
- it does this until:
  [x] there are no more attributes to check——it returns the most common class in the current node
  [x] all the examples in this node have the same output——it returns the result
  [x] there are no more examples (no examples had this value of the previous attribute)——it returns the result of its parent

### What auxiliary functions does it need?
[x] calculate the best attribute
[x] calculate the information gain
    [x] calculate entropy
[x] find examples with a given value
[x] find unique values for an attribute
[x] calculate the frequency of each value of an attribute among a set of examples
[] good representations for all types of data so the tree output is understandable