### What's in a decision tree?

1. **Structure**
   1. A `DataSet`: a set of examples containing different attributes, one of which—typically the last one—is the target variable/class/prediction. A set of attribute names, so that the tree is explainable. 
   2. A `DecisionNode`: a tree node which holds an attribute to test, a dictionary of values for that attribute and branches for those values and a default child, which is a `Leaf` that is used if this node is where the algorithm stops (cannot end on a `DecisionNode`)
   3. A `Leaf`: a tree node which simply holds a result. This result can be structured in different ways, but for simplicity it will just be the most common prediction in each Leaf node.
   4. A `DecisionTreeLearner`: a class that wraps the algorithm. It stores the DataSet and the resulting tree. We need the tree once it's built to predict new examples.
2. **Algorithm**
   1. Conditions for stoppage:
      1. When there are no more attributes to test, return a `Leaf` of the current result
      2. When there are no more examples, return a `Leaf` of the parent's result
      3. When all examples **have** the same prediction, return a `Leaf` of the current result
   2. Find the best attribute to test in this node
   3. Create a `DecisionNode`  with this attribute, and populate its branches (which are the attribute's possible values) with DecisionNodes containing examples for each branch and the remaining attributes.
   4. Return the current tree (= the `DecisionNode`, which points to its children).
3. **Auxiliary functions supporting 1 and 2**
   1. _DataSet-related_
      1. Finding the unique values for an attribute;
      2. Counting the number of examples with each value for an attribute;
      3. Finding the most common value for an attribute;
   2. _DecisionNode-related_
      1. A way to propagate classification through nodes of an already-existing tree;
      2. A way to add subtrees to branches
      3. A way to print the node and its descendants
   3. _Leaf-related_
      1. A way to print the `Leaf` result
   4. _Algorithm-related_
      1. Finding the best attribute using information gain as a measure
      2. Calculating information gain
      3. Calculating entropy for a set of examples
   