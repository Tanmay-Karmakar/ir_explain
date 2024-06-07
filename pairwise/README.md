# Pairwise Scoring with Axioms

This repository contains resources for determining pairwise scores between two documents given a query. All the axioms are stored in the `axioms.py` file. You can import the `pairwise` class from there to calculate the pairwise score using various axioms.

## Axioms

The available axioms are:
- TFC1
- TFC3
- PROX1
- PROX2
- PROX3
- PROX4
- PROX5
- LNC1
- LNC2
- LB1
- STMC1
- AND
- REG
- DIV

## Usage

### Initializing Pairwise

To use the `pairwise` class, initialize it with the query, two documents, and the index path:

```python
from axioms import pairwise

pairwise = pairwise(query, doc1, doc2, index_path)
```
### Defining Axiom Classes

Create a list of axiom classes that you want to use:

```python
axiom_classes = ["TFC1", "STMC1", ...]
```

You can also define abstract classes using binary (+, -, *, /) and unary (-) operators on the axioms. For example:
```python
axiom_classes = ["TFC1+TF_LNC/PROX1"]
```
### Explaining Axiom Results

For getting the score by the list of selected axioms use:

```python
pairwise.explain(axiom_classes)
```

For detailed explanations of the results, use the explain_details method from the explained_details class:

```python
from explain_more import explain_details

pairwise.explain_details(query, doc1, doc2, axiomName)
```

### Examples
For further details and usage examples, refer to the example notebook located in the examples directory under pairwise.
