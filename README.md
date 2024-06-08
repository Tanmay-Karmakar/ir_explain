# ir_explain

`ir_explain` is an open-source Python library that implements a variety
of well-known techniques for Explainable IR (ExIR) within
a common, extensible framework. It supports the three standard
categories of post-hoc explanations, namely pointwise, pairwise,
and listwise explanations.

<!-- `ir_explain` is a post-hoc explainability library of IR. It consists of three componets, i) pointwise, ii) pairwise, iii) listwise.
-->
  ```
@misc{saha2024irexplain,
      title={ir_explain: a Python Library of Explainable IR Methods}, 
      author={Sourav Saha and Harsh Agarwal and Swastik Mohanty and Mandar Mitra and Debapriyo Majumdar},
      year={2024},
      eprint={2404.18546},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
 ```
## Setup

Install via GitHub repository:
```
git clone https://github.com/souravsaha/ir_explain
```

## Installing requirements

The necessary libraries to support `ir_explain` are mentioned in the requirements.txt file.

```
pip3 install -r requirements.txt
```
ir_explain is tested on Python 3.8 and 3.9.

## Usage

We have uploaded notebooks in `ir_explain/examples/` to demonstrate how the modules in `ir_explain` can be used to generate post-hoc explanations

## Pointwise example
`ir_explain/examples/pointwise_demo.ipynb` shows a sample usage of the Pointwise methods EXS and LIRME to generate post-hoc pointwise explanations
 
 The EXS module can be instantiated using the class `ExplainableSearch`
 
 During initialization, the neural reranker and the parameters such as classification method (svm or l.r.) are needed. 

```python
exs_explainer = ExplainableSearch(reranker, 'svm', num_samples=100)
```

Similarly, the lirme module can be instantiated using the class `Lirme`

During initialization, Lirme requires the index_path to be given, so that it can fetch documents and the tf-idf data of the index, which is required while generating the explanations

```python
lirme_explainer = Lirme(index_path)
```

The notebook shows the different functions and how they can be used to generate the explanations using these two methodologies

## Pairwise example
`ir_explain/examples/pairwise/Pairwise_Example_1.ipynb` shows a sample usage of the Pairwise module. The details axiom list can be found at `ir_explain/pairwise/README.md/`.

To use pairwise component, go to the `ir_explain/pairwise` folder, initialize it with the query, two documents, and the index path as follows:

```python
from axioms import pairwise

pairwise = pairwise(query, doc1, doc2, index_path)

# instances of various axiom classes
axiom_classes = ["TFC1", "STMC1", ...]

pairwise.explain(axiom_classes)

```




## Listwise example
`ir_explain/examples/listwise/` folder shows all four Listwise explanation examples. 

To instantiate the `BFS` explainer of the listwise component, go to the `ir_explain/listwise` folder and do the following: 

```python
from bfs_explainer import BFS

# initialize the parameters of BFS
params = {
    "QUEUE_MAX_DEPTH" : 1000,
    "BFS_MAX_EXPLORATION" : 30,
    "BFS_VOCAB_TERMS" : 30,
    "BFS_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    "CORRELATION_MEASURE" : "RBO",
    }

exp_model = "bm25"

# initialize the BFS class
bfs = BFS(index_path, exp_model, params)
```
All other methods and calling conventions are mentioned in `ir_explain/examples/listwise/`.

## Reproducibility experiment

We have conducted a subset of the reproducibility experiments of BFS and Greedy based listwise explainers on the TREC 2019 topic set. The figure shows the reproduced versions of MAP and RBO are denoted as MAP(reprod.) and RBO(reprod.) respectively. As of now, we have reported figures for the entire topic set. 

![image](https://github.com/souravsaha/ir_explain/blob/main/examples/reproducibility-table.png)


