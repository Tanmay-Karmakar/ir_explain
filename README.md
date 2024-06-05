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

```
exs_explainer = ExplainableSearch(reranker, 'svm', num_samples=100)
```

Similarly, the lirme module can be instantiated using the class `Lirme`

During initialization, Lirme requires the index_path to be given, so that it can fetch documents and the tf-idf data of the index, which is required while generating the explanations

```
lirme_explainer = Lirme(index_path)
```

The notebook shows the different functions and how they can be used to generate the explanations using these two methodologies

## Reproducibility experiment

We have conducted a subset of the reproducibility experiments of BFS and Greedy based listwise explainers on the TREC 2019 topic set. The figure shows the reproduced versions of MAP and RBO are denoted as MAP(reprod.) and RBO(reprod.) respectively. As of now, we have reported figures for the entire topic set. 

![image](https://github.com/souravsaha/ir_explain/blob/main/examples/reproducibility-table.png)


