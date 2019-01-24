# DMTE
This repository is the implementation for paper [Diffusion Maps for Textual Network Embedding](https://arxiv.or/pdf/1805.09906.pdf)

## Dependencies
* Tensorflow == 1.12
* python == 2.7

## Run
Run the following command for training:

    python train.py

Run the following command for testing:

    python auc.py

## Datasets
This folder "datasets" contains three datasets used in DMTE, including Cora, DBLP and Zhihu. In each dataset, there are two files named "data.txt" and "graph.txt".

* data.txt: Each line represents the text information of a vertex.    
* graph.txt: The edgelist file of current social network.

Besides, there is an additional "group.txt" file in Cora and DBLP.

* group.txt: Each vertex in Cora has been annotated with a label. This file can be used for vertex classification.

## Reference
The implementation of this paper is based on [CANE](https://github.com/thunlp/CANE)

## Cite
Please cite our paper if it helps with your research
```latex
@inproceedings{zhang2018adversarial,
  title={Diffusion Maps for Textual Network Embedding},
  author={Zhang, Xinyuan and Li, Yitong and Shen, Dinghan and Carin, Lawrence},
  Booktitle={NeurIPS},
  year={2018}
}
```
