## Datasets
This folder "datasets" contains three datasets used in DMTE, including Cora, DBLP and Zhihu. In each dataset, there are two files named "data.txt" and "graph.txt".

* data.txt: Each line represents the text information of a vertex.    
* graph.txt: The edgelist file of current social network.

Besides, there is an additional "group.txt" file in Cora and DBLP.

* group.txt: Each vertex in Cora has been annotated with a label. This file can be used for vertex classification.

## Run
Run the following command for training:

    python train.py

## Dependencies
* Tensorflow == 1.12
* python == 2.7
