## An Exploration of Conditioning Methods in Graph Neural Networks

This repository contains the source code accompanying the paper "An Exploration of Conditioning Methods in Graph Neural Networks" ([arXiv](https://arxiv.org/abs/2305.01933)). This work was accepted to Machine Learning for Drug Discovery workshop at ICLR'23 ([MLDD](https://sites.google.com/view/mldd-2023/accepted-papers_1)).

### Abstract
The flexibility and effectiveness of message passing based graph neural networks (GNNs) induced considerable advances in deep learning on graph-structured data. In such approaches, GNNs recursively update node representations based on their neighbors and they gain expressivity through the use of node and edge attribute vectors. E.g., in computational tasks such as physics and chemistry usage of edge attributes such as relative position or distance proved to be essential. In this work, we address not what kind of attributes to use, but how to condition on this information to improve model performance. We consider three types of conditioning; weak, strong, and pure, which respectively relate to concatenation-based conditioning, gating, and transformations that are causally dependent on the attributes. This categorization provides a unifying viewpoint on different classes of GNNs, from separable convolutions to various forms of message passing networks. We provide an empirical study on the effect of conditioning methods in several tasks in computational chemistry.

### About the code
We provide code for three types of conditional methods in `layers/conditional.py`. Just add ConditionalLayer with specified conditioning method (weak, strong, pure) on top of message network layers of your GNN model.
