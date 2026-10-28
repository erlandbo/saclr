- ImageNet1k : saclr-im1k
- Other image datasets: saclr-otherimages
- Detection: Convert resnet50_last.pth with convert-pretrain-to-detectron2_resnet50.py add weights to MoCo detection codebase: https://github.com/facebookresearch/moco
- Transfer-learning:   Use vissl https://github.com/facebookresearch/vissl with resnet50_last.pth.

## Citation
```
@article{OLSSON2026109098,
  title = {Stochastic approximation to contrastive learning},
  journal = {Neural Networks},
  volume = {203},
  pages = {109098},
  year = {2026},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2026.109098},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608026005587},
  author = {Erland Brandser Olsson and Zhirong Yang},
  keywords = {Contrastive learning, Negative samples, Self-supervised learning, Neighbor embedding},
  abstract = {Contrastive learning pulls positive samples (similar examples) closer and pushes negative samples (dissimilar examples) away to learn a meaningful mapping from inputs to outputs. It is found in a broad range of applications in computer vision that range from image classification to object detection to video processing and has a similarly broad impact and relevance in natural language processing, audio and speech, graphs, recommendation systems, and multimodal learning. Although contrastive learning done right typically achieves state-of-the-art results, there are major limitations with traditional methods because they rely on arbitrary definitions of positive and negative samples and are not decomposable in minibatch optimization. Thus, they require large batchsizes to effectively manage a tradeoff between positive and negative terms. This approach wastes significant computational resources on negative samples that may have minimal learning signals. To address these limitations, we propose a novel method that reformulates contrastive learning as a matrix approximation problem using I-divergence, a non-normalized variant of Kullback-Leibler divergence. Our objective function is decomposable across instance pairs, enabling efficient stochastic approximation algorithms that perform well with fewer negative samples by leveraging neighbor embeddings. Additionally, we generalize the scaling factor beyond standard normalization to adaptively emphasize positive samples with higher learning potential, reducing computational waste from negative samples. Our ambition is to demonstrate that even with a low batchsize and as few as one negative term per image, our method outperforms existing contrastive learning approaches and is competitive in overall performance to other state-of-the-art approaches on ImageNet which use a higher batchsize.}
}
```
