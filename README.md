# Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions

[Tobias Schmidt](https://dynamical-inference.ai/authors/tobias-schmidt/), [Steffen Schneider](https://dynamical-inference.ai/) and [Matthias Bethge](https://bethgelab.org/)

*Accepted at NeurIPS 2025.*

[`[Paper]`](https://openreview.net/forum?id=kvI0QTVRQD) [`[Preprint]`](https://arxiv.org/abs/2510.21706) [`[Poster]`](https://neurips.cc/virtual/2025/loc/san-diego/poster/116330)

🚧 *This repository is currently under construction. You can follow [issue #1](https://github.com/dynamical-inference/ebc/issues/1) if you'd like to be notified about the first code release.*

## Abstract

We propose Equivariance by Contrast (EbC) to learn equivariant embeddings from observation pairs (***y**, g · **y***), where *g* is drawn from a finite group acting on the data. Our method jointly learns a latent space and a group representation in which group actions correspond to invertible linear maps—without relying on group-specific inductive biases. We validate our approach on the infinite dSprites dataset with structured transformations defined by the finite group *G* := (Rm × Zn × Zn), combining discrete rotations and periodic translations. The resulting embeddings exhibit high-fidelity equivariance, with group operations faithfully reproduced in latent space. On synthetic data, we further validate the approach on the nonabelian orthogonal group *O(n)* and the general linear group *GL(n)*. We also provide a theoretical proof for identifiability. While broad evaluation across diverse group types on real-world data remains future work, our results constitute the first successful demonstration of general-purpose encoder-only equivariant learning from group action observations alone, including non-trivial non-abelian groups and a product group motivated by modeling affine equivariances in computer vision.

## Result

<img width="1105" height="837" alt="image" src="https://github.com/user-attachments/assets/a4fc8ea7-db18-4976-9c96-97558db8e44e" />


## Method

<img width="1483" height="677" alt="image" src="https://github.com/user-attachments/assets/d086e8a9-17e8-4205-a3ec-8b025bd0e869" />

<img width="1517" height="572" alt="image" src="https://github.com/user-attachments/assets/9ff414b7-42d5-4e38-83f5-1954b3a93c00" />

## Citation

```
@inproceedings{
  schmidt2025ebc,
  title={Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions},
  author={Tobias Schmidt and Steffen Schneider and Matthias Bethge},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=kvI0QTVRQD}
}
```
