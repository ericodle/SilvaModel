# SilvaModel: DNA Sequence Embedding and Phylogenetic Analysis

This project explores which machine learning architecture best infers phylogenies using unsupervised learning on multi-hot encoded SILVA dataset sequences.

<p align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="img/logo.png" width="350" title="logo">
  </a>
</p>

SilvaModel is a tool for embedding DNA sequences using various deep learning models, performing hierarchical clustering on these embeddings, and analyzing the resulting phylogenetic trees. SilvaModel also supports visualizing the embeddings through PCA and calculating Robinson-Foulds distances between generated and reference trees.

---

## Features

- Supports multiple models for embedding DNA sequences, including:
  - CNNPhyloNet
  - MLPPhyloNet
  - LSTMPhyloNet
  - Transformer-based PhyloNet (TrPhyloNet)
  - Autoencoder PhyloNet (AePhyloNet)
  - Diffusion PhyloNet (DiffPhyloNet)
- Calculates pairwise distances and generates dendrograms using hierarchical clustering.
- Provides visualizations of DNA sequence embeddings using PCA.
- Generates phylogenetic trees in Newick format with branch support values.
- Calculates Robinson-Foulds distance between trees.

---

## Prerequisite

Ensure [Python3](https://www.python.org/downloads/) is installed on your computer.

To verify the version, enter the following in your terminal:

```sh
python --version
