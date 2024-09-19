# SilvaModel
This project explores which machine learning architecture best infers phylogenies using unsupervised learning on multi-hot encoded SILVA dataset sequences. 
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="img/logo.png" width="350" title="logo">
  </a>

<h3 align="center">SilvaModel: DNA Sequence Embedding and Phylogenetic Analysis </h3>

  <p align="center">
  This repository is a tool for embedding DNA sequences using various deep learning models, performing hierarchical clustering on these embeddings, and analyzing the resulting phylogenetic trees. SilvaModel also supports visualizing the embeddings through PCA and calculating Robinson-Foulds distances between generated and reference trees.
    <br />

<!-- ABOUT THE PROJECT -->
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
  
## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.
We used version 3.10 to develop SilvaModel.

## Setup

### Step 1: Download the repository

Download it to your computer. 

### Step 2: Unpack and change directory to the repository

Find where your computer saved the project. 
Unzip/unpack/decompress it, then enter:

  ```sh
  cd /path/to/project/directory
  ```

This is now the working directory.

### Step 3: Create a virtual environment: 
The default size limit on PyPI is 60MB.
Therefore, we will have to take the **virtual environment** route.

Create a virtual environment called *env* inside the working directory.

```sh
python3 -m venv env
```

Then, activate the virtual environment.


```sh
source env/bin/activate
```

### Step 4: Install requirements.txt

Avoid "dependency hell" by installing specific software versions known to work well together.

  ```sh
pip3 install -r requirements.txt
  ```

### Step 5: Run the training script

```sh
python3 ./src/train.py --model CNNPhyloNet --learning_rate 0.001
```
Alternatively, you can run the batch train script
```sh
chmod +x batch_train.sh

./batch_train.sh
```

### Step 6: Run the test script

```sh
python3 test.py --model CNNPhyloNet --learning_rate 0.001 --clade plants
```
Alternatively, you can run the batch test script
```sh
chmod +x batch_test.sh

./batch_test.sh
```


<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
