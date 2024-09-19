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
```

Make sure it returns Python 3.x. If not, upgrade to Python 3. This project was developed using Python 3.10.

## Setup
### Step 1: Download the Repository
Clone or download the repository to your local machine.

### Step 2: Navigate to the Project Directory
Find the directory where your computer saved the project. Unzip the file if necessary, and use the following command to navigate to the working directory:

```sh
cd /path/to/project/directory
```

### Step 3: Create a Virtual Environment
Create a virtual environment called env in the working directory:

```sh
python3 -m venv env
```

Then activate the virtual environment:
```sh
source env/bin/activate
```

### Step 4: Install Dependencies

Install the necessary dependencies listed in requirements.txt:

```sh
pip3 install -r requirements.txt
```

This will ensure compatibility with specific software versions.

## Running the Program

### Step 5: Run the Training Script
You can run the training script with the following command:

```sh
python3 ./src/train.py --model CNNPhyloNet --learning_rate 0.001
```

Alternatively, use the batch training script:

```sh
chmod +x batch_train.sh
./batch_train.sh
```

### Step 6: Run the Test Script
To test the model, run:

```sh
python3 test.py --model CNNPhyloNet --learning_rate 0.001 --clade plants
```

You can also run the batch test script:

```sh
chmod +x batch_test.sh
./batch_test.sh
```

## License
This project is distributed under the MIT License. See the LICENSE file for more information.
