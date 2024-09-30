import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ete3 import Tree
import csv
import numpy as np
import time
import logging
import torch.nn.functional as F
import sys
import os
import argparse

# Adjust sys.path to include the directory containing models.py
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, '..', 'models')  # Move up one directory from src and then into models
sys.path.append(models_dir)

from models import MLPPhyloNet, CNNPhyloNet, LSTMPhyloNet, TrPhyloNet, AePhyloNet, DiffPhyloNet, DNASequenceDataset

# Argument parsing
parser = argparse.ArgumentParser(description="Test trained models with various clades and metrics")
parser.add_argument('--model', type=str, required=True, help='Model to use (e.g., CNNPhyloNet, MLPPhyloNet)')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate used during training')
parser.add_argument('--clade', type=str, required=True, help='Clade for testing (e.g., Clade_A, Clade_B)')

args = parser.parse_args()

# Function to generate Newick format with support values
def generate_newick_with_support(linkage_matrix, labels, support_values):
    def _recursive_newick(tree, node_id):
        if node_id < len(labels):
            return labels[node_id]
        else:
            left = _recursive_newick(tree, int(tree[node_id - len(labels), 0]))
            right = _recursive_newick(tree, int(tree[node_id - len(labels), 1]))
            branch_length = tree[node_id - len(labels), 2]
            support_value = support_values[node_id - len(labels)]
            return f"({left}:{branch_length},{right}:{branch_length})"

    newick_string = _recursive_newick(linkage_matrix, len(labels) + len(linkage_matrix) - 1)
    return f"{newick_string};"

# Function to normalize linkage matrix distances
def normalize_linkage_matrix(linkage_matrix):
    scaler = MinMaxScaler()
    linkage_matrix[:, 2] = scaler.fit_transform(linkage_matrix[:, 2].reshape(-1, 1)).flatten()
    return linkage_matrix

# Function to plot dendrogram with branch support values
def plot_dendrogram_with_support(linkage_matrix, labels, support_values, file_path):
    normalized_support_values = normalize_support_values(support_values)
    
    plt.figure(figsize=(12, 24))
    dendrogram_data = hierarchy.dendrogram(linkage_matrix, orientation='left', labels=labels,
                                           above_threshold_color='gray', leaf_font_size=10)

    # Annotate the dendrogram with normalized support values
    for i, (x, y) in enumerate(zip(dendrogram_data['dcoord'], dendrogram_data['icoord'])):
        x = np.mean(x)
        y = np.mean(y)
        if i < len(normalized_support_values):
            plt.text(x, y, f"{normalized_support_values[i]:.2f}", color='red', fontsize=8, va='center', ha='right')

    plt.xlabel('Distance')
    plt.ylabel('Sequence Names')
    plt.title('Dendrogram of DNA Sequences with Normalized Branch Support Values')
    plt.gca().invert_yaxis()  # Invert y-axis to have the first label at the top
    plt.tight_layout()
    plt.savefig(file_path, dpi=400)
    print(f"Dendrogram with normalized support values saved to {file_path}")

# Function to normalize support values
def normalize_support_values(support_values):
    scaler = MinMaxScaler()
    support_values = np.array(support_values).reshape(-1, 1)
    return scaler.fit_transform(support_values).flatten()

def load_tree(file_path: str) -> Tree:
    """Load a Newick tree from a file."""
    with open(file_path, 'r') as file:
        newick_str = file.read().strip()
    return Tree(newick_str, format=1)

def calculate_rf_distance(tree1: Tree, tree2: Tree) -> int:
    """Calculate Robinson-Foulds distance between two trees, handling unrooted trees."""
    try:
        # Compute RF distance with unrooted trees flag
        rf_distance = tree1.robinson_foulds(tree2, unrooted_trees=True)[0]
    except Tree.TreeError as e:
        print(f"Error calculating RF distance: {e}")
        rf_distance = None
    return rf_distance

def save_metrics_to_csv(rf_metrics: list, output_file: str):
    """Save the RF distances to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group', 'Robinson-Foulds Distance'])
        writer.writerows(rf_metrics)

if __name__ == "__main__":
    # Directory to save the results based on the model, learning rate, and clade
    results_dir = f"./testing_results/{args.model}_lr{args.learning_rate}/{args.clade}"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(f"{args.model}()").to(device)

    # Load the saved model
    model.load_state_dict(torch.load(f"./results/{args.model}_lr{args.learning_rate}/best_model.pt", map_location=device)) 

    # Load the clade sequences
    fasta_file = f"./helper_files/{args.clade}.fasta"
    sequences = []
    sequence_names = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq).upper().replace('T', 'U'))
        sequence_names.append(record.id)

    # Initialize the clade sequences for embeddings
    dataset = DNASequenceDataset(sequences)

    # Generate embeddings
    embeddings = []
    model.eval()
    with torch.no_grad():
        for seq in tqdm(dataset):
            seq = seq.unsqueeze(0).to(device)  # Add batch dimension and move to device
            embedding = model(seq)  # Generate embedding
            embeddings.append(embedding.squeeze(0).cpu().numpy())

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Reshape embeddings for distance calculation
    num_sequences, seq_length, embed_dim = embeddings.shape
    embeddings = embeddings.reshape(num_sequences, seq_length * embed_dim)

    # Save embeddings to file
    embeddings_file = f"{results_dir}/{args.clade}_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved to {embeddings_file}")

    # Calculate pairwise distances between embedded sequences
    distances = pdist(embeddings)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distances, method='single')
    linkage_matrix = normalize_linkage_matrix(linkage_matrix)

    # Generate Newick format with support values
    support_values = np.max(linkage_matrix[:, 2]) - linkage_matrix[:, 2]
    newick_string = generate_newick_with_support(linkage_matrix, sequence_names, support_values)

    # Save Newick format to file
    test_newick = f"{results_dir}/{args.clade}_dendrogram.newick"
    with open(test_newick, 'w') as f:
        f.write(newick_string)
    print(f"Newick format saved to {test_newick}")

    # Save dendrogram with branch support values
    dendrogram_file = f"{results_dir}/{args.clade}_dendrogram.png"
    plot_dendrogram_with_support(linkage_matrix, sequence_names, support_values, dendrogram_file)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Plot PCA result
    pca_plot_file = f"{results_dir}/{args.clade}_pca_plot.png"
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], marker='o', c='blue', alpha=0.7)
    
    for i, seq_name in enumerate(sequence_names):
        plt.text(pca_result[i, 0], pca_result[i, 1], seq_name, fontsize=9)

    plt.title("PCA of DNA Sequence Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(pca_plot_file, dpi=400)
    print(f"PCA plot saved to {pca_plot_file}")

    # Robinson-Foulds distance calculation
    rf_metrics = []

    iqtree_newick = f"./helper_files/{args.clade}.contree"
    clade = args.clade
    
    # Load trees from files
    tree1 = load_tree(test_newick)
    tree2 = load_tree(iqtree_newick)

    # Calculate RF distance
    rf_distance = calculate_rf_distance(tree1, tree2)
    if rf_distance is not None:
        rf_metrics.append((clade, rf_distance))
    
    # Save RF distance metrics to a CSV file
    rf_metrics_file = f"{results_dir}/{args.clade}_rf_metrics.csv"
    save_metrics_to_csv(rf_metrics, rf_metrics_file)
    print(f"Robinson-Foulds metrics saved to {rf_metrics_file}")

# Example execution:
# python3 ./src/test.py --model CNNPhyloNet --learning_rate 0.001 --clade plants

