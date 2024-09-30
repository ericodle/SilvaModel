import os
import sys
import json
import csv
import argparse
from ete3 import Tree

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

def load_tree(file_path: str) -> Tree:
    """Load a Newick tree from a file."""
    with open(file_path, 'r') as file:
        newick_str = file.read().strip()
    return Tree(newick_str, format=1)

def is_monophyletic(tree, taxa_list):
    """Check if a list of taxa forms a monophyletic group in the tree."""
    try:
        common_ancestor = tree.get_common_ancestor(taxa_list)
        return set(common_ancestor.get_leaf_names()) == set(taxa_list)
    except:
        return False

def load_clade_comparison_criteria(file_path):
    """Load the subclade titles and taxa names from the JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def perform_monophyly_tests(iqtree, test_tree, clade_criteria):
    """
    Perform subclade-to-subclade comparison between the ground truth tree and test tree,
    checking if the taxa in each subclade are monophyletically grouped.
    """
    results = []
    
    for subclade_title, taxa_names in clade_criteria.items():
        # Ensure all taxa exist in both trees
        taxa_in_iqtree = [taxon for taxon in taxa_names if taxon in iqtree.get_leaf_names()]
        taxa_in_test_tree = [taxon for taxon in taxa_names if taxon in test_tree.get_leaf_names()]
        
        if not taxa_in_iqtree or not taxa_in_test_tree:
            # If any taxa from the subclade are missing, skip this test
            print(f"Skipping {subclade_title}: Taxa missing in one or both trees.")
            continue
        
        # Check if the taxa are monophyletic in both the ground truth tree and test tree
        iqtree_monophyly = is_monophyletic(iqtree, taxa_in_iqtree)
        test_tree_monophyly = is_monophyletic(test_tree, taxa_in_test_tree)
        
        # Record the result (True if both trees agree on monophyly)
        results.append((subclade_title, iqtree_monophyly, test_tree_monophyly, iqtree_monophyly == test_tree_monophyly))
    
    return results

def save_monophyly_results(results, output_file):
    """Save the monophyly test results to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subclade', 'Ground Truth Monophyly (IQTree)', 'Test Tree Monophyly', 'Match'])
        writer.writerows(results)
    print(f"Monophyly results saved to {output_file}")

if __name__ == "__main__":

    # Directory to save the results based on the model, learning rate, and clade
    comparison_dir = f"comparison_results/{args.model}_lr{args.learning_rate}/{args.clade}"
    os.makedirs(comparison_dir, exist_ok=True)

    # File paths
    iqtree_newick_path = os.path.join(script_dir, '..', f'helper_files/{args.clade}.contree')
    test_newick_path = os.path.join(script_dir, '..', 'testing_results', f'{args.model}_lr{args.learning_rate}', f'{args.clade}', f'{args.clade}_dendrogram.newick')
    clade_criteria_file = os.path.join(script_dir, '..',f"helper_files/clade_comparison_criteria.json")
    
    # Load the trees
    iqtree = load_tree(iqtree_newick_path)  # Ground truth tree
    test_tree = load_tree(test_newick_path)  # Test tree

    # Load clade comparison criteria
    clade_comparison_criteria = load_clade_comparison_criteria(clade_criteria_file)

    # Perform monophyly tests
    monophyly_results = perform_monophyly_tests(iqtree, test_tree, clade_comparison_criteria)

    # Save monophyly test results to CSV
    monophyly_results_file = f"{comparison_dir}/{args.clade}_monophyly_comparison.csv"
    save_monophyly_results(monophyly_results, monophyly_results_file)

# Example execution:
# python3 ./src/clade_comparison.py --model CNNPhyloNet --learning_rate 0.001 --clade plants

