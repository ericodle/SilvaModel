import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to process each CSV file
def process_csv(file_path):
    """Read the CSV file and return a list of subclades with their match results (True/False)."""
    try:
        df = pd.read_csv(file_path, sep=',')  # Use comma as the separator
        print(f"Columns in {file_path}: {df.columns.tolist()}")
        
        subclade_data = []
        
        if 'Subclade' not in df.columns or 'Match' not in df.columns:
            print(f"Warning: Missing required columns in {file_path}.")
            return subclade_data

        for _, row in df.iterrows():
            subclade_name = row['Subclade']
            match_result = row['Match']
            subclade_data.append((subclade_name, match_result))
        
        return subclade_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# Function to convert learning rate to exponential notation
def convert_learning_rate(lr):
    """Convert learning rate to exponential notation."""
    return f"{lr:.1e}"

def plot_main_clades(df):
    """Generate a main plot for all clades."""
    # Convert Match to a numeric value for plotting
    df['Match'] = df['Match'].astype(int)
    
    # Create a combined column for Clade and Subclade
    df['Clade_Subclade'] = df['Clade'] + ' - ' + df['Subclade']
    
    # Define clade order
    clade_order = ['dinoflagellates', 'apicomplexans', 'kelps', 'plants', 'supergroups', 'animals']
    df['Clade'] = pd.Categorical(df['Clade'], categories=clade_order, ordered=True)
    df = df.sort_values('Clade')
    
    # Define model order
    model_order = ['MLPPhyloNet', 'CNNPhyloNet', 'LSTMPhyloNet', 'AePhyloNet', 'TrPhyloNet', 'DiffPhyloNet']
    
    df['Base_Model'] = df['Model'].str.split('_').str[0]
    df['Learning_Rate'] = df['Model'].str.extract(r'_(\d+\.?\d*e?-?\d*)')[0]
    df['Learning_Rate'] = df['Learning_Rate'].astype(float).apply(lambda x: f"{x:.1e}")
    df['Learning_Rate'] = df['Learning_Rate'].apply(lambda x: float(x))
    df['Base_Model'] = pd.Categorical(df['Base_Model'], categories=model_order, ordered=True)
    
    # Sort the DataFrame based on both Clade and Base_Model
    df = df.sort_values(['Clade', 'Base_Model', 'Learning_Rate'], ascending=[True, True, False])
    
    markers = {
        0: {'marker': '.', 'color': 'red', 'size': 50, 'label': 'No Clade Match'},
        1: {'marker': 'o', 'color': 'blue', 'size': 100, 'label': 'Clade Match'}
    }

    plt.figure(figsize=(16, 10))
    
    for match_value, props in markers.items():
        subset = df[df['Match'] == match_value]
        plt.scatter(subset['Model'], subset['Clade_Subclade'], 
                    marker=props['marker'], 
                    color=props['color'], 
                    s=props['size'], 
                    label=props['label'])
    
    plt.title('Model vs Clade-Subclade Matches')
    plt.xlabel('Model')
    plt.ylabel('Clade - Subclade')
    plt.xticks(rotation=90)  
    plt.yticks(sorted(df['Clade_Subclade'].unique()))  # Ensure y-axis labels respect Clade order
    plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid()

    # Save the plot
    plt.savefig('./model_vs_clade_subclade_plot.png', dpi=1200, bbox_inches='tight')
    plt.close()


# Main execution
def main():
    comparison_results_dir = './comparison_results'
    all_results = []

    for root, dirs, files in os.walk(comparison_results_dir):
        for file in files:
            if file.endswith("_monophyly_comparison.csv"):
                file_path = os.path.join(root, file)
                model_lr = os.path.basename(root)
                
                if '_lr' not in model_lr:
                    print(f"Warning: Unexpected folder name format: {model_lr}")
                    continue
                
                model = os.path.basename(os.path.dirname(file_path))
                lr = model_lr.split('_lr')[1]
                lr_exponential = convert_learning_rate(float(lr))
                clade = file.split('_')[0]
                
                subclade_data = process_csv(file_path)
                
                model_with_lr = f"{model.split('_lr')[0]}_{lr_exponential}"
                
                for subclade_name, match_result in subclade_data:
                    all_results.append({
                        'Model': model_with_lr,
                        'Learning Rate': lr_exponential,
                        'Clade': clade,
                        'Subclade': subclade_name,
                        'Match': match_result
                    })
                    
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'Learning Rate', 'Clade', 'Subclade', 'Match']]
    
    output_file_path = './consolidated_results.csv'
    results_df.to_csv(output_file_path, index=False)
    print(f"Consolidated results saved to {output_file_path}")

    # Generate plots
    plot_main_clades(results_df)  # Plot main results

if __name__ == '__main__':
    main()
