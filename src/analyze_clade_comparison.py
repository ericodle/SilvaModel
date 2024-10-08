import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to process each CSV file
def process_csv(file_path):
    """Read the CSV file and return a list of subclades with their match results (True/False)."""
    try:
        df = pd.read_csv(file_path, sep=',')  # Use comma as the separator
        # Print the columns to verify the fix
        print(f"Columns in {file_path}: {df.columns.tolist()}")
        
        subclade_data = []
        
        # Check if required columns exist
        if 'Subclade' not in df.columns or 'Match' not in df.columns:
            print(f"Warning: Missing required columns in {file_path}.")
            return subclade_data

        # Iterate over each row to collect subclade and match results
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

# Main execution
def main():
    comparison_results_dir = './comparison_results'
    all_results = []  # List to store all results

    # Loop through directories and process the data
    for root, dirs, files in os.walk(comparison_results_dir):
        for file in files:
            if file.endswith("_monophyly_comparison.csv"):
                file_path = os.path.join(root, file)
                model_lr = os.path.basename(root)  # Get model and learning rate from folder name
                
                # Ensure the folder name is in the expected format
                if '_lr' not in model_lr:
                    print(f"Warning: Unexpected folder name format: {model_lr}")
                    continue
                
                # Extract model name from the directory structure
                model = os.path.basename(os.path.dirname(file_path))  # Assuming the model is the parent directory
                
                # Get the learning rate from the folder name
                lr = model_lr.split('_lr')[1]  # Extract learning rate part
                lr_exponential = convert_learning_rate(float(lr))  # Convert to exponential notation
                
                # Extract clade from the file name (first word before the underscore)
                clade = file.split('_')[0]  # Get the first word from the file name
                
                # Get subclade match data from the CSV
                subclade_data = process_csv(file_path)
                
                # Update model name to replace learning rate part with the exponential notation
                model_with_lr = f"{model.split('_lr')[0]}_{lr_exponential}"
                
                # Append results to the all_results list with renamed columns
                for subclade_name, match_result in subclade_data:
                    all_results.append({
                        'Model': model_with_lr,  # Use updated model name
                        'Learning Rate': lr_exponential,  # Add Learning Rate in exponential notation
                        'Clade': clade,  # Add Clade extracted from the file name
                        'Subclade': subclade_name,
                        'Match': match_result
                    })
                    
    # Create a DataFrame from the collected results
    results_df = pd.DataFrame(all_results)

    # Reorder columns to ensure Learning Rate is after Model
    results_df = results_df[['Model', 'Learning Rate', 'Clade', 'Subclade', 'Match']]

    # Save the consolidated DataFrame to a new CSV file
    output_file_path = './consolidated_results.csv'
    results_df.to_csv(output_file_path, index=False)
    print(f"Consolidated results saved to {output_file_path}")

#####  Plot Results  #####

    # Load the data from the CSV file
    df = results_df

    # Convert Match to a numeric value for plotting (1 for True, 0 for False)
    df['Match'] = df['Match'].astype(int)

    # Create a combined column for Clade and Subclade for the Y-axis
    df['Clade_Subclade'] = df['Clade'] + ' - ' + df['Subclade']

    # Define the specific order of clades (reverse order)
    clade_order = ['dinoflagellates', 'apicomplexans', 'kelps', 'plants', 'supergroups', 'animals']

    # Ensure Clade_Subclade is ordered based on the specified clade order
    df['Clade'] = pd.Categorical(df['Clade'], categories=clade_order, ordered=True)
    df = df.sort_values('Clade')

    # Define the specific order for models on the X-axis
    model_order = ['MLPPhyloNet', 'CNNPhyloNet', 'LSTMPhyloNet', 'AePhyloNet', 'TrPhyloNet', 'DiffPhyloNet']

    # Extract the base model name and learning rate for ordering
    df['Base_Model'] = df['Model'].str.split('_').str[0]
    df['Learning_Rate'] = df['Model'].str.extract(r'_(\d+\.?\d*e?-?\d*)')[0]

    # Format all learning rates to '1e-x' format
    df['Learning_Rate'] = df['Learning_Rate'].astype(float).apply(lambda x: f"{x:.1e}")

    # Convert Learning_Rate back to float for sorting (this time in scientific notation)
    df['Learning_Rate'] = df['Learning_Rate'].apply(lambda x: float(x))

    # Ensure Base_Model is ordered based on the specified model order
    df['Base_Model'] = pd.Categorical(df['Base_Model'], categories=model_order, ordered=True)

    # Sort the DataFrame by Base_Model and then by Learning_Rate in descending order
    df = df.sort_values(['Base_Model', 'Learning_Rate'], ascending=[True, False])

    # Define marker properties for Match values
    markers = {
        0: {'marker': 'X', 'color': 'red', 'size': 50, 'label': 'No Clade Match'},  # Smaller red X for 0
        1: {'marker': 'o', 'color': 'blue', 'size': 100, 'label': 'Clade Match'}  # Larger blue circle for 1
    }

    # Increase the size of the plot
    plt.figure(figsize=(16, 10))  # Adjust width and height as needed

    # Use the markers based on the Match value
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
    plt.xticks(rotation=90)  # Rotate X-axis labels by 90 degrees
    plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(1.15, 1))  # Adjust legend position if needed
    plt.grid()

    # Save the plot
    plt.savefig('./model_vs_clade_subclade_plot.png', dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    main()

# Example execution
# python3 ./src/analyze_clade_comparison.py

