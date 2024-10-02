import pandas as pd
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

if __name__ == '__main__':
    main()

