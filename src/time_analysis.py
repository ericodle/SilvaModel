import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cumulative_training_time(training_results_folder):
    # Initialize a list to hold results
    results = []

    # Traverse through the subfolders in the training_results_folder
    for subdir, _, files in os.walk(training_results_folder):
        for file in files:
            if file == 'training_log.txt':
                # Construct the full file path
                log_file_path = os.path.join(subdir, file)
                
                # Extract the model name and learning rate from the folder name
                model_learning_rate = os.path.basename(subdir)

                # Debug: Print the model_learning_rate being processed
                print(f"Processing: {model_learning_rate}")

                # Initialize cumulative time
                cumulative_time = 0.0

                # Read the log file
                try:
                    with open(log_file_path, 'r') as log_file:
                        for line in log_file:
                            # Split the line to extract duration
                            if "Duration:" in line:
                                parts = line.split("Duration:")
                                duration_str = parts[1].strip().split()[0]  # Get the duration value
                                try:
                                    # Convert to float and accumulate
                                    cumulative_time += float(duration_str)
                                except ValueError:
                                    print(f"Could not convert duration '{duration_str}' to float in {log_file_path}")
                except Exception as e:
                    print(f"Error reading log file {log_file_path}: {e}")
                    continue  # Skip to the next file if there was an error

                # Format the learning rate
                lr_str = format_learning_rate(model_learning_rate)

                # Append the result with model name and learning rate linked by an underscore
                results.append({
                    'Model_Learning_Rate': lr_str,
                    'Cumulative_Training_Time': cumulative_time,
                })

    # Create a DataFrame
    df = pd.DataFrame(results)

    # Define the model order
    model_order = ['MLPPhyloNet', 'CNNPhyloNet', 'LSTMPhyloNet', 'AePhyloNet', 'TrPhyloNet', 'DiffPhyloNet']

    # Split the Model and Learning Rate
    df[['Model', 'Learning_Rate']] = df['Model_Learning_Rate'].str.split('_lr', expand=True)

    # Sort by Model (custom order) and Learning Rate (keeping as strings)
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values(by=['Model', 'Learning_Rate'], ascending=[True, True])

    # Save to CSV
    df.to_csv('training_times.csv', index=False)
    print("Training summary saved to 'training_times.csv'.")

    # Plot the results
    plot_cumulative_training_time(df)

def format_learning_rate(model_learning_rate):
    """
    Format the learning rate as a string with scientific notation.
    Converts lr0.0001 to MLPPhyloNet_lr1e-04.
    """
    parts = model_learning_rate.split('_lr')
    if len(parts) == 2:
        model = parts[0]
        lr_value = parts[1]
        # Convert to scientific notation with the format lr1e-04
        lr_value = convert_to_scientific(lr_value)
        lr_str = f"{model}_lr{lr_value}"  # Keep as string with prefix
    else:
        lr_str = "Unknown"
        print(f"Could not parse learning rate from: {model_learning_rate}")
    
    return lr_str

def convert_to_scientific(value):
    """
    Convert a learning rate in decimal form to scientific notation.
    """
    try:
        # Convert string to float
        float_value = float(value)
        # Use format for scientific notation
        return "{:.2e}".format(float_value).replace("e", "e").replace("0", "")  # Format for exponent
    except ValueError:
        return value  # Return original value if conversion fails

def plot_cumulative_training_time(df):
    """
    Plot the cumulative training time for each model and learning rate combination with better visibility for lower values.
    """
    plt.figure(figsize=(10, 6))

    # Create a bar plot with logarithmic Y-axis
    plt.bar(df['Model_Learning_Rate'], df['Cumulative_Training_Time'], color='skyblue')

    # Set plot labels and title
    plt.xlabel('Model and Learning Rate', fontsize=12)
    plt.ylabel('Cumulative Training Time (seconds)', fontsize=12)
    plt.title('Cumulative Training Time by Model and Learning Rate', fontsize=14)

    # Set Y-axis to logarithmic scale
    plt.yscale('log')

    # Rotate x labels for readability
    plt.xticks(rotation=90, fontsize=10)

    # Add a grid to improve readability
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save the plot as an image file
    plt.savefig('cumulative_training_time_log_plot.png', bbox_inches='tight', dpi=300)
    print("Plot saved to 'cumulative_training_time_log_plot.png'.")

# Set the path to your training_results folder
training_results_folder = './training_results'
calculate_cumulative_training_time(training_results_folder)

# Example execution
# python3 ./src/time_analysis.py
