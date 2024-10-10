import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths and folder structure relative to the script directory
base_dir = os.path.join(script_dir, '../testing_results')  # Relative path to 'testing_results' folder
model_types = ['MLPPhyloNet', 'CNNPhyloNet', 'LSTMPhyloNet', 'AePhyloNet', 'TrPhyloNet', 'DiffPhyloNet']
learning_rates = ['0.01', '0.001', '0.0001', '1e-05', '1e-06', '1e-07']

# Capitalize clades
clades = ['Dinoflagellates', 'Apicomplexans', 'Kelps', 'Plants', 'Supergroups', 'Animals']

# Initialize an empty DataFrame to store all data
summary_df = pd.DataFrame()

# Loop through all model_type and learning_rate combinations
for model_type in model_types:
    for lr in learning_rates:
        subfolder_path = os.path.join(base_dir, f"{model_type}_lr{lr}")
        
        # Loop through all clades and their respective CSV files
        for clade in clades:
            csv_file = os.path.join(subfolder_path, clade.lower(), f"{clade.lower()}_rf_metrics.csv")  # Use lower case to match file names
            
            # Check if the file exists before reading
            if os.path.exists(csv_file):
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Extract only the 'Robinson-Foulds Distance' column
                df = df[['Robinson-Foulds Distance']]
                
                # Add model concatenated with learning rate as 'Model_LR' and clade as columns
                df['Model_LR'] = f"{model_type}_lr{float(lr):.1e}"  # Model concatenated with learning rate in scientific notation
                df['Clade'] = clade  # Keep the capitalized clade name
                
                # Append to the summary DataFrame, placing columns in the correct order
                summary_df = pd.concat([summary_df, df[['Model_LR', 'Clade', 'Robinson-Foulds Distance']]], ignore_index=True)

# Save the accumulated data to a new CSV file
output_path = os.path.join(script_dir, '../summary_rf_data.csv')  # Save CSV relative to the script's directory
summary_df.to_csv(output_path, index=False)

print(f"Summary data saved to {output_path}")

def plot_rf_distances_bar_3d(summary_df):
    # Set up the 3D plot with a wider figure size
    fig = plt.figure(figsize=(14, 10))  # Increased width for better label visibility
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique Clades and define the position of bars
    bar_width = 0.3  # Width of the bars
    bar_depth = 0.2  # Depth of the bars

    # X and Z positions for the bars
    x = np.arange(len(summary_df['Model_LR'].unique()))  # Positions for Model_LearningRate
    z = np.arange(len(clades))  # Positions for Clades

    # For each clade, plot a set of bars for all Model_LR values
    for i, clade in enumerate(clades):
        clade_data = summary_df[summary_df['Clade'] == clade]
        rob_distances = clade_data['Robinson-Foulds Distance'].to_numpy()

        # Define the bar positions
        x_pos = x + i * bar_depth  # Use a slight offset for each clade

        ax.bar(x_pos, rob_distances, zs=i, zdir='y', width=bar_width, alpha=0.8, label=clade)

    # Set labels and title
    ax.set_xlabel('')  # Hide the Model_LearningRate label
    ax.set_ylabel('')  # Hide the Clade label
    ax.set_zlabel('Robinson-Foulds Distance')  # Keep z-axis label
    ax.set_title('Robinson-Foulds Distance Reduction by Model Type', fontsize=14)

    # Set ticks and tick labels for the y-axis (clades) and x-axis (learning rates)
    ax.set_yticks(z)
    ax.set_yticklabels(clades)  # Capitalized clades displayed on the y-axis
    
    # Adjust x-ticks
    ax.set_xticks(x + bar_depth * (len(clades) / 2 - 0.5))  # Center the x-ticks

    # Show every second label for clarity
    unique_labels = summary_df['Model_LR'].unique()
    ax.set_xticks(x[::2] + bar_depth * (len(clades) / 2 - 0.1))  # Select every second x-tick
    ax.set_xticklabels(unique_labels[::2], rotation=90, ha='right', fontsize=9)  # Rotate for clarity

    # Extend the x-axis range to provide more space
    ax.set_xlim([-0.5, len(unique_labels)])  # Extend the x-axis limits as needed

    # Adjust the viewing angle for a better 3D effect
    ax.view_init(elev=20, azim=30)  # Change elevation and azimuth angles

    # Optional: Add grid lines for better visualization
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')

    # Adjust layout to fit everything properly
    plt.tight_layout()

    # Save the plot to a file
    plot_output_path = os.path.join(script_dir, '../rf_distance_bar_3d_plot.png')  # Save the plot relative to the script's directory
    plt.savefig(plot_output_path, dpi=1200)
    
    print(f"3D bar plot saved to {plot_output_path}")

# Call the 3D plot function to generate and save the plot
plot_rf_distances_bar_3d(summary_df)

# Example execution
# python3 ./src/rf_analysis.py

