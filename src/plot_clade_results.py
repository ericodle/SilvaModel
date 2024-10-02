import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv('./consolidated_results.csv')

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

