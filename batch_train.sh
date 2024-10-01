#!/bin/bash

# Define the models and updated learning rates
models=("MLPPhyloNet" "CNNPhyloNet" "LSTMPhyloNet" "TrPhyloNet" "AePhyloNet" "DiffPhyloNet")
learning_rates=("0.01" "0.001" "0.0001" "0.00001" "0.000001" "0.0000001")

# Directory where the script is located
script_dir="$(pwd)/src"

# Directory to store results (working directory)
results_dir="$(pwd)/results"

# Create the results directory if it does not exist
mkdir -p "$results_dir"

# Iterate over each model and learning rate
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        # Convert learning rate to scientific notation
        lr_formatted=$(printf "%g" "$lr")
        
        echo "Running model: $model with learning rate: $lr_formatted"

        # Create a subdirectory for this model and learning rate
        output_dir="$results_dir/${model}_lr${lr_formatted}"
        mkdir -p "$output_dir"

        # Run the Python script with the current model and learning rate
        python3 "$script_dir/train.py" --model "$model" --learning_rate "$lr"

        echo "Completed model: $model with learning rate: $lr_formatted"
    done
done

echo "All experiments are complete!"


#### How to run ####
# 1) Make it executable:
# chmod +x batch_train.sh
# 2) Run the script:
# ./batch_train.sh
