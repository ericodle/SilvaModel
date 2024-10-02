import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from Bio import SeqIO
from tqdm import tqdm
import matplotlib.pyplot as plt
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

# Setup logging
def setup_logging(log_dir):
    logging.basicConfig(filename=os.path.join(log_dir, 'training_log.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to get the model based on the argument
def get_model(model_name):
    if model_name == 'CNNPhyloNet':
        return CNNPhyloNet()
    elif model_name == 'MLPPhyloNet':
        return MLPPhyloNet()
    elif model_name == 'LSTMPhyloNet':
        return LSTMPhyloNet()
    elif model_name == 'TrPhyloNet':
        return TrPhyloNet()
    elif model_name == 'AePhyloNet':
        return AePhyloNet()
    elif model_name == 'DiffPhyloNet':
        return DiffPhyloNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train PhyloNet models with adjustable hyperparameters.")
    parser.add_argument('--model', type=str, required=True, help="The model to use (CNNPhyloNet, MLPPhyloNet, LSTMPhyloNet, TrPhyloNet, AePhyloNet, DiffPhyloNet)")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for training the model")
    args = parser.parse_args()

    # Create output directory based on model and learning rate
    output_dir = os.path.join(script_dir, '..', 'training_results', f'{args.model}_lr{args.learning_rate}')
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    # Load sequences for training
    training_fasta_file = os.path.join(script_dir, '..', 'helper_files', 'euk_extract.fasta')
    training_sequences = [str(record.seq).upper().replace('T', 'U') for record in SeqIO.parse(training_fasta_file, "fasta")]

    # Create dataset and split into training and validation sets
    dataset = DNASequenceDataset(training_sequences)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss(reduction='mean') 

    # Initialize parameters
    num_epochs = 777777  # Arbitrary high value
    step_losses = []
    train_losses = []
    val_losses = []
    best_train_loss = float('inf')
    improvement_threshold = 0.01  # 1% improvement threshold
    accumulation_steps = 4  # Gradient accumulation steps

    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop (and check for Diffusion Model)
    if isinstance(model, DiffPhyloNet):
        for epoch in range(num_epochs):
            train_loss = 0
            model.train()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                batch = batch.to(device)
                t = torch.rand(batch.size(0), device=device)  # Random tensor 't' for diffusion model
                with torch.cuda.amp.autocast():
                    output = model(batch, t)  # Pass 't' into the diffusion model
                    loss = criterion(output, batch)
                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                train_loss += loss.item()
                step_losses.append(loss.item())

            avg_train_loss = train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # Validation step for diffusion model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = batch.to(device)
                    t = torch.rand(batch.size(0), device=device)
                    with torch.cuda.amp.autocast():
                        output = model(batch, t)
                        loss = criterion(output, batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.4f} seconds")
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.4f} seconds")

            # Check for early stopping
            if epoch > 0 and avg_train_loss >= 0.99 * train_losses[-2]:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                print(f"Stopping early after {epoch+1} epochs due to less than 1% improvement in training loss.")
                break

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                print(f"Model saved with training loss: {avg_train_loss:.12f}")
    else:
        # Regular model training loop (for non-diffusion models)
        for epoch in range(num_epochs):
            start_time = time.time()  # Start timing the epoch
            model.train()
            optimizer.zero_grad()  # Reset gradients at the start of each epoch
            train_loss = 0

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                batch = batch.to(device)
                with torch.cuda.amp.autocast():  # Mixed precision context
                    output = model(batch)
                    loss = criterion(output, batch)  # Compute loss with input as target
                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:  # Perform optimizer step after accumulating gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # Reset gradients after each step
                train_loss += loss.item()
                step_losses.append(loss.item())

            avg_train_loss = train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = batch.to(device)
                    with torch.cuda.amp.autocast():  # Mixed precision context
                        output = model(batch)
                        loss = criterion(output, batch)  # Compute loss with input as target
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
        
            end_time = time.time()  # End timing the epoch
            epoch_duration = end_time - start_time

            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.4f} seconds")
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f} seconds")

            # Check for early stopping
            if epoch > 0 and avg_train_loss >= 0.99 * train_losses[-2]:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                print(f"Stopping early after {epoch+1} epochs due to less than 1% improvement in training loss.")
                break

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                print(f"Model saved with training loss: {avg_train_loss:.12f}")

    # Save training curves
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, len(step_losses) + 1), step_losses, label='Step Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=400)
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epochwise_loss.png'), dpi=400)
    plt.close()
