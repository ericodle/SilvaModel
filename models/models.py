import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import sys
import os

##########################################################################################################################
class MLPPhyloNet(nn.Module):
    def __init__(self, seq_length=2048, embed_dim=64, hidden_dim=4, num_layers=4, input_dim=4):
        super(MLPPhyloNet, self).__init__()
        self.seq_length = seq_length
        self.linear_in = nn.Linear(seq_length * input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, seq_length * input_dim)  # Output dimension matches input for reconstruction

    def forward(self, x):
        x = x.view(-1, self.seq_length * 4)  # Flatten the input
        x = torch.relu(self.linear_in(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.out(x)
        x = x.view(-1, self.seq_length, 4)  # Reshape back to [batch_size, seq_length, input_dim]
        return x

##########################################################################################################################
class CNNPhyloNet(nn.Module):
    def __init__(self, seq_length=2048, input_dim=4):
        super(CNNPhyloNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 1024, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(512, 128, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(seq_length * 128, 64)
        self.fc2 = nn.Linear(64, seq_length * input_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, seq_length) for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 2048, 4)  # Reshape to match input shape
        return x

##########################################################################################################################
class LSTMPhyloNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4, num_layers=4, seq_length=2048):
        super(LSTMPhyloNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * seq_length, 64)
        self.fc2 = nn.Linear(64, seq_length * input_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.contiguous().view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(-1, 2048, 4)
        return out

##########################################################################################################################
class TrPhyloNet(nn.Module):
    def __init__(self, seq_length=2048, embed_dim=64, num_heads=4, num_layers=4, input_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length

        self.positional_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:x.size(1), :].unsqueeze(0)
        x = x.permute(1, 0, 2)  # Change to (seq_length, batch_size, embed_dim) for transformer
        output = self.transformer_encoder(x)
        output = self.decoder(output.permute(1, 0, 2))
        return output

##########################################################################################################################
class AePhyloNet(nn.Module):
    def __init__(self, seq_length=2048, embed_dim=64, latent_dim=4, input_dim=4):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(seq_length * input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, seq_length * input_dim),
            nn.Sigmoid()  # Sigmoid activation for binary outputs (0-1 range)
        )

    def forward(self, x):
        # Ensure correct reshaping
        x = x.view(-1, self.seq_length * self.input_dim)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, self.seq_length, self.input_dim)  # Reshape to match input shape
        return decoded

##########################################################################################################################
class DiffPhyloNet(nn.Module):
    def __init__(self, seq_length=2048, embed_dim=64, num_heads=4, num_layers=4, input_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_in = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.out = nn.Linear(embed_dim, input_dim)
    
    def forward(self, x, t):
        x = self.linear_in(x) + self.pos_embed
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, x.size(1), self.embed_dim)
        x = self.transformer(x)
        x = self.out(x)
        x = x.view(-1, 2048, 4)
        return x

##########################################################################################################################

def diffusion_loss(model, x_0, t):
    noise = torch.randn_like(x_0)
    seq_length = x_0.size(1)  # Get the sequence length
    t = t.view(-1, 1).expand(-1, seq_length)  # Match t to the sequence length
    x_t = torch.sqrt(1 - t.unsqueeze(-1)) * x_0 + torch.sqrt(t.unsqueeze(-1)) * noise
    predicted_noise = model(x_t, t)
    return F.mse_loss(noise, predicted_noise)

##########################################################################################################################

# Define the dataset
class DNASequenceDataset():
    def __init__(self, sequences, max_length=2048):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded_seq = self.multi_hot_encode(seq)
        return encoded_seq

    def multi_hot_encode(self, seq, max_length=2048):
        encoding = {
            'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'U': [0,0,0,1],
            'N': [0.25,0.25,0.25,0.25], 'R': [0.5,0,0.5,0], 'Y': [0,0.5,0,0.5],
            'S': [0,0.5,0.5,0], 'W': [0.5,0,0,0.5], 'K': [0,0,0.5,0.5],
            'M': [0.5,0.5,0,0], 'B': [0,0.33,0.33,0.33], 'D': [0.33,0,0.33,0.33],
            'H': [0.33,0.33,0,0.33], 'V': [0.33,0.33,0.33,0]
        }
        encoded = [encoding.get(nuc, encoding['N']) for nuc in seq[:max_length]]
        encoded += [[0,0,0,0]] * (max_length - len(encoded))  # Padding
        return torch.tensor(encoded, dtype=torch.float)

##########################################################################################################################

# Function to visualize the model architecture
def visualize_model(model, input_tensor, filename, is_diffusion_model=False):
    if is_diffusion_model:
        output = model(*input_tensor)
    else:
        output = model(input_tensor)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(filename)

# Dummy input tensors for each model
dummy_input = torch.randn(1, 2048, 4)
dummy_time = torch.tensor([0.0])

# Instantiate and visualize the MLPPhyloNet
mlp_phylo_net = MLPPhyloNet()
visualize_model(mlp_phylo_net, dummy_input, './models/mlp_phylo_net_architecture')

# Instantiate and visualize the CNNPhyloNet
cnn_phylo_net = CNNPhyloNet()
visualize_model(cnn_phylo_net, dummy_input, './models/cnn_phylo_net_architecture')

# Instantiate and visualize the LSTMPhyloNet
lstm_phylo_net = LSTMPhyloNet()
visualize_model(lstm_phylo_net, dummy_input, './models/lstm_phylo_net_architecture')

# Instantiate and visualize the TrPhyloNet
tr_phylo_net = TrPhyloNet()
visualize_model(tr_phylo_net, dummy_input, './models/tr_phylo_net_architecture')

# Instantiate and visualize the AePhyloNet
ae_phylo_net = AePhyloNet()
visualize_model(ae_phylo_net, dummy_input, './models/ae_phylo_net_architecture')

# Instantiate and visualize the DiffPhyloNet
diff_phylo_net = DiffPhyloNet()
visualize_model(diff_phylo_net, (dummy_input, dummy_time), './models/diff_phylo_net_architecture', is_diffusion_model=True)

