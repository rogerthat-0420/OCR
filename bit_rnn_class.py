import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor([float(bit) for bit in self.sequences[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label
    
class BitCountingRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout = 0.2):
        super(BitCountingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)  
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output, hidden = self.rnn(x, h0)
        last_output = output[:, -1, :]
        normalized = self.layer_norm(last_output)
        prediction = self.fc(normalized)
        return prediction.squeeze()