import numpy as np
import random
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.utils.data import Dataset

module_path_bit = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/rnn'))
sys.path.append(module_path_bit)
from bit_rnn import BitCountingRNN, BitSequenceDataset

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def generate_bit_counting_dataset(num_sequences=100000, max_length=16):
    data = []
    for _ in range(num_sequences):
        length = random.randint(1, max_length)
        sequence = np.random.randint(2, size=length).tolist()
        count = sum(sequence)
        data.append({"sequence": sequence, "count": count})
    return data

dataset=generate_bit_counting_dataset()

train_split = int(0.8 * len(dataset))
val_split = int(0.9 * len(dataset))

train_data = dataset[:train_split]
val_data = dataset[train_split:val_split]
test_data = dataset[val_split:]

train_sequences = [''.join(map(str,item['sequence'])) for item in train_data]
train_labels = [item['count'] for item in train_data]

val_sequences = [''.join(map(str,item['sequence'])) for item in val_data]
val_labels = [item['count'] for item in val_data]

test_sequences = [''.join(map(str,item['sequence'])) for item in test_data]
test_labels = [item['count'] for item in test_data]

train_dataset = BitSequenceDataset(train_sequences, train_labels)
val_dataset = BitSequenceDataset(val_sequences, val_labels)
test_dataset = BitSequenceDataset(test_sequences, test_labels)

BATCH_SIZE = 32

def func(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,collate_fn=func, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE , collate_fn=func)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE , collate_fn=func)


rnn_model = BitCountingRNN(hidden_size=64, num_layers=2, dropout=0.2)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.0001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences
            labels = labels
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences
                labels = labels
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_bitrnnmodel.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation MAE: {val_mae:.4f}')
        print('-' * 50)
    
    return train_losses, val_losses

train_losses, val_losses = train_model(rnn_model, train_loader, val_loader, criterion, optimizer, 20)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, 21), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')
plt.legend()
plt.grid(True)
plt.savefig("train_val_graph.png")



def random_baseline(sequences, labels, max_possible_count):
    predictions = np.random.randint(0, max_possible_count + 1, size=len(labels))
    mae = np.mean(np.abs(predictions - labels))
    return mae

def generate_dataset(num_samples, min_length, max_length):
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        sequence = ''.join(random.choice('01') for _ in range(length))
        label = sum(int(bit) for bit in sequence)
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

def evaluate_sequence_lengths(model, min_length, max_length, samples_per_length):
    model.eval()
    results = []
    
    for length in range(min_length, max_length + 1):
        sequences, labels = generate_dataset(samples_per_length, length, length)
        dataset = BitSequenceDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=32)
        
        total_mae = 0.0
        with torch.no_grad():
            for sequences, labels in loader:
                sequences = sequences
                labels = labels
                
                outputs = model(sequences)
                mae = torch.mean(torch.abs(outputs - labels)).item()
                total_mae += mae
        
        avg_mae = total_mae / len(loader)
        results.append((length, avg_mae))
    
    return results

print("Evaluating generalization...")
generalization_results = evaluate_sequence_lengths(rnn_model, 1, 32, 1000)
lengths, maes = zip(*generalization_results)
plt.figure(figsize=(10, 6))
plt.plot(lengths, maes, marker='o')
plt.xlabel('Sequence Length')
plt.ylabel('Mean Absolute Error')
plt.title('Model Performance vs Sequence Length (Best Hyperparameters)')
plt.grid(True)
plt.savefig('final_generalization_plot.png')
plt.close()




