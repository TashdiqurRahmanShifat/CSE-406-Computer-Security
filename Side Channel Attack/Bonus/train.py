import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import shutil
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintDataset(Dataset):
    """Custom Dataset class for website fingerprinting data."""
    
    def __init__(self, data, transform=None, scaler=None, fit_scaler=True):
        """
        Args:
            data: List of dictionaries containing trace data
            transform: Optional transform to be applied on traces
            scaler: Scaler object for normalization (StandardScaler or MinMaxScaler)
            fit_scaler: Whether to fit the scaler on this data
        """
        self.data = data
        self.transform = transform
        self.scaler = scaler
        
        # Extract traces and labels
        self.traces = []
        self.labels = []
        self.websites = []
        
        # Create website to index mapping
        unique_websites = sorted(list(set(item['website'] for item in data)))  # SORT for consistency
        self.website_to_idx = {website: idx for idx, website in enumerate(unique_websites)}
        self.idx_to_website = {idx: website for website, idx in self.website_to_idx.items()}
        
        # Process all traces first
        raw_traces = []
        for item in data:
            trace_data = item['trace_data']
            website = item['website']
            
            # Pad or truncate traces to INPUT_SIZE
            if len(trace_data) < INPUT_SIZE:
                # Pad with the mean of the trace instead of zeros
                trace_mean = np.mean(trace_data) if len(trace_data) > 0 else 0
                padded_trace = trace_data + [trace_mean] * (INPUT_SIZE - len(trace_data))
            else:
                # Truncate to INPUT_SIZE
                padded_trace = trace_data[:INPUT_SIZE]
            
            raw_traces.append(padded_trace)
            self.labels.append(self.website_to_idx[website])
            self.websites.append(website)
        
        # Apply normalization
        if self.scaler is not None:
            if fit_scaler:
                # Fit scaler on the training data
                self.traces = self.scaler.fit_transform(raw_traces)
            else:
                # Use already fitted scaler for test data
                self.traces = self.scaler.transform(raw_traces)
        else:
            self.traces = raw_traces
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = torch.FloatTensor(self.traces[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        if self.transform:
            trace = self.transform(trace)
            
        return trace, label


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        

class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


# =============================================
# BONUS TASK 4: ADVANCED MODELS
# =============================================

class ResidualBlock1D(nn.Module):
    """1D Residual Block for improved gradient flow"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.skip_connection(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """Self-attention mechanism for sequence data"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentionModule, self).__init__()
        
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        
        # Compute queries, keys, values
        queries = self.query_linear(x)  # (batch_size, seq_len, hidden_dim)
        keys = self.key_linear(x)       # (batch_size, seq_len, hidden_dim)
        values = self.value_linear(x)   # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Project back to original dimension
        output = self.output_linear(attended_values)
        
        return output


class AdvancedResNetClassifier(nn.Module):
    """Advanced ResNet-style model with attention mechanism"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdvancedResNetClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Attention mechanism
        self.attention = AttentionModule(512, hidden_size)
        
        # Classification head
        self.fc1 = nn.Linear(512, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial processing
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)  # (batch_size, 512, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class TransformerClassifier(nn.Module):
    """Transformer-based model for sequence classification"""
    
    def __init__(self, input_size, hidden_size, num_classes, num_heads=8, num_layers=4):
        super(TransformerClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(1, hidden_size)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Reshape and project input
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_size)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size)
        
        # Classification
        x = self.classifier(x)
        
        return x


class HybridCNNTransformer(nn.Module):
    """Hybrid model combining CNN feature extraction with Transformer processing"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(HybridCNNTransformer, self).__init__()
        
        # CNN feature extractor
        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calculate CNN output size
        cnn_output_size = input_size // 8
        
        # Transformer processing
        self.transformer_dim = hidden_size
        self.feature_projection = nn.Linear(256, self.transformer_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=8,
            dim_feedforward=self.transformer_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn_features(x)  # (batch_size, 256, seq_len/8)
        
        # Transpose for transformer: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.feature_projection(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs and apply softmax
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, num_classes)
        
        # Apply softmax to each model's output
        softmax_outputs = F.softmax(stacked_outputs, dim=-1)
        
        # Weighted average using learnable weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = torch.sum(weights.view(-1, 1, 1) * softmax_outputs, dim=0)
        
        # Convert back to logits for loss calculation
        ensemble_output = torch.log(ensemble_output + 1e-8)
        
        return ensemble_output


def load_and_preprocess_data(dataset_path, normalization='standard'):
    """Load and preprocess the dataset from JSON file.
    Args:
        dataset_path: Path to the dataset JSON file
        normalization: Type of normalization ('standard', 'minmax', or 'none')
    Returns:
        dataset, website_names, website_counts, scaler
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load data from JSON
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # SORT data for consistency - this is important!
    data = sorted(data, key=lambda x: (x['website'], x.get('timestamp', 0)))
    
    print(f"Loaded {len(data)} traces")
    
    # Get website statistics
    website_counts = {}
    for item in data:
        website = item['website']
        website_counts[website] = website_counts.get(website, 0) + 1
    
    print("Dataset composition:")
    for website, count in sorted(website_counts.items()):  # Sort for consistent output
        print(f"  {website}: {count} traces")
    
    # Choose scaler based on normalization type
    if normalization == 'standard':
        scaler = StandardScaler()
        print("Using Standard Scaler (mean=0, std=1)")
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
        print("Using Min-Max Scaler (range 0-1)")
    else:
        scaler = None
        print("No normalization applied")
    
    # Create dataset with normalization
    dataset = FingerprintDataset(data, scaler=scaler, fit_scaler=True)
    website_names = list(dataset.idx_to_website.values())
    
    print(f"Number of unique websites: {len(website_names)}")
    print(f"Website names: {website_names}")
    
    return dataset, website_names, website_counts, scaler


def create_data_splits(dataset, train_split=0.8, random_state=None):
    """Create stratified train/test splits with proper normalization.
    Args:
        dataset: FingerprintDataset instance
        train_split: Proportion of data for training
        random_state: Random seed for reproducibility (None for random)
    Returns:
        train_loader, test_loader
    """
    # Get labels for stratification
    labels = [dataset.labels[i] for i in range(len(dataset))]
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=random_state)
    train_indices, test_indices = next(splitter.split(range(len(dataset)), labels))
    
    # print(f"Training samples: {len(train_indices)}")
    # print(f"Testing samples: {len(test_indices)}")
    
    # Get the original data for proper normalization
    train_data = [dataset.data[i] for i in train_indices]
    test_data = [dataset.data[i] for i in test_indices]
    
    # Create train dataset with scaler fitting
    train_dataset = FingerprintDataset(train_data, scaler=dataset.scaler, fit_scaler=True)
    
    # Create test dataset using the same fitted scaler
    test_dataset = FingerprintDataset(test_data, scaler=dataset.scaler, fit_scaler=False)
    
    # Create data loaders without fixed seeds
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, test_loader


def train_with_scheduler(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path, use_scheduler=True):
    """Enhanced training function with learning rate scheduling and early stopping.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
        use_scheduler: Whether to use learning rate scheduling
    Returns:
        best_accuracy: Best test accuracy achieved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Learning rate scheduler
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    patience = 15  # Increased patience for advanced models
    no_improve_count = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Learning rate scheduling
        if use_scheduler:
            scheduler.step(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model and implement early stopping
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Early stopping
        # if no_improve_count >= patience:
        #     print(f'Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs')
        #     break
    
    return best_accuracy


def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
    Returns:
        best_accuracy: Best test accuracy achieved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    patience = 10  # ADD: Early stopping patience
    no_improve_count = 0
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model and implement early stopping
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Early stopping for small datasets
        # if no_improve_count >= patience:
        #     print(f'Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs')
        #     break
    
    return best_accuracy


def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    Returns:
        all_preds, all_labels: Predictions and true labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


def create_ensemble_models(input_size, hidden_size, num_classes):
    """Create individual models for ensemble"""
    models = [
        ComplexFingerprintClassifier(input_size, hidden_size, num_classes),
        AdvancedResNetClassifier(input_size, hidden_size, num_classes),
        HybridCNNTransformer(input_size, hidden_size, num_classes)
    ]
    return models


def detailed_performance_analysis(results, dataset_size, num_classes):
    """
    Provide detailed analysis of model performance and recommendations
    """
    print(f"\n{'='*60}")
    print("DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Dataset size analysis
    samples_per_class = dataset_size // num_classes
    test_samples = int(dataset_size * 0.2)
    test_per_class = test_samples // num_classes
    
    print(f"Dataset Analysis:")
    print(f"  Total samples: {dataset_size}")
    print(f"  Samples per class: ~{samples_per_class}")
    print(f"  Test samples: {test_samples}")
    print(f"  Test per class: ~{test_per_class}")

    
    # Model complexity analysis
    basic_acc = results.get('Basic CNN', {}).get('best_accuracy', 0)
    complex_acc = results.get('Complex CNN', {}).get('best_accuracy', 0)
    
    print(f"\nModel Performance Analysis:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['best_accuracy']:.4f}")
    
    # Find best performing model
    best_model = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
    best_acc = results[best_model]['best_accuracy']
    
    print(f"\nBest Model: {best_model} ({best_acc:.4f})")
    
    if best_acc > complex_acc:
        print(f"✓ Successfully outperformed Complex CNN by {best_acc - complex_acc:.4f}")
    else:
        print(f"✗ Did not outperform Complex CNN")
    
    return best_acc > 0.6 and len(results) >= 2


def main():
    """Enhanced main function with advanced models for Bonus Task 4."""
    
    print("=== Advanced Website Fingerprinting Models (Bonus Task 4) ===\n")
    
    # 1. Load the dataset from the JSON file
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file '{DATASET_PATH}' not found!")
        print("Please make sure you have run the data collection script first.")
        return
    
    # Load data with standard normalization
    dataset, website_names, website_counts, scaler = load_and_preprocess_data(
        DATASET_PATH, normalization='standard'
    )
    # dataset, website_names, website_counts, scaler = load_and_preprocess_data(
    #     DATASET_PATH, normalization='minmax'
    # )
    # Check if we have enough data
    if len(dataset) < 10:
        print("Error: Not enough data in dataset. Please collect more traces.")
        return
    
    # Check if we have at least 3 websites
    if len(website_names) < 3:
        print("Error: Need at least 3 different websites for classification.")
        return
    
    # 2. Split the dataset into training and testing sets
    train_loader, test_loader = create_data_splits(dataset, TRAIN_SPLIT)
    
    num_classes = len(website_names)
    print(f"Training on {num_classes} classes: {website_names}")
    
    # 3. Define all models to train (including advanced ones)
    models_to_train = [
        {
            'name': 'Basic CNN',
            'model': FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'basic_model.pth'),
            'use_scheduler': False
        },
        {
            'name': 'Complex CNN',
            'model': ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'complex_model.pth'),
            'use_scheduler': False
        },
        {
            'name': 'Advanced ResNet',
            'model': AdvancedResNetClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'resnet_model.pth'),
            'use_scheduler': True
        },
        {
            'name': 'Transformer',
            'model': TransformerClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'transformer_model.pth'),
            'use_scheduler': True
        },
        {
            'name': 'Hybrid CNN-Transformer',
            'model': HybridCNNTransformer(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'hybrid_model.pth'),
            'use_scheduler': True
        }
    ]
    
    results = {}
    trained_models = []
    
    # 4. Train and evaluate each model
    for model_info in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_info['name']}")
        print(f"{'='*60}")
        
        model = model_info['model']
        model_save_path = model_info['save_path']
        use_scheduler = model_info.get('use_scheduler', False)
        
        # Use different optimizers for different models
        if 'Transformer' in model_info['name']:
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE/2, weight_decay=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
        
        criterion = nn.CrossEntropyLoss()
        
        # Train with or without scheduler
        if use_scheduler:
            best_accuracy = train_with_scheduler(
                model, train_loader, test_loader, criterion, optimizer, 
                EPOCHS, model_save_path, use_scheduler=True
            )
        else:
            best_accuracy = train(
                model, train_loader, test_loader, criterion, optimizer, 
                EPOCHS, model_save_path
            )
        
        # Load the best model and evaluate
        model.load_state_dict(torch.load(model_save_path))
        
        print(f"\n--- Final Evaluation for {model_info['name']} ---")
        predictions, labels = evaluate(model, test_loader, website_names)
        
        results[model_info['name']] = {
            'best_accuracy': best_accuracy,
            'model_path': model_save_path
        }
        
        trained_models.append(model)
        
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    # 5. Create and train ensemble model
    print(f"\n{'='*60}")
    print("Training Ensemble Model")
    print(f"{'='*60}")
    
    # Create ensemble from top 3 models
    top_models = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)[:3]
    ensemble_models = []
    
    for model_name, model_info in top_models:
        # Find the corresponding model from trained_models
        for i, trained_model_info in enumerate(models_to_train):
            if trained_model_info['name'] == model_name:
                model = trained_model_info['model']
                model.load_state_dict(torch.load(model_info['model_path']))
                ensemble_models.append(model)
                break
    
    if len(ensemble_models) >= 2:
        ensemble_model = EnsembleModel(ensemble_models, num_classes)
        ensemble_save_path = os.path.join(MODELS_DIR, 'ensemble_model.pth')
        
        # Train ensemble (fine-tune the weights)
        optimizer = optim.Adam(ensemble_model.parameters(), lr=LEARNING_RATE/10)
        criterion = nn.CrossEntropyLoss()
        
        ensemble_accuracy = train_with_scheduler(
            ensemble_model, train_loader, test_loader, criterion, optimizer,
            EPOCHS//2, ensemble_save_path, use_scheduler=True
        )
        
        ensemble_model.load_state_dict(torch.load(ensemble_save_path))
        
        print(f"\n--- Final Evaluation for Ensemble Model ---")
        predictions, labels = evaluate(ensemble_model, test_loader, website_names)
        
        results['Ensemble'] = {
            'best_accuracy': ensemble_accuracy,
            'model_path': ensemble_save_path
        }
    
    # 6. Final comparison and analysis
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}: {result['best_accuracy']:.4f}")
    
    best_model_name = sorted_results[0][0]
    best_accuracy = sorted_results[0][1]['best_accuracy']
    complex_cnn_accuracy = results.get('Complex CNN', {}).get('best_accuracy', 0)
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Complex CNN accuracy: {complex_cnn_accuracy:.4f}")
    
    if best_accuracy > complex_cnn_accuracy:
        improvement = best_accuracy - complex_cnn_accuracy
        print(f"✓ Successfully outperformed Complex CNN by {improvement:.4f} ({improvement*100:.2f}%)")
    else:
        print(f"✗ Did not outperform Complex CNN")
    
    # Detailed analysis
    detailed_performance_analysis(results, len(dataset), num_classes)
    
    # Save the best model as main model
    best_model_path = results[best_model_name]['model_path']
    main_model_path = "model.pth"
    
    shutil.copy2(best_model_path, main_model_path)
    print(f"\nBest model ({best_model_name}) saved as '{main_model_path}'")
    
    # Save enhanced model metadata
    model_metadata = {
        'num_classes': num_classes,
        'website_names': website_names,
        'idx_to_website': dataset.idx_to_website,
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'complex_cnn_accuracy': complex_cnn_accuracy,
        'improvement_over_complex': best_accuracy - complex_cnn_accuracy,
        'all_results': {name: result['best_accuracy'] for name, result in results.items()},
        'normalization': 'standard',
        'scaler_params': {
            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"Enhanced model metadata saved to 'model_metadata.json'")
    
    # Bonus Task 4 completion check
    if best_accuracy > complex_cnn_accuracy:
        print(f"\n🎉 BONUS TASK 4 COMPLETED SUCCESSFULLY! 🎉")
        print(f"Advanced model achieved {best_accuracy:.4f} accuracy")
        print(f"Outperformed Complex CNN by {(best_accuracy - complex_cnn_accuracy)*100:.2f}%")
    else:
        print(f"\n⚠️  Bonus Task 4 not completed - no improvement over Complex CNN")
    
    print(f"\nAdvanced model training completed!")


if __name__ == "__main__":
    main()