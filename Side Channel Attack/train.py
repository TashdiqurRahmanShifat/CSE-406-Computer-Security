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
        if len(train_loader.dataset) < 100 and no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs')
            break
    
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


# def analyze_data_distribution(dataset):
#     """Analyze the distribution of trace data for better understanding."""
#     all_traces = np.array([dataset.traces[i] for i in range(len(dataset))])
    
#     print(f"\nData Distribution Analysis:")
#     print(f"  Shape: {all_traces.shape}")
#     print(f"  Mean: {np.mean(all_traces):.4f}")
#     print(f"  Std: {np.std(all_traces):.4f}")
#     print(f"  Min: {np.min(all_traces):.4f}")
#     print(f"  Max: {np.max(all_traces):.4f}")
    
#     # Check for any extreme values
#     percentiles = [1, 5, 25, 50, 75, 95, 99]
#     print(f"  Percentiles:")
#     for p in percentiles:
#         print(f"    {p}%: {np.percentile(all_traces, p):.4f}")



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
    print(f"  Basic CNN:   {basic_acc:.4f}")
    print(f"  Complex CNN: {complex_acc:.4f}")
    print(f"  Difference:  {abs(basic_acc - complex_acc):.4f}")
    
    if basic_acc > complex_acc:
        print(f"\nAnalysis: Basic CNN outperformed Complex CNN")
    else:
        print(f"\nAnalysis: Complex CNN performed better")
    
    return basic_acc > 0.6 and len(results) >= 2

def main():
    """Implement the main function to train and evaluate the models."""
    
    print("=== Website Fingerprinting Model Training ===\n")
    
    # 1. Load the dataset from the JSON file
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file '{DATASET_PATH}' not found!")
        print("Please make sure you have run the data collection script first.")
        return
    
    # Load data with standard normalization (consistent for small datasets)
    dataset, website_names, website_counts, scaler = load_and_preprocess_data(
        DATASET_PATH, normalization='standard'
    )
    
    # Analyze data distribution
    # analyze_data_distribution(dataset)
    
    # Check if we have enough data
    if len(dataset) < 10:
        print("Error: Not enough data in dataset. Please collect more traces.")
        return
    
    # Check if we have at least 3 websites (requirement)
    if len(website_names) < 3:
        print("Error: Need at least 3 different websites for classification.")
        return
    
    # 2. Split the dataset into training and testing sets
    # 3. Create data loader for training and testing
    train_loader, test_loader = create_data_splits(dataset, TRAIN_SPLIT)
    
    num_classes = len(website_names)
    
    # 4. Define the models to train
    models_to_train = [
        {
            'name': 'Basic CNN',
            'model': FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'basic_model.pth')
        },
        {
            'name': 'Complex CNN',
            'model': ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
            'save_path': os.path.join(MODELS_DIR, 'complex_model.pth')
        }
    ]
    
    results = {}
    
    # 5. Train and evaluate each model
    for model_info in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_info['name']} with standard normalization")
        print(f"{'='*50}")
        
        model = model_info['model']
        model_save_path = model_info['save_path']
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        effective_epochs =  EPOCHS
        
        best_accuracy = train(model, train_loader, test_loader, criterion, optimizer, effective_epochs, model_save_path)
        
        model.load_state_dict(torch.load(model_save_path))
        
        print(f"\n--- Final Evaluation for {model_info['name']} ---")
        predictions, labels = evaluate(model, test_loader, website_names)
        
        results[model_info['name']] = {
            'best_accuracy': best_accuracy,
            'model_path': model_save_path
        }
        
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_accuracy']:.4f}")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['best_accuracy'])
    best_accuracy = results[best_model_name]['best_accuracy']
    
    print(f"\nBest performing model: {best_model_name} with {best_accuracy:.4f} accuracy")

    detailed_performance_analysis(results, len(dataset), num_classes)
    
    best_model_path = results[best_model_name]['model_path']
    main_model_path = "model.pth"
    
    shutil.copy2(best_model_path, main_model_path)
    print(f"Best model saved as '{main_model_path}'")
    
    print(f"\n{'='*50}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    
    if best_accuracy >= 0.6:
        print("Successfully achieved >60% classification accuracy!")
    else:
        print("Did not achieve 60% accuracy. Consider:")
    
    print(f"\nDataset statistics:")
    print(f"  Total traces: {len(dataset)}")
    print(f"  Number of websites: {len(website_names)}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Testing samples: {len(test_loader.dataset)}")
    
    # Save model metadata, now including the 'idx_to_website' mapping
    model_metadata = {
        'num_classes': num_classes,
        'website_names': website_names,
        'idx_to_website': dataset.idx_to_website,  # <-- THIS LINE IS ADDED
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'best_model': best_model_name,
        'best_accuracy': best_accuracy,
        'normalization': 'standard',
        'scaler_params': {
            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"Model metadata saved to 'model_metadata.json'")
    print(f"\nTraining completed successfully!")


if __name__ == "__main__":
    main()