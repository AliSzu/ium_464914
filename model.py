import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class Model(nn.Module):
    def __init__(self, input_features=54, hidden_layer1=25, hidden_layer2=30, output_features=8):
        super().__init__()
        self.fc1 = nn.Linear(input_features,output_features)
        self.bn1 = nn.BatchNorm1d(hidden_layer1)  # Add batch normalization
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.bn2 = nn.BatchNorm1d(hidden_layer2)  # Add batch normalization
        self.out = nn.Linear(hidden_layer2, output_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply batch normalization after first linear layer
        #x = F.relu(self.bn2(self.fc2(x)))  # Apply batch normalization after second linear layer
        #x = self.out(x)
        return x

def main():
    forest_train = pd.read_csv('forest_train.csv')
    forest_val = pd.read_csv('forest_val.csv')

    print(forest_train.head())


    X_train = forest_train.drop(columns=['Cover_Type']).values
    y_train = forest_train['Cover_Type'].values

    X_val = forest_val.drop(columns=['Cover_Type']).values
    y_val = forest_val['Cover_Type'].values


    # Initialize model, loss function, and optimizer
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    # Create DataLoader
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=64)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Calculate training loss
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
        

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()
