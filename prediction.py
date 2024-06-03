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
        x = F.relu(self.fc1(x)) 
        return x

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def predict(model, input_data):
    # Convert input data to PyTorch tensor
    
    # Perform forward pass
    with torch.no_grad():
        output = model(input_data)

    _, predicted_class = torch.max(output, 0)
    
    return predicted_class.item()  # Return the predicted class label


def main():
    forest_test = pd.read_csv('forest_test.csv')

    X_test = forest_test.drop(columns=['Cover_Type']).values
    y_test = forest_test['Cover_Type'].values

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = Model().to(device)
    model_path = 'model.pth'  # Path to your saved model file
    load_model(model, model_path)

    predictions = []
    for input_data in X_test:
        predicted_class = predict(model, input_data)
        predictions.append(predicted_class)
    
    with open(r'predictions.txt', 'w') as fp:
        for item in predictions:
            # write each item on a new line
            fp.write("%s\n" % item)
   

if __name__ == "__main__":
    main()