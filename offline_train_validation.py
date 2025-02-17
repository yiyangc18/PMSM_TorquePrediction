# train_validation.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_loader import DataLoader as CustomDataLoader
from config import FNN_HYPERPARAMS, LSTM_HYPERPARAMS
from models import FeedForwardNN, LSTMModel
from result_processor import ResultProcessor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# Custom Dataset
class MotorDataset(Dataset):
    def __init__(self, data, sequence_length=1):
        self.data = data
        self.sequence_length = sequence_length
        self.id = data['i_d'].values
        self.iq = data['i_q'].values
        self.ud = data['u_d'].values
        self.uq = data['u_q'].values
        self.motor_speed = data['motor_speed'].values
        self.Te = data['torque'].values
        self.profile_id = data['profile_id'].values
        
        self.features = np.column_stack((self.id, self.iq, self.ud, self.uq, self.motor_speed))
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
        
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.sequence_length]
        y = self.Te[idx+self.sequence_length-1]
        profile_id = self.profile_id[idx+self.sequence_length-1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), profile_id

# Training function
def train_model(model, train_loader, val_loader, config, device='cpu'):
    """Train the model and return the best model based on validation performance."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('l2_reg', 0.0))
    train_losses = []
    val_losses = []
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Check and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = FeedForwardNN(
                input_dim=config['input_dim'],
                hidden_layers=config['hidden_layers'],
                hidden_units=config['hidden_units']
            ).to(device)
            best_model.load_state_dict(model.state_dict())

        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print(f"Best model with validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses, best_model

# Evaluation and plotting function
def evaluate_and_plot(model, data_loader, data_label, result_processor, device='cpu'):
    model.eval()
    actuals = []
    predictions = []
    profile_ids = []
    with torch.no_grad():
        for inputs, targets, profiles in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.squeeze().cpu().numpy())
            profile_ids.extend(profiles.numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    profile_ids = np.array(profile_ids)
    
    # Calculate error metrics
    error = actuals - predictions
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(error))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error
    }
    params = {}  # Placeholder for any model-specific parameters

    # Use ResultProcessor to save and plot results
    result_processor.save_results(profile_ids, actuals, predictions, data_label, params, metrics)
    result_processor.plot_results(actuals, predictions, data_label, metrics)

    print(f'Error metrics on {data_label} set:')
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, Max Error: {max_error:.4f}')

# Main script
if __name__ == '__main__':
    file_path = "data/measures_v3.csv"
    data_loader = CustomDataLoader(file_path)
    train_data = data_loader.get_train_data()
    val_data = data_loader.get_val_data()
    test_data = data_loader.get_test_data()

    model_config = FNN_HYPERPARAMS  # Default to FNN hyperparameters

    input_dim = 5
    sequence_length = 1
    model_config['input_dim'] = input_dim

    train_dataset = MotorDataset(train_data, sequence_length=sequence_length)
    val_dataset = MotorDataset(val_data, sequence_length=sequence_length)
    test_dataset = MotorDataset(test_data, sequence_length=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = FeedForwardNN(
        input_dim=input_dim,
        hidden_layers=model_config['hidden_layers'],
        hidden_units=model_config['hidden_units']
    ).to(device)

    train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, model_config, device)

    output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(output_dir, f'FNN_model_{timestamp}.pth')
    torch.save(best_model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    result_processor = ResultProcessor(output_dir='data')
    evaluate_and_plot(best_model, val_loader, 'offline_dl_validation', result_processor, device=device)
    evaluate_and_plot(best_model, test_loader, 'offline_dl_test', result_processor, device=device)
