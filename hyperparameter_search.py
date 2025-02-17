# hyperparameter_search.py

import optuna
import subprocess
import os
import sys
import shutil
import logging
import json
from config import FNN_HYPERPARAMS, LSTM_HYPERPARAMS

# Setup logging
logging.basicConfig(
    filename='hyperparameter_search.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the objective function
def objective(trial):
    # Choose model
    model_name = trial.suggest_categorical('model', ['FNN', 'LSTM'])

    # Load existing hyperparameters from config.py
    if model_name == 'FNN':
        hyperparams = FNN_HYPERPARAMS.copy()
    elif model_name == 'LSTM':
        hyperparams = LSTM_HYPERPARAMS.copy()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Suggest new hyperparameters around current values
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
    num_epochs = 1  # For hyperparameter search, we can set epochs to 1

    hyperparams.update({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'l2_reg': l2_reg,
        'num_epochs': num_epochs,
    })

    # Model-specific hyperparameters
    if model_name == 'FNN':
        hidden_layers = trial.suggest_int('hidden_layers', max(1, hyperparams['hidden_layers'] - 1), hyperparams['hidden_layers'] + 1)
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128, 256])
        hyperparams.update({
            'hidden_layers': hidden_layers,
            'hidden_units': hidden_units,
        })
    elif model_name == 'LSTM':
        hidden_layers = trial.suggest_int('hidden_layers', max(1, hyperparams['hidden_layers'] - 1), hyperparams['hidden_layers'] + 1)
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128, 256])
        sequence_length = trial.suggest_int('sequence_length', max(5, hyperparams['sequence_length'] - 5), hyperparams['sequence_length'] + 5)
        grad_clip_norm = trial.suggest_float('grad_clip_norm', 0.1, 5.0)
        hyperparams.update({
            'hidden_layers': hidden_layers,
            'hidden_units': hidden_units,
            'sequence_length': sequence_length,
            'grad_clip_norm': grad_clip_norm,
        })

    # Backup the original config.py
    shutil.copyfile('config.py', 'config_backup.py')

    try:
        # Update config.py with new hyperparameters
        update_config(hyperparams, model_name)

        # Run the training script with the current hyperparameters
        cmd = [sys.executable, 'train.py', '--model', model_name]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Read the validation MAE from the training log or output
        val_mae = parse_validation_mae('training.log', model_name)

        # Log the hyperparameters and result
        logging.info(f'Model: {model_name}, Hyperparameters: {hyperparams}, Validation MAE: {val_mae}')

        # Update best hyperparameters if improved
        update_best_hyperparams(val_mae, hyperparams, model_name)

        return val_mae

    except Exception as e:
        logging.error(f'Error during trial: {e}')
        return float('inf')

    finally:
        # Restore the original config.py
        shutil.move('config_backup.py', 'config.py')


def update_config(hyperparams, model_name):
    # Read the existing config.py
    with open('config.py', 'r') as f:
        lines = f.readlines()

    # Modify the hyperparameters for the specified model
    with open('config.py', 'w') as f:
        inside_model_config = False
        model_config_started = False
        for line in lines:
            if line.strip().startswith(f'{model_name}_HYPERPARAMS'):
                inside_model_config = True
                model_config_started = True
                f.write(f'{model_name}_HYPERPARAMS = {{\n')
                for key, value in hyperparams.items():
                    f.write(f"    '{key}': {repr(value)},\n")
                f.write('}\n')
            elif inside_model_config and line.strip().startswith('}'):
                inside_model_config = False
                continue  # Skip the closing brace since we've already written it
            elif not inside_model_config:
                f.write(line)
            else:
                continue  # Skip lines inside the old hyperparameter definition

        # If the model config was not found, append it at the end
        if not model_config_started:
            f.write(f'\n{model_name}_HYPERPARAMS = {{\n')
            for key, value in hyperparams.items():
                f.write(f"    '{key}': {repr(value)},\n")
            f.write('}\n')


def parse_validation_mae(log_file, model_name):
    # Read the last validation MAE from the log file
    val_mae = None
    with open(log_file, 'r') as f:
        for line in f:
            if f'Validation MAE:' in line:
                val_mae_str = line.strip().split('Validation MAE:')[1]
                val_mae = float(val_mae_str.split(',')[0])
    if val_mae is None:
        raise ValueError('Validation MAE not found in log file.')
    return val_mae


def update_best_hyperparams(val_mae, hyperparams, model_name):
    # Load current best MAE from best_mae.json
    best_mae_file = 'best_mae.json'
    if os.path.exists(best_mae_file):
        with open(best_mae_file, 'r') as f:
            best_mae_data = json.load(f)
    else:
        best_mae_data = {}

    current_best_mae = best_mae_data.get(model_name, {}).get('best_val_mae', float('inf'))

    if val_mae < current_best_mae:
        # Update best_mae.json
        best_mae_data[model_name] = {'best_val_mae': val_mae}
        with open(best_mae_file, 'w') as f:
            json.dump(best_mae_data, f)
        # Update config.py with best hyperparameters
        update_config(hyperparams, model_name)
        logging.info(f'New best hyperparameters found for {model_name} with Validation MAE: {val_mae}')
    else:
        logging.info(f'No improvement for {model_name}, current best Validation MAE: {current_best_mae}')


if __name__ == '__main__':
    # Create an Optuna study
    study_name = 'motor_hyperparameter_optimization'
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        load_if_exists=True
    )

    # Start optimization
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print('Best hyperparameters:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

    print('Hyperparameter search completed.')
