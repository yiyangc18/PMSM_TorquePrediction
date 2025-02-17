
# PMSM Torque Prediction

This project is designed to predict the torque output of Permanent Magnet Synchronous Motors (PMSM) in electric vehicles, focusing on data-driven approaches and machine learning techniques. The aim is to model the torque dynamics based on various motor parameters and operating conditions.

## Overview

This repository contains scripts and models for predicting the output torque of PMSM motors based on an open-source dataset [https://github.com/upb-lea/deep-pmsm]. The project covers multiple phases, including dataset split, online/offline RLS parameter identification, training models, and validating the predictions.


## File Descriptions

| File Name                             | Description                                                        |
|---------------------------------------|--------------------------------------------------------------------|
| `data_loader.py`                      | Contains functions to load data into the model, including batch processing and shuffling. |
| `hyperparameter_search.py`            | Used for searching the optimal hyperparameters for the model.       |
| `models.py`                           | Defines the model architecture for PMSM torque prediction.         |
| `config.py`                           | Contains configuration parameters for the model.                  |
| `motor_online_identification.py`      | Implements online motor RLS parameter identification during operation. |
| `motor_parameter_identification.py`   | Script for RLS offline motor parameter identification.             |
| `offline_train_validation.py`         | Trains and validates the model using train-data.                   |
| `online_pretrain.py`                  | Pre-trains the model online using the training dataset.            |
| `online_train_validation.py`          | Performs online training and validation, updating the model with real-time data. |
| `preprocess/`                         | Contains intermediate steps for preprocessing raw data.            |
| `result_processor.py`                 | Processes the model's prediction results and saves them for comparison. |


## Requirements

The following packages are required to run the project:
- Python 3.x
- NumPy
- PyTorch

Our environment is as follows:
- Python 3.12.0
- NumPy 1.21.3
- PyTorch 1.10.0
- CUDA 12.3

## Usage

### Data Preprocessing

Download data from: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature/data, saving it in the `data` folder.

All the results will be processed by the `result_processor.py` script, which will save the results in the `results` folder.

### Offline/Online RLS

RLS parameter identification can be performed offline or online. 

offline RLS parameter identification can be performed by running:
```bash
python motor_parameter_identification.py
```

online RLS parameter identification can be performed by running:
```bash
python motor_online_identification.py
```

### Offline Training and Validation

To train the model using the preprocessed data in an offline manner, run:

```bash
python offline_train_validation.py
```

This will train the model, validate it, and save the results in `training.log`.
the best model will be selected based on the 'best_mae' metric, and saved in the `models` folder.

### Online Training

Before running the online training, we can use the pre-trained model to initialize the weights. To do this, run:

```bash
python online_pretrain.py
```

Then, to perform online training and validation, run:

```bash
python online_train_validation.py
```

