# config.py

COLUMNS_TO_READ = [
    'u_q',
    'u_d',
    'motor_speed',
    'i_d',
    'i_q',
    'torque',
    'profile_id'
]

# Hyperparameters for FNN model
FNN_HYPERPARAMS = {
    'model': 'FNN',
    'learning_rate': 0.008335468939021625,
    'batch_size': 64,
    'hidden_layers': 3,
    'hidden_units': 128,
    'l2_reg': 1.0566845547962655e-06,
    'num_epochs': 10
}

# LSTM model hyperparameters
LSTM_HYPERPARAMS = {
    'model': 'LSTM',
    'learning_rate': 0.001533393402517762,
    'batch_size': 64,
    'hidden_layers': 4,
    'hidden_units': 128,
    'l2_reg': 1.0850768762049176e-06,
    'sequence_length': 12,
    'tbptt_step': 5,
    'grad_clip_norm': 1.0,
    'num_epochs': 10,
    'dropout': 0.2,
    'bidirectional': False
}

ONLINE_FNN_PRETRAIN_HYPERPARAMS = {
    'learning_rate': 0.0007335468939021625,
    'batch_size': 32,  # 可以与 UpdateWin 重合或为其倍数
    'num_epochs': 30,  # 每次更新时训练的 epoch 数
    'hidden_layers': 3,
    'hidden_units': 64,
    'sequence_length': 1,
    'dropout': 0.25,     # Dropout 概率
}



ONLINE_FNN_HYPERPARAMS = {
    'learning_rate': 0.00000032,
    'batch_size': 32,  # 可以与 UpdateWin 重合或为其倍数
    'num_epochs': 2,  # 每次更新时训练的 epoch 数
    'hidden_layers': 3,
    'hidden_units': 64,
    'sequence_length': 1,
    'UpdateWin': 12800,  # 更新窗口大小
    'Win': 6400,         # 预测窗口大小
    'dropout': 0.25,     # Dropout 概率
}

OFFLINE_FNN_HYPERPARAMS = {
    'learning_rate': 0.001,  # 通常离线训练使用较低的学习率
    'batch_size': 128,
    'num_epochs': 100,
    'hidden_layers': 3,
    'hidden_units': 64,
    'sequence_length': 1,
    'dropout': 0.5,
}

# 在线 LSTM 超参数
ONLINE_LSTM_HYPERPARAMS = {
    'learning_rate': 1e-3,
    'batch_size': 128,
    'num_epochs': 5,
    'hidden_layers': 2,
    'hidden_units': 64,
    'sequence_length': 5,
    'UpdateWin': 128,
    'Win': 64,
}