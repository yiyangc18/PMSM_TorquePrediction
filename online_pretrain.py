import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

from data_loader import DataLoader as CustomDataLoader
from models import FNNOfflineOnlineModel, FNNOnlineModel
from result_processor import ResultProcessor
from config import ONLINE_FNN_PRETRAIN_HYPERPARAMS

def create_features(data, dt=0.5):
    """
    构造 7 维特征:
      i_d, i_q, u_d, u_q, we, did_dt, diq_dt
    并返回扭矩作为 targets
    """
    id_current = data['i_d'].values
    iq_current = data['i_q'].values
    Ud = data['u_d'].values
    Uq = data['u_q'].values
    we = data['motor_speed'].values
    
    # 计算微分
    did_dt = np.concatenate((np.diff(id_current) / dt, [0]))
    diq_dt = np.concatenate((np.diff(iq_current) / dt, [0]))
    
    # 拼成一个7维的输入特征
    features = np.column_stack((id_current, iq_current, Ud, Uq, we, did_dt, diq_dt))
    targets = data["torque"].values
    profile_ids = data["profile_id"].values

    return features, targets, profile_ids


# ----------- 2. 自定义的 Dataset，与之前类似 -----------
class MotorDataset(Dataset):
    def __init__(self, features, targets, profile_ids, sequence_length=1):
        self.features = features
        self.targets = targets
        self.profile_ids = profile_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        pid = self.profile_ids[idx + self.sequence_length - 1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y, pid


# ----------- 3. 物理公式损失函数，与之前保持一致 -----------
def model_loss(model, inputs, Te_actual, P, device):
    model.train()
    inputs = inputs.to(device)
    Te_actual = Te_actual.to(device)

    # Forward pass
    YPred = model(inputs)
    # print("YPred shape:", YPred.shape)  # Debug

    # 如果有序列维度，就取最后一个时刻
    if inputs.dim() == 3:
        id_current = inputs[:, -1, 0]
        iq_current = inputs[:, -1, 1]
    else:
        id_current = inputs[:, 0]
        iq_current = inputs[:, 1]

    Rs = YPred[:, 0]
    Ld = YPred[:, 1]
    Lq = YPred[:, 2]
    Psi_m = YPred[:, 3]

    Te_pred = 1.5 * P * (Psi_m * iq_current + (Ld - Lq) * id_current * iq_current)

    loss = torch.mean((Te_pred - Te_actual) ** 2)
    return loss


# ----------- 4. 带 Warmup 的训练函数 -----------
def train_model(model, train_loader, val_loader, config, P, device='cpu'):
    """
    用 LambdaLR 实现一个简单的线性 Warmup + 线性 Decay 的学习率调度：
    1) 前 warmup_epochs 个epoch: lr从 warmup_start_lr -> base_lr
    2) 之后的 decay_epochs 个epoch: lr从 base_lr -> final_lr
    如果不想要后续衰减，只要把 final_lr 设置成和 base_lr 一样即可。
    """

    # 你可以根据需要自由修改这些参数
    warmup_epochs = 5           # 需要热身的epoch数
    warmup_start_lr = 1e-5      # 热身起始LR
    base_lr = config['learning_rate']  # 热身后到达的LR
    final_lr = 1e-4             # 最终衰减到的LR
    total_epochs = config['num_epochs']
    decay_epochs = total_epochs - warmup_epochs  # 衰减区间

    # 如果 decay_epochs <= 0，说明全程只做warmup，不做衰减
    if decay_epochs <= 0:
        decay_epochs = 0

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # 定义lambda函数来根据 epoch 计算一个 lr_factor
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 在 [0, warmup_epochs) 内做线性上升
            return float(epoch + 1) * (base_lr - warmup_start_lr) / warmup_epochs / base_lr \
                   + warmup_start_lr / base_lr
        else:
            # 如果需要衰减
            if decay_epochs > 0:
                # 从 epoch=warmup_epochs 到 total_epochs 线性衰减
                progress = (epoch - warmup_epochs) / decay_epochs
                # progress 从 0 -> 1
                return max( final_lr / base_lr, 1.0 - progress*(1.0 - final_lr/base_lr) )
            else:
                # 不做衰减，直接返回1.0 (保持base_lr不变)
                return 1.0

    # 创建调度器
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses = []
    val_losses = []
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(total_epochs):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            loss = model_loss(model, inputs, targets, P, device)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * inputs.size(0)

        # 更新训练loss
        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)
                v_loss = model_loss(model, inputs, targets, P, device)
                val_loss_sum += v_loss.item() * inputs.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_loss)

        # 如果验证集更优，则保存当前最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = FNNOnlineModel(
                input_dim=config['input_dim'],
                hidden_layers=config['hidden_layers'],
                hidden_units=config['hidden_units']
            ).to(device)
            best_model.load_state_dict(model.state_dict())

        # 每个epoch 结束后，scheduler.step() 来更新学习率
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']  # 取第一个param_group的lr
        print(f'Epoch [{epoch + 1}/{total_epochs}], '
              f'LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print(f"Best model with validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses, best_model


# ----------- 5. 评估函数，与之前保持一致 -----------
def evaluate_and_plot(model, data_loader, data_label, result_processor, P, device='cpu'):
    model.eval()
    actuals = []
    predictions = []
    profile_ids = []
    with torch.no_grad():
        for inputs, targets, profiles in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            YPred = model(inputs)

            if inputs.dim() == 3:
                id_current = inputs[:, -1, 0]
                iq_current = inputs[:, -1, 1]
            else:
                id_current = inputs[:, 0]
                iq_current = inputs[:, 1]

            Rs = YPred[:, 0]
            Ld = YPred[:, 1]
            Lq = YPred[:, 2]
            Psi_m = YPred[:, 3]

            Te_pred = 1.5 * P * (Psi_m * iq_current + (Ld - Lq) * id_current * iq_current)
            actuals.extend(targets.cpu().numpy())
            predictions.extend(Te_pred.cpu().numpy())
            profile_ids.extend(profiles.numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    profile_ids = np.array(profile_ids)

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
    params = {}

    result_processor.save_results(profile_ids, actuals, predictions, data_label, params, metrics)
    result_processor.plot_results(actuals, predictions, data_label, metrics)

    print(f'Error metrics on {data_label} set:')
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, Max Error: {max_error:.4f}')


# ----------- 6. 主函数 -----------
if __name__ == '__main__':
    # 1. 加载原始数据
    file_path = "data/measures_v3.csv"
    data_loader = CustomDataLoader(file_path)
    train_data = data_loader.get_train_data()
    val_data = data_loader.get_val_data()
    test_data = data_loader.get_test_data()

    # 2. 创建不含微分的特征
    train_features, train_targets, train_pids = create_features(train_data)
    val_features, val_targets, val_pids = create_features(val_data)
    test_features, test_targets, test_pids = create_features(test_data)

    # 3. 用训练集特征 fit StandardScaler, 并transform到val和test
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 4. 组装 Dataset
    config = ONLINE_FNN_PRETRAIN_HYPERPARAMS
    sequence_length = config['sequence_length']  # 你可根据需要改大，获取多步时序信息
    train_dataset = MotorDataset(train_features_scaled, train_targets, train_pids, sequence_length)
    val_dataset = MotorDataset(val_features_scaled, val_targets, val_pids, sequence_length)
    test_dataset = MotorDataset(test_features_scaled, test_targets, test_pids, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # 5. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_dim = 7
    config['input_dim'] = input_dim

    model = FNNOnlineModel(
        input_dim=input_dim,
        hidden_layers=config['hidden_layers'],
        hidden_units=config['hidden_units']
    ).to(device)

    # 6. 训练
    P = 8  # 极对数 
    train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, config, P, device)

    # 7. 保存
    output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)
    model_filename = "FNNOnlineModel_pretrained.pth"
    model_save_path = os.path.join(output_dir, model_filename)
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

    # 8. 评估
    result_processor = ResultProcessor(output_dir='data')
    evaluate_and_plot(best_model, val_loader, 'pretrain_validation', result_processor, P, device)
    evaluate_and_plot(best_model, test_loader, 'pretrain_test', result_processor, P, device)
