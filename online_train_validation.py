import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
import os
import json
import logging

from data_loader import DataLoader as CustomDataLoader
from models import FNNOnlineModel  # 请确保已定义好
from result_processor import ResultProcessor
from config import ONLINE_FNN_HYPERPARAMS

# 电压方程损失
def model_loss(model, inputs, dt, P, device):
    model.train()
    inputs = inputs.to(device)

    YPred = model(inputs)  # [batch_size, 4], 预测 Rs, Ld, Lq, Psi_m

    # 提取输入
    id_current = inputs[:, -1, 0] if inputs.dim() == 3 else inputs[:, 0]
    iq_current = inputs[:, -1, 1] if inputs.dim() == 3 else inputs[:, 1]
    Vd = inputs[:, -1, 2] if inputs.dim() == 3 else inputs[:, 2]
    Vq = inputs[:, -1, 3] if inputs.dim() == 3 else inputs[:, 3]
    we = inputs[:, -1, 4] if inputs.dim() == 3 else inputs[:, 4]
    did_dt = inputs[:, -1, 5] if inputs.dim() == 3 else inputs[:, 5]
    diq_dt = inputs[:, -1, 6] if inputs.dim() == 3 else inputs[:, 6]

    # 预测参数
    Rs = YPred[:, 0]
    Ld = YPred[:, 1]
    Lq = YPred[:, 2]
    Psi_m = YPred[:, 3]

    # 电压残差
    Residual_Vd = Vd - (Rs * id_current + Ld * did_dt - we * Lq * iq_current)
    Residual_Vq = Vq - (Rs * iq_current + Lq * diq_dt + we * (Ld * id_current + Psi_m))
    loss = torch.mean(abs(Residual_Vd) + abs(Residual_Vq))
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online train motor parameter estimation model.')
    parser.add_argument('--pretrained_path', type=str,
                        default='data/output/Pretrained_FNNOnlineModel.pth',
                        help='Path to the pretrained offline model.')
    args = parser.parse_args()

    # 日志 & 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    logging.basicConfig(
        filename='online_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Using device: {device}')

    # 超参数
    config = ONLINE_FNN_HYPERPARAMS
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 5)
    hidden_layers = config.get('hidden_layers', 2)
    hidden_units = config.get('hidden_units', 64)
    sequence_length = config.get('sequence_length', 1)
    UpdateWin = config.get('UpdateWin', 128)
    Win = config.get('Win', 64)

    # 数据
    file_path = "data/measures_v3.csv"
    data_loader = CustomDataLoader(file_path)
    online_data = data_loader.get_test_data()
    Te_data = online_data['torque'].values
    online_data = online_data.drop(columns=['torque'])

    class MotorOnlineDataset:
        def __init__(self, data, sequence_length=1):
            self.sequence_length = sequence_length
            id_current = data['i_d'].values
            iq_current = data['i_q'].values
            Ud = data['u_d'].values
            Uq = data['u_q'].values
            we = data['motor_speed'].values
            dt = 0.5
            did_dt = np.concatenate((np.diff(id_current)/dt, [0]))
            diq_dt = np.concatenate((np.diff(iq_current)/dt, [0]))
            self.X = np.column_stack((id_current, iq_current, Ud, Uq, we, did_dt, diq_dt))
            self.dt = dt

        def __len__(self):
            return len(self.X) - self.sequence_length + 1

        def get_batch(self, idx_start, idx_end):
            inputs = []
            for idx in range(idx_start, idx_end):
                x = self.X[idx: idx + self.sequence_length]
                if self.sequence_length == 1:
                    x = x.squeeze(0)
                inputs.append(torch.tensor(x, dtype=torch.float32))
            return torch.stack(inputs)

    online_dataset = MotorOnlineDataset(online_data, sequence_length)
    total_samples = len(online_dataset)

    # 初始化模型
    model = FNNOnlineModel(
        input_dim=7,  
        hidden_layers=hidden_layers,
        hidden_units=hidden_units
    ).to(device)

    # 加载预训练
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Pretrained model loaded successfully.")
    else:
        print(f"Warning: Pretrained model file '{args.pretrained_path}' not found.")

    # Online过程
    P = 8 # 极对数
    idx = 0
    actuals = []
    predictions = []

    # 这里我们只保存“最后一个样本”的4个参数
    final_Rs, final_Ld, final_Lq, final_Psi_m = None, None, None, None

    while idx + UpdateWin + Win <= total_samples:
        # (a) 用 UpdateWin 做在线训练
        train_inputs = online_dataset.get_batch(idx, idx + UpdateWin)
        train_loader = DataLoader(train_inputs, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        best_epoch_loss = float('inf')
        best_epoch_idx = -1

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_in in train_loader:
                batch_in = batch_in.to(device)
                optimizer.zero_grad()
                loss = model_loss(model, batch_in, online_dataset.dt, P, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                best_epoch_idx = epoch

        print(f"[Index {idx} | {total_samples}] best epoch: {best_epoch_idx+1}, loss: {best_epoch_loss:.6f}")
        logging.info(f"[Index {idx} | {total_samples}] best epoch: {best_epoch_idx+1}, loss: {best_epoch_loss:.6f}")

        # (b) 用 Win 个数据做预测
        model.eval()
        test_inputs = online_dataset.get_batch(idx + UpdateWin, idx + UpdateWin + Win).to(device)
        with torch.no_grad():
            YPred = model(test_inputs).cpu().numpy()  # => [Win, 4]
            Rs_pred = YPred[:, 0]
            Ld_pred = YPred[:, 1]
            Lq_pred = YPred[:, 2]
            Psi_m_pred = YPred[:, 3]

            # 取 Win 段最后一个样本作为“代表”
            final_Rs = float(Rs_pred[-1])
            final_Ld = float(Ld_pred[-1])
            final_Lq = float(Lq_pred[-1])
            final_Psi_m = float(Psi_m_pred[-1])

            # 对应扭矩
            if test_inputs.dim() == 3:
                id_current = test_inputs[:, -1, 0].cpu().numpy()
                iq_current = test_inputs[:, -1, 1].cpu().numpy()
            else:
                id_current = test_inputs[:, 0].cpu().numpy()
                iq_current = test_inputs[:, 1].cpu().numpy()

            Te_pred = 1.5 * P * (Psi_m_pred * iq_current + (Ld_pred - Lq_pred) * id_current * iq_current)
            Te_pred = np.clip(Te_pred, -260, 260)

            predictions.extend(Te_pred)
            Te_actual = Te_data[idx + UpdateWin + sequence_length -1 : idx + UpdateWin + Win + sequence_length -1]
            actuals.extend(Te_actual)

        idx += Win

    # ===== 训练完所有循环后，再用 result_processor 记录并绘图 =====
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    error = actuals - predictions
    mae = np.mean(np.abs(error))
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(error))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error
    }
    # 这里只存最终一次预测的最后一个样本的4个参数
    params = {
        "Rs_pred": final_Rs,
        "Ld_pred": final_Ld,
        "Lq_pred": final_Lq,
        "Psi_m_pred": final_Psi_m
    }

    result_processor = ResultProcessor(output_dir='data')
    data_label = 'online_prediction'

    # 如果没有 profile_id，就弄个简单的索引
    profile_ids = np.arange(len(actuals))

    result_processor.save_results(
        profile_ids, actuals, predictions, data_label, params, metrics
    )
    result_processor.plot_results(actuals, predictions, data_label, metrics)

    print(f"Online results on test data: MAE={mae:.4f}, RMSE={rmse:.4f}, MaxErr={max_error:.4f}")
    print(f"Final motor parameters: Rs={final_Rs:.4f}, Ld={final_Ld:.4f}, Lq={final_Lq:.4f}, Psi_m={final_Psi_m:.4f}")
