# model.py

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=1, hidden_units=64):
        super(FeedForwardNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nonlinearity='tanh', dropout=0.0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, input_dim, sequence_length, num_filters=64, kernel_size=3, dropout=0.0):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
            input_dim, num_filters,
            kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * sequence_length, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch_size, input_dim, sequence_length)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        from torch.nn.utils import weight_norm
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=1, padding=(kernel_size - 1) * dilation_size,
                dilation=dilation_size
            )
            layers += [
                weight_norm(conv),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(out_channels, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch_size, input_dim, sequence_length)
        out = self.network(x)
        out = out[:, :, -1]  # Take the output from the last time step
        out = self.fc(out)
        return out

class FNNOnlineModel(nn.Module):
    def __init__(self, input_dim=7, hidden_layers=2, hidden_units=64):
        super(FNNOnlineModel, self).__init__()
        layers = []
        # 第1层: input_dim -> hidden_units
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())

        # 后续 hidden_layers - 1 层，每次减半
        prev_hidden = hidden_units
        for _ in range(hidden_layers - 1):
            current_hidden = prev_hidden // 2
            layers.append(nn.Linear(prev_hidden, current_hidden))
            layers.append(nn.ReLU())
            prev_hidden = current_hidden

        # 最后一层 -> 输出4
        layers.append(nn.Linear(prev_hidden, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 如果输入是三维 [batch_size, seq_len, feature_dim]，在这里 Flatten。
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # => [batch_size, seq_len*feature_dim]
        out = self.network(x)         # => [batch_size, 4] (或 [batch_size, 1, 4] 若中途有别的操作)

        return out




class LSTMOnlineModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super(LSTMOnlineModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 4)  # 输出 Rs, Ld, Lq, Psi_m
        
    def forward(self, x):
        # x 的形状：(batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out
    
class FNNOfflineOnlineModel(nn.Module):
    def __init__(self, input_dim=7, hidden_layers=2, hidden_units=64, dropout=0.5):
        super(FNNOfflineOnlineModel, self).__init__()
        self.input_filter = nn.Linear(input_dim, hidden_units)
        
        # 创建共享的隐藏层
        self.shared_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.shared_layers.append(nn.Sequential(
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # 创建四个独立的输出塔
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_units, hidden_units // 2),
                nn.BatchNorm1d(hidden_units // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units // 2, 1)
            ) for _ in range(4)
        ])
    
    def forward(self, x):
        # 调整输入形状为 [batch_size, num_features]
        x = x.view(-1, x.size(-1))  # 展平序列维度
        
        x = self.input_filter(x)
        for layer in self.shared_layers:
            x = layer(x)
        
        # 通过四个塔分别预测 Rs, Ld, Lq, Psi_m
        outputs = [tower(x) for tower in self.towers]
        outputs = torch.cat(outputs, dim=1)  # 形状: [batch_size, 4]
        return outputs
