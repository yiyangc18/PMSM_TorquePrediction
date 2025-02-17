import numpy as np
from scipy.signal import savgol_filter
from result_processor import ResultProcessor
from data_loader import DataLoader
import matplotlib.pyplot as plt

# 参数初始化
WINDOW_LENGTH = 64  # 数据窗口长度
FORGET_FACTOR = 1  # 遗忘因子

# 数据加载
file_path = "data/measures_v3.csv"  # 替换为您的数据文件路径
data_loader = DataLoader(file_path)
test_data = data_loader.get_test_data()

# 提取测试集数据
id_test = test_data['i_d'].values
iq_test = test_data['i_q'].values
ud_test = test_data['u_d'].values
uq_test = test_data['u_q'].values
Te_test = test_data['torque'].values
motor_speed_test = test_data['motor_speed'].values
profile_id_test = test_data['profile_id'].values

# 转速转换为电角速度
p = 8  # 电机极对数
omega_test = (2 * np.pi / 60) * motor_speed_test * p

# 初始化参数
initial_theta = np.array([0.0002, 0.0004, 0.4831])  # [Ld, Lq, Psi_m]
P = np.eye(3) * 1e6  # 协方差矩阵

# 阈值设置
current_threshold = 0.5  # 电流阈值
speed_threshold = 10     # 转速阈值 (rpm)
torque_threshold = [-260, 260]  # 扭矩阈值 (N*m)

# 定义结果处理器
processor = ResultProcessor(output_dir="data")

# 使用 Savitzky-Golay 滤波器处理信号
window_length = 51  # 滑动窗口长度
polyorder = 5  # 多项式阶数

id_smooth = savgol_filter(id_test, window_length, polyorder)
iq_smooth = savgol_filter(iq_test, window_length, polyorder)
did_dt = savgol_filter(id_test, window_length, polyorder, deriv=1, delta=0.001)
diq_dt = savgol_filter(iq_test, window_length, polyorder, deriv=1, delta=0.001)

# 预测与参数更新
num_samples = len(id_test)
predicted_torque = np.zeros(num_samples)
updated_theta = initial_theta.copy()

# 汇总所有结果
all_results = []

for start in range(0, num_samples - WINDOW_LENGTH + 1, WINDOW_LENGTH):
    end = start + WINDOW_LENGTH

    # 当前窗口数据
    id_window = id_smooth[start:end]
    iq_window = iq_smooth[start:end]
    ud_window = ud_test[start:end]
    uq_window = uq_test[start:end]
    omega_window = omega_test[start:end]
    did_dt_window = did_dt[start:end]
    diq_dt_window = diq_dt[start:end]
    Te_actual_window = Te_test[start:end]

    for k in range(WINDOW_LENGTH):
        idx = start + k

        # 判断电流和转速是否满足阈值
        current_magnitude = np.sqrt(id_window[k]**2 + iq_window[k]**2)
        if current_magnitude < current_threshold or abs(omega_window[k]) < speed_threshold:
            predicted_torque[idx] = (3/2) * p * (
                updated_theta[2] * iq_window[k] + 
                (updated_theta[0] - updated_theta[1]) * id_window[k] * iq_window[k]
            )
            continue

        # 构建回归矩阵与测量值
        Phi = np.array([
            [did_dt_window[k], -omega_window[k] * iq_window[k], 0],
            [omega_window[k] * id_window[k], diq_dt_window[k], omega_window[k]]
        ])
        y = np.array([
            ud_window[k] - updated_theta[0] * id_window[k],
            uq_window[k] - updated_theta[1] * iq_window[k]
        ])

        # 参数更新
        K = P @ Phi.T @ np.linalg.inv(FORGET_FACTOR * np.eye(2) + Phi @ P @ Phi.T)
        updated_theta += K @ (y - Phi @ updated_theta)
        updated_theta = np.clip(updated_theta, 1e-6, None)
        P = (P - K @ Phi @ P) / FORGET_FACTOR

        # 扭矩预测
        predicted_torque[idx] = (3/2) * p * (
            updated_theta[2] * iq_window[k] + 
            (updated_theta[0] - updated_theta[1]) * id_window[k] * iq_window[k]
        )

        # 阈值检查
        if predicted_torque[idx] < torque_threshold[0]:
            predicted_torque[idx] = torque_threshold[0]
        elif predicted_torque[idx] > torque_threshold[1]:
            predicted_torque[idx] = torque_threshold[1]

    # 记录当前窗口的结果
    error = np.array(predicted_torque[start:end]) - np.array(Te_actual_window)
    metrics = {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "Max Error": np.max(np.abs(error))
    }
    result_df = {
        'Profile ID': profile_id_test[start:end].tolist(),
        'Actual Torque': Te_actual_window.tolist(),
        'Predicted Torque': predicted_torque[start:end].tolist(),
        'Error': error.tolist()
    }
    all_results.append(result_df)

# 将所有结果保存为单个文件
final_results = {
    'Profile ID': [],
    'Actual Torque': [],
    'Predicted Torque': [],
    'Error': []
}
for result in all_results:
    final_results['Profile ID'].extend(result['Profile ID'])
    final_results['Actual Torque'].extend(result['Actual Torque'])
    final_results['Predicted Torque'].extend(result['Predicted Torque'])
    final_results['Error'].extend(result['Error'])


# 处理参数保存为浮点数
final_metrics = {
    "MAE": np.mean(np.abs(predicted_torque - Te_test)),
    "RMSE": np.sqrt(np.mean((predicted_torque - Te_test) ** 2)),
    "Max Error": np.max(np.abs(predicted_torque - Te_test))
}
final_params = {
    "Ld (Inductance, d-axis)": updated_theta[0],  # d轴电感 (Ld)
    "Lq (Inductance, q-axis)": updated_theta[1],  # q轴电感 (Lq)
    "Psi_m (Magnetic flux)": updated_theta[2]   # 磁链 (Psi_m)
}

processor.save_results(
    np.array(final_results['Profile ID']),
    np.array(final_results['Actual Torque']),
    np.array(final_results['Predicted Torque']),
    "lqr_online",
    final_params,
    final_metrics
)

# 绘制整体结果
processor.plot_results(np.array(Te_test), np.array(predicted_torque), "lqr_online", final_metrics)
