import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from result_processor import ResultProcessor

# 示例数据：从你的数据加载器中获取
from data_loader import DataLoader  # 假设已经有这个模块
file_path = "data/measures_v3.csv"  # 替换为实际路径
data_loader = DataLoader(file_path)

train_data = data_loader.get_train_data()
val_data = data_loader.get_val_data()
test_data = data_loader.get_test_data()

# 模型参数辨识
id_train = train_data['i_d'].values
iq_train = train_data['i_q'].values
Te_train = train_data['torque'].values

X_train = np.column_stack((iq_train, id_train * iq_train))
y_train = Te_train

theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
Psi_m = theta[0]
Ld_minus_Lq = theta[1]

# 参数保存
params = {
    "Psi_m": Psi_m,
    "L_d - L_q": Ld_minus_Lq
}

# 定义结果处理器实例
processor = ResultProcessor(output_dir="data")

def evaluate(data, data_label):
    # 提取数据
    id_data = data['i_d'].values
    iq_data = data['i_q'].values
    Te_actual = data['torque'].values
    profile_id = data['profile_id'].values

    # 预测
    X_data = np.column_stack((iq_data, id_data * iq_data))
    Te_predicted = X_data @ theta

    # 误差指标
    mae = mean_absolute_error(Te_actual, Te_predicted)
    mse = mean_squared_error(Te_actual, Te_predicted)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(Te_actual - Te_predicted))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error
    }

    # 保存结果
    processor.save_results(profile_id, Te_actual, Te_predicted, data_label, params, metrics)

    # 绘图
    processor.plot_results(Te_actual, Te_predicted, data_label, metrics)

# 验证集评估
evaluate(val_data, "lqr_offline_Validation")

# 测试集评估
evaluate(test_data, "lqr_offline_Test")
