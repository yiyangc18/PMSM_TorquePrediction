import pandas as pd
import matplotlib.pyplot as plt

def plot_zoomed_comparison(results, dataset, output_file, start_idx, end_idx):
    """
    绘制指定范围的缩放图，比较实际扭矩、预测扭矩以及误差
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    # 固定颜色
    colors = ['blue', 'red']
    labels = list(results.keys())

    # 获取实际扭矩（假设两个方法的 Actual Torque 一致）
    any_method = next(iter(results.values()))
    Te_actual = any_method["Actual Torque"][start_idx:end_idx]
    ax[0].plot(Te_actual, label="Actual Torque", linewidth=2, color='black')  # 实际扭矩固定为黑色

    # 上图：预测扭矩
    for idx, (method, data) in enumerate(results.items()):
        Te_predicted = data["Predicted Torque"][start_idx:end_idx]
        color = colors[idx % len(colors)]  # 分配颜色
        ax[0].plot(Te_predicted, linestyle='--', label=f'{method} Predicted', linewidth=1.5, color=color)

    ax[0].set_title(f'{dataset} (Samples {start_idx}-{end_idx}) - Actual vs Predicted Torque')
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Torque (Nm)')
    ax[0].legend(loc='lower left')
    ax[0].grid(True)

    # 下图：预测误差
    for idx, (method, data) in enumerate(results.items()):
        Te_predicted = data["Predicted Torque"][start_idx:end_idx]
        error = Te_actual - Te_predicted
        color = colors[idx % len(colors)]  # 分配颜色
        ax[1].plot(error, label=f'{method} Error', linewidth=1.5, color=color)

    ax[1].set_title(f'{dataset} (Samples {start_idx}-{end_idx}) - Prediction Error')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Error (Nm)')
    ax[1].legend(loc='lower left')
    ax[1].grid(True)

    # 保存图像
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Zoomed comparison plot saved to {output_file}")


def main():
    # 文件路径
    files = {
        "Offline DL Test": "offline_dl_test_results.csv",
        "Online Prediction": "online_prediction_results.csv"
    }
    results_dir = "data/results"

    # 采样范围
    start_idx = 180000
    end_idx = 220000

    # 读取数据
    results = {}
    for method, file_name in files.items():
        file_path = f"{results_dir}/{file_name}"
        df = pd.read_csv(file_path)
        results[method] = {
            "Actual Torque": df["Actual Torque"],
            "Predicted Torque": df["Predicted Torque"]
        }

    # 绘制缩放图
    output_file = f"{results_dir}/zoomed_comparison_{start_idx}_{end_idx}.png"
    plot_zoomed_comparison(results, "Zoomed Comparison", output_file, start_idx, end_idx)

if __name__ == "__main__":
    main()
