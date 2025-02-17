import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(results, dataset, output_file, prediction_alpha=None, error_alpha=None):
    """
    在同一张图中比较不同方法的预测扭矩，以及误差
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    # 固定颜色列表
    colors = ['red','green', 'yellow','brown','gray','orange', 'purple', 'pink', 'cyan']
    actual_alpha = 1  # Actual Torque 的透明度
    prediction_alpha = prediction_alpha or [0.5] * len(results)  # 预测曲线透明度列表
    error_alpha = error_alpha or [0.6] * len(results)  # 误差曲线透明度列表

    # 获取 Actual Torque（假设所有方法的 Actual Torque 一致）
    any_method = next(iter(results.values()))
    Te_actual = any_method["Actual Torque"]
    ax[0].plot(Te_actual, label="Actual Torque", linewidth=2, color='blue', alpha=actual_alpha)  # 单独绘制实际扭矩

    # 上图：预测扭矩
    for idx, (method, data) in enumerate(results.items()):
        Te_predicted = data["Predicted Torque"]
        color = colors[idx % len(colors)]  # 循环分配颜色
        alpha = prediction_alpha[idx % len(prediction_alpha)]  # 从透明度列表取值
        ax[0].plot(Te_predicted, linestyle='--', label=f'{method} Predicted', linewidth=1.5, alpha=alpha, color=color)

    ax[0].set_title(f'{dataset} - Actual vs Predicted Torque')
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Torque (Nm)')
    ax[0].legend(loc='lower left')  # 图例位置统一为左下角
    ax[0].grid(True)

    # 下图：预测误差
    for idx, (method, data) in enumerate(results.items()):
        Te_predicted = data["Predicted Torque"]
        error = Te_actual - Te_predicted
        color = colors[idx % len(colors)]  # 使用相同颜色
        alpha = error_alpha[idx % len(error_alpha)]  # 从透明度列表取值
        ax[1].plot(error, label=f'{method} Error', linewidth=1.5, alpha=alpha, color=color)

    ax[1].set_title(f'{dataset} - Prediction Error')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Error (Nm)')
    ax[1].legend(loc='lower left')  # 图例位置统一为左下角
    ax[1].grid(True)

    # 保存图像
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Comparison plot saved to {output_file}")


def main():
    # 定义数据路径和方法
    methods = {
        # "Validation": [ "offline_dl_validation_results.csv","lqr_offline_validation_results.csv", "pretrain_validation_results.csv"],
        "Test": ["online_prediction_results.csv","pretrain_test_results.csv","lqr_online_results.csv"]
    }
    results_dir = "data/results"

    # 自定义透明度
    prediction_alpha = [0.4, 0.4, 0.4, 0.5, 0.5]  # 对应预测曲线的透明度
    error_alpha = [0.7, 0.4, 0.4, 0.5, 0.5]  # 对应误差曲线的透明度

    # 遍历 Validation 和 Test 数据集
    for dataset, files in methods.items():
        results = {}
        for file in files:
            method_name = file.replace("_results.csv", "").replace("_", " ").title()
            file_path = f"{results_dir}/{file}"

            # 读取数据
            df = pd.read_csv(file_path)
            results[method_name] = {
                "Actual Torque": df["Actual Torque"],
                "Predicted Torque": df["Predicted Torque"]
            }

        # 绘制图像
        output_file = f"{results_dir}/{dataset.lower()}_comparison.png"
        plot_comparison(results, dataset, output_file, prediction_alpha, error_alpha)

if __name__ == "__main__":
    main()
