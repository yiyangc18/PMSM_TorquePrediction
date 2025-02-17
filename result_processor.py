import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultProcessor:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.fig_dir = os.path.join(self.output_dir, "figs")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def save_results(self, profile_id, Te_actual, Te_predicted, data_label, params, metrics):
        """
        保存预测结果到文件，包括参数、误差指标和详细预测结果
        """
        # 计算误差
        error = Te_actual - Te_predicted

        # 保存详细预测结果到 CSV
        result_df = pd.DataFrame({
            'Profile ID': profile_id,
            'Actual Torque': Te_actual,
            'Predicted Torque': Te_predicted,
            'Error': error
        })
        result_file = os.path.join(self.result_dir, f"{data_label.lower()}_results.csv")
        result_df.to_csv(result_file, index=False)

        # 保存参数和误差指标到文本文件
        metrics_file = os.path.join(self.result_dir, f"{data_label.lower()}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Identified Parameters:\n")
            for param_name, param_value in params.items():
                f.write(f"{param_name}: {param_value:.4f}\n")
            f.write(f"\nError metrics on {data_label} set:\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")

        print(f"Results saved to {result_file}")
        print(f"Metrics saved to {metrics_file}")

    def plot_results(self, Te_actual, Te_predicted, data_label, metrics):
        """
        根据预测结果绘图并保存到文件，同时在图上显示误差指标
        """
        # 计算误差
        error = Te_actual - Te_predicted

        # 绘图
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # 上图: 实际扭矩 vs 预测扭矩
        ax[0].plot(Te_actual, label='Actual Torque', linewidth=1.5)
        ax[0].plot(Te_predicted, label='Predicted Torque', linestyle='--', linewidth=1.5)
        ax[0].legend()
        ax[0].set_title(f'Actual vs Predicted Torque on {data_label} Set')
        ax[0].set_xlabel('Sample')
        ax[0].set_ylabel('Torque (Nm)')
        ax[0].grid(True)

        # 下图: 预测误差
        ax[1].plot(error, color='red', linewidth=1.5)
        ax[1].set_title('Prediction Error')
        ax[1].set_xlabel('Sample')
        ax[1].set_ylabel('Error (Nm)')
        ax[1].grid(True)

        # 在下图上显示误差指标
        mae = metrics["MAE"]
        rmse = metrics["RMSE"]
        max_error = metrics["Max Error"]
        ax[1].text(
            0.05, 0.95,
            f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMax Error: {max_error:.4f}',
            transform=ax[1].transAxes,
            verticalalignment='top',
            color='red',  # 设置文本为红色
            fontweight='bold'  # 设置文本加粗
        )

        # 保存图像
        fig.tight_layout()
        plot_file = os.path.join(self.fig_dir, f"{data_label.lower()}.png")
        fig.savefig(plot_file)
        plt.close(fig)

        print(f"Plot saved to {plot_file}")
