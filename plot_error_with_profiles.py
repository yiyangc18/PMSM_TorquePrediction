import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np

def extract_profile_change_points(test_data, start_idx, end_idx):
    """
    提取测试集中 profile_id 变化的采样点位置和对应的 profile_id，限定在指定范围内。
    """
    profile_ids = test_data['profile_id']
    change_points = profile_ids[profile_ids != profile_ids.shift()].index.tolist()
    change_points_in_range = [point for point in change_points if start_idx <= point <= end_idx]
    profile_ids_in_range = [test_data['profile_id'].iloc[point] for point in change_points_in_range]
    return change_points_in_range, profile_ids_in_range

def plot_zoomed_error_with_profile_changes(results, profile_changes, profile_ids, output_file, start_idx, end_idx):
    """
    绘制指定范围的误差图，标注 profile_id 变化点，并标注工况变化初期最大误差。
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # 固定颜色和透明度
    colors = ['blue', 'red']
    line_styles = ['--', '-.', ':', '--']  # 多种虚线样式
    alpha = 0.7  # 透明度

    # 绘制误差曲线
    for idx, (method, data) in enumerate(results.items()):
        Te_actual = data["Actual Torque"][start_idx:end_idx]
        Te_predicted = data["Predicted Torque"][start_idx:end_idx]
        error = Te_actual - Te_predicted
        color = colors[idx % len(colors)]
        ax.plot(error.index, error, label=f'{method} Error', linewidth=1.5, color=color, alpha=alpha)

        # # 标注 profile_id 变化后的初期最大误差
        # for i, change_point in enumerate(profile_changes):
        #     if start_idx <= change_point <= end_idx:  # 确保变化点在范围内
        #         relative_start = max(change_point, start_idx)
        #         relative_end = min(change_point + 500, end_idx)
        #         segment = error[relative_start:relative_end]  # 取出500点的误差段
        #         if len(segment) > 0:
        #             max_error_idx = segment.abs().idxmax()  # 找到绝对值最大的点索引
        #             max_error_value = segment[max_error_idx]
        #             # 确保标注文字在图像区域内
        #             x_pos = max_error_idx
        #             y_pos = max_error_value * 1.1 if max_error_value > 0 else max_error_value * 0.9
        #             ax.text(
        #                 x_pos, y_pos,
        #                 f'{max_error_value:.2f}',  # 显示最大误差值
        #                 color=color, fontsize=10, fontweight='bold',
        #                 ha='left', va='center', clip_on=False  # 确保文字不被剪裁
        #             )

    # 标注 profile_id 变化点
    for i, (change_point, profile_id) in enumerate(zip(profile_changes, profile_ids)):
        if start_idx <= change_point <= end_idx:  # 确保变化点在范围内
            line_color = f'C{i % 10}'  # 使用 matplotlib 的 10 种颜色
            line_style = line_styles[i % len(line_styles)]
            ax.axvline(x=change_point, color=line_color, linestyle=line_style, linewidth=2.5, alpha=0.4)  # 更明显的竖线
            ax.text(change_point, ax.get_ylim()[1] * 0.8, f'Profile ID: {profile_id}', 
                    rotation=90, verticalalignment='center', fontsize=12, color=line_color)

    # 图像设置
    ax.set_title(f"Torque Estimation Error (Samples {start_idx}~{end_idx}) with Profile Changes")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Error (Nm)")
    ax.legend(loc='lower left')  # 将 Legend 移动到左下角
    ax.grid(True)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Zoomed error plot with profile changes saved to {output_file}")


def main():
    # 数据加载器初始化
    data_file = "data/measures_v3.csv"  # 替换为实际数据文件路径
    loader = DataLoader(data_file)

    # 获取测试数据
    test_data = loader.get_test_data()

    # 定义缩放区间
    start_idx = 13000
    end_idx = 320000

    # 提取 profile_id 变化点和对应的 profile_id
    profile_changes, profile_ids = extract_profile_change_points(test_data, start_idx, end_idx)

    # 文件路径
    files = {
        "LQR Online": "lqr_online_results.csv",
        "PINN Online Prediction": "online_prediction_results.csv"
    }
    results_dir = "data/results"

    # 读取误差数据
    results = {}
    for method, file_name in files.items():
        file_path = f"{results_dir}/{file_name}"
        df = pd.read_csv(file_path)
        results[method] = {
            "Actual Torque": df["Actual Torque"],
            "Predicted Torque": df["Predicted Torque"]
        }

    # 绘制图像
    output_file = f"{results_dir}/error_with_profile_changes.png"
    plot_zoomed_error_with_profile_changes(results, profile_changes, profile_ids, output_file, start_idx, end_idx)

if __name__ == "__main__":
    main()
