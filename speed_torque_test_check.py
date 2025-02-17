import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader

def extract_profile_change_points(test_data):
    """
    提取测试集中 profile_id 变化的采样点位置和对应的 profile_id。
    """
    profile_ids = test_data['profile_id']
    change_points = profile_ids[profile_ids != profile_ids.shift()].index.tolist()
    profile_ids_in_range = [profile_ids.iloc[point] for point in change_points]
    return change_points, profile_ids_in_range

def plot_speed_and_torque_with_profile_changes(test_data, profile_changes, profile_ids, output_file):
    """
    绘制转速和扭矩图，并标注 profile_id 变化点。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 绘制转速曲线
    axes[0].plot(test_data.index, test_data['motor_speed'], label='Motor Speed (RPM)', color='blue', linewidth=1.5)
    axes[0].set_ylabel("Motor Speed (RPM)")
    axes[0].set_title("Motor Speed with Profile Changes")
    axes[0].grid(True)

    # 绘制扭矩曲线
    axes[1].plot(test_data.index, test_data['torque'], label='Torque (Nm)', color='green', linewidth=1.5)
    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_title("Torque with Profile Changes")
    axes[1].set_xlabel("Sample")
    axes[1].grid(True)

    # 标注 profile_id 变化点
    for change_point, profile_id in zip(profile_changes, profile_ids):
        for ax in axes:
            ax.axvline(x=change_point, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
            ax.text(change_point, ax.get_ylim()[1] * 0.9, f'Profile ID: {profile_id}', 
                    rotation=90, verticalalignment='center', fontsize=10, color='red')

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Speed and Torque plot with profile changes saved to {output_file}")

def main():
    # 数据加载器初始化
    data_file = "data/measures_v3.csv"  # 替换为实际数据文件路径
    loader = DataLoader(data_file)

    # 获取测试数据
    test_data = loader.get_test_data()

    # 提取 profile_id 变化点和对应的 profile_id
    profile_changes, profile_ids = extract_profile_change_points(test_data)

    # 绘制图像
    output_file = "data/results/speed_torque_with_profile_changes.png"
    plot_speed_and_torque_with_profile_changes(test_data, profile_changes, profile_ids, output_file)

if __name__ == "__main__":
    main()
