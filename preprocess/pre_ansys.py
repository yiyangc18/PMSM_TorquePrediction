import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
file_path = "data/measures_v2.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 统计每个 profile_id 的数量
profile_counts = data['profile_id'].value_counts().sort_index()

# 定义数据集划分
train_range = (2, 41)
val_range = (42, 71)
test_range = (72, 81)

# 绘制柱状图
plt.figure(figsize=(12, 8))
ax = profile_counts.plot(kind='bar', color='lightblue', edgecolor='black', alpha=0.8)

# 获取 x 轴的刻度标签
xticks = ax.get_xticks()
xtick_labels = [int(label.get_text()) for label in ax.get_xticklabels()]

# 确定红线位置
train_start = xticks[xtick_labels.index(train_range[0])]
train_end = xticks[xtick_labels.index(train_range[1])]
val_start = xticks[xtick_labels.index(val_range[0])]
val_end = xticks[xtick_labels.index(val_range[1])]
test_start = xticks[xtick_labels.index(test_range[0])]
test_end = xticks[xtick_labels.index(test_range[1])]

# 添加划分框
plt.axvline(x=train_start - 0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(x=train_end + 0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(x=val_start - 0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(x=val_end + 0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(x=test_start - 0.5, color='red', linestyle='--', linewidth=1)
plt.axvline(x=test_end + 0.5, color='red', linestyle='--', linewidth=1)

# 添加文字标注
plt.text((train_start + train_end) / 2, profile_counts.max() * 1.02, "Training Set",
         color='red', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text((val_start + val_end) / 2, profile_counts.max() * 1.02, "Validation Set",
         color='red', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))
plt.text((test_start + test_end) / 2, profile_counts.max() * 1.02, "Real-time Test Set",
         color='red', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7))

# 图表标题和轴标签
plt.title("Number of Data Points per Profile ID", fontsize=14)
plt.xlabel("Profile ID", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图表
plt.show()


# 总数据量
total_samples = len(data)

# 采样频率 (Hz)
sampling_frequency = 2  # 2Hz

# 计算总时长 (秒)
total_duration_seconds = total_samples / sampling_frequency

# 转换为时、分、秒
hours = int(total_duration_seconds // 3600)
minutes = int((total_duration_seconds % 3600) // 60)
seconds = int(total_duration_seconds % 60)

# 输出结果
print(f"总数据量: {total_samples}")
print(f"采样频率: {sampling_frequency} Hz")
print(f"总时长: {hours} 小时 {minutes} 分钟 {seconds} 秒")

# 提取需要的列
motor_speed = data['motor_speed']
torque = data['torque']

# 创建二维直方图，降低分辨率
heatmap, xedges, yedges = np.histogram2d(motor_speed, torque, bins=[120, 60])  # 调整 bins 值降低分辨率

# 对频率取对数（防止对数溢出，设置一个小的非零值）
heatmap_log = np.log1p(heatmap)  # 使用 log(1 + x) 避免 log(0) 问题

# 绘制热力图
plt.figure(figsize=(12, 8))
img = plt.imshow(
    heatmap_log.T, 
    origin='lower', 
    aspect='auto', 
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
    cmap='viridis',  # 选择颜色映射
    interpolation='nearest'  # 提高可视性
)

# 自定义 colorbar 的刻度标签
cbar = plt.colorbar(img)
log_ticks = [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10 ,11, 12]  # 对应 log(1 + x) 的刻度值
freq_ticks = [int(np.expm1(tick)) for tick in log_ticks]  # 转换为原始频率值
cbar.set_ticks(log_ticks)
cbar.set_ticklabels(freq_ticks)
cbar.set_label('Frequency')

# 图表标题和轴标签
plt.title('Heatmap of Motor Speed vs Torque (Log-scaled)')
plt.xlabel('Motor Speed (rpm)')
plt.ylabel('Torque (Nm)')
plt.grid(alpha=0.3, linestyle='--')

# 设置背景为白色
plt.gca().set_facecolor('white')
plt.tight_layout()

# 显示图表
# plt.show()
plt.savefig("data/figs/dataset_overview.png")