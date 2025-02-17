import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# 读取数据集
file_path = "data/measures_v2.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 选择需要按 6:4 划分的 profile_id 区间
data_2_20_41_60 = data[((data['profile_id'] >= 2) & (data['profile_id'] <= 20)) |
                       ((data['profile_id'] >= 45) & (data['profile_id'] <= 65))]

# 选择实时测试集的数据 (不在上述区间内的 profile_id)
test_data = data[~(((data['profile_id'] >= 2) & (data['profile_id'] <= 20)) |
                   ((data['profile_id'] >= 45) & (data['profile_id'] <= 65)))]

# 按每个 profile_id 划分训练集和验证集
train_data_list = []
val_data_list = []

for pid, group in data_2_20_41_60.groupby('profile_id'):
    n = len(group)
    split_index = int(0.6 * n)  # 按 6:4 划分
    train_data_list.append(group.iloc[:split_index])
    val_data_list.append(group.iloc[split_index:])

# 合并训练集和验证集
train_data = pd.concat(train_data_list)
val_data = pd.concat(val_data_list)


datasets = {
    "All Data": data,
    "Training Set": train_data,
    "Validation Set": val_data,
    "Real-time Test Set": test_data
}

# 准备柱状图数据
total_counts = data['profile_id'].value_counts().sort_index()
profile_ids = total_counts.index

train_counts = train_data['profile_id'].value_counts().reindex(profile_ids, fill_value=0).sort_index()
val_counts = val_data['profile_id'].value_counts().reindex(profile_ids, fill_value=0).sort_index()
test_counts = test_data['profile_id'].value_counts().reindex(profile_ids, fill_value=0).sort_index()

# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(profile_ids))

# 绘制训练集部分
ax.bar(x, train_counts.values, width=0.8, color='red', label='Training Set')

# 绘制验证集部分
ax.bar(x, val_counts.values, width=0.8, bottom=train_counts.values, color='green', label='Validation Set')

# 绘制测试集部分
train_val_counts = train_counts.values + val_counts.values
ax.bar(x, test_counts.values, width=0.8, bottom=train_val_counts, color='blue', label='Real-time Test Set')

# 设置图表属性并调整字体大小
ax.set_title("Dataset Split by Profile ID", fontsize=18)
ax.set_xlabel("Profile ID", fontsize=16)
ax.set_ylabel("Count", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(profile_ids, rotation=90, fontsize=10)
ax.legend(loc='upper right', fontsize=14)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("data/figs/dataset_split_with_profiles.png")

# 绘制 2x2 热力图
fig2 = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 2)

axs = []
for i in range(2):
    for j in range(2):
        axs.append(fig2.add_subplot(gs[i, j]))

bins = [120, 60]  # 热力图分辨率

# 记录所有热力图，以便共享颜色尺度
heatmaps = []
vmin = None
vmax = None

for ax, (name, subset) in zip(axs, datasets.items()):
    motor_speed = subset['motor_speed']
    torque = subset['torque']
    heatmap, xedges, yedges = np.histogram2d(motor_speed, torque, bins=bins)

    # 对频率取对数
    heatmap_log = np.log1p(heatmap)
    heatmaps.append(heatmap_log)

    # 确定颜色尺度的最小值和最大值
    if vmin is None or heatmap_log.min() < vmin:
        vmin = heatmap_log.min()
    if vmax is None or heatmap_log.max() > vmax:
        vmax = heatmap_log.max()

    # 绘制热力图
    img = ax.imshow(
        heatmap_log.T,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis',
        interpolation='nearest'
    )
    ax.set_title(f"{name} Heatmap", fontsize=16, pad=15)
    ax.set_xlabel("Motor Speed (rpm)", fontsize=14)
    ax.set_ylabel("Torque (Nm)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(alpha=0.3, linestyle='--')

# 调整所有热力图的颜色尺度
for ax in axs:
    img = ax.images[0]
    img.set_clim(vmin, vmax)

# 调整布局以为颜色条腾出空间
fig2.subplots_adjust(right=0.85, hspace=0.3)

# 在靠近主图的位置添加颜色条
cbar_ax = fig2.add_axes([0.87, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]
cbar = fig2.colorbar(img, cax=cbar_ax)
cbar.set_label('Frequency', fontsize=14)

# 设置颜色条的刻度和标签
# 定义需要显示的原始频率值
max_freq = np.max([np.expm1(hm).max() for hm in heatmaps])
freq_values = [0, 1, 10, 100, 1000, 10000, int(max_freq)]
freq_values = [f for f in freq_values if f <= max_freq]  # 过滤超过最大值的频率

# 计算对应的对数值
log_values = np.log1p(freq_values)

# 设置颜色条的刻度和标签
cbar.set_ticks(log_values)
cbar.set_ticklabels([str(int(f)) for f in freq_values])

# plt.show()
plt.savefig("data/figs/dataset_split_hotmap.png")


# # 绘制 Torque 分布图
# fig, ax = plt.subplots(figsize=(16, 8))

# # 按采样点绘制 torque 值
# ax.plot(data.index, data['torque'], label='Torque', color='blue', alpha=0.7)

# # 标注每个 profile_id 的分界
# profile_changes = data['profile_id'].drop_duplicates().index
# for idx in profile_changes:
#     ax.axvline(idx, color='red', linestyle='--', alpha=0.5)

# # 标注 profile_id
# profile_ids = data['profile_id'].drop_duplicates().values
# for idx, pid in zip(profile_changes, profile_ids):
#     ax.text(idx, max(data['torque']) * 0.9, f'P{pid}', color='red', fontsize=8, rotation=90, verticalalignment='bottom')

# # 设置图表属性
# ax.set_title("Torque Distribution Across Profiles", fontsize=18)
# ax.set_xlabel("Sample Index", fontsize=16)
# ax.set_ylabel("Torque (Nm)", fontsize=16)
# ax.legend(fontsize=14)
# ax.grid(alpha=0.3, linestyle='--')
# plt.tight_layout()

# # 保存图像
# plt.savefig("data/figs/torque_distribution_by_profile.png")
# plt.show()
