import os
import pandas as pd

# 文件路径
results_dir = "data/results"

# 目标汇总表
summary_data = []

# 遍历所有 metrics 文件
for file_name in os.listdir(results_dir):
    if file_name.endswith("_metrics.txt"):
        method_name = file_name.replace("_metrics.txt", "")
        file_path = os.path.join(results_dir, file_name)
        
        # 解析文件内容
        metrics = {'Method': method_name}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ")
                    metrics[key.strip()] = float(value.strip())
        
        # 收集结果
        summary_data.append(metrics)

# 转换为 DataFrame
summary_df = pd.DataFrame(summary_data)

# 对表格按列排序
summary_df = summary_df[['Method', 'MAE', 'RMSE', 'Max Error']]

# 保存结果为 CSV 文件
output_path = os.path.join(results_dir, "performance_summary.csv")
summary_df.to_csv(output_path, index=False)

print(f"Performance metrics summary saved to {output_path}")
