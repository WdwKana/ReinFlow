import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ 配置部分 - 直接复制粘贴文件路径 ============

# ShellGameTouch-v0 三个方法对比
method1_name = "Reinflow-Baseline"
method1_files = [
    "log/mikasa/finetune/ShellGameTouch-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-15_03-32-29_42/training_metrics.csv",
]

method2_name = "Reinflow-Memory (ours)"
method2_files = [
    "log/mikasa/finetune/ShellGameTouch-v0_ft_reflow_membank_ta4_td5_tdf5/2025-10-15_03-31-04_42/training_metrics.csv",
]

method3_name = "Reinflow-Memory Slight Improved (ours)"
method3_files = [
    "log/mikasa/finetune/ShellGameTouch-v0_ft_reflow_mem_learned_vector_ta4_td5_tdf5/2025-10-15_23-22-33_42/training_metrics.csv",
]

# 数据模式: "train" 或 "eval"
mode = "eval"

# 指标选择: "episode" (avg_episode_reward) 或 "success" (success_rate)
metric = "success"

# 任务名称（用于图表标题）
task_name = "ShellGameTouch-v0"

# ============ 数据处理和绘图 ============

# 指标名称映射
metric_mapping = {
    "episode": "avg_episode_reward",
    "success": "success_rate"
}
metric_col = metric_mapping[metric]

def load_and_average(file_paths, mode, metric_col):
    """加载多个种子的数据并求平均"""
    all_data = []
    
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        df_filtered = df[df['mode'] == mode][['total_env_steps', metric_col]]
        all_data.append(df_filtered)
    
    # 找到所有共同的steps
    common_steps = set(all_data[0]['total_env_steps'])
    for df in all_data[1:]:
        common_steps &= set(df['total_env_steps'])
    common_steps = sorted(common_steps)
    
    # 计算每个step的平均值和标准误
    mean_values = []
    sem_values = []
    for step in common_steps:
        values = [df[df['total_env_steps'] == step][metric_col].values[0] 
                  for df in all_data]
        mean_values.append(np.mean(values))
        sem_values.append(np.std(values) / np.sqrt(len(values)))
    
    return common_steps, mean_values, sem_values

# 加载三组数据
print(f"Processing {method1_name}...")
steps1, mean1, sem1 = load_and_average(method1_files, mode, metric_col)

print(f"\nProcessing {method2_name}...")
steps2, mean2, sem2 = load_and_average(method2_files, mode, metric_col)

print(f"\nProcessing {method3_name}...")
steps3, mean3, sem3 = load_and_average(method3_files, mode, metric_col)

# 绘图
plt.figure(figsize=(10, 6))

# 定义三种不同的颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
markers = ['o', 's', '^']  # 圆圈、方块、三角形

# 绘制第一种方法
plt.plot(steps1, mean1, label=method1_name, linewidth=2, 
         color=colors[0], marker=markers[0], markersize=6, markevery=5)
plt.fill_between(steps1, 
                 np.array(mean1) - np.array(sem1), 
                 np.array(mean1) + np.array(sem1), 
                 alpha=0.2, color=colors[0])

# 绘制第二种方法
plt.plot(steps2, mean2, label=method2_name, linewidth=2,
         color=colors[1], marker=markers[1], markersize=6, markevery=5)
plt.fill_between(steps2, 
                 np.array(mean2) - np.array(sem2), 
                 np.array(mean2) + np.array(sem2), 
                 alpha=0.2, color=colors[1])

# 绘制第三种方法
plt.plot(steps3, mean3, label=method3_name, linewidth=2,
         color=colors[2], marker=markers[2], markersize=6, markevery=5)
plt.fill_between(steps3, 
                 np.array(mean3) - np.array(sem3), 
                 np.array(mean3) + np.array(sem3), 
                 alpha=0.2, color=colors[2])

# 图表设置
metric_labels = {
    "episode": "Average Episode Reward",
    "success": "Success Rate"
}
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel(metric_labels[metric], fontsize=12)
plt.title(f'{task_name} - {mode.capitalize()} {metric_labels[metric]}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存
output_file = f"{task_name}_{mode}_{metric}_3methods.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved: {output_file}")
print(f"{method1_name} final: {mean1[-1]:.4f} ± {sem1[-1]:.4f}")
print(f"{method2_name} final: {mean2[-1]:.4f} ± {sem2[-1]:.4f}")
print(f"{method3_name} final: {mean3[-1]:.4f} ± {sem3[-1]:.4f}")

plt.show()