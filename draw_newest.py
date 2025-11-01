import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 配置部分 - 文件路径清单
# ============================================================================

# PPO-MLP 实验路径 (MIKASA-Robo workspace)
PPO_MLP_BASE = "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense"

PPO_MLP_FILES = {
    "InterceptGrabMedium-v0": {
        33: f"{PPO_MLP_BASE}/InterceptGrabMedium-v0/ppo-mlp-dense-intercept-grab-medium-v0__33__rgb_joints__20251026_080157/20251026_080157/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/InterceptGrabMedium-v0/ppo-mlp-dense-intercept-grab-medium-v0__42__rgb_joints__20251026_063403/20251026_063403/training_metrics.csv",
    },
    "RememberColor5-v0": {
        33: f"{PPO_MLP_BASE}/RememberColor5-v0/ppo-mlp-dense-remember-color-5-v0-s33__33__rgb_joints__20251025_145012/20251025_145012/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberColor5-v0/ppo-mlp-dense-remember-color-5-v0-s42__42__rgb_joints__20251025_130501/20251025_130501/training_metrics.csv",
    },
    "RememberColor9-v0": {
        33: f"{PPO_MLP_BASE}/RememberColor9-v0/ppo-mlp-dense-remember-color-9-v0-s33__33__rgb_joints__20251025_184336/20251025_184336/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberColor9-v0/ppo-mlp-dense-remember-color-9-v0-s42__42__rgb_joints__20251025_163412/20251025_163412/training_metrics.csv",
    },
    "RememberShape5-v0": {
        33: f"{PPO_MLP_BASE}/RememberShape5-v0/ppo-mlp-dense-remember-shape-5-v0__33__rgb_joints__20251026_013133/20251026_013133/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShape5-v0/ppo-mlp-dense-remember-shape-5-v0__42__rgb_joints__20251025_233537/20251025_233537/training_metrics.csv",
    },
    "RememberShapeAndColor3x2-v0": {
        33: f"{PPO_MLP_BASE}/RememberShapeAndColor3x2-v0/ppo-mlp-dense-remember-shape-and-color-3x2-v0__33__rgb_joints__20251025_150018/20251025_150018/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShapeAndColor3x2-v0/ppo-mlp-dense-remember-shape-and-color-3x2-v0__42__rgb_joints__20251025_131224/20251025_131224/training_metrics.csv",
    },
    "RememberShapeAndColor3x3-v0": {
        33: f"{PPO_MLP_BASE}/RememberShapeAndColor3x3-v0/ppo-mlp-dense-remember-shape-and-color-3x3-v0__33__rgb_joints__20251025_190032/20251025_190032/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShapeAndColor3x3-v0/ppo-mlp-dense-remember-shape-and-color-3x3-v0__42__rgb_joints__20251025_165820/20251025_165820/training_metrics.csv",
    },
}

# PPO-LSTM 实验路径 (MIKASA-Robo workspace)
PPO_LSTM_FILES = {
    "InterceptGrabMedium-v0": {
        33: f"{PPO_MLP_BASE}/InterceptGrabMedium-v0/ppo-lstm-dense-InterceptGrabMedium-v0__33__rgb_joints__20251026_094517/20251026_094517/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/InterceptGrabMedium-v0/ppo-lstm-dense-InterceptGrabMedium-v0__42__rgb_joints__20251026_073152/20251026_073152/training_metrics.csv",
    },
    "RememberColor5-v0": {
        33: f"{PPO_MLP_BASE}/RememberColor5-v0/ppo-lstm-dense-remember-color-5-v0__33__rgb_joints__20251026_010048/20251026_010048/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberColor5-v0/ppo-lstm-dense-remember-color-5-v0__42__rgb_joints__20251025_231732/20251025_231732/training_metrics.csv",
    },
    "RememberColor9-v0": {
        33: f"{PPO_MLP_BASE}/RememberColor9-v0/ppo-lstm-dense-remember-color-9-v0__33__rgb_joints__20251026_051106/20251026_051106/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberColor9-v0/ppo-lstm-dense-remember-color-9-v0__42__rgb_joints__20251026_024434/20251026_024434/training_metrics.csv",
    },
    "RememberShape5-v0": {
        33: f"{PPO_MLP_BASE}/RememberShape5-v0/ppo-lstm-dense-remember-Shape-5-v0__33__rgb_joints__20251026_052053/20251026_052053/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShape5-v0/ppo-lstm-dense-remember-Shape-5-v0__42__rgb_joints__20251026_031129/20251026_031129/training_metrics.csv",
    },
    "RememberShapeAndColor3x2-v0": {
        33: f"{PPO_MLP_BASE}/RememberShapeAndColor3x2-v0/ppo-lstm-dense-remember-shape-and-color-3x2-v0__33__rgb_joints__20251026_014249/20251026_014249/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShapeAndColor3x2-v0/ppo-lstm-dense-remember-shape-and-color-3x2-v0__42__rgb_joints__20251025_233618/20251025_233618/training_metrics.csv",
    },
    "RememberShapeAndColor3x3-v0": {
        33: f"{PPO_MLP_BASE}/RememberShapeAndColor3x3-v0/ppo-lstm-dense-remember-shape-and-color-3x3-v0__33__rgb_joints__20251026_045833/20251026_045833/training_metrics.csv",
        42: f"{PPO_MLP_BASE}/RememberShapeAndColor3x3-v0/ppo-lstm-dense-remember-shape-and-color-3x3-v0__42__rgb_joints__20251026_024803/20251026_024803/training_metrics.csv",
    },
}

# Reinflow-MLP-IMG 实验路径 (ReinFlow workspace)
# 注意：这里选择的是最新的两次实验 (total_steps ~1.1e7 或 ~1.38e7)
REINFLOW_BASE = "/local/s4176650/ReinFlow/log/mikasa/finetune"

REINFLOW_FILES = {
    "InterceptGrabMedium-v0": {
        33: f"{REINFLOW_BASE}/InterceptGrabMedium-v0_ft_reflow_mlp_img_ta3_td5_tdf5/2025-10-25_10-17-23_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/InterceptGrabMedium-v0_ft_reflow_mlp_img_ta3_td5_tdf5/2025-10-25_02-56-41_42/training_metrics.csv",
    },
    "RememberColor5-v0": {
        33: f"{REINFLOW_BASE}/RememberColor5-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_22-40-00_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/RememberColor5-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_09-32-42_42/training_metrics.csv",
    },
    "RememberColor9-v0": {
        33: f"{REINFLOW_BASE}/RememberColor9-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-25_03-48-58_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/RememberColor9-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_09-32-50_42/training_metrics.csv",
    },
    "RememberShape5-v0": {
        33: f"{REINFLOW_BASE}/RememberShape5-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-25_05-47-25_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/RememberShape5-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_20-49-42_42/training_metrics.csv",
    },
    "RememberShapeAndColor3x2-v0": {
        33: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-25_05-47-06_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_09-32-58_42/training_metrics.csv",
    },
    "RememberShapeAndColor3x3-v0": {
        33: f"{REINFLOW_BASE}/RememberShapeAndColor3x3-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_22-39-09_33/training_metrics.csv",
        42: f"{REINFLOW_BASE}/RememberShapeAndColor3x3-v0_ft_reflow_mlp_img_ta4_td5_tdf5/2025-10-24_09-32-36_42/training_metrics.csv",
    },
}

# Reinflow-MEM (learned matrix) 实验路径 (ReinFlow workspace)
# 选择每个任务最新的 42/33 种子训练记录
REINFLOW_MEM_FILES = {
    "InterceptGrabMedium-v0": {
        42: f"{REINFLOW_BASE}/InterceptGrabMedium-v0_ft_reflow_mem_learned_matrix_ta3_td5_tdf5/2025-10-29_03-34-02_42/training_metrics.csv",
        33: f"{REINFLOW_BASE}/InterceptGrabMedium-v0_ft_reflow_mem_learned_matrix_ta3_td5_tdf5/2025-10-29_09-43-11_33/training_metrics.csv",
    },
    "RememberColor5-v0": {
        42: f"{REINFLOW_BASE}/RememberColor5-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_03-51-50_42/training_metrics.csv",
        33: f"{REINFLOW_BASE}/RememberColor5-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_10-51-12_33/training_metrics.csv",
    },
    "RememberShapeAndColor3x2-v0": {
        42: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mem_learned_matrix_Temp0.01_0.02_tdf5/2025-11-01_03-59-25_42/training_metrics.csv",
        33: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mem_learned_matrix_Temp0.01_0.02_tdf5/2025-11-01_11-08-30_33/training_metrics.csv",
        #42: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_03-32-39_42/training_metrics.csv",
        #33: f"{REINFLOW_BASE}/RememberShapeAndColor3x2-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_10-26-51_33/training_metrics.csv",
    },
    "RememberShapeAndColor3x3-v0": {
        42: f"{REINFLOW_BASE}/RememberShapeAndColor3x3-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_03-33-51_42/training_metrics.csv",
        33: f"{REINFLOW_BASE}/RememberShapeAndColor3x3-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_10-41-57_33/training_metrics.csv",
    },
    "RememberShape5-v0": {
        42: f"{REINFLOW_BASE}/RememberShape5-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-29_20-18-18_42/training_metrics.csv",
        #33: f"{REINFLOW_BASE}/RememberShape5-v0_ft_reflow_mem_learned_matrix_ta4_td5_tdf5/2025-10-30_03-24-05_33/training_metrics.csv",
    },
}

# ============================================================================
# 实验配置 - 可根据需要修改这些参数
# ============================================================================

# 选择要绘制的任务（从上面定义的任务中选择一个）
#TASK_NAME = "RememberColor5-v0"
#TASK_NAME = "RememberColor9-v0"
#TASK_NAME = "RememberShape5-v0"
#TASK_NAME = "RememberShapeAndColor3x3-v0"
TASK_NAME = "RememberShapeAndColor3x2-v0"
#TASK_NAME = "InterceptGrabMedium-v0"
# 数据模式: "train" 或 "eval"
MODE = "eval"

# 指标选择: "episode" (avg_episode_reward/return), "success" (success_rate/success_once), 或 "success_end" (success_at_end)
METRIC = "success"

# ============================================================================
# 数据处理和绘图
# ============================================================================

# 指标名称映射 (PPO vs Reinflow column names)
METRIC_MAPPING = {
    "ppo": {
        "return": "return",
        "success": "success_once",
        "success_end": "success_at_end"
    },
    "reinflow": {
        "return": "avg_episode_reward",
        "success": "success_rate_once",
        "success_end": "success_rate_at_end"
    }
}

def load_and_average(file_dict, mode, metric_col):
    """加载多个种子的数据并求平均"""
    all_data = []
    
    for seed, file_path in file_dict.items():
        print(f"  Loading seed {seed}: {file_path}")
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
print(f"\n{'='*60}")
print(f"Processing Task: {TASK_NAME}")
print(f"Mode: {MODE}, Metric: {METRIC}")
print(f"{'='*60}\n")

print("Loading PPO-MLP data...")
ppo_metric_col = METRIC_MAPPING["ppo"][METRIC]
steps_ppo, mean_ppo, sem_ppo = load_and_average(
    PPO_MLP_FILES[TASK_NAME], MODE, ppo_metric_col
)

print("\nLoading PPO-LSTM data...")
steps_lstm, mean_lstm, sem_lstm = load_and_average(
    PPO_LSTM_FILES[TASK_NAME], MODE, ppo_metric_col
)

print("\nLoading Reinflow-MLP-IMG data...")
reinflow_metric_col = METRIC_MAPPING["reinflow"][METRIC]
steps_reinflow, mean_reinflow, sem_reinflow = load_and_average(
    REINFLOW_FILES[TASK_NAME], MODE, reinflow_metric_col
)

print("\nLoading Reinflow-MEM data...")
steps_mem, mean_mem, sem_mem = load_and_average(
    REINFLOW_MEM_FILES[TASK_NAME], MODE, reinflow_metric_col
)

# 绘图
plt.figure(figsize=(10, 6))

# PPO-MLP
plt.plot(steps_ppo, mean_ppo, label='PPO-MLP', linewidth=2, alpha=0.8)
plt.fill_between(steps_ppo, 
                 np.array(mean_ppo) - np.array(sem_ppo), 
                 np.array(mean_ppo) + np.array(sem_ppo), 
                 alpha=0.2)

# PPO-LSTM
plt.plot(steps_lstm, mean_lstm, label='PPO-LSTM', linewidth=2, alpha=0.8)
plt.fill_between(steps_lstm, 
                 np.array(mean_lstm) - np.array(sem_lstm), 
                 np.array(mean_lstm) + np.array(sem_lstm), 
                 alpha=0.2)

# Reinflow-MLP-IMG
plt.plot(steps_reinflow, mean_reinflow, label='Reinflow', linewidth=2, alpha=0.8)
plt.fill_between(steps_reinflow, 
                 np.array(mean_reinflow) - np.array(sem_reinflow), 
                 np.array(mean_reinflow) + np.array(sem_reinflow), 
                 alpha=0.2)

# Reinflow-MEM (learned matrix)
plt.plot(steps_mem, mean_mem, label='Reinflow-MEM (ours)', linewidth=2, alpha=0.8)
plt.fill_between(steps_mem, 
                 np.array(mean_mem) - np.array(sem_mem), 
                 np.array(mean_mem) + np.array(sem_mem), 
                 alpha=0.2)

# 图表设置
metric_labels = {
    "return": "Average Episode Reward",
    "success": "Success Rate (Once)",
    "success_end": "Success Rate (At End)"
}
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel(metric_labels[METRIC], fontsize=12)
plt.title(f'{TASK_NAME} - {MODE.capitalize()} {metric_labels[METRIC]}', fontsize=14)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存
output_file = f"{TASK_NAME}_{MODE}_{METRIC}_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"\n{'='*60}")
print(f"Plot saved: {output_file}")
print(f"{'='*60}")
print(f"PPO-MLP final:         {mean_ppo[-1]:.4f} ± {sem_ppo[-1]:.4f}")
print(f"PPO-LSTM final:        {mean_lstm[-1]:.4f} ± {sem_lstm[-1]:.4f}")
print(f"Reinflow-MLP-IMG final: {mean_reinflow[-1]:.4f} ± {sem_reinflow[-1]:.4f}")
print(f"Reinflow-MEM final:     {mean_mem[-1]:.4f} ± {sem_mem[-1]:.4f}")
print(f"{'='*60}\n")

plt.show()