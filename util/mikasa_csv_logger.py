# util/mikasa_csv_logger.py
import csv
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

log = logging.getLogger(__name__)

class MikasaMetricsCSVLogger:
    """记录与mikasa-example.py完全相同的评估指标的CSV记录器，包含steps信息"""
    
    def __init__(self, logdir: str, filename: str = "training_metrics.csv"):
        """
        初始化CSV记录器
        
        Args:
            logdir: 日志目录
            filename: CSV文件名
        """
        self.csv_path = os.path.join(logdir, filename)
        self.fieldnames = [
            'iteration',
            'global_step', 
            'env_steps',  # 新增：环境步数
            'train_steps',  # 新增：训练步数
            'mode',  # 'train' or 'eval'
            'success_once_mean',
            'success_once_std',
            'return_mean', 
            'return_std',
            'episode_len_mean',
            'episode_len_std', 
            'reward_mean',
            'reward_std',
            'success_at_end_mean',
            'success_at_end_std',
            'num_episodes',
            'total_env_steps',  # 新增：累积环境步数
            'total_train_steps',  # 新增：累积训练步数
            'timestamp'
        ]
        
        # 检查文件是否存在，不存在则创建并写入表头
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            log.info(f"Created new CSV file: {self.csv_path}")
        else:
            log.info(f"Appending to existing CSV file: {self.csv_path}")
    
    def log_metrics(self, 
                   iteration: int, 
                   global_step: int, 
                   mode: str, 
                   metrics_dict: Dict[str, List], 
                   num_episodes: int,
                   env_steps: int = None,
                   train_steps: int = None,
                   total_env_steps: int = None,
                   total_train_steps: int = None):
        """
        记录指标到CSV文件
        
        Args:
            iteration: 训练迭代次数
            global_step: 全局步数
            mode: 'train' 或 'eval'
            metrics_dict: 包含指标列表的字典，如 {'success_once': [tensor1, tensor2, ...]}
            num_episodes: 完成的episode数量
            env_steps: 当前迭代的环境步数
            train_steps: 当前迭代的训练步数
            total_env_steps: 累积环境步数
            total_train_steps: 累积训练步数
        """
        import torch
        from datetime import datetime
        
        # 计算每个指标的均值和标准差
        row_data = {
            'iteration': iteration,
            'global_step': global_step,
            'env_steps': env_steps if env_steps is not None else 0,
            'train_steps': train_steps if train_steps is not None else 0,
            'mode': mode,
            'num_episodes': num_episodes,
            'total_env_steps': total_env_steps if total_env_steps is not None else 0,
            'total_train_steps': total_train_steps if total_train_steps is not None else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 期望的指标列表
        expected_metrics = ['success_once', 'return', 'episode_len', 'reward', 'success_at_end']
        
        for metric_name in expected_metrics:
            if metric_name in metrics_dict and len(metrics_dict[metric_name]) > 0:
                # 将张量转换为浮点数
                values = []
                for tensor in metrics_dict[metric_name]:
                    if torch.is_tensor(tensor):
                        # 处理可能的多维张量
                        if tensor.numel() == 1:
                            values.append(float(tensor.item()))
                        else:
                            # 如果是多维张量，取平均值
                            values.append(float(tensor.float().mean().item()))
                    else:
                        values.append(float(tensor))
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values) if len(values) > 1 else 0.0
                else:
                    mean_val = 0.0
                    std_val = 0.0
            else:
                mean_val = 0.0
                std_val = 0.0
            
            row_data[f'{metric_name}_mean'] = round(mean_val, 6)
            row_data[f'{metric_name}_std'] = round(std_val, 6)
        
        # 写入CSV文件
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)
        
        log.info(f"Logged {mode} metrics for iteration {iteration} "
                f"(env_steps: {env_steps}, train_steps: {train_steps}): "
                f"success_once={row_data['success_once_mean']:.4f}, "
                f"return={row_data['return_mean']:.4f}, "
                f"episode_len={row_data['episode_len_mean']:.1f}, "
                f"reward={row_data['reward_mean']:.4f}, "
                f"success_at_end={row_data['success_at_end_mean']:.4f}")
    
    def load_metrics(self) -> pd.DataFrame:
        """
        加载CSV文件为DataFrame
        
        Returns:
            pandas.DataFrame: 包含所有记录的指标数据
        """
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        else:
            log.warning(f"CSV file not found: {self.csv_path}")
            return pd.DataFrame()
    
    def get_latest_metrics(self, mode: str = 'eval') -> Dict[str, float]:
        """
        获取最新的指标值
        
        Args:
            mode: 'train' 或 'eval'
            
        Returns:
            Dict: 最新的指标值
        """
        df = self.load_metrics()
        if df.empty:
            return {}
        
        mode_df = df[df['mode'] == mode]
        if mode_df.empty:
            return {}
        
        latest_row = mode_df.iloc[-1]
        return latest_row.to_dict()
    
    def plot_metrics(self, save_path: str = None, show_std: bool = True, x_axis: str = 'iteration'):
        """
        绘制指标变化曲线
        
        Args:
            save_path: 保存图片的路径，如果为None则显示图片
            show_std: 是否显示标准差阴影
            x_axis: X轴选择 ('iteration', 'total_env_steps', 'total_train_steps')
        """
        import matplotlib.pyplot as plt
        
        df = self.load_metrics()
        if df.empty:
            log.warning("No data to plot")
            return
        
        # 分别绘制训练和评估指标
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Metrics Over {x_axis.replace("_", " ").title()}')
        
        metrics = ['success_once', 'return', 'episode_len', 'reward', 'success_at_end']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            for mode in ['train', 'eval']:
                mode_df = df[df['mode'] == mode]
                if not mode_df.empty and x_axis in mode_df.columns:
                    x = mode_df[x_axis]
                    y = mode_df[f'{metric}_mean']
                    ax.plot(x, y, label=f'{mode}', marker='o', markersize=3)
                    
                    if show_std and f'{metric}_std' in mode_df.columns:
                        std = mode_df[f'{metric}_std']
                        ax.fill_between(x, y - std, y + std, alpha=0.2)
            
            ax.set_title(f'{metric}')
            ax.set_xlabel(x_axis.replace('_', ' ').title())
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 删除多余的子图
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Metrics plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_steps_relationship(self, save_path: str = None):
        """
        绘制不同steps之间的关系图
        """
        import matplotlib.pyplot as plt
        
        df = self.load_metrics()
        if df.empty:
            log.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制环境步数与训练步数的关系
        if 'total_env_steps' in df.columns and 'total_train_steps' in df.columns:
            axes[0].scatter(df['total_env_steps'], df['total_train_steps'], alpha=0.6)
            axes[0].set_xlabel('Total Environment Steps')
            axes[0].set_ylabel('Total Training Steps')
            axes[0].set_title('Environment vs Training Steps')
            axes[0].grid(True, alpha=0.3)
        
        # 绘制步数随iteration的变化
        axes[1].plot(df['iteration'], df['total_env_steps'], label='Environment Steps', marker='o')
        axes[1].plot(df['iteration'], df['total_train_steps'], label='Training Steps', marker='s')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Cumulative Steps')
        axes[1].set_title('Steps Over Iterations')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 绘制每个iteration的步数增量
        if len(df) > 1:
            env_step_diff = df['total_env_steps'].diff().fillna(0)
            train_step_diff = df['total_train_steps'].diff().fillna(0)
            axes[2].plot(df['iteration'], env_step_diff, label='Env Steps per Iter', marker='o')
            axes[2].plot(df['iteration'], train_step_diff, label='Train Steps per Iter', marker='s')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Steps per Iteration')
            axes[2].set_title('Step Increments')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Steps relationship plot saved to: {save_path}")
        else:
            plt.show()