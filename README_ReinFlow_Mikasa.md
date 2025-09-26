# ReinFlow-Mikasa 集成

这个项目将 ReinFlow 算法集成到了 MikasaRobo 环境的 PPO 训练框架中，保持了原 mikasa-example 的简洁性，同时添加了 ReinFlow 的功能。

## 主要特性

- ✅ **兼容接口**: ReinFlowAgent 实现了与原 Agent 相同的接口 (`get_action`, `get_value`, `get_action_and_value`)
- ✅ **动作序列管理**: 自动管理 ReinFlow 的动作序列，每次只返回单步动作
- ✅ **简化 Buffer**: 继续使用原有的 DictArray，避免复杂的 GPU buffer 逻辑
- ✅ **灵活切换**: 通过 `--use_reinflow` 参数轻松在标准 PPO 和 ReinFlow 之间切换
- ✅ **保持训练循环**: 主要训练逻辑保持不变，最小化代码修改

## 文件结构

```
.
├── reinflow-mikasa.py          # 主要的集成文件
├── example_run_reinflow.py     # 使用示例
├── README_ReinFlow_Mikasa.md   # 本文档
└── mikasa-example.py           # 原始 mikasa 示例（参考）
```

## 快速开始

### 1. 环境依赖

确保你已经安装了以下依赖：

```bash
# 基础依赖
pip install torch torchvision gymnasium numpy tqdm colorama tyro

# ManiSkill 和 MikasaRobo
pip install mani-skill
pip install mikasa-robo-suite

# ReinFlow 相关（需要确保 model/ 目录下的文件可用）
```

### 2. 基础使用

**使用标准 PPO（原始行为）：**
```bash
python reinflow-mikasa.py --env_id RememberShapeAndColor3x2-v0 --include_rgb --include_joints
```

**使用 ReinFlow：**
```bash
python reinflow-mikasa.py --use_reinflow --env_id RememberShapeAndColor3x2-v0 --include_rgb --include_joints
```

### 3. 关键参数

#### ReinFlow 特定参数：
- `--use_reinflow`: 启用 ReinFlow agent
- `--horizon_steps`: 动作序列长度（默认: 4）
- `--inference_steps`: 推理去噪步数（默认: 1）
- `--ft_denoising_steps`: 微调去噪步数（默认: 1）
- `--pretrained_model_path`: 预训练模型路径
- `--min_sampling_denoising_std`: 采样时最小标准差（默认: 0.08）
- `--max_logprob_denoising_std`: 计算log概率时最大标准差（默认: 0.14）

#### 环境参数（继承自原版）：
- `--env_id`: 环境名称
- `--include_rgb`: 使用图像输入
- `--include_state`: 使用状态输入
- `--include_joints`: 包含关节信息
- `--num_envs`: 并行环境数量
- `--num_steps`: 每次rollout的步数

## 核心实现

### ReinFlowAgent 类

```python
class ReinFlowAgent(nn.Module):
    def __init__(self, envs, sample_obs, args):
        # 初始化特征提取器（复用 NatureCNN）
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        
        # 初始化 PPOFlow
        self.ppoflow = PPOFlow(...)
        
        # 动作序列管理
        self.current_action_sequence = None
        self.action_step = 0
    
    def get_action(self, x, deterministic=False):
        # 如果序列用完或没有序列，生成新序列
        if self.current_action_sequence is None or self.action_step >= self.horizon_steps:
            self._generate_new_action_sequence(x, deterministic)
            self.action_step = 0
        
        # 返回当前步动作
        action = self.current_action_sequence[:, self.action_step]
        self.action_step += 1
        return action
```

### 关键设计决策

1. **接口兼容性**: ReinFlowAgent 实现了与原 Agent 相同的方法签名
2. **动作序列管理**: 内部维护动作序列状态，外部调用者无感知
3. **简化实现**: 避免复杂的 buffer 和调度逻辑，专注核心功能
4. **渐进式集成**: 可以逐步添加更多 ReinFlow 特性

## 最新更新 - 完整 PPOFlow 集成

### ✅ **已实现的完整功能**：
1. **✅ 完整的 PPOFlow.loss()**: 现在使用原生的 `agent.ppoflow.loss()` 方法
2. **✅ ViT Critic 支持**: 添加了完整的 ViT Critic，与预训练模型兼容
3. **✅ Chains 存储和传递**: 完整的动作链存储和传递机制
4. **✅ 全面的参数控制**: 添加了所有 ReinFlow 特定的参数控制
5. **✅ 预训练模型加载**: 支持预训练模型的加载和使用

### 🔥 **核心改进**：

#### 1. **真正的 PPOFlow.loss() 集成**
```python
# 使用完整的PPOFlow损失函数
pg_loss, entropy_loss, v_loss, bc_loss, \
clipfrac, approx_kl, ratio, \
oldlogprob_min, oldlogprob_max, oldlogprob_std, \
newlogprob_min, newlogprob_max, newlogprob_std, \
noise_std, newQ_values = agent.ppoflow.loss(
    mb_obs, mb_chains, mb_returns, mb_values, mb_advantages, mb_logprobs,
    use_bc_loss=args.use_bc_loss,
    normalize_denoising_horizon=args.normalize_denoising_horizon,
    # ... 所有完整参数
)
```

#### 2. **ViT Critic 自动检测**
```python
def _create_vit_critic(self, args):
    if args.include_rgb:
        # 使用完整的ViT Critic
        return ViTCritic(backbone=VitEncoder(...))
    else:
        # 回退到MLP Critic
        return nn.Sequential(...)
```

#### 3. **完整的参数控制**
- `--use_bc_loss`: 行为克隆损失
- `--normalize_denoising_horizon`: 归一化去噪步数
- `--normalize_act_space_dimension`: 归一化动作空间维度
- `--clip_intermediate_actions`: 裁剪中间动作
- `--account_for_initial_stochasticity`: 考虑初始随机性

### 🚀 **现在这是一个真正的 ReinFlow 实现**

与原始的复杂实现相比，这个版本：
- ✅ **功能完整**: 使用真正的 PPOFlow.loss()
- ✅ **结构清晰**: 保持 mikasa-example 的简洁性  
- ✅ **易于调试**: 清晰的代码流程和错误处理
- ✅ **高度兼容**: 支持 ViT 和 MLP critic 的自动切换

### 剩余的小改进方向：
1. **更好的错误处理**: 增强模型加载失败时的回退机制
2. **性能优化**: 针对大规模训练的内存优化  
3. **更多可视化**: 添加训练过程的可视化工具

## 使用建议

### 开发测试：
```bash
# 小规模测试
python reinflow-mikasa.py --use_reinflow --env_id RememberShapeAndColor3x2-v0 \
  --include_rgb --include_joints --num_envs 4 --num_steps 20 --total_timesteps 5000
```

### 完整训练：
```bash
# 完整训练（需要预训练模型）
python reinflow-mikasa.py --use_reinflow --env_id RememberShapeAndColor3x2-v0 \
  --include_rgb --include_joints --pretrained_model_path /path/to/model.pt \
  --num_envs 50 --num_steps 180 --total_timesteps 50000000
```

## 调试和故障排除

### 常见问题：

1. **模型初始化错误**: 确保 model/ 目录下的文件完整
2. **环境错误**: 确保 MikasaRobo 环境正确安装
3. **CUDA 内存不足**: 减少 `num_envs` 或 `horizon_steps`

### 调试模式：
```bash
# 启用详细日志
python reinflow-mikasa.py --use_reinflow --verbose
```

## 贡献和扩展

这个集成版本的目标是提供一个清晰、简洁的 ReinFlow 测试平台。如果你需要添加更多功能：

1. 在 `ReinFlowAgent` 类中添加新方法
2. 在 `Args` 类中添加新参数
3. 在训练循环中添加新逻辑
4. 保持与原 mikasa-example 的兼容性

## 致谢

基于 ReinFlow 和 MikasaRobo 项目构建，感谢原作者的优秀工作。
