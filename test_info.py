import numpy as np
import torch
import gymnasium as gym
import mikasa_robo_suite            # 注册 Mikasa 环境
import mani_skill.envs              # 注册 ManiSkill 任务
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# 创建基础 Mikasa 环境（不套任何额外 wrapper）
raw_env = gym.make(
    "RememberShapeAndColor3x2-v0",
    num_envs=1,
    obs_mode="rgb",                 # 按需调整
    control_mode="pd_joint_delta_pos",
    sim_backend="gpu",              # 或 "cpu"
    reward_mode="dense",
)

# ManiSkill 向量化封装，打开 record_metrics 以便拿到 final_info
venv = ManiSkillVectorEnv(
    raw_env,
    num_envs=1,
    ignore_terminations=True,
    record_metrics=True,
)

obs, info = venv.reset()
print(f"reset obs keys: {list(obs.keys())}")

for step in range(10_000):
    action = venv.single_action_space.sample()
    obs, reward, terminated, truncated, info = venv.step(action)

    mask_tensor = info.get("_final_info")
    if mask_tensor is None:
        continue

    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.detach().cpu().numpy().astype(bool)
    else:
        mask = np.asarray(mask_tensor, dtype=bool)

    if not mask.any():
        continue

    episode_info = info["final_info"]["episode"]
    print("episode_info keys:", list(episode_info.keys()))
    print("episode_info values for finished envs:")
    for k, v in episode_info.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        print(f"  {k}: {arr[mask]}")
    break
else:
    print("没有采到任何 episode 结束（可增加步数或调整动作）")

venv.close()