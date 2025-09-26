import numpy as np, os

file_path = "/root/ReinFlow/data/mikasa/RememberShapeAndColor3x2-v0/train_chw_cam0.npz"
data = np.load(file_path, allow_pickle=False)
# 读取
data = np.load(file_path, allow_pickle=False)
states = data['states']; actions = data['actions']

# 计算归一化参数（但只用于action）
obs_min = states.min(axis=0); obs_max = states.max(axis=0)
action_min = actions.min(axis=0); action_max = actions.max(axis=0)

# 保存 normalization.npz (保留obs_min/max用于兼容性，但实际不用于state归一化)
normalize_path = os.path.join(os.path.dirname(file_path), 'normalization.npz')
np.savez(normalize_path, obs_min=obs_min, obs_max=obs_max,
         action_min=action_min, action_max=action_max)

# 只归一化actions，保持states原始值
states_norm = states  # 不做归一化，直接使用原始值
actions_norm = 2 * (actions - action_min) / (action_max - action_min + 1e-6) - 1

print(f"States processing: 原始范围 [{states.min():.3f}, {states.max():.3f}] -> 保持不变")
print(f"Actions processing: 原始范围 [{actions.min():.3f}, {actions.max():.3f}] -> 归一化到 [{actions_norm.min():.3f}, {actions_norm.max():.3f}]")

# 处理图像键与通道顺序
def get_images(d):
    if 'images' in d.files:
        img = d['images']
    elif 'rgb' in d.files:
        img = d['rgb']
    else:
        return None
    if img.ndim != 4:
        raise ValueError(f'images ndim must be 4, got {img.ndim}')
    # HWC -> CHW
    if img.shape[-1] in (3, 6):
        img = np.transpose(img, (0, 3, 1, 2))
    elif img.shape[1] in (3, 6):
        pass
    else:
        raise ValueError(f'Cannot infer channel dim from {img.shape}')
    return img.astype(np.uint8)

images = get_images(data)

# 保存到不覆盖的新文件
out_path = os.path.join(os.path.dirname(file_path), 'train_norm_cam0.npz')
save_dict = dict(
    states=states_norm.astype(np.float32),  # 现在是原始值，不是归一化值
    actions=actions_norm.astype(np.float32),  # 仍然是归一化值
    traj_lengths=data['traj_lengths'].astype(np.int64),
)
if 'rewards' in data.files:   save_dict['rewards'] = data['rewards'].astype(np.float32)
if 'terminals' in data.files: save_dict['terminals'] = data['terminals'].astype(np.float32)
if images is not None:        save_dict['images'] = images

np.savez_compressed(out_path, **save_dict)
print('Saved:', out_path)