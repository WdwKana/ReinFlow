import numpy as np
import os

in_path  = "/root/ReinFlow/data/mikasa/RememberShapeAndColor3x2-v0/train.npz"
out_path = "/root/ReinFlow/data/mikasa/RememberShapeAndColor3x2-v0/train_chw.npz"

d = np.load(in_path, allow_pickle=False)

def pick_images(d):
    if "images" in d.files:
        img = d["images"]
    elif "rgb" in d.files:
        img = d["rgb"]
    else:
        raise KeyError("No 'images' or 'rgb' in npz")
    if img.ndim != 4:
        raise ValueError(f"images ndim must be 4, got {img.ndim}")
    # HWC -> CHW if needed
    if img.shape[-1] in (3, 6):         # (T, H, W, C)
        img = np.transpose(img, (0, 3, 1, 2))
    elif img.shape[1] in (3, 6):        # already (T, C, H, W)
        pass
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {img.shape}")
    return img

images = pick_images(d)

states = d["states"].astype(np.float32)
actions = d["actions"].astype(np.float32)
traj_lengths = d["traj_lengths"].astype(np.int64)

save_dict = dict(states=states, actions=actions, images=images, traj_lengths=traj_lengths)
# 可选字段保留
if "rewards" in d.files:   save_dict["rewards"]   = d["rewards"].astype(np.float32)
if "terminals" in d.files: save_dict["terminals"] = d["terminals"].astype(np.float32)

np.savez_compressed(out_path, **save_dict)
print("Saved:", out_path)
print("images shape:", images.shape)  # (T, C, H, W)