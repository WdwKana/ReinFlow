import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image


def to_tchw(img: np.ndarray) -> np.ndarray:
    """Ensure image is (T, C, H, W). Accepts (T, H, W, C) or (T, C, H, W)."""
    if img.ndim != 4:
        raise ValueError(f"images ndim must be 4, got {img.ndim}")
    if img.shape[-1] in (3, 6):
        # (T, H, W, C) -> (T, C, H, W)
        return np.transpose(img, (0, 3, 1, 2))
    if img.shape[1] in (3, 6):
        return img
    raise ValueError(f"Cannot infer channel dim from shape {img.shape}")


def save_frame_pair(images_tchw: np.ndarray, t: int, outdir: str) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    if not (0 <= t < images_tchw.shape[0]):
        raise IndexError(f"t={t} out of range [0, {images_tchw.shape[0]-1}]")
    c = images_tchw.shape[1]
    if c < 6:
        raise ValueError(f"Expect at least 6 channels (2 cams x RGB). Got C={c}")
    cam0 = images_tchw[t, :3].transpose(1, 2, 0)
    cam1 = images_tchw[t, 3:6].transpose(1, 2, 0)

    cam0_path = os.path.join(outdir, f"cam0_t{t}.png")
    cam1_path = os.path.join(outdir, f"cam1_t{t}.png")
    Image.fromarray(cam0.astype(np.uint8)).save(cam0_path)
    Image.fromarray(cam1.astype(np.uint8)).save(cam1_path)
    return cam0_path, cam1_path


def mean_temporal_diff(x: np.ndarray) -> float:
    """Mean absolute diff across time between consecutive frames. x: (T, C, H, W)."""
    if x.shape[0] < 2:
        return 0.0
    return float(np.abs(x[1:] - x[:-1]).mean())


def main():
    parser = argparse.ArgumentParser(description="Inspect Mikasa cam channels and export sample frames.")
    parser.add_argument(
        "--npz",
        type=str,
        default="/root/ReinFlow/data/mikasa/RememberShapeAndColor3x2-v0/train_chw.npz",
        help="Path to NPZ containing images (T,C,H,W or T,H,W,C)",
    )
    parser.add_argument("--t", type=int, default=0, help="Frame index to export")
    parser.add_argument("--outdir", type=str, default="/root/tmp_cam_check", help="Output directory")
    parser.add_argument(
        "--motion", action="store_true", help="Print simple temporal motion stats per cam"
    )
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    if "images" in d.files:
        images = d["images"]
    elif "rgb" in d.files:
        images = d["rgb"]
    else:
        raise KeyError("No 'images' or 'rgb' in NPZ")

    images = to_tchw(images)
    T, C, H, W = images.shape
    print(f"images: T={T}, C={C}, H={H}, W={W}, dtype={images.dtype}")

    cam0_path, cam1_path = save_frame_pair(images, args.t, args.outdir)
    print(f"Saved: {cam0_path}\nSaved: {cam1_path}")

    if args.motion:
        cam0 = images[:, :3]
        cam1 = images[:, 3:6]
        d0 = mean_temporal_diff(cam0)
        d1 = mean_temporal_diff(cam1)
        print(f"mean_abs_temporal_diff: cam0={d0:.6f}, cam1={d1:.6f}")
        if d0 != d1:
            likely = "cam1 (wrist)" if d1 > d0 else "cam0 (base)"
            print(f"Heuristic guess (more motion -> wrist): likely {likely}")


if __name__ == "__main__":
    main()


