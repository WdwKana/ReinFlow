import os
import re
import glob
import argparse
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Mikasa shards into a stitched NPZ containing only successful episodes, preserving episode and within-episode order."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train_data_*.npz shards",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output NPZ path (e.g., /path/to/train_success.npz)",
    )
    parser.add_argument(
        "--success_key",
        type=str,
        default="success",
        help="Key in shard files indicating success per step",
    )
    parser.add_argument(
        "--done_key",
        type=str,
        default="done",
        help="Key in shard files indicating terminals per step",
    )
    parser.add_argument(
        "--success_rule",
        type=str,
        choices=["last", "any", "all"],
        default="last",
        help=(
            "Criterion to define a successful episode from the per-step success array: "
            "'last' -> success[-1] == 1; 'any' -> any(success==1); 'all' -> all(success==1)."
        ),
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Optionally limit the number of shards to process (after sorting).",
    )
    return parser.parse_args()


def list_shards(input_dir: str, max_shards: int | None) -> List[str]:
    shard_paths = sorted(
        glob.glob(os.path.join(input_dir, "train_data_*.npz")),
        key=lambda p: int(re.search(r"(\d+)\.npz$", os.path.basename(p)).group(1)),
    )
    if max_shards is not None:
        shard_paths = shard_paths[: max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {input_dir}")
    return shard_paths


def is_success_episode(success_arr: np.ndarray, rule: str) -> bool:
    if success_arr is None:
        return False
    if rule == "last":
        return bool(success_arr[-1] == 1)
    if rule == "any":
        return bool(np.any(success_arr == 1))
    if rule == "all":
        return bool(np.all(success_arr == 1))
    raise ValueError(f"Unknown success rule: {rule}")


def infer_and_validate_shapes(rgb: np.ndarray, joints: np.ndarray, action: np.ndarray) -> Tuple[int, int, int, int, int]:
    if joints.ndim != 2 or action.ndim != 2:
        raise ValueError(f"Unexpected shapes: joints={joints.shape}, action={action.shape}")
    obs_dim = int(joints.shape[1])
    action_dim = int(action.shape[1])

    if rgb.ndim != 4:
        raise ValueError(f"Unexpected rgb ndim {rgb.ndim}, rgb.shape={rgb.shape}")

    # Support either (T, H, W, C) or (T, C, H, W)
    if rgb.shape[-1] in (3, 6):
        T, H, W, C = rgb.shape
    elif rgb.shape[1] in (3, 6):
        T, C, H, W = rgb.shape
    else:
        raise ValueError(f"Cannot infer channels from rgb shape {rgb.shape}")

    return obs_dim, action_dim, C, H, W


def to_tchw(rgb: np.ndarray, expected_C: int | None = None) -> np.ndarray:
    if rgb.ndim != 4:
        raise ValueError(f"rgb must be 4D, got {rgb.shape}")
    if rgb.shape[-1] in (3, 6):
        # (T, H, W, C) -> (T, C, H, W)
        rgb = np.transpose(rgb, (0, 3, 1, 2))
    elif rgb.shape[1] in (3, 6):
        # already (T, C, H, W)
        pass
    else:
        raise ValueError(f"Cannot interpret rgb shape {rgb.shape}")
    if expected_C is not None and rgb.shape[1] != expected_C:
        raise ValueError(f"Channel mismatch: expected C={expected_C}, got {rgb.shape[1]}")
    return rgb


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    shard_paths = list_shards(args.input_dir, args.max_shards)
    print(f"Found {len(shard_paths)} shards in {args.input_dir}")

    # First pass: decide which episodes to keep and compute total sizes
    keep_mask: List[bool] = []
    traj_lengths: List[int] = []

    obs_dim = None
    action_dim = None
    C = None
    H = None
    W = None

    for sp in shard_paths:
        d = np.load(sp, allow_pickle=True)
        rgb = d["rgb"]
        joints = d["joints"]
        action = d["action"]
        success = d.get(args.success_key)

        # shapes
        od, ad, Cc, Hc, Wc = infer_and_validate_shapes(rgb, joints, action)
        if obs_dim is None:
            obs_dim = od
            action_dim = ad
            C, H, W = Cc, Hc, Wc
        else:
            if od != obs_dim or ad != action_dim or Cc != C or Hc != H or Wc != W:
                raise ValueError(
                    f"Inconsistent shapes across shards. Got (obs={od}, act={ad}, C={Cc}, H={Hc}, W={Wc}) in {sp},"
                    f" expected (obs={obs_dim}, act={action_dim}, C={C}, H={H}, W={W})."
                )

        keep = is_success_episode(success, args.success_rule)
        keep_mask.append(keep)
        traj_lengths.append(int(joints.shape[0]) if keep else 0)

    num_kept = int(np.sum(keep_mask))
    if num_kept == 0:
        raise RuntimeError("No successful episodes found with the given rule.")

    total_steps = int(np.sum(traj_lengths))
    print(
        f"Keeping {num_kept}/{len(shard_paths)} episodes. Total steps kept: {total_steps}."
    )

    # Prepare memmaps for streaming write
    tmp_dir = os.path.dirname(args.output_path)
    states_mm_path = os.path.join(tmp_dir, "tmp_states.dat")
    actions_mm_path = os.path.join(tmp_dir, "tmp_actions.dat")
    images_mm_path = os.path.join(tmp_dir, "tmp_images.dat")
    rewards_mm_path = os.path.join(tmp_dir, "tmp_rewards.dat")
    terms_mm_path = os.path.join(tmp_dir, "tmp_terminals.dat")

    states_mm = np.memmap(
        states_mm_path, dtype=np.float32, mode="w+", shape=(total_steps, obs_dim)
    )
    actions_mm = np.memmap(
        actions_mm_path, dtype=np.float32, mode="w+", shape=(total_steps, action_dim)
    )
    images_mm = np.memmap(
        images_mm_path, dtype=np.uint8, mode="w+", shape=(total_steps, C, H, W)
    )
    rewards_mm = np.memmap(
        rewards_mm_path, dtype=np.float32, mode="w+", shape=(total_steps,)
    )
    terms_mm = np.memmap(
        terms_mm_path, dtype=np.float32, mode="w+", shape=(total_steps,)
    )

    # Second pass: fill data in original shard order, skipping unsuccessful episodes
    write_cursor = 0
    kept_traj_lengths: List[int] = []
    for idx, sp in enumerate(shard_paths):
        if not keep_mask[idx]:
            continue
        d = np.load(sp, allow_pickle=True)
        rgb = d["rgb"]
        joints = d["joints"].astype(np.float32)
        action = d["action"].astype(np.float32)
        reward = d.get("reward")
        done = d.get(args.done_key)

        rgb_tchw = to_tchw(rgb, expected_C=C).astype(np.uint8)

        T = int(joints.shape[0])
        sl = slice(write_cursor, write_cursor + T)

        states_mm[sl] = joints
        actions_mm[sl] = action
        images_mm[sl] = rgb_tchw
        rewards_mm[sl] = (reward.astype(np.float32) if reward is not None else 0.0)
        terms_mm[sl] = (done.astype(np.float32) if done is not None else 0.0)

        kept_traj_lengths.append(T)
        write_cursor += T
        if (len(kept_traj_lengths)) % 50 == 0:
            print(f"Merged {len(kept_traj_lengths)} successful episodes so far...")

    # Flush to disk
    states_mm.flush(); actions_mm.flush(); images_mm.flush(); rewards_mm.flush(); terms_mm.flush()

    traj_lengths_np = np.asarray(kept_traj_lengths, dtype=np.int64)
    print(f"Saving compressed NPZ to {args.output_path}")
    np.savez_compressed(
        args.output_path,
        states=states_mm,
        actions=actions_mm,
        images=images_mm,
        traj_lengths=traj_lengths_np,
        rewards=rewards_mm,
        terminals=terms_mm,
    )
    print(f"Saved to {args.output_path}")

    # Cleanup temp files
    for p in [states_mm_path, actions_mm_path, images_mm_path, rewards_mm_path, terms_mm_path]:
        try:
            os.remove(p)
        except Exception as e:
            print("Warning: failed to remove", p, e)


if __name__ == "__main__":
    main()


