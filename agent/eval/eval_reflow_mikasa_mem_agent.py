# agent/eval/eval_reflow_mikasa_mem_agent.py
import copy
import logging
import math
import os

import cv2
import hydra
import numpy as np
import torch
from tqdm import tqdm

from agent.eval.eval_reflow_mikasa_agent import EvalReFlowMikasaAgent
from util.timer import Timer


log = logging.getLogger(__name__)

class EvalReFlowMikasaMemAgent(EvalReFlowMikasaAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.memory_cfg = cfg.get("memory", None)
        #self.memory = None


        if self.memory_cfg is None:
            log.warning("No memory configuration found; proceeding without an external memory module.")
        else:
            instantiate_kwargs = {"device": self.device}
            target = self.memory_cfg.get("_target_", "")
            needs_num_slots = "PatchWorkingMemory" in target or "num_slots" in self.memory_cfg

            if needs_num_slots:
                num_slots = self.memory_cfg.get("num_slots", None)
                if num_slots is None:
                    num_slots = getattr(self.model.network.backbone, "num_patch", None)
                if num_slots is not None:
                    instantiate_kwargs["num_slots"] = num_slots

            self.memory = hydra.utils.instantiate(
                self.memory_cfg,
                **instantiate_kwargs,
            )
            self.memory = self.memory.to(self.device)


            if hasattr(self.memory, "eval"):
                self.memory.eval()

        self.record_memory_heatmap = cfg.get("record_memory_heatmap", False)
        self.memory_record_env_index = int(cfg.get("memory_record_env_index", 0))
        self.memory_heatmap_max_steps = cfg.get("memory_heatmap_max_steps", None)
        if self.memory_heatmap_max_steps is not None:
            try:
                self.memory_heatmap_max_steps = int(self.memory_heatmap_max_steps)
            except (TypeError, ValueError):
                self.memory_heatmap_max_steps = None

        self.memory_heatmap_dir = None
        self._memory_debug_entries = []
        self._memory_debug_active = False
        self._memory_debug_hook_registered = False
        self._memory_episode_finished = False
        self._memory_debug_env_index = 0
        self._current_rgb_frame = None

    @torch.no_grad()
    def _reset_memory_all(self):
        if self.memory is None:
            return
        self.memory.reset(B=self.n_envs, device=self.device)

    @torch.no_grad()
    def _clear_memory(self, mask):
        if self.memory is None:
            return
        if mask is None:
            return
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(bool))
        storage = self._memory_storage()
        if storage.numel() == 0:
            return
        mask = mask.to(storage.device)
        if mask.any():
            storage[mask] = 0.0

    def _memory_storage(self) -> torch.Tensor:
        if self.memory is None:
            raise AttributeError("Memory storage requested but no memory module is instantiated.")
        if hasattr(self.memory, "memory"):
            return self.memory.memory
        if hasattr(self.memory, "m"):
            return self.memory.m
        raise AttributeError("Memory module must expose a tensor buffer named 'memory' or 'm'.")

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        obs = super().reset_env_all(verbose=verbose, options_venv=options_venv, **kwargs)
        self._reset_memory_all()
        return obs

    def infer(self, cond, num_denoising_steps):
        cond_with_mem = dict(cond)
        if self.memory is not None:
            cond_with_mem["wm"] = self.memory
        timer = Timer()
        samples = self.model.sample(
            cond=cond_with_mem,
            inference_steps=num_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.clip_intermediate_actions,
        )
        duration = timer()
        return samples, duration

    def single_run(self, num_denoising_steps, options_venv):
        self.create_video_recorder(num_denoising_steps)
        self._init_memory_debugging(num_denoising_steps)
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        single_step_duration_list = np.zeros(self.n_steps)
        success_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        success_once_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        success_at_end_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        episode_return_trajs = np.zeros((self.n_steps, self.n_envs))
        episode_length_trajs = np.zeros((self.n_steps, self.n_envs))
        episode_finished_mask_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)

        log.info(
            f"Evaluating {self.model.__class__.__name__} model in {self.env_name} environment with {num_denoising_steps} step(s)."
        )

        for step in tqdm(range(self.n_steps), dynamic_ncols=True, desc=f"{num_denoising_steps} step(s):"):
            if self._memory_debug_active and not self._memory_episode_finished:
                self._current_rgb_frame = self._extract_memory_rgb(prev_obs_venv)
            else:
                self._current_rgb_frame = None
            with torch.no_grad():
                if hasattr(self, "obs_dims"):
                    cond = self._obs_to_device(prev_obs_venv)
                else:
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                samples, single_step_duration = self.infer(cond, num_denoising_steps)

                single_step_duration_list[step] = single_step_duration

                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]

            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                action_venv
            )

            if self.is_mikasa:
                obs_venv = self._obs_to_numpy(obs_venv)
                reward_venv = self._ensure_numpy_array(reward_venv, dtype=np.float32)
                terminated_venv = self._ensure_numpy_array(terminated_venv, dtype=bool)
                truncated_venv = self._ensure_numpy_array(truncated_venv, dtype=bool)
                done_mask = terminated_venv | truncated_venv
                info_dict = info_venv if isinstance(info_venv, dict) else {}
                if step < 5:
                    debug_keys = list(info_dict.keys())
                    debug_mask = self._ensure_numpy_array(info_dict.get("_final_info"), dtype=bool)
                    debug_mask_sum = int(debug_mask.sum()) if debug_mask is not None else 0
                    debug_final_ep = info_dict.get("final_info", {})
                    debug_ep_keys = (
                        list(debug_final_ep.get("episode", {}).keys())
                        if isinstance(debug_final_ep, dict)
                        else []
                    )
                    print(
                        f"[EvalDebug] step={step} keys={debug_keys} mask_sum={debug_mask_sum} episode_keys={debug_ep_keys}"
                    )
                success_once = self._ensure_numpy_array(info_dict.get("success_once"), dtype=bool)
                if success_once is None:
                    success_once = np.zeros(self.n_envs, dtype=bool)
                success_at_end = self._ensure_numpy_array(info_dict.get("success_at_end"), dtype=bool)
                if success_at_end is None:
                    success_at_end = np.zeros(self.n_envs, dtype=bool)
                success_trajs[step] = success_once | success_at_end
                success_once_trajs[step] = success_once
                success_at_end_trajs[step] = success_at_end

                mask = self._ensure_numpy_array(info_dict.get("_final_info"), dtype=bool)
                if mask is None:
                    mask = np.zeros(self.n_envs, dtype=bool)
                episode_finished_mask_trajs[step] = mask

                episode_info = info_dict.get("final_info", {}).get("episode", {}) if mask.any() else {}
                episode_returns = self._ensure_numpy_array(episode_info.get("return"), dtype=np.float32)
                if episode_returns is None:
                    episode_returns = reward_venv.copy()
                episode_lengths = self._ensure_numpy_array(episode_info.get("episode_len"), dtype=np.int32)
                if episode_lengths is None:
                    episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
                episode_return_trajs[step] = episode_returns
                episode_length_trajs[step] = episode_lengths
            else:
                info_list = info_venv if isinstance(info_venv, list) else []
                success_venv = []
                for i, info in enumerate(info_list):
                    if "success" in info:
                        success_val = info["success"]
                        if hasattr(success_val, "item"):
                            if hasattr(success_val, "numel") and success_val.numel() > 1:
                                success_venv.append(bool(success_val[i].item()))
                            else:
                                success_venv.append(bool(success_val.item()))
                        else:
                            success_venv.append(bool(success_val))
                    else:
                        success_venv.append(False)
                success_arr = np.array(success_venv, dtype=bool)
                success_trajs[step] = success_arr
                success_once_trajs[step] = success_arr
                success_at_end_trajs[step] = success_arr
                terminated_venv = np.asarray(terminated_venv, dtype=bool)
                truncated_venv = np.asarray(truncated_venv, dtype=bool)
                done_mask = terminated_venv | truncated_venv

            self._clear_memory(done_mask)

            if self._memory_debug_active and not self._memory_episode_finished:
                env_idx = self._memory_debug_env_index
                if 0 <= env_idx < done_mask.shape[0] and bool(done_mask[env_idx]):
                    self._memory_episode_finished = True
                    self._deactivate_memory_debug_hook()

            if self.render_onscreen:
                self.venv.render(mode="human")
            if self.record_video:
                if self.is_mikasa and not self.mikasa_use_custom_video:
                    continue
                else:
                    if "kitchen" in self.env_name.lower():
                        raise ValueError(
                            f"Cannot record video for kitchen environments with the current setup. self.env_name={self.env_name}"
                        )

                    if self.is_mikasa:
                        frame_batch = self.venv.render()
                    else:
                        frame_batch = self.venv.render(
                            mode="rgb_array",
                            height=self.frame_height,
                            width=self.frame_width,
                        )

                    frame = frame_batch[self.record_env_index]
                    if frame is None or frame == []:
                        raise ValueError(
                            f"frame is {frame} (empty), check your environment rendering settings."
                        )
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu().numpy()
                    elif not isinstance(frame, np.ndarray):
                        frame = np.asarray(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    cv2.putText(
                        frame,
                        self.video_title,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    self.video_writer.write(frame)

            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_mask
            prev_obs_venv = obs_venv

        self._current_rgb_frame = None

        if self.record_video and self.video_writer is not None:
            self.video_writer.release()
            self.all_video_paths.append(self.video_path)
            print(f"Video saved to {self.video_path}")

        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episodes_finished = len(reward_trajs_split)
            episode_reward = np.array([
                np.sum(reward_traj) for reward_traj in reward_trajs_split
            ])
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)

            episode_success = []
            episode_success_once = []
            episode_success_at_end = []
            for env_ind, start, end in episodes_start_end:
                episode_success.append(
                    np.any(success_trajs[start : end + 1, env_ind])
                )
                episode_success_once.append(
                    np.any(success_once_trajs[start : end + 1, env_ind])
                )
                episode_success_at_end.append(
                    np.any(success_at_end_trajs[start : end + 1, env_ind])
                )
            episode_success = np.array(episode_success, dtype=float)
            episode_success_once = np.array(episode_success_once, dtype=float)
            episode_success_at_end = np.array(episode_success_at_end, dtype=float)

            success_rate = float(episode_success.mean())
            success_rate_std = float(episode_success.std())
            success_rate_once = float(episode_success_once.mean())
            success_rate_once_std = float(episode_success_once.std())
            success_rate_at_end = float(episode_success_at_end.mean())
            success_rate_at_end_std = float(episode_success_at_end.std())
        else:
            episode_reward = np.array([])
            num_episodes_finished = 0
            avg_episode_reward = 0
            avg_episode_reward_std = 0
            avg_best_reward = 0
            avg_best_reward_std = 0
            success_rate = 0
            success_rate_std = 0
            success_rate_once = 0
            success_rate_once_std = 0
            success_rate_at_end = 0
            success_rate_at_end_std = 0
            log.info("[WARNING] No episode completed within the iteration!")

        episode_lengths = np.array([
            (end - start + 1) * self.act_steps for _, start, end in episodes_start_end
        ])
        avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        avg_traj_length_std = (
            np.std(episode_lengths) if len(episode_lengths) > 0 else 0
        )

        avg_single_step_duration = single_step_duration_list.mean()
        avg_single_step_duration_std = single_step_duration_list.std()
        single_step_frequency_list = np.divide(
            1.0,
            np.clip(single_step_duration_list, a_min=1e-8, a_max=None),
        )
        avg_single_step_freq = single_step_frequency_list.mean()
        avg_single_step_freq_std = single_step_frequency_list.std()

        BOLDSTART = "\033[1m"
        BOLDEND = "\033[0m"
        log.info(
            f"""
            #############################################################
            {BOLDSTART}Evaluation{BOLDEND}
            Model:                    {self.model.__class__.__name__:>30}
            Environment:              {self.env_name + ' x ' + str(self.n_envs):>30}
            denoising steps:          {num_denoising_steps:>30}

            success_rate:             {success_rate*100:>8.3f} % ± {success_rate_std*100:>8.3f} %
            avg_episode_reward:       {avg_episode_reward:>8.1f} ± {avg_episode_reward_std:>2.1f}


            avg_single_step_freq:     {avg_single_step_freq:>3.1f} ± {avg_single_step_freq_std:>3.1f} HZ

            avg_traj_length:          {avg_traj_length:>3.1f} ± {avg_traj_length_std:>3.1f} steps
            avg_best_reward:          {avg_best_reward:>8.1f} ± {avg_best_reward_std:>2.1f}
            num_episode:              {num_episodes_finished:>4d}
            #############################################################
            """
        )
        result = (
            num_denoising_steps,
            avg_single_step_freq,
            avg_single_step_freq_std,
            avg_single_step_duration,
            avg_single_step_duration_std,
            avg_traj_length,
            avg_traj_length_std,
            avg_episode_reward,
            avg_episode_reward_std,
            avg_best_reward,
            avg_best_reward_std,
            num_episodes_finished,
            success_rate,
            success_rate_std,
        )

        self._teardown_memory_debugging(num_denoising_steps)

        return result

    def _init_memory_debugging(self, num_denoising_steps: int) -> None:
        self._memory_debug_entries = []
        self._memory_episode_finished = False
        self._memory_debug_hook_registered = False
        self._memory_debug_env_index = int(np.clip(self.memory_record_env_index, 0, max(self.n_envs - 1, 0)))
        self._memory_debug_active = (
            bool(self.record_memory_heatmap)
            and self.memory is not None
            and hasattr(self.memory, "register_debug_hook")
            and hasattr(self.memory, "clear_debug_hook")
        )
        log.info(f"heatmap_on={self.record_memory_heatmap}, mem={type(self.memory)}, can_hook={hasattr(self.memory,'register_debug_hook') and hasattr(self.memory,'clear_debug_hook')}")

        if not self._memory_debug_active:
            return

        if self.eval_log_dir is None:
            self.memory_heatmap_dir = None
            self._memory_debug_active = False
            return

        self.memory_heatmap_dir = os.path.join(self.eval_log_dir, "memory_heatmap")
        os.makedirs(self.memory_heatmap_dir, exist_ok=True)

        self.memory.register_debug_hook(self._memory_debug_callback)
        self._memory_debug_hook_registered = True

    def _teardown_memory_debugging(self, num_denoising_steps: int) -> None:
        if self._memory_debug_hook_registered and hasattr(self.memory, "clear_debug_hook"):
            self.memory.clear_debug_hook()
        self._memory_debug_hook_registered = False
        self._memory_debug_active = False

        if not self._memory_debug_entries or not self.memory_heatmap_dir:
            self._memory_debug_entries = []
            return

        episode_dir = os.path.join(
            self.memory_heatmap_dir,
            f"denoise_{int(num_denoising_steps):03d}"
        )
        os.makedirs(episode_dir, exist_ok=True)

        for step_idx, entry in enumerate(self._memory_debug_entries):
            self._save_memory_heatmap(entry, step_idx, episode_dir)

        self._memory_debug_entries = []

    def _memory_debug_callback(self, payload: dict) -> None:
        if not self._memory_debug_active or self._memory_episode_finished:
            return

        if (
            self.memory_heatmap_max_steps is not None
            and len(self._memory_debug_entries) >= self.memory_heatmap_max_steps
        ):
            self._memory_episode_finished = True
            self._deactivate_memory_debug_hook()
            return

        weights = payload.get("weights")
        read_out = payload.get("read_out")
        #log.info(f"wm_hook_step={len(self._memory_debug_entries)}")

        if not isinstance(weights, torch.Tensor) or not isinstance(read_out, torch.Tensor):
            return

        if weights.dim() != 3 or read_out.dim() != 3:
            return

        env_idx = self._memory_debug_env_index
        if env_idx >= weights.shape[0] or env_idx >= read_out.shape[0]:
            return

        try:
            weights_np = weights.detach().cpu().numpy()
            w_env = weights_np[env_idx]
            if w_env.ndim == 2 and w_env.shape[1] > 0:
                max_vals = w_env.max(axis=1)
                mean_vals = w_env.mean(axis=1)
                log.info(
                    "[WMStats] step=%d max_mean=%.4f max_std=%.4f mean_mean=%.4f mean_std=%.4f",
                    len(self._memory_debug_entries),
                    float(max_vals.mean()),
                    float(max_vals.std()),
                    float(mean_vals.mean()),
                    float(mean_vals.std()),
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("Failed to compute WMStats: %s", exc)

        entry = {
            "weights": self._tensor_to_numpy(weights[env_idx]),
            "read_out": self._tensor_to_numpy(read_out[env_idx]),
            "rgb": self._current_rgb_frame.copy() if self._current_rgb_frame is not None else None,
            "step_index": len(self._memory_debug_entries),
        }

        memory_tensor = payload.get("memory")
        if isinstance(memory_tensor, torch.Tensor) and memory_tensor.dim() >= 2 and env_idx < memory_tensor.shape[0]:
            entry["memory"] = self._tensor_to_numpy(memory_tensor[env_idx])

        write_weights = payload.get("write_weights")
        if isinstance(write_weights, torch.Tensor) and write_weights.dim() == 3 and env_idx < write_weights.shape[0]:
            entry["write_weights"] = self._tensor_to_numpy(write_weights[env_idx])

        erase_tensor = payload.get("erase")
        if isinstance(erase_tensor, torch.Tensor) and erase_tensor.dim() >= 2 and env_idx < erase_tensor.shape[0]:
            entry["erase"] = self._tensor_to_numpy(erase_tensor[env_idx])

        self._memory_debug_entries.append(entry)

    def _deactivate_memory_debug_hook(self) -> None:
        if self._memory_debug_hook_registered and hasattr(self.memory, "clear_debug_hook"):
            self.memory.clear_debug_hook()
        self._memory_debug_hook_registered = False
        self._memory_debug_active = False

    @staticmethod
    def _tensor_to_numpy(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.to(dtype=torch.float32)
            return tensor.numpy()
        return np.asarray(tensor)

    def _extract_memory_rgb(self, obs_dict):
        if obs_dict is None or "rgb" not in obs_dict:
            return None

        rgb = obs_dict["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        elif not isinstance(rgb, np.ndarray):
            rgb = np.asarray(rgb)

        if rgb.ndim < 3:
            return None

        env_idx = int(np.clip(self.memory_record_env_index, 0, rgb.shape[0] - 1))
        frame = rgb[env_idx]

        if frame.ndim == 4:
            frame = frame[-1]

        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            return None

        frame = frame.astype(np.float32)
        max_val = float(frame.max()) if frame.size > 0 else 0.0
        if max_val <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _infer_patch_grid(num_patches: int) -> tuple[int, int]:
        if num_patches <= 0:
            return 0, 0
        height = int(math.sqrt(num_patches))
        height = max(height, 1)
        while height > 1 and num_patches % height != 0:
            height -= 1
        width = num_patches // height if height > 0 else num_patches
        width = max(width, 1)
        return height, width

    def _save_memory_heatmap(self, entry: dict, step_idx: int, episode_dir: str) -> None:
        read_out = entry.get("read_out")
        if read_out is None:
            return

        read_out = np.asarray(read_out)
        if read_out.ndim != 2:
            return

        num_patches = read_out.shape[0]
        height, width = self._infer_patch_grid(num_patches)
        if height == 0 or width == 0:
            return

        rgb = entry.get("rgb")
        #rgb = entry.get("rgb")
        frame_bgr = None
        target_hw = None
        if rgb is not None:
            target_hw = (rgb.shape[1], rgb.shape[0])
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def render_map(map_2d: np.ndarray, tag: str) -> None:
            if map_2d is None:
                return
            # 百分位归一化（更稳），也可改回 min-max
            lo, hi = np.percentile(map_2d, [2, 98])
            denom = max(hi - lo, 1e-8)
            norm = np.clip((map_2d - lo) / denom, 0, 1)
            map_uint8 = (norm * 255.0).astype(np.uint8)

            # 调整到与原图同尺寸（若无原图则放大显示）
            if frame_bgr is not None and target_hw is not None:
                resized = cv2.resize(map_uint8, target_hw, interpolation=cv2.INTER_LINEAR)
            else:
                scale = max(height, width) * 16
                resized = cv2.resize(map_uint8, (scale, scale), interpolation=cv2.INTER_NEAREST)

            # 纯热图（不叠加）
            color_only = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            heat_path = os.path.join(episode_dir, f"step_{step_idx:04d}_{tag}_heat.png")
            cv2.imwrite(heat_path, color_only)

            # 叠加到原图（若有原图）
            if frame_bgr is not None:
                overlay = cv2.addWeighted(frame_bgr, 0.6, color_only, 0.4, 0)
                overlay_path = os.path.join(episode_dir, f"step_{step_idx:04d}_{tag}.png")
                cv2.imwrite(overlay_path, overlay)

        # 读出强度 |read_out|
        read_map = np.linalg.norm(read_out, axis=-1).reshape(height, width)
        render_map(read_map, "read")

        # 读注意力（更显著：尖锐度 peak）
        weights = entry.get("weights")
        if weights is not None:
            w = np.asarray(weights)                  # [P,S] or [P, S]
            if w.ndim == 2 and w.shape[0] == num_patches:
                S = w.shape[1]
                peak = (w.max(axis=-1) - 1.0 / S) / max(1.0 - 1.0 / S, 1e-8)
                attn_map = peak.reshape(height, width)
                render_map(attn_map, "attn")

        # 写入权重
        write_weights = entry.get("write_weights")
        if write_weights is not None:
            beta = np.asarray(write_weights)
            if beta.ndim == 2 and beta.shape[0] == num_patches:
                write_map = beta.max(axis=-1).reshape(height, width)
                render_map(write_map, "write")