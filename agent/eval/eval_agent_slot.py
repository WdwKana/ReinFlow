# agent/eval/eval_agent_slot.py
import logging
import numpy as np
import torch

from agent.eval.eval_reflow_mikasa_agent import EvalReFlowMikasaAgent
from util.timer import Timer


log = logging.getLogger(__name__)

class EvalReFlowSlotAgent(EvalReFlowMikasaAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.slot_memory_cfg = cfg.get("slot_memory", None)

        if self.slot_memory_cfg is None:
            log.warning("No slot_memory configuration found; proceeding without slot memory module.")
            self.slot_mem = None
        else:
            # Import SlotMemory
            from model.common.learned_memory import SlotMemory
            
            # Get slot memory parameters from config
            num_slots = self.slot_memory_cfg.get("num_slots", None)
            slot_dim = self.slot_memory_cfg.get("slot_dim", None)
            
            # Try to infer from model if not provided
            if num_slots is None and hasattr(self.model, "network"):
                num_slots = getattr(self.model.network, "num_slots", None)
            if slot_dim is None and hasattr(self.model, "network"):
                slot_dim = getattr(self.model.network, "slot_dim", None)
            
            if num_slots is None or slot_dim is None:
                raise ValueError(
                    f"slot_memory requires num_slots and slot_dim. "
                    f"Got num_slots={num_slots}, slot_dim={slot_dim}"
                )
            
            log.info(f"Initializing SlotMemory with num_slots={num_slots}, slot_dim={slot_dim}")
            
            self.slot_mem = SlotMemory(
                num_slots=num_slots,
                slot_dim=slot_dim,
                device=self.device,
            )
            self.slot_mem = self.slot_mem.to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(self.base_policy_path, map_location=self.device)
            if "slot_memory" in checkpoint and self.slot_mem is not None:
                missing, unexpected = self.slot_mem.load_state_dict(
                    checkpoint["slot_memory"], strict=False
                )
                log.info(f"Loaded slot_memory state (missing={missing}, unexpected={unexpected})")
            else:
                log.warning("Slot memory state not found in checkpoint; using fresh initialization.")

            if hasattr(self.slot_mem, "eval"):
                self.slot_mem.eval()

    @torch.no_grad()
    def _reset_slot_memory_all(self):
        """Reset slot memory for all environments."""
        if self.slot_mem is None:
            return
        # SlotMemory.slots_for_batch handles initialization
        self.slot_mem.slots_for_batch(B=self.n_envs, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _clear_slot_memory(self, mask):
        """Clear slot memory for environments indicated by mask."""
        if self.slot_mem is None:
            return
        if mask is None:
            return
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(bool))
        
        storage = self._slot_memory_storage()
        if storage.numel() == 0:
            return
        
        mask = mask.to(storage.device)
        if mask.any():
            storage[mask] = 0.0

    def _slot_memory_storage(self) -> torch.Tensor:
        """Get the slot memory tensor."""
        if self.slot_mem is None:
            raise AttributeError("Slot memory storage requested but no slot memory module is instantiated.")
        if hasattr(self.slot_mem, "memory"):
            return self.slot_mem.memory
        if hasattr(self.slot_mem, "m"):
            return self.slot_mem.m
        raise AttributeError("Slot memory module must expose a tensor buffer named 'memory' or 'm'.")

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        """Reset all environments and slot memory."""
        obs = super().reset_env_all(verbose=verbose, options_venv=options_venv, **kwargs)
        self._reset_slot_memory_all()
        return obs

    def infer(self, cond, num_denoising_steps):
        """Inference with slot memory."""
        cond_with_mem = dict(cond)
        if self.slot_mem is not None:
            cond_with_mem["slot_mem"] = self.slot_mem
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
        """Run evaluation for one episode with slot memory management."""
        import cv2
        from tqdm import tqdm
        
        self.create_video_recorder(num_denoising_steps)
        self.model.eval()
        
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        single_step_duration_list = np.zeros(self.n_steps)
        success_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        success_once_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        success_at_end_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)

        log.info(
            f"Evaluating {self.model.__class__.__name__} model in {self.env_name} environment "
            f"with {num_denoising_steps} step(s)."
        )

        for step in tqdm(range(self.n_steps), dynamic_ncols=True, desc=f"{num_denoising_steps} step(s):"):
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
                
                success_once = self._ensure_numpy_array(info_dict.get("success_once"), dtype=bool)
                if success_once is None:
                    success_once = np.zeros(self.n_envs, dtype=bool)
                success_at_end = self._ensure_numpy_array(info_dict.get("success_at_end"), dtype=bool)
                if success_at_end is None:
                    success_at_end = np.zeros(self.n_envs, dtype=bool)
                success_trajs[step] = success_once | success_at_end
                success_once_trajs[step] = success_once
                success_at_end_trajs[step] = success_at_end
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

            # Clear slot memory for done environments
            self._clear_slot_memory(done_mask)

            # Video recording
            if self.render_onscreen:
                self.venv.render(mode="human")
            if self.record_video:
                if self.is_mikasa and not self.mikasa_use_custom_video:
                    pass  # Skip video for non-custom mikasa
                else:
                    if "kitchen" in self.env_name.lower():
                        raise ValueError(
                            f"Cannot record video for kitchen environments. env_name={self.env_name}"
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

        if self.record_video and self.video_writer is not None:
            self.video_writer.release()
            self.all_video_paths.append(self.video_path)
            print(f"Video saved to {self.video_path}")

        # Compute statistics
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

        return result