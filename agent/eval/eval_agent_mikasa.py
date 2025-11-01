# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




"""
Parent eval agent class with state input, for openai-gym environment
"""
import os
import numpy as np
import torch
import hydra
import logging
import random
from tqdm import tqdm as tqdm
log = logging.getLogger(__name__)
from env.gym_utils import make_async, make_mikasa_efficient
from omegaconf import OmegaConf
import torch.nn as nn
import os
import cv2
from agent.eval.visualize.utils import read_eval_statistics
from util.dirs import REINFLOW_DIR 
class EvalAgent:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.device = cfg.device
        self.base_policy_path = cfg.base_policy_path
        if not self.base_policy_path:
            raise ValueError("base_policy_path must be set in the config file!")
        self.eval_log_dir = cfg.get('eval_log_dir', None)
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        ############ could be overload #############
        self.load_ema = cfg.get("load_ema", False)
        self.denoising_steps = cfg.get("denoising_step_list", None)
        self.record_video = cfg.get("record_video", False)
        self.frame_width = cfg.get("frame_width", 640)
        self.frame_height = cfg.get("frame_height", 480)
        self.all_video_paths=[] # a list of video paths for each denoising step.
        self.record_env_index = cfg.get("record_env_index", 0)
        self.render_onscreen = False
        self.denoising_steps_trained = None
        self.plot_scale = cfg.get("plot_scale", "semilogx")  # Default to semilogx, can be "standard" or "semilogx"
        self.plot_scale_options=['standard, semilogx']
        # if self.plot_scale not in self.plot_scale_options:
        #     raise ValueError(f"plot scale must be one of {self.plot_scale_options}, but received {self.plot_scale}!")
        ############################################
        
        # Make vectorized env
        self.env_name: str = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.is_mikasa = env_type == "mikasa"
        self.mikasa_video_dir = cfg.env.get("video_dir", None)
        self.mikasa_save_video = cfg.env.get("save_video", False)
        self.mikasa_use_custom_video = cfg.env.get("use_custom_video", False)

        if self.is_mikasa:
            wrappers_cfg = cfg.env.get("wrappers", None)
            specific_cfg = cfg.env.get("specific", {})
            self.venv = make_mikasa_efficient(
                env_name=cfg.env.name,
                num_envs=cfg.env.n_envs,
                wrappers=wrappers_cfg,
                use_image_obs=cfg.env.get("use_image_obs", False),
                max_episode_steps=cfg.env.max_episode_steps,
                render_mode=cfg.env.get("render_mode", "rgb_array"),
                sim_backend=cfg.env.get("sim_backend", "gpu"),
                reward_mode=cfg.env.get("reward_mode", "normalized_dense"),
                save_video=cfg.env.get("save_video", False),
                video_dir=cfg.env.get("video_dir", None),
                video_fps=cfg.env.get("video_fps", 30),
                save_trajectory=cfg.env.get("save_trajectory", False),
                save_video_trigger=cfg.env.get("save_video_trigger", None),
                **(specific_cfg if specific_cfg is not None else {}),
            )
        else:
            self.venv = make_async(
                cfg.env.name,
                env_type=env_type,
                num_envs=cfg.env.n_envs,
                asynchronous=True,
                max_episode_steps=cfg.env.max_episode_steps,
                wrappers=cfg.env.get("wrappers", None),
                robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
                shape_meta=cfg.get("shape_meta", None),
                use_image_obs=cfg.env.get("use_image_obs", False),
                render=cfg.env.get("render", False),
                render_offscreen=cfg.env.get("save_video", False),
                obs_dim=cfg.obs_dim,
                action_dim=cfg.action_dim,
                **cfg.env.specific if "specific" in cfg.env else {},
            )
            if not env_type == "furniture":
                self.venv.seed(
                    [self.seed + i for i in range(cfg.env.n_envs)]
                )
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )
        
        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        
        # Eval params
        self.n_steps = cfg.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.render_num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        
    
    def load_model_for_eval(self):
        data = torch.load(self.base_policy_path, weights_only=True, map_location=self.device)
        self.model: nn.Module        
        print(f"loading model...")
        if self.load_ema:
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=False)
                else:
                    actor_policy_state_dict = {key.replace('actor_ft.policy.', 'network.'): value 
                                        for key, value in data["ema"].items() 
                                        if key.startswith('actor_ft.policy.')}
                    if actor_policy_state_dict == {}:
                        raise ValueError(f"No parameter starting with actor_ft.policy in ={data['ema'].keys()}")
                    self.model.load_state_dict(actor_policy_state_dict, strict=False)
            else:
                raise ValueError(f"You set self.load_ema={self.load_ema}, but your state dictionary does not contain key: ema. It only contains keys: {data.keys()}")
            log.info(f"Loaded EMA model dict from {self.base_policy_path}")
        else:
            if 'policy' in data.keys():
                self.model.load_state_dict(data["policy"], strict=True)
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=True)
            elif 'model' in data.keys():
                actor_policy_state_dict = {} 
                for key, value in data["model"].items():
                    if key.startswith('actor_ft.mlp_logvar') or key.startswith('actor_ft.logvar') or key.startswith('actor_ft.explore_noise_net.'):
                        continue
                    if key.startswith('actor_ft.policy.'):
                        actor_policy_state_dict[key.replace('actor_ft.policy.', 'network.')] = value
                    elif key.startswith('actor_ft.'):
                        actor_policy_state_dict[key.replace('actor_ft.', 'network.')] = value
                if actor_policy_state_dict == {}:
                    raise ValueError(f"No parameter starting with actor_ft.policy or actor_ft. in ={data['model'].keys()}")
                self.model.load_state_dict(actor_policy_state_dict, strict=True)
            else:
                raise ValueError(f"Your state dictionary is not correct, it does not contain keys: policy or model. It only contains keys: {data.keys()}")
            log.info(f"Loaded model dict from {self.base_policy_path}")

    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if self.is_mikasa:
            result = self.venv.reset(**kwargs)
            if isinstance(result, tuple):
                obs, _ = result
            else:
                obs = result
            obs = self._obs_to_numpy(obs)
            if verbose:
                logging.info("Reset Mikasa vector env")
            return obs

        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        if self.is_mikasa:
            raise NotImplementedError("Per-environment reset is not supported for Mikasa envs.")
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs

    # ---------------------------- helper utils ---------------------------------
    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, list):
            return np.asarray(value)
        return value

    def _obs_to_numpy(self, obs_dict):
        return {k: self._to_numpy(v) for k, v in obs_dict.items()}

    def _obs_to_device(self, obs_dict):
        result = {}
        for key in self.obs_dims:
            value = obs_dict[key]
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device, dtype=torch.float32)
            else:
                tensor = torch.from_numpy(value).float().to(self.device)
            result[key] = tensor
        return result

    def _ensure_numpy_array(self, array, dtype=None):
        arr = self._to_numpy(array)
        if dtype is not None and arr is not None:
            arr = arr.astype(dtype, copy=False)
        return arr
    
    
    def run(self):
        if self.render_onscreen and self.n_envs > 1:
            raise ValueError(f"Cannot render on screen with more than one parallel envs. self.render_onscreen={self.render_onscreen}, cfg.env.n_envs={self.n_envs}")
        if self.eval_log_dir is None:
            self.eval_log_dir = f'visualize/{self.model.__class__.__name__}/{self.env_name}/{self.current_time()}/'
        os.makedirs(self.eval_log_dir, exist_ok=True)
        cfg_path = os.path.join(self.eval_log_dir, "cfg.yaml")
        with open(cfg_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        print(f"Configuration saved to {cfg_path}")
        
        options_venv = None if self.is_mikasa else [{} for _ in range(self.n_envs)]
        if self.render_video:
            if options_venv is None:
                # for mikasa, RecordEpisode is already handled inside env builder; fallback to manual writer
                pass
            else:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"eval_trial-{env_ind}.mp4"
                    )
        
        self.load_model_for_eval()
        denoising_steps_set = self.denoising_steps
        
        # Lists to store the results
        num_denoising_steps_list = []
        avg_single_step_freq_list = []
        avg_single_step_freq_std_list = []
        avg_single_step_duration_list = []
        avg_single_step_duration_std_list = []
        avg_traj_length_list = []
        avg_traj_length_std_list = []  # Added
        avg_episode_reward_list = []
        avg_episode_reward_std_list = []
        avg_best_reward_list = []
        avg_best_reward_std_list = []
        success_rate_list = []
        success_rate_std_list = []  # Added
        num_episodes_finished_list = []
        
        for num_denoising_steps in denoising_steps_set:
            if not self.is_mikasa:
                self.venv.reset()
            result = self.single_run(num_denoising_steps, options_venv)
            
            num_denoising_steps, avg_single_step_freq, avg_single_step_freq_std, \
                avg_single_step_duration, avg_single_step_duration_std, \
                avg_traj_length, avg_traj_length_std, \
                avg_episode_reward, avg_episode_reward_std, \
                avg_best_reward, avg_best_reward_std, \
                num_episodes_finished, success_rate, success_rate_std = result
            
            num_denoising_steps_list.append(num_denoising_steps)
            avg_single_step_freq_list.append(avg_single_step_freq)
            avg_single_step_freq_std_list.append(avg_single_step_freq_std)
            avg_single_step_duration_list.append(avg_single_step_duration)
            avg_single_step_duration_std_list.append(avg_single_step_duration_std)
            avg_traj_length_list.append(avg_traj_length)
            avg_traj_length_std_list.append(avg_traj_length_std)  # Added
            avg_episode_reward_list.append(avg_episode_reward)
            avg_episode_reward_std_list.append(avg_episode_reward_std)
            avg_best_reward_list.append(avg_best_reward)
            avg_best_reward_std_list.append(avg_best_reward_std)
            success_rate_list.append(success_rate)
            success_rate_std_list.append(success_rate_std)  # Added
            num_episodes_finished_list.append(num_episodes_finished)
        
        # Save evaluation statistics as an npz
        dtype = [
            ('num_denoising_steps', int),
            ('avg_single_step_freq', float),
            ('avg_single_step_freq_std', float),
            ('avg_single_step_duration', float),
            ('avg_single_step_duration_std', float),
            ('avg_traj_length', float),
            ('avg_traj_length_std', float),  # Added
            ('avg_episode_reward', float),
            ('avg_episode_reward_std', float),
            ('avg_best_reward', float),
            ('avg_best_reward_std', float),
            ('success_rate', float),
            ('success_rate_std', float),  # Added
            ('num_episodes_finished', int)
        ]

        data = np.zeros(len(num_denoising_steps_list), dtype=dtype)
        data['num_denoising_steps'] = num_denoising_steps_list
        data['avg_single_step_freq'] = avg_single_step_freq_list
        data['avg_single_step_freq_std'] = avg_single_step_freq_std_list
        data['avg_single_step_duration'] = avg_single_step_duration_list
        data['avg_single_step_duration_std'] = avg_single_step_duration_std_list
        data['avg_traj_length'] = avg_traj_length_list
        data['avg_traj_length_std'] = avg_traj_length_std_list  # Added
        data['avg_episode_reward'] = avg_episode_reward_list
        data['avg_episode_reward_std'] = avg_episode_reward_std_list
        data['avg_best_reward'] = avg_best_reward_list
        data['avg_best_reward_std'] = avg_best_reward_std_list
        data['success_rate'] = success_rate_list
        data['success_rate_std'] = success_rate_std_list  # Added
        data['num_episodes_finished'] = num_episodes_finished_list

        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')
        np.savez(eval_statistics_path, data=data)
        
        statistics = read_eval_statistics(npz_file_path=eval_statistics_path)
        self.plot_eval_statistics(statistics, self.eval_log_dir)


    def create_video_recorder(self, num_denoising_steps:int):
        self.video_writer = None
        if self.record_video and self.is_mikasa and not self.mikasa_use_custom_video:
            # rely on RecordEpisode wrapper; no manual writer
            self.video_path = None
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps"
            return
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_path = os.path.join(self.eval_log_dir, f'{self.model.__class__.__name__}_{self.env_name}_step{num_denoising_steps}.mp4')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (self.frame_width, self.frame_height))
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps"
        
    def single_run(self, num_denoising_steps, options_venv):
        
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
        episode_return_trajs = np.zeros((self.n_steps, self.n_envs))
        episode_length_trajs = np.zeros((self.n_steps, self.n_envs))
        episode_finished_mask_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        
        # Log message can remain as is
        log.info(f"Evaluating {self.model.__class__.__name__} model in {self.env_name} environment with {num_denoising_steps} step(s).")
        
        for step in tqdm(range(self.n_steps), dynamic_ncols=True, desc=f'{num_denoising_steps} step(s):'):
            with torch.no_grad():
                # Generic observation handling
                if hasattr(self, 'obs_dims'):  # For image-based agents
                    cond = self._obs_to_device(prev_obs_venv)
                else:  # For state-based agents
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                samples, single_step_duration= self.infer(cond, num_denoising_steps)
                
                single_step_duration_list[step] = single_step_duration
                
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]
            
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )

            if self.is_mikasa:
                obs_venv = self._obs_to_numpy(obs_venv)
                reward_venv = self._ensure_numpy_array(reward_venv, dtype=np.float32)
                terminated_venv = self._ensure_numpy_array(terminated_venv, dtype=bool)
                truncated_venv = self._ensure_numpy_array(truncated_venv, dtype=bool)
                info_dict = info_venv if isinstance(info_venv, dict) else {}
                if step < 5:
                    debug_keys = list(info_dict.keys())
                    debug_mask = self._ensure_numpy_array(info_dict.get('_final_info'), dtype=bool)
                    debug_mask_sum = int(debug_mask.sum()) if debug_mask is not None else 0
                    debug_final_ep = info_dict.get('final_info', {})
                    debug_ep_keys = list(debug_final_ep.get('episode', {}).keys()) if isinstance(debug_final_ep, dict) else []
                    print(f"[EvalDebug] step={step} keys={debug_keys} mask_sum={debug_mask_sum} episode_keys={debug_ep_keys}")
                success_once = self._ensure_numpy_array(info_dict.get('success_once'), dtype=bool)
                if success_once is None:
                    success_once = np.zeros(self.n_envs, dtype=bool)
                success_at_end = self._ensure_numpy_array(info_dict.get('success_at_end'), dtype=bool)
                if success_at_end is None:
                    success_at_end = np.zeros(self.n_envs, dtype=bool)
                success_trajs[step] = success_once | success_at_end
                success_once_trajs[step] = success_once
                success_at_end_trajs[step] = success_at_end

                mask = self._ensure_numpy_array(info_dict.get('_final_info'), dtype=bool)
                if mask is None:
                    mask = np.zeros(self.n_envs, dtype=bool)
                episode_finished_mask_trajs[step] = mask

                episode_info = info_dict.get('final_info', {}).get('episode', {}) if mask.any() else {}
                episode_returns = self._ensure_numpy_array(episode_info.get('return'), dtype=np.float32)
                if episode_returns is None:
                    episode_returns = reward_venv.copy()
                episode_lengths = self._ensure_numpy_array(episode_info.get('episode_len'), dtype=np.int32)
                if episode_lengths is None:
                    episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
                episode_return_trajs[step] = episode_returns
                episode_length_trajs[step] = episode_lengths
            else:
                # add success by dawei (legacy path)
                info_list = info_venv if isinstance(info_venv, list) else []
                success_venv = []
                for i, info in enumerate(info_list):
                    if 'success' in info:
                        success_val = info['success']
                        if hasattr(success_val, 'item'):
                            if hasattr(success_val, 'numel') and success_val.numel() > 1:
                                success_venv.append(bool(success_val[i].item()))
                            else:
                                success_venv.append(bool(success_val.item()))
                        else:
                            success_venv.append(bool(success_val))
                    else:
                        success_venv.append(False)
                success_trajs[step] = np.array(success_venv, dtype=bool)

            if self.render_onscreen:
                self.venv.render(mode='human')
            if self.record_video:
                if self.is_mikasa and not self.mikasa_use_custom_video:
                    continue

                if 'kitchen' in self.env_name.lower():  # Kitchen
                    raise ValueError(
                        f"Cannot record video for kitchen environments with the current setup. self.env_name={self.env_name}"
                    )

                if self.is_mikasa:
                    frame_batch = self.venv.render()
                else:  # legacy gym / robomimic / d3il envs
                    frame_batch = self.venv.render(
                        mode='rgb_array',
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
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            prev_obs_venv = obs_venv
        
        if self.video_writer is not None:
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
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)

            episode_success = []
            episode_success_once = []
            episode_success_at_end = []
            for env_ind, start, end in episodes_start_end:
                episode_success.append(np.any(success_trajs[start:end + 1, env_ind]))
                episode_success_once.append(np.any(success_once_trajs[start:end + 1, env_ind]))
                episode_success_at_end.append(np.any(success_at_end_trajs[start:end + 1, env_ind]))
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

        episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
        avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        avg_traj_length_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
        
        avg_single_step_duration = single_step_duration_list.mean()
        avg_single_step_duration_std = single_step_duration_list.std()
        single_step_frequency_list = 1 / single_step_duration_list
        avg_single_step_freq = single_step_frequency_list.mean()
        avg_single_step_freq_std = single_step_frequency_list.std()
        
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'
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
        
        return num_denoising_steps, \
            avg_single_step_freq, avg_single_step_freq_std, \
            avg_single_step_duration, avg_single_step_duration_std, \
            avg_traj_length, avg_traj_length_std, \
            avg_episode_reward, avg_episode_reward_std, \
            avg_best_reward, avg_best_reward_std, \
            num_episodes_finished, success_rate, success_rate_std
    
    def current_time(self):
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
        return formatted_time
    
    def plot_eval_statistics(self, eval_statistics, log_dir: str):
        num_denoising_steps_list, \
            avg_single_step_freq_list, avg_single_step_freq_std_list, \
            avg_single_step_duration_list, avg_single_step_duration_std_list, \
            avg_traj_length_list, avg_traj_length_std_list, \
            avg_episode_reward_list, avg_episode_reward_std_list, \
            avg_best_reward_list, avg_best_reward_std_list, \
            success_rate_list, success_rate_std_list, \
            num_episodes_finished_list = eval_statistics
        
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 8))
        
        # Define plotting function based on self.plot_scale
        plot_func = plt.semilogx if self.plot_scale == "semilogx" else plt.plot

         # Plot success rate
        plt.subplot(2, 3, 1)
        plot_func(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.fill_between(num_denoising_steps_list,
                        [sr - std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        [sr + std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        color='y', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()


        # Plot average episode reward
        plt.subplot(2, 3, 2)
        plot_func(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
        plt.fill_between(num_denoising_steps_list,
                        [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        color='b', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 5)
        plot_func(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.fill_between(num_denoising_steps_list,
                        [avg_traj - std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        [avg_traj + std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        color='r', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Trajectory Length')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Trajectory Length')
        plt.grid(True)
        plt.legend()

        # Plot inference duration
        plt.subplot(2, 3, 3)
        plot_func(num_denoising_steps_list, avg_single_step_duration_list, marker='o', label='Time', color='purple')
        plt.fill_between(num_denoising_steps_list,
                        [duration - std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        [duration + std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        color='purple', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Duration')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Time (s)')
        plt.grid(True)
        plt.legend()
        
        
        # Plot average best reward
        plt.subplot(2, 3, 4)
        plot_func(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
        plt.fill_between(num_denoising_steps_list,
                        [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        color='g', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

       
        # Plot inference frequency
        plt.subplot(2, 3, 6)
        plot_func(num_denoising_steps_list, avg_single_step_freq_list, marker='o', label='Frequency', color='brown')
        plt.fill_between(num_denoising_steps_list,
                        [freq - std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        [freq + std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        color='brown', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Frequency')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Frequency (Hz)')
        plt.grid(True)
        plt.legend()

        plt.suptitle(f"{self.model.__class__.__name__}, {self.env_name}\nsteps = {', '.join(map(str, num_denoising_steps_list))}", fontsize=25)
        plt.tight_layout()
        
        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')


        fig_path = os.path.join(REINFLOW_DIR, log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"Finished evaluating {self.model.__class__.__name__} in environment {self.env_name}")
        print(f"Base_policy_path: {os.path.join(REINFLOW_DIR,self.base_policy_path)}")
        print(f"Figure saved to {fig_path}")
        print(f"Evaluation statistics saved to  {eval_statistics_path}")
        if self.record_video:
            print(f"Video(s) saved to {self.all_video_paths}")   
        plt.close()