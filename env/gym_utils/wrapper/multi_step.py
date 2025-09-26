# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Multi-step wrapper. 
Allow executing multiple environmnt steps. 
Returns stacked observation and optionally stacked previous action.

Modified from 
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

TODO: allow cond_steps != img_cond_steps (should be implemented in training scripts, not here)
"""

import gymnasium as gym
from typing import Optional
from gymnasium import spaces
import numpy as np
import torch
from collections import defaultdict, deque


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )

'''
def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")
'''
def repeated_space(space, n):
    space_type_name = type(space).__name__
    
    # 检查是否为 Box 类型（支持 gym 和 gymnasium）
    if space_type_name == 'Box':
        return repeated_box(space, n)
    # 检查是否为 Dict 类型（支持 gym 和 gymnasium）
    elif space_type_name == 'Dict':
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    # Aggregate over time steps (axis=0), preserve per-env shape
    arr = np.asarray(data)
    if method == "max":
        return np.max(arr, axis=0)
    elif method == "min":
        return np.min(arr, axis=0)
    elif method == "mean":
        return np.mean(arr, axis=0)
    elif method == "sum":
        return np.sum(arr, axis=0)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    """Apply padding"""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStep(gym.Wrapper):
    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        reward_agg_method="sum",  # never use other types
        prev_action=True,
        reset_within_step=False,
        pass_full_observations=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose
    '''
    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: dict = {},
    ):
        """Resets the environment."""
        obs = self.env.reset(
            seed=seed,
            options=options,
            return_info=return_info,
        )
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps)

        self.cnt = 0
        return obs
    '''
    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: dict = {},
    ):
        """Resets the environment."""
        #print(f"DEBUG MultiStep: reset called with return_info={return_info}")
        try:
            result = self.env.reset(
                seed=seed,
                options=options,
                #return_info=return_info,
            )
            print(f"DEBUG: env.reset() returned type: {type(result)}, content: {result}")  # 添加这行

            #print(f"DEBUG MultiStep: env.reset returned type: {type(result)}")
            #print(f"DEBUG MultiStep: env.reset result: {result}")
            
            # 处理gymnasium格式的返回值 (obs, info) 或者只有obs
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                #print(f"DEBUG MultiStep: got tuple, obs type: {type(obs)}, info type: {type(info)}")
            else:
                obs = result
                info = {}
                #print(f"DEBUG MultiStep: got single value, obs type: {type(obs)}")
            
            #print(f"DEBUG MultiStep: final obs type: {type(obs)}")
            #if isinstance(obs, dict):
            #    print(f"DEBUG MultiStep: obs keys: {list(obs.keys())}")
                
        except Exception as e:
            print(f"ERROR in MultiStep reset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps)

        self.cnt = 0
        
        if return_info:
            return obs, info
        else:
            return obs
    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        # Normalize action layout to (T, N, D)
        try:
            import numpy as _np
        except Exception:
            import numpy as _np
        # torch -> numpy
        if hasattr(action, "detach") and hasattr(action, "cpu"):
            try:
                action = action.detach().cpu().numpy()
            except Exception:
                action = _np.array(action)
        else:
            action = _np.array(action)

        # Now standardize shapes
        # Expected per-time-step env action is (N, D) for vectorized env with N envs;
        # We want action overall as (T, N, D)
        if action.ndim == 1:  # (D,) -> (T=1, N=1, D)
            action = action[_np.newaxis, _np.newaxis, :]
        elif action.ndim == 2:
            # Could be (T, D) for single env -> (T, 1, D)
            # or (N, D) for single time step -> (1, N, D)
            if action.shape[0] == self.n_action_steps:
                action = action[:, _np.newaxis, :]
            else:
                action = action[_np.newaxis, :, :]
        elif action.ndim == 3:
            # Could be (T, N, D) or (N, T, D)
            if action.shape[0] != self.n_action_steps and action.shape[1] == self.n_action_steps:
                # (N, T, D) -> (T, N, D)
                action = _np.swapaxes(action, 0, 1)
            # else assume already (T, N, D)
        first_step = True
        agg_mask = None
        agg_success = None
        for act_step, act in enumerate(action):
            self.cnt += 1
            # Step vectorized env one chunk step - 处理 gymnasium 格式
            result = self.env.step(act)
            
            if len(result) == 5:
                # gymnasium 格式
                observation, reward, terminated, truncated, info = result
                
                # 转换 tensor 为 numpy 以便后续处理
                if isinstance(terminated, torch.Tensor):
                    terminated = terminated.detach().cpu().numpy()
                if isinstance(truncated, torch.Tensor):
                    truncated = truncated.detach().cpu().numpy()
                
                done = np.logical_or(terminated, truncated)  # 现在可以安全合并
            elif len(result) == 4:
                # gym 格式
                observation, reward, done, info = result
                
                # 确保 done 也是 numpy
                if isinstance(done, torch.Tensor):
                    done = done.detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected step result length: {len(result)}")

            # 转换 reward 为 numpy 以便聚合
            if isinstance(reward, torch.Tensor):
                reward = reward.detach().cpu().numpy()

            self.obs.append(observation)
            self.action.append(act)
            self.reward.append(reward)
            # done is per-env boolean array; just record it
            self.done.append(done)
            self._add_info(info)
            
            # 初始化聚合变量（首次循环时）
            if first_step:
                agg_mask = np.zeros_like(reward, dtype=bool)
                agg_success = np.zeros_like(reward, dtype=bool)
                first_step = False
                
            # 聚合 ManiSkill 的 final_info
            if isinstance(info, dict) and 'final_info' in info:
                mask = info.get('_final_info', None)
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy().astype(bool)
                if mask is not None:
                    agg_mask = np.logical_or(agg_mask, mask)
                    ep = info['final_info'].get('episode', {})
                    succ = ep.get('success_at_end', ep.get('success_once', None))
                    if succ is not None:
                        if isinstance(succ, torch.Tensor):
                            succ = succ.detach().cpu().numpy()
                        succ = np.asarray(succ).astype(bool)
                        agg_success[mask] = succ[mask]
        
        # 处理最终输出
        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        
        # 保留最后一步的 info 结构，并添加聚合信息
        info_out = info if isinstance(info, dict) else {}
        
        # 添加聚合的成功信息
        if agg_success is not None:
            info_out['success'] = agg_success
            
        # 更新 _final_info 为整个 chunk 的聚合
        if agg_mask is not None and agg_mask.any():
            info_out['_final_info'] = agg_mask
        info = info_out
        if self.pass_full_observations:
            info_out["full_obs"] = self._get_obs(act_step + 1)
        # Convert info to a list of per-env dicts (length = batch size)
        # Determine batch size from observation
        if isinstance(observation, dict):
            sample_key = next(iter(observation))
            obs_sample = observation[sample_key]
        else:
            obs_sample = observation
        # obs_sample expected shape (B, T, ...)
        B = obs_sample.shape[0] if hasattr(obs_sample, 'shape') and len(obs_sample.shape) > 0 else 1
        '''
        info_list = []
        for b in range(B):
            per_env = {}
            for k, v in info.items():
                # v is shape (T, ...) or (T,) of values; take the last time step
                try:
                    v_last = v[-1]
                except Exception:
                    v_last = v
                # if v_last is per-env array/list, index by b
                if hasattr(v_last, '__len__') and not isinstance(v_last, (str, bytes)):
                    try:
                        per_env[k] = v_last[b]
                    except Exception:
                        per_env[k] = v_last
                else:
                    per_env[k] = v_last
            info_list.append(per_env)
        info = info_list
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1)
        '''
        # Optional: reset within step if any env is done (vectorized-safe)
        if self.reset_within_step:
            done_last = np.asarray(self.done[-1])
            if done_last.any():
                observation = self.reset()
                self.verbose and print("Reset env within wrapper.")

        # reset reward and done for next step
        self.reward = list()
        self.done = list()
        # Ensure terminated/truncated are per-env boolean arrays
        if isinstance(observation, dict):
            sample_key = next(iter(observation))
            obs_sample = observation[sample_key]
        else:
            obs_sample = observation
        B = obs_sample.shape[0] if hasattr(obs_sample, 'shape') and len(obs_sample.shape) > 0 else 1

        done_vec = done
        if not isinstance(done_vec, np.ndarray):
            done_vec = np.array([done_vec] * B, dtype=bool)
        terminated_vec = done_vec.astype(bool)
        truncated_vec = np.zeros_like(terminated_vec, dtype=bool)

        # 确保terminated_vec反映真实的episode完成情况
        if agg_mask is not None and agg_mask.any():
            # 有episode真正完成时，使用聚合的mask作为terminated信号
            terminated_vec = agg_mask.astype(bool)
        else:
            # 没有episode完成时，terminated为False
            terminated_vec = np.zeros_like(done_vec, dtype=bool)
            
        # truncated保持原来的逻辑
        truncated_vec = np.logical_and(done_vec, ~terminated_vec)

        # Debug output for single env case
        if len(terminated_vec) == 1:
            has_success = False
            success_val = 'N/A'
            if info_out:
                if isinstance(info_out, list) and len(info_out) > 0:
                    has_success = 'success' in info_out[0]
                    success_val = info_out[0].get('success', 'N/A') if has_success else 'N/A'
                elif isinstance(info_out, dict):
                    has_success = 'success' in info_out
                    success_val = info_out.get('success', 'N/A') if has_success else 'N/A'
            print(f"[CHK1] MultiStep.step return: terminated={terminated_vec[0]}, truncated={truncated_vec[0]}, has_success={has_success}, success={success_val}, info_type={type(info_out)}")

        return observation, reward, terminated_vec, truncated_vec, info_out


#change by Dawei for env
    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0

        #debug by Dawei Wang 2025-09-01
        first_obs = self.obs[0]
        print(f"DEBUG: _get_obs first_obs type: {type(first_obs)}")
        if isinstance(self.observation_space, spaces.Box):
            #return stack_last_n_obs(self.obs, n_steps)
            out = stack_last_n_obs(self.obs, n_steps)  # (T, ...) with batch inside obs shape
            if out.ndim >= 2:
                out = np.swapaxes(out, 0, 1)  # (T, B, ...) -> (B, T, ...)
            return out
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            if isinstance(first_obs, dict):
                actual_keys = first_obs.keys()
            else:
                print(f"ERROR: Expected dict, got {type(first_obs)}: {first_obs}")
                raise RuntimeError(f"obs[0] should be dict, got {type(first_obs)}")
            #use the actual keys from the obs changed by Dawei Wang 2025-09-01
            #actual_keys = self.obs[0].keys() if isinstance(self.obs[0], dict) else self.observation_space.keys()
            #for key in self.observation_space.keys():
            for key in actual_keys:
                out = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
                if out.ndim >= 2:
                    out = np.swapaxes(out, 0, 1)
                #result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
                result[key] = out
            return result
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps)
    '''
    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    '''
    def _add_info(self, info):
        for key, value in info.items():
            # normalize tensors to numpy/python for safe numpy stacking
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                try:
                    value = value.detach().cpu()
                    value = value.item() if value.ndim == 0 else value.numpy()
                except Exception:
                    value = value.detach().cpu().numpy()
            self.info[key].append(value)
    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs)


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt
    from env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = MultiStep(
        env=RobomimicImageWrapper(
            env=env,
            shape_meta=shape_meta,
            image_keys=["robot0_eye_in_hand_image"],
        ),
        n_obs_steps=1,
        n_action_steps=1,
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")
