
import gymnasium as gym
import torch
from typing import Dict
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv


from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper

import numpy as np

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=False, state=False, oracle=False, joints=True, target_camera="base_camera", normalization_path=None, clamp_obs=False) -> None:
        self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints
        self.target_camera = target_camera

        # normalization
        self.normalize = normalization_path is not None
        self.clamp_obs = clamp_obs
        if self.normalize:
            #import numpy as np
            norm = np.load(normalization_path)
            self.obs_min = norm["obs_min"].astype(np.float32)
            self.obs_max = norm["obs_max"].astype(np.float32)
            self.action_min = norm["action_min"].astype(np.float32)
            self.action_max = norm["action_max"].astype(np.float32)
        
        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.base_env.update_obs_space(new_obs)
        '''
        # expose transformed observation_space
        import gym
        obs_space = {}
        for k, v in new_obs.items():
            shape = tuple(v.shape)
            if k == "state":
                obs_space[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=v.dtype)
            else:
                obs_space[k] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=v.dtype)
        self.observation_space = gym.spaces.Dict(obs_space)
        '''
        
    

    def reset(self, **kwargs):
        """Override reset to properly transform observations"""
        try:
            
            
            
            filtered_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['return_info']:  
                    filtered_kwargs[key] = value  
            
            #print(f"DEBUG: calling env.reset with filtered kwargs: {filtered_kwargs}")
            result = self.env.reset(**filtered_kwargs)
            
            
            
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                transformed_obs = self.observation(obs)
                return transformed_obs, info
            else:
                obs = result
                transformed_obs = self.observation(obs)
                return transformed_obs
                
        except Exception as e:
            print(f"ERROR in mikasa wrapper reset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def seed(self, seed=None):

        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env.unwrapped, 'seed'):
            return self.env.unwrapped.seed(seed)
        else:

            import numpy as np
            if seed is not None:
                np.random.seed(seed)
            return [seed]

    def observation(self, observation: Dict):
        #print(f"Mikasa observation keys: {list(observation.keys())}")
        #for key, value in observation.items():
        #    if hasattr(value, 'shape'):
        #        print(f"  {key}: shape {value.shape}")
        #    else:
        #        print(f"  {key}: {type(value)}")
        
        
        ret = dict()

        if self.include_rgb or self.include_depth:
            #ret['oracle_info'] = observation['oracle_info']
            #ret['prompt'] = observation['prompt']
            sensor_data = observation.pop("sensor_data")

            del observation["sensor_param"]
            images = []
            #for cam_data in sensor_data.values():
            if self.target_camera in sensor_data:
                cam_data = sensor_data[self.target_camera]
                if self.include_rgb:
                    rgb_img = cam_data["rgb"]
                    #print(f"DEBUG: Original RGB shape: {rgb_img.shape}")
                    #print(f"DEBUG: Original RGB type: {type(rgb_img)}")
                    #print(f"DEBUG: RGB ndim: {rgb_img.ndim}")
                    
                    if isinstance(rgb_img, torch.Tensor):
                        if rgb_img.ndim == 5:  
                            rgb_img = rgb_img.squeeze(1)  # → (num_envs, H, W, C)
                            rgb_img = rgb_img.permute(0, 3, 1, 2)  # → (num_envs, C, H, W)
                        elif rgb_img.ndim == 4:
                            if rgb_img.shape[0] == 1 and rgb_img.shape[-1] in [3, 4]:  # (1, H, W, C)
                                rgb_img = rgb_img.squeeze(0)  # → (H, W, C)
                                rgb_img = rgb_img.permute(2, 0, 1)  # → (C, H, W)
                            elif rgb_img.shape[-1] in [3, 4]:  # (num_envs, H, W, C)
                                rgb_img = rgb_img.permute(0, 3, 1, 2)  # → (num_envs, C, H, W)
                            elif rgb_img.shape[1] in [3, 4] and rgb_img.shape[1] < rgb_img.shape[2]:
                                #print(f"DEBUG: Already in (batch, C, H, W) format")
                                pass
                            #else:
                                #print(f"DEBUG: Unexpected 4D shape: {rgb_img.shape}")
                        elif rgb_img.ndim == 3 and rgb_img.shape[-1] in [3, 4]:  # (H, W, C)
                            #print(f"DEBUG: 3D tensor detected: {rgb_img.shape}")
                            rgb_img = rgb_img.permute(2, 0, 1)  # → (C, H, W)
                        #else:
                            #print(f"DEBUG: Unexpected RGB tensor shape: {rgb_img.shape}")
                    
                    #print(f"DEBUG: Processed RGB shape: {rgb_img.shape}")
                    images.append(rgb_img)
                if self.include_depth:
                    depth_img = cam_data["depth"]
                    if isinstance(depth_img, torch.Tensor):
                        if depth_img.dim() == 2:  # (H, W)
                            depth_img = depth_img.unsqueeze(0)  # → (1, H, W)
                        elif depth_img.dim() == 3 and depth_img.shape[-1] == 1:  # (H, W, 1)
                            depth_img = depth_img.permute(2, 0, 1)  # → (1, H, W)
                    images.append(depth_img)

            if len(images) > 0:
                images = torch.cat(images, dim=0)

        # flatten the rest of the data which should just be state data
        if self.include_state and not (self.include_rgb or self.include_depth):
            pass
            '''
            if not self.include_oracle:
                observation.pop("oracle_info")
            else:
                observation = observation
            '''
        else:
            if not self.include_joints:
                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info']}
            else:
                # Create extra_agent dict with 'extra' and 'agent' keys
                extra_agent = {}
                for key in ['extra', 'agent']:
                    if key in observation:
                        extra_agent[key] = observation.pop(key)

                # Flatten the extra_agent dict
                extra_agent_flat = common.flatten_state_dict(extra_agent, use_torch=True, device=self.base_env.device)
                if isinstance(extra_agent_flat, torch.Tensor) and extra_agent_flat.ndim == 2 and extra_agent_flat.shape[0] == 1:
                    extra_agent_flat = extra_agent_flat.squeeze(0)  # (1, 25) → (25)
                ret['state'] = extra_agent_flat

                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info', 'extra']}

            observation = common.flatten_state_dict(
                filtered_obs, use_torch=True, device=self.base_env.device
            )
        
        if self.include_state and not (self.include_rgb or self.include_depth):
            ret = observation
        elif not self.include_joints:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images

        if 'state' in ret.keys() and not self.include_state and not self.include_joints:
            ret.pop('state')
        if 'oracle_info' in ret.keys() and not self.include_oracle and ret['oracle_info'] is not None:
            ret.pop('oracle_info')

        if 'oracle_info' in ret.keys() and (ret['oracle_info'] == 4242424242).any().item():
            ret.pop('oracle_info')

        if 'prompt' in ret.keys() and (ret['prompt'] == 4242424242).any().item():
            ret.pop('prompt')
        

        if 'joints' in ret.keys() and not self.include_joints:
            ret.pop('joints')
        #print(f"Final wrapper output type: {type(ret)}")
        #print(f"Final wrapper output keys: {list(ret.keys()) if isinstance(ret, dict) else 'Not a dict'}")
        #if 'rgb' in ret:
            #print(f"RGB shape: {ret['rgb'].shape}")
        #if 'state' in ret:
            #print(f"State shape: {ret['state'].shape}")
            #print(f"State type: {type(ret['state'])}")

        
        assert isinstance(ret, dict), f"Wrapper output must be dict, got {type(ret)}"
        #print(f"=== WRAPPER DEBUG ===")
        #print(f"Return type: {type(ret)}")
        #print(f"Return is dict: {isinstance(ret, dict)}")
        #print(f"Return keys: {list(ret.keys()) if isinstance(ret, dict) else 'N/A'}")
        
        #print(f"DEBUG: Before numpy conversion:")
        #for key, value in ret.items():
            #if isinstance(value, torch.Tensor):
                #print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            #else:
                #print(f"  {key}: type {type(value)}")

        for key, value in ret.items():
            if isinstance(value, torch.Tensor):
                if key in ['rgb', 'rgbd', 'depth']:
                    if value.ndim == 3:  # (C, H, W) → (1, C, H, W)
                        value = value.unsqueeze(0)
                elif key == 'state':
                    if value.ndim == 1:  # (D,) → (1, D)
                        value = value.unsqueeze(0)
                converted = value.detach().cpu().numpy().astype(np.float32)
                ret[key] = converted

        #print(f"DEBUG: After numpy conversion:")
        #for key, value in ret.items():
        #    if isinstance(value, np.ndarray):
        #        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        #    else:
        #       print(f"  {key}: type {type(value)}")

        return ret
    
    def step(self, action):
        """Override step to convert gymnasium format and handle tensors"""
        result = self.env.step(action)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            if isinstance(done, torch.Tensor):
                done = done.detach().cpu().numpy().astype(bool)
            elif not isinstance(done, (np.ndarray, list)):
                done = np.array([bool(done)], dtype=bool)
            else:
                done = np.asarray(done).astype(bool)
            
            terminated = done
            truncated = np.zeros_like(done, dtype=bool)
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")
        
        obs = self.observation(obs)
        
        return obs, reward, terminated, truncated, info