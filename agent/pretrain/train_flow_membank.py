"""
Pre-training ReFlow policy with working memory support
"""
import logging
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.reflow import ReFlow
from model.common.working_memory import WorkingMemory
import torch

class TrainReFlowAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ReFlow
        self.ema_model: ReFlow
        
        
        self.use_memory = 'mlp_memorybank' in cfg.model.network._target_
        if self.use_memory:
            embed_dim = cfg.model.network.backbone.cfg.embed_dim
            
            self.wm_train = WorkingMemory(
                embed_dim=embed_dim,
                ema=0.9,
                temperature=0.07,
                device=self.device
            )
            log.info(f"Initialized WorkingMemory for training with embed_dim={embed_dim}")
        else:
            self.wm_train = None
        
        self.verbose_train=False
        self.verbose_loss= True
        self.verbose_test= False
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test=  False
            self.test_denoising_steps=4
            self.test_clip_intermediate_actions=True
            self.test_model_type='ema'
            
            if self.use_memory:
                self.wm_test = WorkingMemory(
                    embed_dim=embed_dim,
                    ema=0.9,
                    temperature=0.07,
                    device=self.device
                )
                self.test_episode_steps = None
    
    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        act, cond = batch_data
        
        if self.use_memory and self.wm_train is not None:
            episode_steps = cond.get('episode_step', None)
            if episode_steps is not None:
                if not isinstance(episode_steps, torch.Tensor):
                    if isinstance(episode_steps, (int, float)):
                        B = act.shape[0]  # 从 action 获取 batch size
                        episode_steps = torch.full((B,), episode_steps, dtype=torch.long, device=self.device)
                    else:
                        episode_steps = torch.tensor(episode_steps, dtype=torch.long, device=self.device)
                elif episode_steps.dim() == 0:
                    B = act.shape[0]
                    episode_steps = episode_steps.unsqueeze(0).expand(B)
                
                phase_ids = torch.where(episode_steps < 5, 0, 
                           torch.where(episode_steps < 10, 1, 2))
                
                if (episode_steps == 0).any():
                    B = act.shape[0]
                    self.wm_train.reset(B=B, device=self.device)
                
                cond['wm'] = self.wm_train
                cond['phase_ids'] = phase_ids
        
        (xt, t), v = self.model.generate_target(act)
        loss= self.model.loss(xt, t, cond, v)
        return loss
    
    def inference(self, cond:dict):
        '''for testing purpose in mujoco'''
        if self.use_memory and hasattr(self, 'wm_test'):
            B = cond['state'].shape[0]
            
            if self.test_episode_steps is None:
                self.test_episode_steps = torch.zeros(B, dtype=torch.long, device=self.device)
                self.wm_test.reset(B=B, device=self.device)
            
            phase_ids = torch.where(self.test_episode_steps < 5, 0, 
                       torch.where(self.test_episode_steps < 10, 1, 2))
            
            cond['wm'] = self.wm_test
            cond['phase_ids'] = phase_ids
        
        if self.test_model_type == 'ema':
            samples = self.ema_model.sample(cond, 
                                            inference_steps=self.test_denoising_steps, 
                                            record_intermediate=False,
                                            clip_intermediate_actions=self.test_clip_intermediate_actions)
        else:
            samples = self.model.sample(cond, 
                                        inference_steps=self.test_denoising_steps, 
                                        record_intermediate=False,
                                        clip_intermediate_actions=self.test_clip_intermediate_actions)
        
        if self.use_memory and hasattr(self, 'wm_test'):
            self.test_episode_steps += 1
        
        return samples
    
    def test(self):
        if not self.test_in_mujoco:
            return
        
        log.info(f"Evaluating {self.model.__class__.__name__} in environment {self.env_name} with denoising steps = {self.test_denoising_steps}")
        
        if self.use_memory:
            self.test_episode_steps = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            self.wm_test.reset(B=self.n_envs, device=self.device)
        
        from util.timer import Timer
        log_all= self.test_log_all
        timer = Timer()
        
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        
        for step in tqdm(range(self.n_steps)) if self.verbose_test else range(self.n_steps):
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                
                samples = self.inference(cond=cond)
                output_venv = samples.trajectories.cpu().numpy()
            
            action_venv = output_venv[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            reward_trajs[step] = reward_venv
            done_venv = terminated_venv | truncated_venv
            firsts_trajs[step + 1] = done_venv
            
            if self.use_memory and done_venv.any():
                self.test_episode_steps[done_venv] = 0
                self.wm_test.m[done_venv] = 0.0
            
            prev_obs_venv = obs_venv
        
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
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
            
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array([
                    np.max(reward_traj) / self.act_steps
                    for reward_traj in reward_trajs_split
                ])
            
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_episode_reward_std = np.std(episode_reward)
            self.success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            
            if log_all:
                self.avg_best_reward = np.mean(episode_best_reward)
                self.avg_best_reward_std = np.std(episode_best_reward)
                episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
                self.avg_episode_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                self.avg_episode_length_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0
        else:
            self.avg_episode_reward = 0
            self.avg_episode_reward_std = 0.0
            self.success_rate = 0
            if log_all:
                self.avg_best_reward = 0
                self.avg_best_reward_std = 0.0
                self.avg_episode_length = 0
                self.avg_episode_length_std = 0
            log.info("[WARNING] No episode completed within the iteration!")
        
        if self.avg_episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.avg_episode_reward
            self.save_best_model()
            log.info(f"Current Best model saved at epoch {self.epoch}")
        
        time = timer()
        if log_all:
            log.info(
                f"eval: success rate {self.success_rate:8.4f} | avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f} | avg_episode_length {self.avg_episode_length:4.2f}±{self.avg_episode_length_std:4.2f} | num episode {num_episode_finished:4d} | avg best reward {self.avg_best_reward:8.1f}±{self.avg_best_reward_std:2.1f} |"
            )
        else:
            log.info(
                f"eval: success rate {self.success_rate*100:2.2f}%| avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f}"
            )
        
        import numpy as np
        if log_all:
            np.savez(
                self.result_path,
                num_episode=num_episode_finished,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_best_reward=self.avg_best_reward,
                time=time,
            )
        else:
            np.savez(
                self.result_path,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_episode_reward_std=self.avg_episode_reward_std,
                time=time,
            )