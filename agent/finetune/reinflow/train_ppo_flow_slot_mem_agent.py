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
DPPO fine-tuning.
run this line to finetune hopper-v2: 
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_reflow_mlp device=cuda:7
"""
import os
from tqdm import tqdm as tqdm
import torch
import logging
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from model.common.modules import RandomShiftsAug
import numpy as np
from model.flow.ft_ppo.ppoflow import PPOFlow
from agent.finetune.reinflow.buffer import PPOFlowImgBuffer, PPOFlowImgBufferGPU
from model.common.learned_memory import SlotMemory, SlotAttention
from omegaconf import OmegaConf
class TrainPPOImgFlowAgent(TrainPPOFlowAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)
        
        self.initial_ratio_error_threshold = 1e-6 # for image input tasks with random augmentation, the log prob at eopch=0 batch=0 could be different. so we relax the threshold.
        
        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate
        
        self.verbose = cfg.train.get('verbose', False)
        
        self.buffer_device = self.device # 'cpu'
        #self.slot_cfg = cfg.get('slot_memory', None)
        self.num_slots = self.model.actor_ft.policy.num_slots
        self.slot_dim = self.model.actor_ft.policy.slot_dim

        self.minibatch_duplicate_multiplier= 1    #self.ft_denoising_steps for robomimic

        self.skip_initial_eval =False
        
        self.use_early_stop = False
        self.memory_cfg = cfg.get('memory', None)
        
        self.fix_nextvalue_augment_bug=True #False
        self.actor_weight_decay = cfg.train.actor_weight_decay
        self.critic_weight_decay = cfg.train.get('critic_weight_decay', 0)
    # overload
    def init_buffer(self):
        log.info(f"self.buffer_device={self.buffer_device}")
        log_prob_cfg_dict={
            'normalize_denoising_horizon':self.normalize_denoising_horizon,
            'normalize_act_space_dimension': self.normalize_act_space_dim, 
            'clip_intermediate_actions':self.clip_intermediate_actions,
            'account_for_initial_stochasticity':self.account_for_initial_stochasticity
        }       
                                                           
        if self.buffer_device=='cpu':
            self.buffer = PPOFlowImgBuffer(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.inference_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims,  # this is different from state obs
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug,       # debug bugfix 
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug, # debug bugfix 2
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        else:
            self.buffer = PPOFlowImgBufferGPU(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.inference_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims,  # this is different from state obs
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug,       # debug bugfix 
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug, # debug bugfix 2
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        log.info(f"created buffer: {self.buffer.__class__} on {self.buffer_device}")

    # overload, do not return log probabilities during sampling.
    @torch.no_grad()
    def get_samples(self, 
                    cond:dict, 
                    ret_device='cpu', 
                    save_chains=True, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False, 
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True):
        # returns: action_samples are still numpy because mujoco engine receives np.
        if save_chains:
            action_samples, chains_venv  = self.model.get_actions(cond, 
                                                                eval_mode=self.eval_mode, 
                                                                save_chains=save_chains, 
                                                                normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                clip_intermediate_actions=clip_intermediate_actions,
                                                                account_for_initial_stochasticity=account_for_initial_stochasticity,
                                                                ret_logprob=False)        # n_envs , horizon_steps , act_dim
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv
        else:
            action_samples  = self.model.get_actions(cond, 
                                                    eval_mode=self.eval_mode, 
                                                    save_chains=save_chains, 
                                                    normalize_denoising_horizon=normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=normalize_act_space_dimension, 
                                                    clip_intermediate_actions=clip_intermediate_actions,
                                                    account_for_initial_stochasticity=account_for_initial_stochasticity,
                                                    ret_logprob=False)
            return action_samples.cpu().numpy()
    
    
    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        #from model.common.working_memory import WorkingMemory
        #from model.common.learned_memory import LearnedWorkingMemoryLite
        #from model.common.learned_memory import PatchWorkingMemory
        #wm = LearnedWorkingMemoryLite(embed_dim=128, temperature=0.07, normalize=True, device=self.device)
        #wm = WorkingMemory(embed_dim=128, ema=0.9, temperature=0.07, normalize=True, device=self.device)
        
        self.slot_mem = SlotMemory(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            device=self.device,
        ).to(self.device)


        mem_ckpt_path = getattr(self.cfg.model, "actor_policy_path", None)
        if mem_ckpt_path:
            try:
                ckpt = torch.load(mem_ckpt_path, map_location=self.device, weights_only=True)
            except TypeError:
                ckpt = torch.load(mem_ckpt_path, map_location=self.device)
            slot_memory_state = ckpt.get("slot_memory")
            if slot_memory_state is not None:
                missing, unexpected = self.slot_mem.load_state_dict(slot_memory_state, strict=False)
                log.info(f"Loaded slot memory from actor_policy_path (missing={missing}, unexpected={unexpected})")
            else:
                log.info("actor_policy_path checkpoint has no slot memory; skip slot memory restore.")
        #wm = wm.to(self.device)
        slot_memory_params = list(self.slot_mem.parameters())
        log.info(f"Adding {len(slot_memory_params)} slot memory parameters to actor optimizer")
        log.info(f"Slot memory param shapes: {[p.shape for p in slot_memory_params]}")
        
        
        actor_params = list(self.model.actor_ft.parameters()) + slot_memory_params
        self.actor_optimizer = torch.optim.AdamW(
            actor_params,
            lr=self.actor_lr,
            weight_decay=self.actor_weight_decay
        )
        episode_steps = np.zeros(self.n_envs, dtype=np.int32)
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            if self.eval_mode:
                eval_env = self.eval_venv
                n_eval_envs = getattr(eval_env, "num_envs", self.eval_n_envs)

                seed = None
                if not self._eval_seeded:
                    seed = self.seed
                    self._eval_seeded = True

                self.reset_env(
                    buffer_device=self.buffer_device,
                    venv=eval_env,
                    done_cache=self.eval_done_venv,
                    seed=seed,
                )
                self.slot_mem.reset(B=n_eval_envs, device=self.device)

                episode_return = np.zeros(n_eval_envs, dtype=float)
                episode_best = np.full(n_eval_envs, -np.inf, dtype=float)
                episode_length = np.zeros(n_eval_envs, dtype=int)

                finished_returns = []
                finished_best = []
                finished_lengths = []
                finished_success_once = []
                finished_success_end = []

                for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                    if not self.verbose and step % 100 == 0:
                        print(f"Processed {step} of {self.n_steps}")

                    with torch.no_grad():
                        cond = {
                            key: torch.from_numpy(self.prev_obs_venv[key]).float().to(self.device)
                            for key in self.obs_dims
                        }
                        cond["slot_mem"] = self.slot_mem
                        action_samples, _ = self.get_samples(
                            cond=cond,
                            ret_device=self.buffer_device,
                            normalize_denoising_horizon=self.normalize_denoising_horizon,
                            normalize_act_space_dimension=self.normalize_act_space_dim,
                            clip_intermediate_actions=self.clip_intermediate_actions,
                            account_for_initial_stochasticity=self.account_for_initial_stochasticity,
                        )

                    action_venv = action_samples[:, : self.act_steps]
                    obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = eval_env.step(action_venv)

                    episode_return += reward_venv
                    episode_best = np.maximum(episode_best, reward_venv)
                    episode_length += 1
                    self.prev_obs_venv = obs_venv

                    done_venv = terminated_venv | truncated_venv
                    if done_venv.any():
                        self.slot_mem.memory[done_venv] = 0.0

                    has_final_info = isinstance(info_venv, dict) and "_final_info" in info_venv
                    if has_final_info:
                        mask = info_venv["_final_info"]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.detach().cpu().numpy().astype(bool)
                        if mask.any():
                            episode_dict = info_venv.get("final_info", {}).get("episode", {})

                            def to_numpy(item, dtype):
                                if item is None:
                                    return None
                                if isinstance(item, torch.Tensor):
                                    item = item.detach().cpu().numpy()
                                return item.astype(dtype)

                            succ_once = to_numpy(episode_dict.get("success_once"), bool)
                            succ_end = to_numpy(episode_dict.get("success_at_end"), bool)
                            returns = to_numpy(episode_dict.get("return"), float)
                            lengths = to_numpy(episode_dict.get("episode_len"), int)

                            finished_returns.extend(
                                episode_return[mask] if returns is None else returns[mask]
                            )
                            finished_best.extend(episode_best[mask] / self.act_steps)
                            finished_lengths.extend(
                                episode_length[mask] * self.act_steps if lengths is None else lengths[mask]
                            )
                            if succ_once is not None:
                                finished_success_once.extend(succ_once[mask])
                            if succ_end is not None:
                                finished_success_end.extend(succ_end[mask])

                            episode_return[mask] = 0.0
                            episode_best[mask] = -np.inf
                            episode_length[mask] = 0
                            self.slot_mem.memory[mask] = 0.0

                num_finished = len(finished_returns)
                if num_finished:
                    returns_arr = np.asarray(finished_returns, dtype=float)
                    best_arr = np.asarray(finished_best, dtype=float)
                    length_arr = np.asarray(finished_lengths, dtype=float)

                    self.buffer.avg_episode_reward = float(returns_arr.mean())
                    self.buffer.std_episode_reward = float(returns_arr.std())
                    self.buffer.avg_best_reward = float(best_arr.mean())
                    self.buffer.std_best_reward = float(best_arr.std())
                    self.buffer.avg_episode_length = float(length_arr.mean())
                    self.buffer.std_episode_length = float(length_arr.std())

                    if finished_success_once:
                        success_once_arr = np.asarray(finished_success_once, dtype=float)
                        self.buffer.success_rate_once = float(success_once_arr.mean())
                        self.buffer.std_success_rate_once = float(success_once_arr.std())
                    else:
                        self.buffer.success_rate_once = 0.0
                        self.buffer.std_success_rate_once = 0.0

                    if finished_success_end:
                        success_end_arr = np.asarray(finished_success_end, dtype=float)
                        self.buffer.success_rate_at_end = float(success_end_arr.mean())
                        self.buffer.std_success_rate_at_end = float(success_end_arr.std())
                    else:
                        self.buffer.success_rate_at_end = 0.0
                        self.buffer.std_success_rate_at_end = 0.0
                else:
                    self.buffer.avg_episode_reward = 0.0
                    self.buffer.std_episode_reward = 0.0
                    self.buffer.avg_best_reward = 0.0
                    self.buffer.std_best_reward = 0.0
                    self.buffer.avg_episode_length = 0.0
                    self.buffer.std_episode_length = 0.0
                    self.buffer.success_rate_once = 0.0
                    self.buffer.std_success_rate_once = 0.0
                    self.buffer.success_rate_at_end = 0.0
                    self.buffer.std_success_rate_at_end = 0.0

                self.buffer.num_episode_finished = num_finished

                self.log()
                self.update_lr()
                self.adjust_finetune_schedule()
                self.save_model()
                self.itr += 1
                self.clear_cache()
                self.inspect_memory()
                continue
            else:
                seed = None
                if not self._train_seeded:
                    seed = self.seed
                    self._train_seeded = True

                self.reset_env(
                    buffer_device=self.buffer_device,
                    venv=self.venv,
                    done_cache=self.done_venv,
                    seed=seed,
                )
                #self.reset_env(buffer_device=self.buffer_device)
                self.buffer.update_full_obs()
                episode_steps[:] = 0
                self.slot_mem.reset(B=self.n_envs,device=self.device)
                
                for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                    if not self.verbose and step % 100 == 0: print(f"Processed {step} of {self.n_steps}")
                    with torch.no_grad():
                        #phase_ids = np.where(episode_steps < 5,0,np.where(episode_steps < 10,1,2))

                        ####### visual input #########################
                        cond = {
                            key: torch.from_numpy(self.prev_obs_venv[key])
                            .float()
                            .to(self.device)
                            for key in self.obs_dims
                        }
                        cond['slot_mem'] = self.slot_mem
                        #cond['phase_ids'] = torch.from_numpy(phase_ids).to(self.device)
                        ## overload bug fix
                        action_samples, chains_venv = self.get_samples(cond=cond, 
                                                                    ret_device=self.buffer_device,
                                                                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                                        normalize_act_space_dimension=self.normalize_act_space_dim, 
                                                                        clip_intermediate_actions=self.clip_intermediate_actions,
                                                                        account_for_initial_stochasticity=self.account_for_initial_stochasticity)
                    
                    # Apply multi-step action
                    action_venv = action_samples[:, : self.act_steps]
                    obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                    episode_steps += self.act_steps
                    done_venv = terminated_venv | truncated_venv
                    if done_venv.any():
                        episode_steps[done_venv] = 0
                        self.slot_mem.memory[done_venv] = 0.0
                    # overload, bug fix
                    #self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, terminated_venv, truncated_venv)
                    
                    # add success info
                    # extract success info from info
                    success_once_venv = np.zeros(self.n_envs, dtype=bool)
                    success_at_end_venv = np.zeros(self.n_envs, dtype=bool)
                    episode_return_venv = np.zeros(self.n_envs, dtype=float)
                    episode_length_venv = np.zeros(self.n_envs, dtype=int)
                    episode_finished_mask_venv = np.zeros(self.n_envs, dtype=bool)
                    has_final_info = isinstance(info_venv, dict) and '_final_info' in info_venv
                    if has_final_info:
                        mask = info_venv['_final_info']
                        if isinstance(mask, torch.Tensor):
                            mask = mask.detach().cpu().numpy().astype(bool)
                        
                        if mask.any():  
                            episode_return_venv[mask] = reward_venv[mask]
                            episode_finished_mask_venv[mask] = True
                            final_info = info_venv.get('final_info', {})
                            episode_info = final_info.get('episode', {})
                            print(list(episode_info.keys()))
                            #succ = episode_info.get('success_at_end', episode_info.get('success_once', None))
                            succ_once = episode_info.get('success_once', None)
                            ep_return = episode_info.get('return', None)
                            ep_length = episode_info.get('episode_len', None)
                            if succ_once is not None:
                                if isinstance(succ_once, torch.Tensor):
                                    succ_once = succ_once.detach().cpu().numpy().astype(bool)
                                success_once_venv[mask] = succ_once[mask]
                            succ_at_end = episode_info.get('success_at_end', None)
                            if succ_at_end is not None:
                                if isinstance(succ_at_end, torch.Tensor):
                                    succ_at_end = succ_at_end.detach().cpu().numpy().astype(bool)
                                success_at_end_venv[mask] = succ_at_end[mask]
                            if ep_return is not None:
                                if isinstance(ep_return, torch.Tensor):
                                    ep_return = ep_return.detach().cpu().numpy().astype(float)
                                episode_return_venv[mask] = ep_return[mask]
                            if ep_length is not None:
                                if isinstance(ep_length, torch.Tensor):
                                    ep_length = ep_length.detach().cpu().numpy().astype(int)
                                episode_length_venv[mask] = ep_length[mask]
                            
                            # Debug output for single env case
                            #if self.n_envs == 1:
                            #    print(f"[CHK2] Success extraction: has_final_info={has_final_info}, mask={mask[0] if len(mask) > 0 else 'N/A'}, episode_keys={list(episode_info.keys())}, success_value={success_venv[0]}")
                    else:
                        # Try direct success from info
                        if isinstance(info_venv, list) and len(info_venv) > 0:
                            success_once_venv[0] = info_venv[0]['success_once']
                            success_at_end_venv[0] = info_venv[0]['success_at_end']
                            #if self.n_envs == 1:
                            #    print(f"[CHK2] Success from direct info: success={success_venv[0]}")
                        #elif self.n_envs == 1:
                            #print(f"[CHK2] No success info found in info_venv structure")

                    
                    self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, 
                                    terminated_venv, truncated_venv, success_once_venv, success_at_end_venv, episode_return_venv, episode_length_venv, episode_finished_mask_venv)
                    
                    self.prev_obs_venv = obs_venv
                    self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
                
                self.buffer.summarize_episode_reward()

                if not self.eval_mode:
                    ### bug fix
                    self.buffer: PPOFlowImgBufferGPU
                    self.buffer.update_img(obs_venv, self.model)
                    self.agent_update(verbose=self.verbose)
                
                self.log()
                self.update_lr()
                self.adjust_finetune_schedule()# update finetune scheduler of ReFlow Policy
                self.save_model()
                                
                
                self.itr += 1
                # early stopping
                if self.use_early_stop and (self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 2.0):
                    log.info(f"Your finetuning failed. success_rate={self.buffer.success_rate*100:.2f}% and avg_episode_reward={self.buffer.avg_episode_reward:.2f}")
                    exit()
                
                self.clear_cache()
                self.inspect_memory()
            
    # overload to accomodate gradaccum
    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        actor_norm=0.0
        critic_norm=0.0
        
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():

            # minibatch gradient descent
            self.model: PPOFlow
            
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, newQ_values= self.model.loss(*minibatch, 
                                                    use_bc_loss=self.use_bc_loss, 
                                                    bc_loss_type=self.bc_loss_type, 
                                                    normalize_denoising_horizon=self.normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                log.info(f"Warning: ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            loss.backward()
            # overload, bugfix to support gradient accumulation
            if (batch_id + 1) % self.grad_accumulate == 0:
                # debug the losses
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                if verbose:
                    log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")
                
                # update actor: after critic warmup update the actor less frequently but more times. 
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                # update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # release gradient accumulation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # report
                log.info(f"run grad update at batch {batch_id}")
                log.info(
                    f"approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps //self.batch_size}"
                )
        
        clip_fracs=np.mean(clipfracs_list)
        noise_stds=np.mean(noise_std_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": self.approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": self.explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor_norm": actor_norm,
                "critic_norm": critic_norm,
                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                "min_logprob_noise_std": self.model.min_logprob_denoising_std,
                "min_sampling_noise_std": self.model.min_sampling_denoising_std,
                "noise_std": noise_stds,
                "Q_values": self.Q_values   # # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values but also remember to re-define Q values in agent/finetune/reinflow/train_ppo_diffusion_img_agent.py
            }
    
    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values but also remember to re-define Q values in agent/finetune/reinflow/train_ppo_diffusion_img_agent.py
        self.Q_values = oldvalues.mean().item()
        
        duplicate_multiplier = self.minibatch_duplicate_multiplier 
        
        self.total_steps = self.n_steps * self.n_envs *  duplicate_multiplier
        
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {k: obs[k][batch_inds_b] for k in obs}, # rgb and state. overload
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                
                if (self.lr_schedule=='fixed' 
                    and self.target_kl 
                    and self.approx_kl > self.target_kl
                    and self.itr >= self.n_critic_warmup_itr  # bug fix
                ): # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch
    def save_model(self, only_save_policy_network=False):
        policy_network_state_dict = {
            "network." + key: value
            for key, value in self.model.actor_ft.policy.state_dict().items()
        }

        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            "slot_memory": self.slot_mem.state_dict(),
        }

        if only_save_policy_network:
            data["policy"] = policy_network_state_dict
        else:
            data["model"] = self.model.state_dict()
            data["policy"] = policy_network_state_dict

        def _save_checkpoint(filename: str) -> None:
            path = os.path.join(self.checkpoint_dir, filename)
            torch.save(data, path)

        _save_checkpoint("last.pt")
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            _save_checkpoint(f"state_{self.itr}.pt")
        if self.is_best_so_far:
            _save_checkpoint("best.pt")
            log.info(
                f"\n Saved best checkpoint with reward {self.current_best_reward:4.3f} "
                f"to {os.path.join(self.checkpoint_dir, 'best.pt')}\n "
            )
            self.is_best_so_far = False
    def resume_training(self):
        super().resume_training()

        if getattr(self, "wm", None) is None:
            log.warning("Checkpoint has no memory state; continuing with freshly initialized memory.")
            return

        checkpoint = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        slot_memory_state = checkpoint.get("slot_memory")
        if memory_state is not None:
            missing, unexpected = self.slot_mem.load_state_dict(slot_memory_state, strict=False)
            log.info(f"Loaded slot memory state (missing={missing}, unexpected={unexpected})")
        else:
            log.warning("Checkpoint has no memory state; continuing with freshly initialized memory.")
                
        
        