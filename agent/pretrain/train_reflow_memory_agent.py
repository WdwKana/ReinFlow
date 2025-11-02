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
# 
# 

"""
Pre-training ReFlow policy
"""
import os
import logging
import torch
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.reflow import ReFlow
class TrainReFlowAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ReFlow
        self.ema_model: ReFlow
        self._wm_cache = {}
        
        self.verbose_train=False #True #False #True #False #True #False
        self.verbose_loss= True #False #False #True #False #True #False # True
        self.verbose_test= False #True #False
        from model.common.learned_memory import PatchWorkingMemory

        self.wm = PatchWorkingMemory(
            embed_dim=cfg.memory.embed_dim,
            num_slots=self.model.network.backbone.num_patch,
            read_temperature=cfg.memory.read_temperature,
            write_temperature=cfg.memory.write_temperature,
            use_topk=cfg.memory.use_topk,
            topk=cfg.memory.topk,
            normalize=cfg.memory.normalize,
            device=self.device,
        )
        self.wm = self.wm.to(self.device)
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test=  False                      # when toggled, test and then exit code.
            
            self.test_denoising_steps=4                 #self.model.max_denoising_steps!!!!!!!!!!!!
            
            self.test_clip_intermediate_actions=True    #True    # this will affect performance. !!!!!!!!!!!!
            
            self.test_model_type='ema'                  # minor difference!!!!!!!!!!!!
    def _ensure_wm_batch(self, batch_size: int):
        if self.wm.memory.shape[0] != batch_size:
            self.wm.reset(B=batch_size, device=self.device)

    def _restore_memory(self, episode_id: torch.Tensor, episode_start: torch.Tensor):
        B = episode_id.shape[0]
        episode_id_cpu = episode_id.detach().cpu().tolist()
        episode_start_cpu = episode_start.detach().cpu().tolist()

        for i in range(B):
            eid = int(episode_id_cpu[i])
            if episode_start_cpu[i] or eid not in self._wm_cache:
                self.wm.memory[i].zero_()
            else:
                cached = self._wm_cache[eid].to(self.wm.memory.device, non_blocking=True)
                self.wm.memory[i].copy_(cached)

    def _stash_memory(self, episode_id: torch.Tensor, episode_end: torch.Tensor):
        B = episode_id.shape[0]
        episode_id_cpu = episode_id.detach().cpu().tolist()
        episode_end_cpu = episode_end.detach().cpu().tolist()

        with torch.no_grad():
            for i in range(B):
                eid = int(episode_id_cpu[i])
                if episode_end_cpu[i]:
                    self._wm_cache.pop(eid, None)
                else:
                    self._wm_cache[eid] = self.wm.memory[i].detach().cpu().clone()
    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        # here *batch_data = actions, observation, according to StitchedSequenceDataset
        act, cond = batch_data
        episode_id = cond.pop('episode_id')
        episode_start = cond.pop("episode_start")
        episode_end = cond.pop("episode_end")
        #cond.pop("episode_step", None)
        self._ensure_wm_batch(batch_size=act.shape[0])
        self._restore_memory(episode_id, episode_start)
        cond['wm'] = self.wm
        (xt, t), v = self.model.generate_target(act)  # here *batch_train = actions, observation, according to StitchedSequenceDataset

        loss= self.model.loss(xt, t, cond, v)
        if self.model.training:
            with torch.no_grad():
                self.wm.memory = self.wm.memory.detach()
            self._stash_memory(episode_id, episode_end)
        return loss
    
    def inference(self, cond:dict):
        '''for testing purpose'''
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
        return samples

    def save_model(self):
        """
        saves model, ema and memory to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "memory": self.wm.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model with memory to {savepath}\n")

    def save_best_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "memory": self.wm.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best.pt")
        torch.save(data, savepath)
        log.info(
            f"Saved the best model with memory to {savepath}\t It has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}."
        )

    def save_best_ema_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "memory": self.wm.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best_ema.pt")
        torch.save(data, savepath)
        log.info(
            f"Saved the best EMA model with memory to {savepath}, which has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}."
        )

    def save_last_model(self):
        """for resume purpose"""
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "memory": self.wm.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"last.pt")
        torch.save(data, savepath)

    def load(self, epoch, custom_path=None):
        """
        loads model, ema and memory from disk
        """
        if custom_path:
            loadpath = os.path.join(custom_path)
        else:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")

        data = torch.load(loadpath, weights_only=True)

        self.epoch = data["epoch"]
        self.first_epoch = self.epoch + 1
        log.info(
            f"Resume from self.epoch={self.epoch}. Will start from self.first_epoch={self.first_epoch} and train for another {self.n_epochs} epochs. "
        )
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        memory_state = data.get("memory")
        if memory_state is not None:
            missing, unexpected = self.wm.load_state_dict(memory_state, strict=False)
            log.info(f"Loaded memory state (missing={missing}, unexpected={unexpected})")
        else:
            log.warning("Checkpoint has no memory state; continuing with freshly initialized memory.")
    
    
            
            