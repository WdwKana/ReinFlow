"""
Pre-training data loader. 
Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- 

we always normalize the data when pre-processing it with a different script, 

and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
from typing import NamedTuple

import numpy as np
import torch
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class SequenceSample(NamedTuple):
    start: int
    num_before_start: int
    episode_id: int
    episode_step: int
    is_episode_start: bool
    is_episode_end: bool


class StitchedSequenceMemoryDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=-1,
        use_img=False,
        device="cuda:0",
        return_episode_info=True,
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        
        print(f"max_n_episodes={max_n_episodes}")
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.return_episode_info = return_episode_info

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        if max_n_episodes == -1:
            max_n_episodes = len(dataset["traj_lengths"])
            log.info(f"max_n_episodes specified as -1, fall back to maximum value {max_n_episodes}") 
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)
        # Set up indices for sampling
        self._all_indices = self.make_indices(traj_lengths, horizon_steps)
        self.indices = list(self._all_indices)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, obs_dim)
        
        self.actions = (
            torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)

        log.info(f"Successfully loaded dataset from {dataset_path}")
        n_eps=min(max_n_episodes, len(traj_lengths))
        if n_eps<=0:
            raise ValueError(f"number of episodes less than 1, check where is wrong...")
        log.info(f"Number of episodes: {n_eps}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                device
            )  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")
        log.info(f"Finished creating {self.__class__.__name__} from {dataset_path}")

        self._build_cached_metadata()

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        sample_index = self.indices[idx]
        start = sample_index.start
        num_before_start = sample_index.num_before_start
        end = start + self.horizon_steps
        # print(f"start={start}, num_before_start={num_before_start}, (start - num_before_start) : (start + 1)={(start - num_before_start)} : {(start + 1)}")
        # print(f"self.states length: {self.states.shape}")
        
        # print(f"self.states[673463 : 673758]")
        # try:
        #     print(self.states[673463])
        # except RuntimeError as e:
        #     print(f"RuntimeError: {e}")
        
        # print(self.states[673463])
        # exit()

        
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        self._add_episode_info(conditions, idx)
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for episode_id, traj_length in enumerate(traj_lengths):
            episode_start = cur_traj_index
            episode_end = cur_traj_index + traj_length
            max_start = episode_end - horizon_steps
            for start in range(episode_start, max_start + 1):
                num_before_start = start - episode_start
                end = start + horizon_steps
                indices.append(
                    SequenceSample(
                        start=start,
                        num_before_start=num_before_start,
                        episode_id=episode_id,
                        episode_step=num_before_start,
                        is_episode_start=(num_before_start == 0),
                        is_episode_end=(end >= episode_end),
                    )
                )
            cur_traj_index += traj_length
        return indices

    def _build_cached_metadata(self):
        if not self.return_episode_info:
            self._episode_ids = None
            self._episode_steps = None
            self._episode_starts = None
            self._episode_ends = None
            return

        device = self.states.device if hasattr(self, "states") else torch.device(self.device)
        if len(self.indices) == 0:
            self._episode_ids = torch.empty(0, dtype=torch.long, device=device)
            self._episode_steps = torch.empty(0, dtype=torch.long, device=device)
            self._episode_starts = torch.empty(0, dtype=torch.bool, device=device)
            self._episode_ends = torch.empty(0, dtype=torch.bool, device=device)
            return

        self._episode_ids = torch.tensor(
            [sample.episode_id for sample in self.indices],
            dtype=torch.long,
            device=device,
        )
        self._episode_steps = torch.tensor(
            [sample.episode_step for sample in self.indices],
            dtype=torch.long,
            device=device,
        )
        self._episode_starts = torch.tensor(
            [sample.is_episode_start for sample in self.indices],
            dtype=torch.bool,
            device=device,
        )
        self._episode_ends = torch.tensor(
            [sample.is_episode_end for sample in self.indices],
            dtype=torch.bool,
            device=device,
        )

    def _add_episode_info(self, conditions, idx):
        if not self.return_episode_info or self._episode_ids is None:
            return
        conditions["episode_id"] = self._episode_ids[idx]
        conditions["episode_step"] = self._episode_steps[idx]
        conditions["episode_start"] = self._episode_starts[idx]
        conditions["episode_end"] = self._episode_ends[idx]

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        total = len(self._all_indices)
        num_train = int(total * train_split)
        all_positions = list(range(total))
        random.shuffle(all_positions)
        train_positions = sorted(all_positions[:num_train])
        val_positions = sorted(all_positions[num_train:])
        self.set_indices(train_positions)
        return val_positions

    def set_indices(self, subset_indices):
        self.indices = [self._all_indices[i] for i in subset_indices]
        self._build_cached_metadata()

    def __len__(self):
        return len(self.indices)


class StitchedSequenceMemoryQLearningDataset(StitchedSequenceMemoryDataset):
    """
    Extends StitchedSequenceMemoryDataset to include rewards and dones for Q learning
    
    **Returns:**
    batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
    
    Do not load the last step of **truncated** episodes since we do not have the correct next state for the final step of each episode. 
    Truncation can be determined by terminal=False by the end of the episode.
    """

    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        device="cuda:0",
        get_mc_return=False,
        **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device).float()
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")
        
        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = (
                        traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for episode_id, traj_length in enumerate(traj_lengths):
            episode_start = cur_traj_index
            episode_end = cur_traj_index + traj_length
            max_start = episode_end - horizon_steps
            if not self.dones[episode_end - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            for start in range(episode_start, max_start + 1):
                num_before_start = start - episode_start
                end = start + horizon_steps
                indices.append(
                    SequenceSample(
                        start=start,
                        num_before_start=num_before_start,
                        episode_id=episode_id,
                        episode_step=num_before_start,
                        is_episode_start=(num_before_start == 0),
                        is_episode_end=(end >= episode_end),
                    )
                )
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        start = sample_index.start
        num_before_start = sample_index.num_before_start
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start : (start + 1)]
        dones = self.dones[start : (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps) : start
                + 1
                + self.horizon_steps
            ]  # even if this uses the first state(s) of the next episode, done=True will prevent bootstrapping. We have already filtered out cases where done=False but end of episode (truncation).
        else:
            # prevents indexing error, but ignored since done=True
            next_states = torch.zeros_like(states)

        # stack obs history
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        next_states = torch.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        self._add_episode_info(conditions, idx)
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start : (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch
