"""
将Mikasa专家数据转换为DFBT训练格式
处理图像数据 + 轨迹分割
"""
from pathlib import Path
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm

class MikasaDFBTDataConverter:
    def __init__(
        self,
        data_path: str,
        history_len: int = 10,  # DFBT的序列长度
        save_path: str = None,
        use_image: bool = True,  # 是否使用图像
        resize_image: Tuple[int, int] = None,  # 是否resize图像 (H, W)
    ):
        self.data_path = Path(data_path)
        self.history_len = history_len
        self.use_image = use_image
        self.resize_image = resize_image
        if save_path is None:
            self.save_path = self.data_path.parent / "dfbt_format"
        else:
            self.save_path = Path(save_path)  # ← 添加这行
        # self.save_path = save_path or self.data_path.parent / "dfbt_format"
        self.save_path.mkdir(exist_ok=True, parents=True)
    
    def load_mikasa_data(self) -> Dict:
        """
        加载Mikasa专家数据
        
        Keys:
            states: (N, 25) - 本体感知状态
            actions: (N, 8) - 动作
            rewards: (N,) - 奖励
            terminals: (N,) - episode结束标志
            images: (N, 3, 128, 128) - RGB图像 (CHW格式, uint8)
            traj_lengths: (num_traj,) - 每条轨迹的长度
        """
        print(f"Loading Mikasa data from {self.data_path}...")
        data = np.load(self.data_path)
        
        print("\n=== Mikasa Dataset Info ===")
        for key in data.keys():
            if hasattr(data[key], 'shape'):
                print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
            else:
                print(f"  {key}: {data[key]}")
        
        return {
            'states': data['states'],      # (N, 25)
            'actions': data['actions'],    # (N, 8)
            'rewards': data['rewards'],    # (N,)
            'terminals': data['terminals'],# (N,)
            'images': data['images'],      # (N, 3, 128, 128)
            'traj_lengths': data['traj_lengths']  # (num_traj,)
        }
    
    def split_trajectories(self, data: Dict) -> List[Dict]:
        """
        根据traj_lengths将数据分割成轨迹
        
        Returns:
            List of trajectory dicts, each containing:
                - states: (traj_len, 25)
                - actions: (traj_len, 8)
                - rewards: (traj_len,)
                - images: (traj_len, 3, H, W)
        """
        traj_lengths = data['traj_lengths']
        num_trajs = len(traj_lengths)
        
        trajectories = []
        start_idx = 0
        
        print(f"\nSplitting {num_trajs} trajectories...")
        for traj_idx in tqdm(range(num_trajs)):
            traj_len = int(traj_lengths[traj_idx])
            end_idx = start_idx + traj_len
            
            traj = {
                'states': data['states'][start_idx:end_idx],
                'actions': data['actions'][start_idx:end_idx],
                'rewards': data['rewards'][start_idx:end_idx],
                'images': data['images'][start_idx:end_idx],
                'length': traj_len
            }
            
            trajectories.append(traj)
            start_idx = end_idx
        
        print(f"Split into {len(trajectories)} trajectories")
        print(f"  Min length: {min(t['length'] for t in trajectories)}")
        print(f"  Max length: {max(t['length'] for t in trajectories)}")
        print(f"  Avg length: {np.mean([t['length'] for t in trajectories]):.1f}")
        
        return trajectories
    
    def resize_images_if_needed(self, images: np.ndarray) -> np.ndarray:
        """
        可选：将图像resize到指定大小（用于减少计算量）
        
        Args:
            images: (N, 3, H_orig, W_orig)
        Returns:
            images_resized: (N, 3, H_new, W_new)
        """
        if self.resize_image is None:
            return images
        
        import cv2
        N, C, H_orig, W_orig = images.shape
        H_new, W_new = self.resize_image
        
        if (H_orig, W_orig) == (H_new, W_new):
            return images
        
        print(f"\nResizing images from {H_orig}x{W_orig} to {H_new}x{W_new}...")
        
        images_resized = np.zeros((N, C, H_new, W_new), dtype=images.dtype)
        
        for i in tqdm(range(N)):
            # 转换为HWC格式用于cv2
            img_hwc = np.transpose(images[i], (1, 2, 0))  # (H, W, C)
            img_resized = cv2.resize(img_hwc, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            images_resized[i] = np.transpose(img_resized, (2, 0, 1))  # 转回CHW
        
        return images_resized
    
    def create_sequences_from_trajectories(
        self, 
        trajectories: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从轨迹列表创建DFBT训练序列（滑动窗口）
        
        对每条轨迹:
            如果长度 > history_len:
                创建 (length - history_len) 个样本
                每个样本:
                    输入: 过去history_len步的 (img, state, action, reward)
                    输出: 第history_len+1步的 (img, state)
        
        Returns:
            img_history: (num_seq, history_len, 3, H, W)
            state_history: (num_seq, history_len, 25)
            act_history: (num_seq, history_len, 8)
            rew_history: (num_seq, history_len, 1)
            img_next: (num_seq, 3, H, W)
            state_next: (num_seq, 25)
        """
        img_seqs = []
        state_seqs = []
        act_seqs = []
        rew_seqs = []
        img_nexts = []
        state_nexts = []
        
        print(f"\nCreating sequences (history_len={self.history_len})...")
        
        for traj in tqdm(trajectories):
            traj_len = traj['length']
            
            # 跳过太短的轨迹
            if traj_len <= self.history_len:
                continue
            
            images = traj['images']
            states = traj['states']
            actions = traj['actions']
            rewards = traj['rewards']
            
            # 对每条轨迹创建滑动窗口序列
            for i in range(traj_len - self.history_len):
                # 输入：过去history_len步
                img_seqs.append(images[i:i+self.history_len])
                state_seqs.append(states[i:i+self.history_len])
                act_seqs.append(actions[i:i+self.history_len])
                rew_seqs.append(rewards[i:i+self.history_len])
                
                # 输出：下一步
                img_nexts.append(images[i+self.history_len])
                state_nexts.append(states[i+self.history_len])
        
        print(f"\nCreated {len(img_seqs)} training sequences")
        
        return (
            np.array(img_seqs),      # (N, history_len, 3, H, W)
            np.array(state_seqs),    # (N, history_len, 25)
            np.array(act_seqs),      # (N, history_len, 8)
            np.array(rew_seqs)[..., None],  # (N, history_len, 1)
            np.array(img_nexts),     # (N, 3, H, W)
            np.array(state_nexts)    # (N, 25)
        )
    
    def save_dfbt_dataset(
        self, 
        img_seq, state_seq, act_seq, rew_seq, 
        img_next, state_next
    ):
        """保存DFBT格式数据"""
        save_file = self.save_path / "dfbt_train_data.npz"
        
        print(f"\nSaving DFBT dataset...")
        np.savez_compressed(
            save_file,
            img_history=img_seq,
            state_history=state_seq,
            act_history=act_seq,
            rew_history=rew_seq,
            img_next=img_next,
            state_next=state_next
        )
        
        print(f"\n=== Saved DFBT Dataset ===")
        print(f"  Path: {save_file}")
        print(f"  img_history: {img_seq.shape}")
        print(f"  state_history: {state_seq.shape}")
        print(f"  act_history: {act_seq.shape}")
        print(f"  rew_history: {rew_seq.shape}")
        print(f"  img_next: {img_next.shape}")
        print(f"  state_next: {state_next.shape}")
        
        # 计算文件大小
        file_size_mb = save_file.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    def convert(self):
        """执行完整转换流程"""
        # 1. 加载数据
        data = self.load_mikasa_data()
        
        # 2. 可选：resize图像
        if self.resize_image is not None:
            data['images'] = self.resize_images_if_needed(data['images'])
        
        # 3. 分割轨迹
        trajectories = self.split_trajectories(data)
        
        # 4. 创建序列
        img_seq, state_seq, act_seq, rew_seq, img_next, state_next = \
            self.create_sequences_from_trajectories(trajectories)
        
        # 5. 保存
        self.save_dfbt_dataset(
            img_seq, state_seq, act_seq, rew_seq,
            img_next, state_next
        )
        
        print("\n✅ Conversion completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Mikasa expert data to DFBT format")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Mikasa .npz file")
    parser.add_argument("--history_len", type=int, default=10,
                        help="Length of history sequence for DFBT")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save DFBT format data")
    parser.add_argument("--resize_image", type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help="Resize images to H W (e.g., --resize_image 96 96)")
    args = parser.parse_args()
    
    converter = MikasaDFBTDataConverter(
        data_path=args.data_path,
        history_len=args.history_len,
        save_path=args.save_path,
        resize_image=tuple(args.resize_image) if args.resize_image else None
    )
    
    converter.convert()