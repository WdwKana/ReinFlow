"""
DFBT的视觉版本：用ViT编码图像 + Transformer预测下一步潜变量
适配Mikasa数据集（图像 + 状态）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model.common.vit import VitEncoder
from typing import Tuple
import numpy as np


# ============================================================================
# 1. 视觉自编码器
# ============================================================================

class VisionAutoEncoder(nn.Module):
    """视觉自编码器：RGB → latent → RGB特征"""
    def __init__(
        self,
        backbone: VitEncoder,
        latent_dim: int = 256,
        compress_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        
        # Encoder: ViT patches → latent
        self.encoder = nn.Sequential(
            nn.Linear(backbone.repr_dim, compress_dim),
            nn.LayerNorm(compress_dim),
            nn.ReLU(),
            nn.Linear(compress_dim, latent_dim),
            nn.ReLU(),
        )
        
        # Decoder: latent → ViT features (用于重建loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, compress_dim),
            nn.ReLU(),
            nn.Linear(compress_dim, backbone.repr_dim),
        )
    
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, C, H, W) uint8 or float32 images
        Returns:
            z: (B, latent_dim)
        """
        # 确保是float并归一化
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()
        
        rgb = rgb / 255.0 - 0.5
        feat = self.backbone(rgb, flatten=True)  # (B, repr_dim)
        z = self.encoder(feat)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """重建ViT特征（用于训练loss）"""
        return self.decoder(z)
    
    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的前向传播（用于计算重建loss）
        Returns:
            feat_rec: 重建的特征
            feat_gt: 真实的特征
        """
        if rgb.dtype == torch.uint8:
            rgb = rgb.float()
        
        rgb = rgb / 255.0 - 0.5
        feat_gt = self.backbone(rgb, flatten=True)
        z = self.encoder(feat_gt)
        feat_rec = self.decoder(z)
        return feat_rec, feat_gt


# ============================================================================
# 2. DFBT Belief Model（结合视觉和状态）
# ============================================================================

class DirectForecastingBeliefVisionState(nn.Module):
    """
    DFBT的视觉+状态版：预测下一步的视觉潜变量和状态
    Input: 
        - z_img_current: 当前视觉潜变量 (B, latent_dim)
        - state_current: 当前状态 (B, state_dim)
        - state_history: 历史状态 (B, seq_len, state_dim)
        - actions: 动作历史 (B, seq_len, action_dim)
        - rewards: 奖励历史 (B, seq_len, 1)
    Output: 
        - z_img_next: 预测的下一步视觉潜变量 (B, latent_dim)
        - state_next: 预测的下一步状态 (B, state_dim)
    """
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        seq_len: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.latent_proj = nn.Linear(latent_dim, latent_dim)
        self.state_proj = nn.Linear(state_dim, latent_dim)
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.reward_proj = nn.Linear(1, latent_dim)
        
        # Token embedding: [z_img, state, action, reward] concat
        # 每个时间步的token维度：latent_dim * 4
        self.token_emb = nn.Linear(latent_dim * 4, hidden_dim)
        self.timestep_emb = nn.Embedding(seq_len, hidden_dim)
        
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(residual_dropout)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=attention_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影（分别预测图像latent和state）
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj_img = nn.Linear(hidden_dim, latent_dim)
        self.out_proj_state = nn.Linear(hidden_dim, state_dim)
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        z_img_current: torch.Tensor,   # (B, latent_dim) - 当前视觉latent
        state_current: torch.Tensor,   # (B, state_dim) - 当前状态
        state_history: torch.Tensor,   # (B, seq_len, state_dim) - 状态历史
        actions: torch.Tensor,         # (B, seq_len, action_dim)
        rewards: torch.Tensor,         # (B, seq_len, 1)
        masks: torch.Tensor = None     # (B, seq_len) - padding mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_img_next_pred: (B, latent_dim) - 预测的下一步视觉latent
            state_next_pred: (B, state_dim) - 预测的下一步状态
        """
        B, L = actions.shape[:2]
        device = actions.device
        
        # 1. 扩展当前视觉latent到序列长度
        z_img_seq = z_img_current.unsqueeze(1).repeat(1, L, 1)  # (B, L, latent_dim)
        
        # 2. 投影各种输入
        z_img_emb = self.latent_proj(z_img_seq)         # (B, L, latent_dim)
        state_emb = self.state_proj(state_history)      # (B, L, latent_dim)
        actions_emb = self.action_proj(actions)         # (B, L, latent_dim)
        rewards_emb = self.reward_proj(rewards)         # (B, L, latent_dim)
        
        # 3. 拼接token
        tokens = torch.cat([z_img_emb, state_emb, actions_emb, rewards_emb], dim=-1)  # (B, L, latent_dim*4)
        tokens = self.token_emb(tokens)  # (B, L, hidden_dim)
        
        # 4. 添加位置编码
        timesteps = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)  # (B, L)
        tokens = tokens + self.timestep_emb(timesteps)
        
        tokens = self.norm_in(tokens)
        tokens = self.dropout(tokens)
        
        # 5. Transformer encoding
        if masks is not None:
            tokens = self.transformer(tokens, src_key_padding_mask=masks)
        else:
            tokens = self.transformer(tokens)
        
        # 6. 输出预测（取最后一个时间步）
        tokens = self.norm_out(tokens)
        last_token = tokens[:, -1, :]  # (B, hidden_dim)
        
        z_img_next_pred = self.out_proj_img(last_token)   # (B, latent_dim)
        state_next_pred = self.out_proj_state(last_token)  # (B, state_dim)
        
        return z_img_next_pred, state_next_pred


# ============================================================================
# 3. 完整的Mikasa-DFBT模型
# ============================================================================

class MikasaDFBT(nn.Module):
    """完整的Mikasa-DFBT模型：处理图像+状态"""
    def __init__(
        self,
        backbone: VitEncoder,
        action_dim: int,
        state_dim: int,
        seq_len: int = 10,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.autoencoder = VisionAutoEncoder(backbone, latent_dim)
        self.belief_model = DirectForecastingBeliefVisionState(
            latent_dim=latent_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
    
    def forward(
        self,
        img_history: torch.Tensor,     # (B, L, 3, H, W) uint8
        state_history: torch.Tensor,   # (B, L, state_dim)
        actions: torch.Tensor,         # (B, L, action_dim)
        rewards: torch.Tensor,         # (B, L, 1)
        img_next: torch.Tensor,        # (B, 3, H, W) uint8
        state_next: torch.Tensor,      # (B, state_dim)
        masks: torch.Tensor = None
    ) -> dict:
        """
        训练时的forward，计算重建loss和预测loss
        """
        B, L = img_history.shape[:2]
        
        # 1. Encode图像历史
        img_flat = img_history.reshape(B * L, *img_history.shape[2:])
        z_img_history = self.autoencoder.encode(img_flat).reshape(B, L, -1)
        
        # 2. 取最后一个作为当前状态
        z_img_current = z_img_history[:, -1]  # (B, latent_dim)
        state_current = state_history[:, -1]  # (B, state_dim)
        
        # 3. 预测下一步
        z_img_next_pred, state_next_pred = self.belief_model(
            z_img_current,
            state_current,
            state_history,
            actions,
            rewards,
            masks
        )
        
        # 4. Ground truth
        z_img_next_gt = self.autoencoder.encode(img_next)
        
        # 5. 重建loss（可选，用于正则化）
        feat_rec, feat_gt = self.autoencoder(img_next)
        recon_loss = nn.functional.mse_loss(feat_rec, feat_gt)
        
        # 6. 预测loss
        img_pred_loss = nn.functional.mse_loss(z_img_next_pred, z_img_next_gt)
        state_pred_loss = nn.functional.mse_loss(state_next_pred, state_next)
        
        pred_loss = img_pred_loss + 0.5 * state_pred_loss  # 可调权重
        
        return {
            'pred_loss': pred_loss,
            'img_pred_loss': img_pred_loss,
            'state_pred_loss': state_pred_loss,
            'recon_loss': recon_loss,
            'total_loss': pred_loss + 0.1 * recon_loss,  # 可调权重
            'z_img_next_pred': z_img_next_pred,
            'state_next_pred': state_next_pred,
            'z_img_next_gt': z_img_next_gt,
            'state_next_gt': state_next
        }
    
    @torch.no_grad()
    def get_belief(
        self,
        img_history: torch.Tensor,     # (B, L, 3, H, W) uint8
        state_history: torch.Tensor,   # (B, L, state_dim)
        actions: torch.Tensor,         # (B, L, action_dim)
        rewards: torch.Tensor,         # (B, L, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理时获取belief state（预测的下一步latent和状态）
        
        Returns:
            z_img_belief: (B, latent_dim) - 预测的视觉belief
            state_belief: (B, state_dim) - 预测的状态belief
        """
        B, L = img_history.shape[:2]
        
        # Encode图像历史
        img_flat = img_history.reshape(B * L, *img_history.shape[2:])
        z_img_history = self.autoencoder.encode(img_flat).reshape(B, L, -1)
        
        z_img_current = z_img_history[:, -1]
        state_current = state_history[:, -1]
        
        # 预测下一步
        z_img_belief, state_belief = self.belief_model(
            z_img_current,
            state_current,
            state_history,
            actions,
            rewards
        )
        
        return z_img_belief, state_belief


# ============================================================================
# 4. Dataset加载器
# ============================================================================

class DFBTDataset(Dataset):
    """加载转换后的DFBT数据"""
    def __init__(self, data_path: str):
        print(f"Loading DFBT dataset from {data_path}...")
        data = np.load(data_path)
        
        # 图像历史：(N, history_len, 3, H, W) uint8
        self.img_history = data['img_history']
        
        # 状态历史：(N, history_len, state_dim)
        self.state_history = torch.from_numpy(data['state_history']).float()
        
        # 动作历史：(N, history_len, action_dim)
        self.act_history = torch.from_numpy(data['act_history']).float()
        
        # 奖励历史：(N, history_len, 1)
        self.rew_history = torch.from_numpy(data['rew_history']).float()
        
        # 下一步图像：(N, 3, H, W) uint8
        self.img_next = data['img_next']
        
        # 下一步状态：(N, state_dim)
        self.state_next = torch.from_numpy(data['state_next']).float()
        
        print(f"  ✓ Loaded {len(self.img_history)} samples")
        print(f"  ✓ Image shape: {self.img_history.shape[2:]}")
        print(f"  ✓ State dim: {self.state_history.shape[-1]}")
        print(f"  ✓ Action dim: {self.act_history.shape[-1]}")
        print(f"  ✓ Sequence length: {self.img_history.shape[1]}")
    
    def __len__(self):
        return len(self.img_history)
    
    def __getitem__(self, idx):
        # 图像保持uint8格式，在模型中再转换（节省内存）
        return (
            self.img_history[idx],    # (seq_len, 3, H, W) uint8
            self.state_history[idx],  # (seq_len, state_dim) float32
            self.act_history[idx],    # (seq_len, action_dim) float32
            self.rew_history[idx],    # (seq_len, 1) float32
            self.img_next[idx],       # (3, H, W) uint8
            self.state_next[idx]      # (state_dim,) float32
        )


# ============================================================================
# 5. 测试代码
# ============================================================================

if __name__ == "__main__":
    # 创建测试数据
    B = 4
    L = 10
    H, W = 96, 96
    state_dim = 25
    action_dim = 8
    latent_dim = 256
    
    # 创建backbone
    from model.common.vit import VitEncoderConfig
    backbone = VitEncoder(
        obs_shape=[3, H, W],
        cfg=VitEncoderConfig(
            patch_size=8,
            depth=2,
            embed_dim=128,
            num_heads=4
        ),
        num_channel=3,
        img_h=H,
        img_w=W
    )
    
    # 创建模型
    model = MikasaDFBT(
        backbone=backbone,
        action_dim=action_dim,
        state_dim=state_dim,
        seq_len=L,
        latent_dim=latent_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
    )
    
    # 创建测试输入
    img_hist = torch.randint(0, 256, (B, L, 3, H, W), dtype=torch.uint8)
    state_hist = torch.randn(B, L, state_dim)
    actions = torch.randn(B, L, action_dim)
    rewards = torch.randn(B, L, 1)
    img_next = torch.randint(0, 256, (B, 3, H, W), dtype=torch.uint8)
    state_next = torch.randn(B, state_dim)
    
    # Forward pass
    outputs = model(img_hist, state_hist, actions, rewards, img_next, state_next)
    
    print("\n=== Test Results ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, mean={value.mean().item():.4f}")
        else:
            print(f"{key}: {value}")
    
    # 测试belief获取
    z_belief, s_belief = model.get_belief(img_hist, state_hist, actions, rewards)
    print(f"\nBelief shapes:")
    print(f"  z_img_belief: {z_belief.shape}")
    print(f"  state_belief: {s_belief.shape}")
    
    print("\n✅ All tests passed!")