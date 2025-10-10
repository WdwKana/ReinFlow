import torch
import torch.nn as nn

class ARMemoryEncoder(nn.Module):
    def __init__(
        self,
        vis_feat_dim,         # 来自 ViT+compress 的特征维度
        action_dim,
        hidden_dim=256,
        latent_dim=128,
        num_layers=4,
        num_heads=4,
        seq_len=8,
        dropout=0.1,
    ):
        super().__init__()
        # 将三路输入对齐到 latent_dim
        self.vis_proj = nn.Linear(vis_feat_dim, latent_dim, bias=False)
        self.act_proj = nn.Linear(action_dim, latent_dim, bias=False)
        self.rew_proj = nn.Linear(1, latent_dim, bias=False)

        self.in_proj  = nn.Linear(latent_dim * 3, hidden_dim, bias=False)
        self.tok_drop = nn.Dropout(dropout)
        self.norm_in  = nn.LayerNorm(hidden_dim)

        self.tok_time = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim, bias=False)

    def forward(self, vis_seq, act_seq, rew_seq, mask=None):
        """
        vis_seq: [B, T, vis_feat_dim]
        act_seq: [B, T, action_dim]
        rew_seq: [B, T, 1]
        mask:    [B, T]  True=pad
        """
        B, T, _ = vis_seq.shape
        v = self.vis_proj(vis_seq)
        a = self.act_proj(act_seq)
        r = self.rew_proj(rew_seq)
        x = torch.cat([v, a, r], dim=-1)       # [B, T, 3*latent_dim]
        x = self.in_proj(x)

        time_ids = torch.arange(T, device=x.device)
        x = x + self.tok_time(time_ids)[None, :, :]

        x = self.norm_in(x)
        x = self.tok_drop(x)
        x = self.encoder(x, src_key_padding_mask=mask)  # [B, T, hidden_dim]

        x = self.norm_out(x)
        x = self.out_proj(x)                             # [B, T, latent_dim]

        # 取最后一个非 pad 的时刻（若无 mask，取最后一步）
        if mask is None:
            m = x[:, -1]
        else:
            last_idx = (~mask).long().sum(dim=1).clamp(min=1) - 1   # [B]
            m = x[torch.arange(B, device=x.device), last_idx]
        return m    # [B, latent_dim]