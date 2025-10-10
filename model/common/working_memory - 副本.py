import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):
    """
    Single-vector working memory for RememberShapeAndColor task.
    - Writes target prototype during observation phase (steps 0-4)
    - Maintains memory during delay phase (steps 5-9)
    - Attends to matching patches during selection phase (steps 10+)
    """

    def __init__(self, embed_dim, ema=0.9, temperature=0.07, normalize=True, device=None, dtype=torch.float32):
        super().__init__()
        self.E = int(embed_dim)
        self.ema = float(ema)
        self.T = float(temperature)
        self.normalize = normalize
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # Per-env memory: [B, E]
        self.register_buffer("m", torch.zeros(0, self.E, dtype=dtype, device=device))

    def _ensure_b(self, B, device, dtype):
        if self.m.numel() == 0 or self.m.shape[0] != B:
            self.m = torch.zeros(B, self.E, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B=None, device=None, dtype=None):
        """Reset memory (call at episode start or when batch size changes)"""
        if B is None and self.m.numel() > 0:
            B = self.m.shape[0]
        if B is not None:
            device = device or self.m.device
            dtype = dtype or self.m.dtype
            self.m = torch.zeros(B, self.E, device=device, dtype=dtype)
        else:
            self.m.zero_()

    @torch.no_grad()
    def write(self, tokens, env_indices=None):
        """
        Write target prototype from observation phase.
        tokens: [B, P, E] 或 [num_selected, P, E]
        env_indices: 可选，torch.Tensor bool mask，指示哪些环境需要写入
        """
        B, P, E = tokens.shape
        
        # 如果传入的是部分环境的 tokens，需要知道它们在完整 batch 中的位置
        if env_indices is None:
            # 全量更新
            self._ensure_b(B, tokens.device, tokens.dtype)
            target_indices = torch.arange(B, device=tokens.device)
        else:
            # 部分更新：env_indices 是 bool mask [full_B]
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_indices = torch.where(env_indices)[0]  # 转换为索引
        
        proto = tokens.mean(dim=1)  # [num_selected, E]
        if self.normalize:
            proto = F.normalize(proto, dim=-1)
        
        # EMA 更新
        if self.m[target_indices].abs().sum() < 1e-6:  # 首次写入
            self.m[target_indices] = proto
        else:
            m_old = F.normalize(self.m[target_indices], dim=-1) if self.normalize else self.m[target_indices]
            self.m[target_indices] = self.ema * m_old + (1.0 - self.ema) * proto

    def attend(self, tokens, env_indices=None):
        """
        Compute attention weights for selection phase.
        tokens: [B, P, E] 或 [num_selected, P, E]
        env_indices: 可选，torch.Tensor bool mask
        returns: weights [num_selected, P]
        """
        B, P, E = tokens.shape
        
        if env_indices is None:
            target_indices = torch.arange(B, device=tokens.device)
        else:
            target_indices = torch.where(env_indices)[0]
        
        q = F.normalize(tokens, dim=-1) if self.normalize else tokens  # [num_selected, P, E]
        k = self.m[target_indices]  # [num_selected, E]
        k = F.normalize(k, dim=-1) if self.normalize else k
        
        scores = torch.einsum("bpe,be->bp", q, k)  # [num_selected, P]
        weights = torch.softmax(scores / self.T, dim=1)
        return weights