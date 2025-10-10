# ReinFlow/model/common/working_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkingMemory(nn.Module):
    def __init__(self, embed_dim, ema=0.9, temperature=0.07, normalize=True, device=None, dtype=torch.float32,
                 write_temperature=None, alpha_min=0.02, alpha_max=0.25):
        super().__init__()
        self.E = int(embed_dim)
        self.ema = float(ema)
        self.T = float(temperature)                      # the temperature of attend
        self.Tw = float(write_temperature or self.T)     # the temperature of write
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.normalize = normalize
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.register_buffer("m", torch.zeros(0, self.E, dtype=dtype, device=device))

    def _ensure_b(self, B, device, dtype):
        if self.m.numel() == 0 or self.m.shape[0] != B:
            self.m = torch.zeros(B, self.E, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B=None, device=None, dtype=None):
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
        Memory-guided selective aggregation + confidence-gated update.
        tokens: [B, P, E] or [num_selected, P, E]
        env_indices: optional bool mask, indicating the positions to write in the full batch
        """
        B_sel, P, E = tokens.shape
        if env_indices is None:
            self._ensure_b(B_sel, tokens.device, tokens.dtype)
            target_idx = torch.arange(B_sel, device=tokens.device)
        else:
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_idx = torch.where(env_indices)[0]

        m_sel = self.m[target_idx]  # [B_sel, E]
        # whether the memory is empty (sum of L1 norm is close to zero)
        empty = (m_sel.abs().sum(dim=1, keepdim=True) < 1e-6)  # [B_sel,1]

        # normalization
        t_norm = F.normalize(tokens, dim=-1) if self.normalize else tokens
        m_norm = F.normalize(m_sel, dim=-1) if self.normalize else m_sel
        # for empty memory, avoid NaN: zero vector does not produce bias, because softmax(0)=uniform
        m_norm = torch.where(empty, torch.zeros_like(m_norm), m_norm)

        # memory-guided weights
        # scores: [B_sel, P]
        scores = torch.einsum("bpe,be->bp", t_norm, m_norm) / self.Tw
        w = torch.softmax(scores, dim=1)  # [B_sel, P]
        # candidate prototype
        proto = torch.einsum("bp,bpe->be", w, tokens)  # [B_sel, E]
        if self.normalize:
            proto = F.normalize(proto, dim=-1)

        # kurtosis -> confidence gate
        # w_max ∈ [1/P, 1]; mapped to [0,1]
        w_max = w.max(dim=1).values
        peak = (w_max - (1.0 / P)) / (1.0 - (1.0 / P) + 1e-6)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * peak  # [B_sel]
        alpha = alpha.unsqueeze(-1)

        # update (equivalent to uniform mean write when memory is empty)
        if self.normalize:
            m_old = F.normalize(m_sel, dim=-1)
        else:
            m_old = m_sel
        m_new = (1.0 - alpha) * m_old + alpha * proto
        if self.normalize:
            m_new = F.normalize(m_new, dim=-1)

        self.m[target_idx] = m_new

    def attend(self, tokens, env_indices=None):
        """
        read: use memory vectors to compute attention weights for current tokens.
        returns: weights [num_selected, P]
        """
        B_sel, P, E = tokens.shape
        if env_indices is None:
            target_idx = torch.arange(self.m.shape[0], device=tokens.device)[:B_sel]
        else:
            target_idx = torch.where(env_indices)[0]

        t_norm = F.normalize(tokens, dim=-1) if self.normalize else tokens
        k = self.m[target_idx]
        k = F.normalize(k, dim=-1) if self.normalize else k
        scores = torch.einsum("bpe,be->bp", t_norm, k) / self.T
        weights = torch.softmax(scores, dim=1)
        return weights