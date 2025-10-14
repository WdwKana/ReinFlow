# model/common/learned_momory.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedWorkingMemoryLite(nn.Module):
    def __init__(self, embed_dim, temperature=0.07, write_temperature=None,
                 alpha_min=0.02, alpha_max=0.25, normalize=True,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.E = int(embed_dim)
        self.T = float(temperature)
        self.Tw = float(write_temperature or (self.T * 2.0))  # 写温度默认更高
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.normalize = normalize
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # 动态记忆缓冲（每环境1向量）
        self.register_buffer("m", torch.zeros(0, self.E, dtype=dtype, device=device))

        # 可学习投影（让相似度定义可学习）
        self.t_proj = nn.Linear(self.E, self.E, bias=False)
        self.m_proj = nn.Linear(self.E, self.E, bias=False)
        
        # 初始化为接近单位阵（稳定训练初期）
        nn.init.eye_(self.t_proj.weight)
        nn.init.eye_(self.m_proj.weight)
        # 添加小扰动
        with torch.no_grad():
            self.t_proj.weight.add_(torch.randn_like(self.t_proj.weight) * 0.01)
            self.m_proj.weight.add_(torch.randn_like(self.m_proj.weight) * 0.01)

        # 写门控 α(s)∈[alpha_min, alpha_max]（根据 m 和 proto 学习）
        hid = max(32, self.E // 4)
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * self.E, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
            nn.Sigmoid(),   # -> [0,1]
        )

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

    def attend(self, tokens, env_indices=None, return_weights_only=True):
        """读记忆：返回注意力权重 [B,P]"""
        B, P, E = tokens.shape
        if env_indices is None:
            self._ensure_b(B, tokens.device, tokens.dtype)
            target_idx = torch.arange(B, device=tokens.device)
        else:
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_idx = torch.where(env_indices)[0]

        m_sel = self.m[target_idx]  # [B,E]
        
        # 可学习投影
        t = self.t_proj(tokens)     # [B,P,E]
        k = self.m_proj(m_sel)      # [B,E]
        
        if self.normalize:
            t = F.normalize(t, dim=-1, eps=1e-8)
            k_norm = k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k = k / k_norm
            # 检测空记忆
            empty_mask = (k_norm.squeeze(-1) < 1e-6).unsqueeze(-1)  # [B,1]
        else:
            empty_mask = None

        scores = torch.einsum("bpe,be->bp", t, k) / self.T  # [B,P]
        
        # 空记忆时强制均匀权重
        if empty_mask is not None:
            uniform = torch.ones_like(scores) / P
            scores = torch.where(empty_mask.expand_as(scores), 
                                torch.zeros_like(scores), scores)
        
        w = torch.softmax(scores, dim=1)  # [B,P]
        
        if empty_mask is not None:
            uniform = torch.ones_like(w) / P
            w = torch.where(empty_mask.expand_as(w), uniform, w)
        
        if return_weights_only:
            return w
        return w, k

    def write(self, tokens, env_indices=None):
        """写记忆：可学习门控的 EMA 更新"""
        B_sel, P, E = tokens.shape
        if env_indices is None:
            self._ensure_b(B_sel, tokens.device, tokens.dtype)
            target_idx = torch.arange(B_sel, device=tokens.device)
        else:
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_idx = torch.where(env_indices)[0]

        m_sel = self.m[target_idx]  # [B_sel,E]

        # 可学习投影
        t = self.t_proj(tokens)
        k = self.m_proj(m_sel)
        
        if self.normalize:
            t = F.normalize(t, dim=-1, eps=1e-8)
            k_norm = k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k = k / k_norm
            empty_mask = (k_norm.squeeze(-1) < 1e-6)  # [B_sel]
        else:
            empty_mask = None
        
        scores = torch.einsum("bpe,be->bp", t, k) / self.Tw
        
        # 空记忆时用均匀权重
        if empty_mask is not None:
            uniform_w = torch.ones_like(scores) / P
            scores = torch.where(empty_mask.unsqueeze(-1), 
                                torch.zeros_like(scores), scores)
        
        w = torch.softmax(scores, dim=1)  # [B_sel,P]
        
        if empty_mask is not None:
            w = torch.where(empty_mask.unsqueeze(-1), uniform_w, w)
        
        proto = torch.einsum("bp,bpe->be", w, tokens)  # [B_sel,E]
        if self.normalize:
            proto = F.normalize(proto, dim=-1, eps=1e-8)

        # 学习门控 α
        gate_in = torch.cat([m_sel, proto], dim=-1)  # [B_sel,2E]
        a01 = self.alpha_head(gate_in)  # [B_sel,1] in [0,1]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * a01

        # EMA 更新
        m_old = F.normalize(m_sel, dim=-1, eps=1e-8) if self.normalize else m_sel
        m_new = (1.0 - alpha) * m_old + alpha * proto
        if self.normalize:
            m_new = F.normalize(m_new, dim=-1, eps=1e-8)

        # 空记忆第一次写入直接用原型
        if empty_mask is not None:
            m_new = torch.where(empty_mask.unsqueeze(-1), proto, m_new)

        self.m[target_idx] = m_new