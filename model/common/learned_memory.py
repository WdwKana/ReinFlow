# model/common/learned_momory.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class LearnedWorkingMemoryLite(nn.Module):
    def __init__(self, embed_dim, temperature=0.07, write_temperature=None,
                 alpha_min=0.02, alpha_max=0.25, normalize=True,
                 device=None, dtype=torch.float32):
        super().__init__()
        self.E = int(embed_dim)
        self.T = float(temperature)
        self.Tw = float(write_temperature or (self.T * 2.0))  
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.normalize = normalize
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self._debug_hook: Optional[Callable[[dict], None]] = None

        # memory buffer
        self.register_buffer("m", torch.zeros(0, self.E, dtype=dtype, device=device), persistent=False)

        # learnable projection
        self.t_proj = nn.Linear(self.E, self.E, bias=False)
        self.m_proj = nn.Linear(self.E, self.E, bias=False)
        
        # initialize to identity matrix (stable training in the early stages)
        nn.init.eye_(self.t_proj.weight)
        nn.init.eye_(self.m_proj.weight)
        # add small perturbation
        with torch.no_grad():
            self.t_proj.weight.add_(torch.randn_like(self.t_proj.weight) * 0.01)
            self.m_proj.weight.add_(torch.randn_like(self.m_proj.weight) * 0.01)

        # write gate α(s)∈[alpha_min, alpha_max] (learned from m and proto)
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
        """read memory: return attention weights [B,P]"""
        B, P, E = tokens.shape
        if env_indices is None:
            self._ensure_b(B, tokens.device, tokens.dtype)
            target_idx = torch.arange(B, device=tokens.device)
        else:
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_idx = torch.where(env_indices)[0]

        m_sel = self.m[target_idx]  # [B,E]
        
        # learnable projection
        t = self.t_proj(tokens)     # [B,P,E]
        k = self.m_proj(m_sel)      # [B,E]
        
        if self.normalize:
            t = F.normalize(t, dim=-1, eps=1e-8)
            k_norm = k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k = k / k_norm
            # check empty memory
            empty_mask = (k_norm.squeeze(-1) < 1e-6).unsqueeze(-1)  # [B,1]
        else:
            empty_mask = None

        scores = torch.einsum("bpe,be->bp", t, k) / self.T  # [B,P]
        
        # force uniform weights when empty memory
        if empty_mask is not None:
            uniform = torch.ones_like(scores) / P
            scores = torch.where(empty_mask.expand_as(scores), 
                                torch.zeros_like(scores), scores)
        
        w = torch.softmax(scores, dim=1)  # [B,P]
        
        if empty_mask is not None:
            uniform = torch.ones_like(w) / P
            w = torch.where(empty_mask.expand_as(w), uniform, w)
        
        if self._debug_hook is not None:
            read_out = w.unsqueeze(-1) * m_sel.unsqueeze(1)
            payload = {
                "weights": w.unsqueeze(-1).detach(),
                "read_out": read_out.detach(),
                "memory": self.m[target_idx].detach(),
            }
            try:
                self._debug_hook(payload)
            except Exception:
                pass
        if return_weights_only:
            return w
        return w, k

    def write(self, tokens, env_indices=None):
        """write memory: EMA update with learnable gate"""
        B_sel, P, E = tokens.shape
        if env_indices is None:
            self._ensure_b(B_sel, tokens.device, tokens.dtype)
            target_idx = torch.arange(B_sel, device=tokens.device)
        else:
            full_B = env_indices.shape[0]
            self._ensure_b(full_B, tokens.device, tokens.dtype)
            target_idx = torch.where(env_indices)[0]

        m_sel = self.m[target_idx]  # [B_sel,E]

        # learnable projection
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
        
        # use uniform weights when empty memory
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

        # learnable gate α
        gate_in = torch.cat([m_sel, proto], dim=-1)  # [B_sel,2E]
        a01 = self.alpha_head(gate_in)  # [B_sel,1] in [0,1]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * a01

        # EMA update
        m_old = F.normalize(m_sel, dim=-1, eps=1e-8) if self.normalize else m_sel
        m_new = (1.0 - alpha) * m_old + alpha * proto
        if self.normalize:
            m_new = F.normalize(m_new, dim=-1, eps=1e-8)

        # write prototype directly when empty memory
        if empty_mask is not None:
            m_new = torch.where(empty_mask.unsqueeze(-1), proto, m_new)

        self.m[target_idx] = m_new

    def register_debug_hook(self, hook: Callable[[dict], None]) -> None:
        self._debug_hook = hook

    def clear_debug_hook(self) -> None:
        self._debug_hook = None


class PatchWorkingMemory(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_slots: int,
            read_temperature: float = 0.07,
            write_temperature: float | None = None,
            use_topk: bool = False,
            topk: int = 8,
            residual_scale: float = 1.0,
            normalize: bool = True,
            device=None,
            dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.E = int(embed_dim)
        self.S = int(num_slots)
        self.normalize = normalize
        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature or read_temperature)
        self.use_topk = use_topk
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self._debug_hook: Optional[Callable[[dict], None]] = None

        self.register_buffer(
            "memory",
            torch.zeros(0, self.S, self.E, dtype=dtype, device=device),
        )

        self.slot_emb = nn.Parameter(
            torch.randn(self.S, self.E, dtype=dtype, device=device) * 0.01
        )

        # projections for read (content address)
        self.read_query = nn.Linear(self.E, self.E, bias=False)
        self.read_key = nn.Linear(self.E, self.E, bias=False)

        # projections for write (strength + value)
        self.write_query = nn.Linear(self.E, self.E, bias=False)
        self.write_key = nn.Linear(self.E, self.E, bias=False)
        self.write_value = nn.Linear(self.E, self.E, bias=False)

        self.erase_value = nn.Linear(self.E, self.E, bias=False)
        self.add_value = nn.Linear(self.E, self.E, bias=False)

    def _ensure(self, B: int, device, dtype) -> None:
        if self.memory.numel() == 0 or self.memory.shape[0] != B:
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B: int | None = None, device=None, dtype=None) -> None:
        if B is None and self.memory.numel() > 0:
            B = self.memory.shape[0]
        if B is not None:
            device = device or self.memory.device
            dtype = dtype or self.memory.dtype
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)
        else:
            self.memory.zero_()

    def forward(
            self,
            tokens: torch.Tensor,
            write: bool = True,
            return_weights: bool = False,
    ):
        """
        Args:
            tokens: [B, P, E]  ViT patch embeddings
            write:  是否执行写入更新
            return_weights: 是否返回 w (B,P,S)

        Returns:
            enhanced_tokens: [B, P, E]  （tokens + memory读出）
            (optional) w: 阅读寻址权重
        """
        B, P, E = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        self._ensure(B, device, dtype)

        memory = self.memory  # [B, S, E]
        slot_bias = self.slot_emb.to(device=device, dtype=dtype).unsqueeze(0)
        memory_with_bias = memory + slot_bias

        # ---------- Read ----------
        q_r = self.read_query(memory_with_bias)          # [B, S, E]
        #q_r = self.read_query(memory)
        k_r = self.read_key(tokens)            # [B, P, E]

        if self.normalize:
            q_r = F.normalize(q_r, dim=-1, eps=1e-6)
            k_r = F.normalize(k_r, dim=-1, eps=1e-6)

        read_logits = torch.einsum("bse,bpe->bsp", q_r, k_r)  # [B, S, P]
        read_logits = read_logits / max(self.read_temperature, 1e-6)
        read_logits = self._apply_topk(read_logits)

        w = torch.softmax(read_logits.transpose(1, 2), dim=-1)  # [B, P, S]

        # ---------- Write ----------
        if write:
            q_w = self.write_query(memory_with_bias)         # [B, S, E]
            #q_w = self.write_query(memory)
            k_w = self.write_key(tokens)           # [B, P, E]
            v_w = self.write_value(tokens)         # [B, P, E]

            if self.normalize:
                q_w = F.normalize(q_w, dim=-1, eps=1e-6)
                k_w = F.normalize(k_w, dim=-1, eps=1e-6)
                v_w = v_w

            write_logits = torch.einsum("bse,bpe->bsp", q_w, k_w)
            write_logits = write_logits / max(self.write_temperature, 1e-6)
            write_logits = self._apply_topk(write_logits)

            beta = torch.softmax(write_logits.transpose(1, 2), dim=-1)  # [B, P, S]

            #erase = (w * (1.0 - beta)).sum(dim=1)  # [B, S]
            erase = torch.sigmoid(self.erase_value(tokens))
            erase_vec = torch.einsum("bps,bpe->bse", w *(1.0 - beta), erase)
            erase_vec = erase_vec.clamp(0,1)
            w_beta = w * beta 
            #denom = w_beta.sum(dim=1, keepdim=True).clamp_min(1e-6)
            #add = torch.einsum("bps,bpe->bse", w_beta / denom, v_w)
            add = torch.einsum("bps,bpe->bse", w * beta, v_w)  # [B, S, E]

            #memory = memory * (1.0 - erase.unsqueeze(-1)) + add
            memory = memory * (1.0 - erase_vec) + add
            self.memory = memory

        
        read_out = torch.einsum("bps,bse->bpe", w, memory)  # [B, P, E]
        #read_out = torch.einsum("bps,bse->bpe", w_beta, memory)  # [B, P, E]
        enhanced = tokens + self.residual_scale * read_out
        #enhanced = read_out

        if self._debug_hook is not None:
            payload = {
                "weights": w.detach(),
                "read_out": read_out.detach(),
                "memory": memory.detach(),
            }
            if write:
                payload["write_weights"] = beta.detach()
                payload["erase"] = erase_vec.detach()
            try:
                self._debug_hook(payload)
            except Exception:
                pass

        if return_weights:
            return enhanced, w
        return enhanced

    def register_debug_hook(self, hook: Callable[[dict], None]) -> None:
        self._debug_hook = hook

    def clear_debug_hook(self) -> None:
        self._debug_hook = None

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_topk or self.topk <= 0 or self.topk >= self.S:
            return logits
        k = min(self.topk, self.S)
        values, indices = torch.topk(logits, k=k, dim=1)   # [B, k, P]
        mask = torch.zeros_like(logits)
        mask.scatter_(1, indices, 1.0)
        zero_mask = mask.sum(dim=1, keepdim=True) == 0
        if zero_mask.any():
            fallback_idx = logits.argmax(dim=1, keepdim=True)
            mask.scatter_(1, fallback_idx, 1.0)
        return logits.masked_fill(mask == 0, float("-inf"))
    @property
    def m(self):
        return self.memory

    @m.setter
    def m(self, value):
        self.memory = value

class PatchActionWorkingMemory(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_slots: int,
            read_temperature: float = 0.07,
            write_temperature: float | None = None,
            use_topk: bool = False,
            topk: int = 8,
            residual_scale: float = 1.0,
            normalize: bool = True,
            device=None,
            dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.E = int(embed_dim)
        self.S = int(num_slots)
        self.normalize = normalize
        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature or read_temperature)
        self.use_topk = use_topk
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self._debug_hook: Optional[Callable[[dict], None]] = None

        self.register_buffer(
            "memory",
            torch.zeros(0, self.S, self.E, dtype=dtype, device=device),
        )

        self.slot_emb = nn.Parameter(
            torch.randn(self.S, self.E, dtype=dtype, device=device) * 0.01
        )

        # projections for read (content address)
        self.read_query = nn.Linear(self.E, self.E, bias=False)
        self.read_key = nn.Linear(self.E, self.E, bias=False)

        # projections for write (strength + value)
        self.write_query = nn.Linear(self.E, self.E, bias=False)
        self.write_key = nn.Linear(self.E, self.E, bias=False)
        self.write_value = nn.Linear(self.E, self.E, bias=False)

        self.erase_value = nn.Linear(self.E, self.E, bias=False)
        self.add_value = nn.Linear(self.E, self.E, bias=False)



    def _ensure(self, B: int, device, dtype) -> None:
        if self.memory.numel() == 0 or self.memory.shape[0] != B:
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B: int | None = None, device=None, dtype=None) -> None:
        if B is None and self.memory.numel() > 0:
            B = self.memory.shape[0]
        if B is not None:
            device = device or self.memory.device
            dtype = dtype or self.memory.dtype
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)
        else:
            self.memory.zero_()

    def forward(
            self,
            tokens: torch.Tensor,
            action_query: torch.Tensor,
            write: bool = True,
            return_weights: bool = False,
    ):
        """
        Args:
            tokens: [B, P, E]  ViT patch embeddings
            action_query: [B, E]  action query embeddings
            write: whether to execute write update
            return_weights: whether to return w (B,P,S)

        Returns:
            enhanced_tokens: [B, P, E]  (tokens + memory read out + action query)
            (optional) w: read attention weights
        """
        B, P, E = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        self._ensure(B, device, dtype)

        memory = self.memory  # [B, S, E]
        slot_bias = self.slot_emb.to(device=device, dtype=dtype).unsqueeze(0)
        memory_with_bias = memory + slot_bias

        # ---------- Read ----------
        q_r = self.read_query(memory_with_bias)          # [B, S, E]
        #q_r = self.read_query(memory)
        k_r = self.read_key(tokens)            # [B, P, E]
        if self.normalize:
            q_r = F.normalize(q_r, dim=-1, eps=1e-6)
            k_r = F.normalize(k_r, dim=-1, eps=1e-6)

        read_logits = torch.einsum("bse,bpe->bsp", q_r, k_r)  # [B, S, P]
        read_logits = read_logits / max(self.read_temperature, 1e-6)
        read_logits = self._apply_topk(read_logits)

        w = torch.softmax(read_logits.transpose(1, 2), dim=-1)  # [B, P, S]
        a = action_query

        if self.normalize:
            a = F.normalize(a, dim=-1, eps=1e-6) # [B, E]
        token_sim = torch.einsum("bpe,be->bp",k_r,a) #[B, P]
        alpha = torch.softmax(token_sim, dim=-1) #[B, P]
        w = w * alpha.unsqueeze(-1)
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-6))

        # ---------- Write ----------
        if write:
            q_w = self.write_query(memory_with_bias)         # [B, S, E]
            #q_w = self.write_query(memory)
            k_w = self.write_key(tokens)           # [B, P, E]
            v_w = self.write_value(tokens)         # [B, P, E]

            if self.normalize:
                q_w = F.normalize(q_w, dim=-1, eps=1e-6)
                k_w = F.normalize(k_w, dim=-1, eps=1e-6)
                v_w = v_w

            write_logits = torch.einsum("bse,bpe->bsp", q_w, k_w)
            write_logits = write_logits / max(self.write_temperature, 1e-6)
            write_logits = self._apply_topk(write_logits)

            beta = torch.softmax(write_logits.transpose(1, 2), dim=-1)  # [B, P, S]

            #erase = (w * (1.0 - beta)).sum(dim=1)  # [B, S]
            erase = torch.sigmoid(self.erase_value(tokens))
            erase_vec = torch.einsum("bps,bpe->bse", w *(1.0 - beta), erase)
            erase_vec = erase_vec.clamp(0,1)
            w_beta = w * beta 
            #denom = w_beta.sum(dim=1, keepdim=True).clamp_min(1e-6)
            #add = torch.einsum("bps,bpe->bse", w_beta / denom, v_w)
            add = torch.einsum("bps,bpe->bse", w * beta, v_w)  # [B, S, E]

            #memory = memory * (1.0 - erase.unsqueeze(-1)) + add
            memory = memory * (1.0 - erase_vec) + add
            self.memory = memory

        
        read_out = torch.einsum("bps,bse->bpe", w, memory)  # [B, P, E]
        #read_out = torch.einsum("bps,bse->bpe", w_beta, memory)  # [B, P, E]
        enhanced = tokens + self.residual_scale * read_out
        #enhanced = read_out

        if self._debug_hook is not None:
            payload = {
                "weights": w.detach(),
                "read_out": read_out.detach(),
                "memory": memory.detach(),
            }
            if write:
                payload["write_weights"] = beta.detach()
                payload["erase"] = erase_vec.detach()
            try:
                self._debug_hook(payload)
            except Exception:
                pass

        if return_weights:
            return enhanced, w
        return enhanced

    def register_debug_hook(self, hook: Callable[[dict], None]) -> None:
        self._debug_hook = hook

    def clear_debug_hook(self) -> None:
        self._debug_hook = None

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_topk or self.topk <= 0 or self.topk >= self.S:
            return logits
        k = min(self.topk, self.S)
        values, indices = torch.topk(logits, k=k, dim=1)   # [B, k, P]
        mask = torch.zeros_like(logits)
        mask.scatter_(1, indices, 1.0)
        zero_mask = mask.sum(dim=1, keepdim=True) == 0
        if zero_mask.any():
            fallback_idx = logits.argmax(dim=1, keepdim=True)
            mask.scatter_(1, fallback_idx, 1.0)
        return logits.masked_fill(mask == 0, float("-inf"))
    @property
    def m(self):
        return self.memory

    @m.setter
    def m(self, value):
        self.memory = value

class PatchSHMWorkingMemory(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_slots: int,
            read_temperature: float = 0.07,
            write_temperature: float | None = None,
            use_topk: bool = False,
            topk: int = 8,
            residual_scale: float = 1.0,
            normalize: bool = True,
            device=None,
            add_space: int = 7,
            dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.E = int(embed_dim)
        self.S = int(num_slots)
        self.normalize = normalize
        self.read_temperature = float(read_temperature)
        self.write_temperature = float(write_temperature or read_temperature)
        self.use_topk = use_topk
        self.topk = int(topk)
        self.residual_scale = float(residual_scale)
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self._debug_hook: Optional[Callable[[dict], None]] = None

        self.register_buffer(
            "memory",
            torch.zeros(0, self.S, self.E, dtype=dtype, device=device),
        )

        self.slot_emb = nn.Parameter(
            torch.randn(self.S, self.E, dtype=dtype, device=device) * 0.01
        )

        # projections for read (content address)
        self.read_query = nn.Linear(self.E, self.E, bias=False)
        self.read_key = nn.Linear(self.E, self.E, bias=False)

        # projections for write (strength + value)
        self.write_query = nn.Linear(self.E, self.E, bias=False)
        self.write_key = nn.Linear(self.E, self.E, bias=False)
        self.write_value = nn.Linear(self.E, self.E, bias=False)

        self.erase_value = nn.Linear(self.E, self.E, bias=False)
        self.add_value = nn.Linear(self.E, self.E, bias=False)
        self.add_space = int(add_space)
        self.L = 2 ** self.add_space
        self.theta_selector = nn.Linear(self.E, self.L, bias=False)
        self.theta_matrix = nn.Parameter(torch.empty(self.L, self.E, dtype=dtype, device=device))
        nn.init.xavier_uniform_(self.theta_matrix)
        self.gumbel_tau = 1.0



    def _ensure(self, B: int, device, dtype) -> None:
        if self.memory.numel() == 0 or self.memory.shape[0] != B:
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B: int | None = None, device=None, dtype=None) -> None:
        if B is None and self.memory.numel() > 0:
            B = self.memory.shape[0]
        if B is not None:
            device = device or self.memory.device
            dtype = dtype or self.memory.dtype
            self.memory = torch.zeros(B, self.S, self.E, device=device, dtype=dtype)
        else:
            self.memory.zero_()

    def forward(
            self,
            tokens: torch.Tensor,
            action_query: torch.Tensor,
            write: bool = True,
            return_weights: bool = False,
    ):
        """
        Args:
            tokens: [B, P, E]  ViT patch embeddings
            action_query: [B, E]  action query embeddings
            write: whether to execute write update
            return_weights: whether to return w (B,P,S)

        Returns:
            enhanced_tokens: [B, P, E]  (tokens + memory read out + action query)
            (optional) w: read attention weights
        """
        B, P, E = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        self._ensure(B, device, dtype)

        memory = self.memory  # [B, S, E]
        slot_bias = self.slot_emb.to(device=device, dtype=dtype).unsqueeze(0)
        # ---- SHM calibration: replace alpha with Ct gating over memory ----
        a = action_query
        if self.normalize:
            a = F.normalize(a, dim=-1, eps=1e-6)  # [B,E]
        theta_logits = self.theta_selector(a)  # [B,L]
        gamma = (F.gumbel_softmax(theta_logits, tau=self.gumbel_tau, hard=False)
                 if self.training else torch.softmax(theta_logits, dim=-1))  # [B,L]
        theta_t = torch.matmul(gamma, self.theta_matrix)  # [B,E]
        g_vec = 1.0 + torch.tanh(theta_t * a)  # [B,E], in (0,2)
        C = g_vec.unsqueeze(1)  # [B,1,E]

        memory = memory * C
        memory_with_bias = memory + slot_bias

        # ---------- Read ----------
        q_r = self.read_query(memory_with_bias)          # [B, S, E]
        #q_r = self.read_query(memory)
        k_r = self.read_key(tokens)            # [B, P, E]
        if self.normalize:
            q_r = F.normalize(q_r, dim=-1, eps=1e-6)
            k_r = F.normalize(k_r, dim=-1, eps=1e-6)

        read_logits = torch.einsum("bse,bpe->bsp", q_r, k_r)  # [B, S, P]
        read_logits = read_logits / max(self.read_temperature, 1e-6)
        read_logits = self._apply_topk(read_logits)

        w = torch.softmax(read_logits.transpose(1, 2), dim=-1)  # [B, P, S]


        # ---------- Write ----------
        if write:
            q_w = self.write_query(memory_with_bias)         # [B, S, E]
            #q_w = self.write_query(memory)
            k_w = self.write_key(tokens)           # [B, P, E]
            v_w = self.write_value(tokens)         # [B, P, E]

            if self.normalize:
                q_w = F.normalize(q_w, dim=-1, eps=1e-6)
                k_w = F.normalize(k_w, dim=-1, eps=1e-6)
                v_w = v_w

            write_logits = torch.einsum("bse,bpe->bsp", q_w, k_w)
            write_logits = write_logits / max(self.write_temperature, 1e-6)
            write_logits = self._apply_topk(write_logits)

            beta = torch.softmax(write_logits.transpose(1, 2), dim=-1)  # [B, P, S]

            #erase = (w * (1.0 - beta)).sum(dim=1)  # [B, S]
            erase = torch.sigmoid(self.erase_value(tokens))
            erase_vec = torch.einsum("bps,bpe->bse", w *(1.0 - beta), erase)
            erase_vec = erase_vec.clamp(0,1)
            w_beta = w * beta 
            #denom = w_beta.sum(dim=1, keepdim=True).clamp_min(1e-6)
            #add = torch.einsum("bps,bpe->bse", w_beta / denom, v_w)
            add = torch.einsum("bps,bpe->bse", w * beta, v_w)  # [B, S, E]

            #memory = memory * (1.0 - erase.unsqueeze(-1)) + add
            memory = memory * (1.0 - erase_vec) + add
            self.memory = memory

        
        read_out = torch.einsum("bps,bse->bpe", w, memory)  # [B, P, E]
        #read_out = torch.einsum("bps,bse->bpe", w_beta, memory)  # [B, P, E]
        enhanced = tokens + self.residual_scale * read_out
        #enhanced = read_out

        if self._debug_hook is not None:
            payload = {
                "weights": w.detach(),
                "read_out": read_out.detach(),
                "memory": memory.detach(),
            }
            if write:
                payload["write_weights"] = beta.detach()
                payload["erase"] = erase_vec.detach()
            try:
                self._debug_hook(payload)
            except Exception:
                pass

        if return_weights:
            return enhanced, w
        return enhanced

    def register_debug_hook(self, hook: Callable[[dict], None]) -> None:
        self._debug_hook = hook

    def clear_debug_hook(self) -> None:
        self._debug_hook = None

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_topk or self.topk <= 0 or self.topk >= self.S:
            return logits
        k = min(self.topk, self.S)
        values, indices = torch.topk(logits, k=k, dim=1)   # [B, k, P]
        mask = torch.zeros_like(logits)
        mask.scatter_(1, indices, 1.0)
        zero_mask = mask.sum(dim=1, keepdim=True) == 0
        if zero_mask.any():
            fallback_idx = logits.argmax(dim=1, keepdim=True)
            mask.scatter_(1, fallback_idx, 1.0)
        return logits.masked_fill(mask == 0, float("-inf"))

    @property
    def m(self):
        return self.memory

    @m.setter
    def m(self, value):
        self.memory = value
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 1, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(
        self, 
        inputs, 
        num_slots = None, 
        initial_slots: torch.Tensor | None = None, 
        detach_init: bool = True,
    ) -> torch.Tensor:
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        #mu = self.slots_mu.expand(b, n_s, -1)
        #sigma = self.slots_sigma.expand(b, n_s, -1)
        #slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs) 
        if initial_slots is not None:
            if initial_slots.shape[1] != n_s:
                raise ValueError(
                    f"initial_slots.shape[1]={initial_slots.shape[1]} "
                    f"does not match requested num_slots={n_s}"
                )
            slots = initial_slots.to(device=inputs.device, dtype=inputs.dtype)
            if detach_init:
                slots = slots.detach()
            slots = slots.clone()
        else:
            mu = self.slots_mu.to(device=inputs.device, dtype=inputs.dtype).expand(b, n_s, -1)
            sigma = self.slots_sigma.to(device=inputs.device, dtype=inputs.dtype).expand(b, n_s, -1)
            slots = torch.normal(mu, sigma)
                      
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            slot_strength = attn.sum(dim=-1, keepdim=True)
            self.slot_strength = slot_strength.detach()
            #attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
        

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

class SlotMemory(nn.Module):
    """
    Episode-level slot memory that keeps persistent slots across timesteps.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        detach_initial: bool = True,
        detach_update: bool = True,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.detach_initial = detach_initial
        self.detach_update = detach_update
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.register_buffer(
            "memory",
            torch.zeros(0, self.num_slots, self.slot_dim, device=device, dtype=dtype),
            persistent=False,
        )

    def _ensure(self, B: int, device, dtype) -> None:
        if self.memory.numel() == 0 or self.memory.shape[0] != B:
            self.memory = torch.zeros(B, self.num_slots, self.slot_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B: int | None = None, device=None, dtype=None) -> None:
        if B is None and self.memory.numel() > 0:
            B = self.memory.shape[0]
        if B is not None:
            device = device or self.memory.device
            dtype = dtype or self.memory.dtype
            self.memory = torch.zeros(B, self.num_slots, self.slot_dim, device=device, dtype=dtype)
        else:
            self.memory.zero_()

    def slots_for_batch(self, B: int, device, dtype) -> torch.Tensor:
        self._ensure(B, device, dtype)
        return self.memory

    def forward(
        self,
        slot_attention: SlotAttention,
        tokens: torch.Tensor,
        env_indices: torch.Tensor | None = None,
        detach_initial: bool | None = None,
        detach_update: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            slot_attention: SlotAttention module used for updates
            tokens: [B, P, slot_dim] encoded tokens
            env_indices: optional boolean mask of shape [B] to update a subset
            detach_initial: override default detach behaviour
            detach_update: override default detach behaviour after update
        Returns:
            slots: [B, num_slots, slot_dim] current slot memory (after update)
        """
        B, _, _ = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        self._ensure(B, device, dtype)

        use_detach_initial = self.detach_initial if detach_initial is None else detach_initial
        use_detach_update = self.detach_update if detach_update is None else detach_update

        if env_indices is None:
            init_slots = self.memory
            updated_slots = slot_attention(
                tokens,
                num_slots=self.num_slots,
                initial_slots=init_slots,
                detach_init=use_detach_initial,
            )
            self.memory = updated_slots.detach() if use_detach_update else updated_slots
            return self.memory

        indices = torch.where(env_indices)[0]
        if indices.numel() == 0:
            return self.memory

        init_slots = self.memory.index_select(0, indices)
        token_subset = tokens.index_select(0, indices)
        updated_slots = slot_attention(
                token_subset,
                num_slots=self.num_slots,
                initial_slots=init_slots,
                detach_init=use_detach_initial,
        )
        if use_detach_update:
            updated_slots = updated_slots.detach()
        self.memory.index_copy_(0, indices, updated_slots)
        return self.memory

class SlotAttentionwithCrossAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 1, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(
        self, 
        inputs, 
        num_slots = None, 
        initial_slots: torch.Tensor | None = None, 
        detach_init: bool = True,
    ) -> torch.Tensor:
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        #mu = self.slots_mu.expand(b, n_s, -1)
        #sigma = self.slots_sigma.expand(b, n_s, -1)
        #slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs) 
        if initial_slots is not None:
            if initial_slots.shape[1] != n_s:
                raise ValueError(
                    f"initial_slots.shape[1]={initial_slots.shape[1]} "
                    f"does not match requested num_slots={n_s}"
                )
            slots = initial_slots.to(device=inputs.device, dtype=inputs.dtype)
            if detach_init:
                slots = slots.detach()
            slots = slots.clone()
        else:
            mu = self.slots_mu.to(device=inputs.device, dtype=inputs.dtype).expand(b, n_s, -1)
            sigma = self.slots_sigma.to(device=inputs.device, dtype=inputs.dtype).expand(b, n_s, -1)
            slots = torch.normal(mu, sigma)
                      
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            slot_strength = attn.sum(dim=-1, keepdim=True)
            self.slot_strength = slot_strength.detach()
            #attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
        

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

class SlotMemorywithCrossAttention(nn.Module):
    """
    Episode-level slot memory that keeps persistent slots across timesteps.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        detach_initial: bool = True,
        detach_update: bool = True,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.detach_initial = detach_initial
        self.detach_update = detach_update
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.register_buffer(
            "memory",
            torch.zeros(0, self.num_slots, self.slot_dim, device=device, dtype=dtype),
            persistent=False,
        )

    def _ensure(self, B: int, device, dtype) -> None:
        if self.memory.numel() == 0 or self.memory.shape[0] != B:
            self.memory = torch.zeros(B, self.num_slots, self.slot_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, B: int | None = None, device=None, dtype=None) -> None:
        if B is None and self.memory.numel() > 0:
            B = self.memory.shape[0]
        if B is not None:
            device = device or self.memory.device
            dtype = dtype or self.memory.dtype
            self.memory = torch.zeros(B, self.num_slots, self.slot_dim, device=device, dtype=dtype)
        else:
            self.memory.zero_()

    def slots_for_batch(self, B: int, device, dtype) -> torch.Tensor:
        self._ensure(B, device, dtype)
        return self.memory

    def forward(
        self,
        slot_attention: SlotAttention,
        tokens: torch.Tensor,
        env_indices: torch.Tensor | None = None,
        detach_initial: bool | None = None,
        detach_update: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            slot_attention: SlotAttention module used for updates
            tokens: [B, P, slot_dim] encoded tokens
            env_indices: optional boolean mask of shape [B] to update a subset
            detach_initial: override default detach behaviour
            detach_update: override default detach behaviour after update
        Returns:
            slots: [B, num_slots, slot_dim] current slot memory (after update)
        """
        B, _, _ = tokens.shape
        device, dtype = tokens.device, tokens.dtype
        self._ensure(B, device, dtype)

        use_detach_initial = self.detach_initial if detach_initial is None else detach_initial
        use_detach_update = self.detach_update if detach_update is None else detach_update

        if env_indices is None:
            init_slots = self.memory
            updated_slots = slot_attention(
                tokens,
                num_slots=self.num_slots,
                initial_slots=init_slots,
                detach_init=use_detach_initial,
            )
            self.memory = updated_slots.detach() if use_detach_update else updated_slots
            return self.memory

        indices = torch.where(env_indices)[0]
        if indices.numel() == 0:
            return self.memory

        init_slots = self.memory.index_select(0, indices)
        token_subset = tokens.index_select(0, indices)
        updated_slots = slot_attention(
                token_subset,
                num_slots=self.num_slots,
                initial_slots=init_slots,
                detach_init=use_detach_initial,
        )
        if use_detach_update:
            updated_slots = updated_slots.detach()
        self.memory.index_copy_(0, indices, updated_slots)
        return self.memory



