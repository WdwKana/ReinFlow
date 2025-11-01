# MIT License
#
# Copyright ...

import logging

from agent.eval.eval_agent_mikasa import EvalAgent as EvalAgentMikasa
from model.flow.reflow import ReFlow
from util.timer import Timer

log = logging.getLogger(__name__)


class EvalReFlowMikasaAgent(EvalAgentMikasa):
    """Mikasa 环境下评估 ReFlow 模型的子类。"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # ReFlow 专属的采样开关
        self.clip_intermediate_actions = cfg.get("clip_intermediate_actions", True)
        self.shape_meta = cfg.shape_meta
        self.obs_dims = {k: self.shape_meta.obs[k]["shape"] for k in self.shape_meta.obs}
        # 兼容只提供单个 denoising_steps 的配置
        if self.denoising_steps is None:
            single = cfg.get("denoising_steps", None)
            if single is not None:
                self.denoising_steps = [single]
        if self.denoising_steps is None:
            raise ValueError(
                "必须在配置里提供 denoising_step_list 或 denoising_steps"
            )

        # 方便日志里区分
        log.info(
            f"Evaluation: load_ema={self.load_ema}, "
            f"clip_intermediate_actions={self.clip_intermediate_actions}, "
            f"denoising_step_list={self.denoising_steps}"
        )

    def infer(self, cond: dict, num_denoising_steps: int):
        """调用 ReFlow.sample 生成动作序列，并返回耗时。"""
        self.model: ReFlow
        timer = Timer()
        samples = self.model.sample(
            cond=cond,
            inference_steps=num_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.clip_intermediate_actions,
        )
        duration = timer()
        return samples, duration