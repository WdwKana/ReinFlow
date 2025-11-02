# MIT License
#
# Copyright ...

import logging

from agent.eval.eval_agent_mikasa import EvalAgent as EvalAgentMikasa
from model.flow.reflow import ReFlow
from util.timer import Timer

log = logging.getLogger(__name__)


class EvalReFlowMikasaAgent(EvalAgentMikasa):
    """for mikasa environment."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.clip_intermediate_actions = cfg.get("clip_intermediate_actions", True)
        self.shape_meta = cfg.shape_meta
        self.obs_dims = {k: self.shape_meta.obs[k]["shape"] for k in self.shape_meta.obs}
        if self.denoising_steps is None:
            single = cfg.get("denoising_steps", None)
            if single is not None:
                self.denoising_steps = [single]
        if self.denoising_steps is None:
            raise ValueError(
                "must provide denoising_step_list or denoising_steps in the config"
            )

        log.info(
            f"Evaluation: load_ema={self.load_ema}, "
            f"clip_intermediate_actions={self.clip_intermediate_actions}, "
            f"denoising_step_list={self.denoising_steps}"
        )

    def infer(self, cond: dict, num_denoising_steps: int):
        """call ReFlow.sample to generate action sequence, and return duration."""
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