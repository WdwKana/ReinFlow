import os, sys, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

# 注册本项目用到的 resolver（YAML 里有 ${eval:...} 等）
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# 组合 Mikasa 配置（不走 Hydra CLI）
with initialize_config_dir(config_dir="/root/ReinFlow/cfg/mikasa/pretrain", version_base=None):
    cfg = compose(
        config_name="pre_reflow_mlp_img",
        overrides=[
            "train_dataset_path=/abs/path/to/your/train.npz",
            "test_in_mujoco=false",
            "wandb=null",
        ],
    )

# 兜底，确保有 _target_
target = cfg.get("_target_", "agent.pretrain.train_reflow_agent.TrainReFlowAgent")
Agent = hydra.utils.get_class(target)
agent = Agent(cfg)
agent.run()