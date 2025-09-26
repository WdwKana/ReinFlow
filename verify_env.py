#!/usr/bin/env python3
"""
独立的环境验证脚本
用法: python verify_env.py --config-path=cfg/mikasa/finetune/RememberShapeAndColor3x2-v0 --config-name=ft_ppo_reflow_mlp_img
"""

import os
import sys
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import psutil
import gc
from env.gym_utils import make_async

def simple_env_check(venv):
    """简单但有效的环境检查"""
    print("=" * 60)
    print("🔍 开始环境验证")
    print("=" * 60)
    
    # 基础信息
    print(f"📊 环境基础信息:")
    print(f"   环境数量: {venv.num_envs}")
    print(f"   观测空间: {venv.observation_space}")
    print(f"   动作空间: {venv.action_space}")
    print(f"   单环境观测空间: {venv.single_observation_space}")
    print(f"   单环境动作空间: {venv.single_action_space}")
    
    # 测试重置
    print(f"\n🔄 测试环境重置...")
    try:
        obs = venv.reset()
        print(f"   ✅ 重置成功")
        
        if isinstance(obs, dict):
            print(f"   📝 观测字典包含键: {list(obs.keys())}")
            for key, value in obs.items():
                print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"      {key}: min={value.min():.3f}, max={value.max():.3f}")
        else:
            print(f"   📝 观测: shape={obs.shape}, dtype={obs.dtype}")
            print(f"   📝 数值范围: min={obs.min():.3f}, max={obs.max():.3f}")
            
    except Exception as e:
        print(f"   ❌ 重置失败: {e}")
        return False
    
    # 测试动作执行
    print(f"\n🎮 测试动作执行...")
    try:
        # 生成随机动作
        if hasattr(venv.action_space, 'sample'):
            actions = venv.action_space.sample()
        else:
            actions = np.array([venv.single_action_space.sample() for _ in range(venv.num_envs)])
        
        print(f"   🎯 动作形状: {actions.shape}")
        print(f"   🎯 动作范围: min={actions.min():.3f}, max={actions.max():.3f}")
        
        # 执行动作
        obs, rewards, terminated, truncated, infos = venv.step(actions)
        
        print(f"   ✅ 动作执行成功")
        print(f"   🏆 奖励: {rewards} (平均: {np.mean(rewards):.3f})")
        print(f"   🏁 终止状态: terminated={terminated}, truncated={truncated}")
        print(f"   ℹ️  信息数量: {len(infos)}")
        
        # 检查返回的观测
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"   📝 {key}: shape={value.shape}")
        else:
            print(f"   📝 观测: shape={obs.shape}")
            
    except Exception as e:
        print(f"   ❌ 动作执行失败: {e}")
        return False
    
    # 连续测试几步
    print(f"\n🔁 连续测试10步...")
    try:
        for step in range(10):
            if hasattr(venv.action_space, 'sample'):
                actions = venv.action_space.sample()
            else:
                actions = np.array([venv.single_action_space.sample() for _ in range(venv.num_envs)])
            
            obs, rewards, terminated, truncated, infos = venv.step(actions)
            
            # 检查异常值
            if not np.all(np.isfinite(rewards)):
                print(f"   ⚠️  步骤{step}: 发现无效奖励值 {rewards}")
            
            # 如果环境结束了，重置
            done = terminated | truncated
            if np.any(done):
                print(f"   🔄 步骤{step}: 环境 {np.where(done)[0]} 需要重置")
                obs = venv.reset()
        
        print(f"   ✅ 连续测试完成")
        
    except Exception as e:
        print(f"   ❌ 连续测试失败: {e}")
        return False
    
    # 最终重置验证
    print(f"\n🔄 最终重置验证...")
    try:
        final_obs = venv.reset()
        print(f"   ✅ 最终重置成功")
        
        # 比较重置前后观测是否合理变化
        if isinstance(final_obs, dict) and isinstance(obs, dict):
            for key in final_obs.keys():
                if key in obs:
                    diff = not np.array_equal(final_obs[key], obs[key])
                    print(f"   📊 {key} 重置后变化: {'是' if diff else '否'}")
        
    except Exception as e:
        print(f"   ❌ 最终重置失败: {e}")
        return False
    
    print(f"\n" + "=" * 60)
    print(f"🎉 环境验证完成 - 所有测试通过！")
    print(f"✅ 环境可以安全用于训练")
    print("=" * 60)
    
    return True

def memory_stress_test(venv, n_iterations=100):
    """内存压力测试"""
    print(f"\n🧠 内存压力测试 ({n_iterations}次迭代)...")
    
    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        for i in range(n_iterations):
            venv.reset()
            
            # 执行几步
            for _ in range(5):
                if hasattr(venv.action_space, 'sample'):
                    actions = venv.action_space.sample()
                else:
                    actions = np.array([venv.single_action_space.sample() for _ in range(venv.num_envs)])
                venv.step(actions)
            
            if i % 20 == 0:
                gc.collect()
                mem_current = process.memory_info().rss / 1024 / 1024
                print(f"   迭代 {i}: 内存使用 {mem_current:.1f}MB")
        
        gc.collect()
        mem_end = process.memory_info().rss / 1024 / 1024
        mem_growth = mem_end - mem_start
        
        print(f"   📊 内存统计:")
        print(f"      开始: {mem_start:.1f}MB")
        print(f"      结束: {mem_end:.1f}MB") 
        print(f"      增长: {mem_growth:.1f}MB")
        
        if mem_growth > 200:  # 200MB阈值
            print(f"   ⚠️  内存增长较大，可能存在内存泄漏")
        else:
            print(f"   ✅ 内存使用正常")
            
    except Exception as e:
        print(f"   ❌ 内存测试失败: {e}")

@hydra.main(version_base=None)
def main(cfg):
    print("🚀 环境验证脚本启动")
    print(f"📋 配置: {cfg.env.name}")
    print(f"🎯 环境类型: {cfg.env.get('env_type', 'default')}")
    
    # 创建环境（完全按照训练代码的方式）
    try:
        venv = make_async(
            cfg.env.name,
            env_type=cfg.env.get("env_type", None),
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        
        # 设置种子（如果不是furniture环境）
        if cfg.env.get("env_type") != "furniture":
            seed = cfg.get('seed', 42)
            venv.seed([seed + i for i in range(cfg.env.n_envs)])
            print(f"🌱 已设置种子: {seed}")
        
        print(f"✅ 环境创建成功")
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行验证
    success = simple_env_check(venv)
    
    if success:
        # 如果基础验证通过，执行内存测试
        memory_stress_test(venv, n_iterations=50)
        
        print(f"\n🎊 全部验证完成！")
        print(f"✨ 环境 {cfg.env.name} 可以安全用于训练")
    else:
        print(f"\n💥 验证失败！")
        print(f"❌ 环境存在问题，请检查配置")
    
    # 清理
    try:
        venv.close()
        print(f"🧹 环境已清理")
    except:
        pass

if __name__ == "__main__":
    main()