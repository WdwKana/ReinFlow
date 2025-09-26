#!/usr/bin/env python3
"""
ManiSkill + mikasa_robo_suite info调试
"""
import gymnasium as gym
import mikasa_robo_suite

print("=== MikasaRobo环境Info调试 ===")

try:
    env = gym.make('RememberShapeAndColor3x2-v0')
    print("✅ MikasaRobo环境创建成功")
    
    obs, info = env.reset()
    print(f"重置info keys: {list(info.keys())}")
    print(f"🎯 发现success key: {info['success']}")
    
    # 执行几步，专注看success
    for i in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 转换tensor为float避免格式化错误
        reward_val = float(reward) if hasattr(reward, 'item') else reward
        success_val = bool(info['success']) if hasattr(info['success'], 'item') else info['success']
        
        print(f"步骤{i}: reward={reward_val:.3f}, success={success_val}, done={terminated or truncated}")
        
        if terminated or truncated:
            print("🏁 Episode结束!")
            break
    
    env.close()
    
    print("\n✅ 调试完成")
    print("🎯 结论: success的key就是 'success'")
    
except Exception as e:
    print(f"❌ 错误: {e}")