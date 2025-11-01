import numpy as np

data = np.load('visualize/ReFlow/RememberColor5-v0/25-10-26-21-29-35/eval_statistics.npz', allow_pickle=True)['data']

print("评估结果汇总：")
print("-" * 80)
for record in data:
    print(f"Denoising Steps: {record['num_denoising_steps']:>3d}")
    print(f"  成功率:         {record['success_rate']*100:>6.2f}% ± {record['success_rate_std']*100:>5.2f}%")
    print(f"  平均奖励:       {record['avg_episode_reward']:>8.2f} ± {record['avg_episode_reward_std']:>6.2f}")
    print(f"  推理频率:       {record['avg_single_step_freq']:>6.1f} Hz")
    print(f"  推理时长:       {record['avg_single_step_duration']*1000:>6.1f} ms")
    print(f"  轨迹长度:       {record['avg_traj_length']:>6.1f} steps")
    print(f"  完成 episode:   {record['num_episodes_finished']:>4d}")
    print("-" * 80)