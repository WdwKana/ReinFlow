import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV数据
df = pd.read_csv("training_metrics (2).csv")
train_df = df[df['mode'] == 'eval']

plt.figure(figsize=(12, 6))
plt.plot(train_df['total_env_steps'], train_df['success_rate'],
         label='success_rate', color='blue', alpha=0.8)
plt.xlabel('Total Environment Steps')
plt.ylabel('success_rate')
plt.title('slot_eval_success_rate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
plt.savefig("success_rate.png")