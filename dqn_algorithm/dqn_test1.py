'''
在dqn_test基础上，优化评估流程和指标：去除可视化，frame达到500即停止并记录episode，重复30次计算总体平均episode。
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os # 引入 os 模块
import statistics # 引入 statistics 模块 for mean calculation
import time

# 超参数
BATCH_SIZE = 100
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100
MEMORY_CAPACITY = 1000
maximum_episode_length = 500 # Renamed for clarity

# 定义最大训练回合数常量，用于失败时计入总和
MAX_TRAIN_EPISODES_PER_RUN = 200 # 从 demo 函数的 while 循环条件推断

# 创建环境
env = gym.make("CartPole-v1", render_mode="rgb_array") # Added render_mode as in previous versions, though not used in report
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.out(x)


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # Set network to eval mode for inference
        self.eval_net.eval()
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # No gradient tracking for action selection
        with torch.no_grad():
            if np.random.uniform() < EPSILON:
                actions_value = self.eval_net(x)
                action = torch.max(actions_value, 1)[1].data.numpy()
                action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            else:
                action = np.random.randint(0, N_ACTIONS)
                action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        # Set network back to train mode
        self.eval_net.train()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Ensure memory is full enough for batching
        current_memory_size = min(self.memory_counter, MEMORY_CAPACITY)
        if current_memory_size < BATCH_SIZE:
             return # Not enough samples to learn

        sample_index = np.random.choice(current_memory_size, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def demo():
    dqn = DQN()
    max_frames_achieved = False
    achieved_episode = None

    # Removed alive_bar from here to avoid nesting/confusion with outer bar
    # with alive_bar() as bar: # Original bar removed
    i_episode = 0
    # Use MAX_TRAIN_EPISODES_PER_RUN for consistency
    while i_episode < MAX_TRAIN_EPISODES_PER_RUN and not max_frames_achieved:
        s = env.reset()[0]
        ep_r = 0
        step_count = 0

        while True:
            # bar() # Original bar step removed
            a = dqn.choose_action(s)
            s_, r, done, truncated, _ = env.step(a) # Added truncated from gym API

            # 自定义奖励
            x, _, theta, _ = s_
            rx = -(x / env.x_threshold) ** 2
            rtheta = -(theta / env.theta_threshold_radians) ** 2
            r += rx + rtheta

            dqn.store_transition(s, a, r, s_)
            ep_r += r

            # Ensure memory is full enough before learning attempts
            if dqn.memory_counter > MEMORY_CAPACITY and dqn.memory_counter >= BATCH_SIZE: # Check both conditions
                dqn.learn()

            # Termination condition: done OR truncated OR reached frame limit
            # Check for reaching frame limit first for the success condition
            if step_count + 1 >= maximum_episode_length: # Check step_count + 1 as step_count is 0-indexed
                 print(f'\nReached {maximum_episode_length} frames at episode {i_episode}!')
                 max_frames_achieved = True
                 achieved_episode = i_episode # Record 0-indexed episode
                 # Print episode reward for successful episode
                 print(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)}')
                 break # Stop episode loop

            # Check for natural termination (pole fell, went out of bounds)
            # This check happens *after* the success check
            if done or truncated:
                # If done/truncated before hitting frame limit, it's a failure for this episode
                # print(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)}') # Print episode reward for failed episode
                break # Stop episode loop


            s = s_
            step_count += 1 # Increment step_count AFTER using s

        if max_frames_achieved:
            break # Stop training loop for this run
        i_episode += 1 # Increment episode counter AFTER the episode loop

    # Return achieved_episode (0-indexed) or None if not successful
    return achieved_episode


if __name__ == "__main__":
    # Construct the output path relative to the script's potential location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in dqn_algorithm/dqn_test1.py and output is in dqn_output/dqn_test1.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir
    output_dir = os.path.join(project_root, "dqn_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dqn_test1.txt")

    # --- MODIFICATION START ---

    # Open the output file in append mode ('a')
    with open(output_file, "a") as f:
        # Add a header for this specific run block
        f.write("\n" + "="*40 + "\n")
        f.write("===== 训练结果报告 =====\n")
        f.write(f"运行时间: {time.ctime()}\n") # 添加时间戳
        f.write(f"评价指标: 计算所有运行的平均达成episode（失败记为{MAX_TRAIN_EPISODES_PER_RUN}）\n") # 明确评价指标
        f.write("-" * 40 + "\n")
        f.write("运行次数 | 是否成功 | 达成episode\n")
        f.write("-" * 40 + "\n")

        success_count = 0
        success_episodes = [] # 存储成功运行的达成episode (0-indexed)
        all_run_episode_counts = [] # 存储所有运行的达成episode (成功为 0-indexed+1, 失败为 MAX_TRAIN_EPISODES_PER_RUN)


        print(f"开始进行 30 次独立训练运行...")
        # 使用 alive_bar 显示整体进度
        with alive_bar(30, title="总体进度") as bar:
            for run_id in range(30):
                f.write(f"  第{run_id + 1:02d}次  |")
                achieved_episode = demo() # demo 函数返回 0-indexed episode 或 None

                if achieved_episode is not None:
                    success_count += 1
                    # 存储成功运行的 0-indexed episode
                    success_episodes.append(achieved_episode)
                    # 存储所有运行的 episode 计数 (成功为 1-based)
                    all_run_episode_counts.append(achieved_episode + 1)
                    f.write(f"    成功    |   {achieved_episode + 1}\n") # 报告中显示 1-based episode
                else:
                    # 失败的运行，计为达到最大episode数
                    all_run_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN)
                    f.write(f"    失败    |     --    \n")

                bar() # 更新整体进度条

        # 关闭环境
        env.close()

        # 写入统计摘要
        f.write("\n===== 统计摘要 =====\n")
        f.write(f"总运行次数: 30\n")
        f.write(f"成功次数 (达到{maximum_episode_length}帧): {success_count}\n") # 明确成功标准
        f.write(f"失败次数: {30 - success_count}\n")

        # 计算总体平均episode数 (包含失败计为 MAX_TRAIN_EPISODES_PER_RUN)
        if all_run_episode_counts: # 确保列表非空
            avg_episode_overall = statistics.mean(all_run_episode_counts)
            worst_episode_overall = max(all_run_episode_counts)
            f.write(f"\n平均达成episode (总体, 失败记为{MAX_TRAIN_EPISODES_PER_RUN}): {avg_episode_overall:.1f}\n") # 修改描述
            f.write(f"最差成绩 (总体): {worst_episode_overall}\n") # 最差成绩是所有运行中的最大值

        if success_count > 0:
            # 仅计算成功运行的平均和最佳成绩
            avg_episode_successful = statistics.mean([ep + 1 for ep in success_episodes]) # 使用 1-based calculation for successful runs
            best_episode_successful = min([ep + 1 for ep in success_episodes])
            f.write(f"平均达成episode (成功运行): {avg_episode_successful:.1f}\n") # 仅成功运行的平均
            f.write(f"最佳成绩 (成功运行): {best_episode_successful} (episode越小越好)\n")
            # 最差成绩 (成功运行) 也可以加，但总体最差更重要
            # worst_episode_successful = max([ep + 1 for ep in success_episodes])
            # f.write(f"最差成绩 (成功运行): {worst_episode_successful}\n")
        else:
            f.write("\n所有运行均未达到500帧，无成功运行统计\n")

        f.write("\n===== 报告结束 =====")
        f.write("\n" + "="*40 + "\n") # 结束本次报告块

    # --- MODIFICATION END ---

    print(f"完整报告已保存至 {output_file}")