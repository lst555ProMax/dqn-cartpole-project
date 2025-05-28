'''
标准DQN，应用dqn_test1_optuna找到的优化超参数。
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import os
import statistics
import time

# from alive_progress import alive_bar # 如果不需要进度条，可以注释掉

# 更新后的超参数
BATCH_SIZE = 256
LR = 0.0024303199955094507
GAMMA = 0.9556460180442318
TARGET_NETWORK_REPLACE_FREQ = 300
MEMORY_CAPACITY = 1000
EPSILON_START = 0.9212305525092727
EPSILON_END = 0.045039045287420844
EPSILON_DECAY_STEPS = 7000
HIDDEN_SIZE = 64
maximum_episode_length = 500

# 定义最大训练回合数常量，用于失败时计入总和
MAX_TRAIN_EPISODES_PER_RUN = 200 # 从 demo 函数的 while 循环条件推断


# 注意：env应该在外部定义，DQN类不应该拥有自己的env实例，以保持通用性
# env = gym.make("CartPole-v1") # 这行将被移到run_single_experiment函数内部或main中
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size):  # 接收参数
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)


class DQN(object):
    def __init__(self, n_states, n_actions, env_a_shape):  # 接收环境参数
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape

        self.eval_net = Net(self.n_states, self.n_actions, HIDDEN_SIZE)
        self.target_net = Net(self.n_states, self.n_actions, HIDDEN_SIZE)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.total_steps = 0
        self.memory = np.zeros((MEMORY_CAPACITY, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def _get_current_epsilon(self):
        if self.total_steps >= EPSILON_DECAY_STEPS:
            return EPSILON_END
        return EPSILON_END + \
            (EPSILON_START - EPSILON_END) * \
            np.exp(-1. * self.total_steps / (EPSILON_DECAY_STEPS / 5))

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        epsilon = self._get_current_epsilon()
        self.total_steps += 1

        if np.random.uniform() < epsilon:
            action = np.random.randint(0, self.n_actions)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else:
            self.eval_net.eval()
            with torch.no_grad():
                actions_value = self.eval_net(x)
            self.eval_net.train()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
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

        current_memory_size = min(self.memory_counter, MEMORY_CAPACITY)
        if current_memory_size < BATCH_SIZE:
            return

        sample_index = np.random.choice(current_memory_size, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]

        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 将单次实验运行逻辑封装成一个函数，以便重复调用
def run_single_experiment(env_instance, n_states, n_actions, env_a_shape):
    dqn = DQN(n_states, n_actions, env_a_shape)
    # MAX_EPISODES_PER_RUN: 在单次运行中，智能体尝试解决问题的最大episode数
    # 如果在这个数量内没有达到500帧，则认为本次运行失败
    MAX_EPISODES_PER_RUN = 200  # 与你之前设定的类似，可以调整
    achieved_episode = None

    for i_episode in range(MAX_EPISODES_PER_RUN):
        s, _ = env_instance.reset()
        ep_r = 0  # 记录原始环境奖励
        step_count = 0

        while True:
            a = dqn.choose_action(s)
            s_, r, done, truncated, _ = env_instance.step(a)

            x, _, theta, _ = s_
            rx = -(x / env_instance.x_threshold) ** 2
            rtheta = -(theta / env_instance.theta_threshold_radians) ** 2
            modified_r = r + (rx + rtheta) * 0.1  # 使用修改后的奖励进行存储

            if done and step_count < maximum_episode_length - 1:  # 如果提前失败，给一个大的负奖励
                modified_r = -100

            dqn.store_transition(s, a, modified_r, s_)
            ep_r += r  # 累加原始奖励用于判断是否成功

            if dqn.memory_counter > BATCH_SIZE:
                dqn.learn()

            if done or truncated or step_count >= maximum_episode_length - 1:
                if step_count >= maximum_episode_length - 1:
                    print(f'Reached {maximum_episode_length} frames at episode {i_episode}!')
                    achieved_episode = i_episode
                # 移除了这里的打印，让外部循环控制打印或日志记录
                # print(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Steps: {step_count} | Epsilon: {dqn._get_current_epsilon():.3f}')
                break
            s = s_
            step_count += 1

        if achieved_episode is not None:  # 如果已经达到500帧，则本次实验成功，提前结束
            break

    return achieved_episode


if __name__ == "__main__":
    # 在main函数中创建环境，获取环境参数
    # 这样每次run_single_experiment都会使用同一个环境的定义，但可以有新的实例
    # 或者，如果环境没有内部状态会影响下一次运行，可以重用env_main并reset
    # 这里为了简单，我们假设env_main.reset()足以重置状态
    env_main = gym.make("CartPole-v1")
    env_main = env_main.unwrapped  # 确保使用未包装的环境以访问x_threshold等
    N_ACTIONS_MAIN = env_main.action_space.n
    N_STATES_MAIN = env_main.observation_space.shape[0]
    ENV_A_SHAPE_MAIN = 0 if isinstance(env_main.action_space.sample(), int) else env_main.action_space.sample().shape

    # --- MODIFICATION START ---

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in dqn_algorithm/dqn_test2.py and output is in dqn_output/dqn_test2.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir
    output_dir = os.path.join(project_root, "dqn_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dqn_test2.txt")


    # Open the output file in WRITE mode ('w') to overwrite previous content
    with open(output_file, "w") as f: # Changed "a" to "w"
        f.write("===== 训练结果报告 (使用优化参数) =====\n")
        f.write(f"运行时间: {time.ctime()}\n") # Added timestamp as in previous reports
        f.write(
            f"Parameters: BATCH_SIZE={BATCH_SIZE}, LR={LR:.6f}, GAMMA={GAMMA:.6f}, TARGET_NETWORK_REPLACE_FREQ={TARGET_NETWORK_REPLACE_FREQ}, MEMORY_CAPACITY={MEMORY_CAPACITY}, EPSILON_START={EPSILON_START:.6f}, EPSILON_END={EPSILON_END:.6f}, EPSILON_DECAY_STEPS={EPSILON_DECAY_STEPS}, HIDDEN_SIZE={HIDDEN_SIZE}, MAX_EP_LEN={maximum_episode_length}\n")
        # Clarify the evaluation metric in the header
        f.write(f"评价指标: 计算所有运行的平均达成episode（失败记为{MAX_TRAIN_EPISODES_PER_RUN}）\n")
        f.write("-" * 50 + "\n")
        f.write("运行次数 | 是否成功 | 达成episode\n")
        f.write("-" * 50 + "\n")

        success_count = 0
        success_episodes = [] # Stores 0-indexed episode numbers ONLY for successful runs
        N_RUNS = 30  # 你期望的运行次数
        # !!! New list to store episode counts for ALL runs (successful or failed) !!!
        all_run_episode_counts = [] # Stores 1-based episode counts (failures = MAX_EPISODES_PER_RUN)


        print(f"开始进行 {N_RUNS} 次独立训练运行...")
        # Consider adding an outer alive_bar here for the 30 runs if desired
        # from alive_progress import alive_bar # Uncomment at the top if needed
        # with alive_bar(N_RUNS, title="Overall Progress") as bar: # Uncomment if adding outer bar
        for run_id in range(N_RUNS):
            print(f"\n--- Starting Run {run_id + 1}/{N_RUNS} with optimized params ---")
            f.write(f"  第{run_id + 1:02d}次  |")

            # 每次运行都使用相同的环境定义，但可以是一个新的实例（如果需要完全隔离）
            # 或者，如果环境没有内部状态会影响下一次运行，可以重用env_main并reset
            # 这里为了简单，我们假设env_main.reset()足以重置状态
            achieved_episode_in_run = run_single_experiment(env_main, N_STATES_MAIN, N_ACTIONS_MAIN, ENV_A_SHAPE_MAIN) # This returns 0-indexed episode or None

            if achieved_episode_in_run is not None:
                success_count += 1
                success_episodes.append(achieved_episode_in_run) # Store 0-indexed for internal success stats
                # For overall stats, store 1-based episode count
                all_run_episode_counts.append(achieved_episode_in_run + 1)
                f.write(f"    成功    |   {achieved_episode_in_run + 1:3d}\n")  # Report 1-based episode
                print(f"Run {run_id + 1}: Success, solved in {achieved_episode_in_run + 1} episodes.") # Print 1-based
            else:
                # If failed, append the maximum number of episodes allowed for the run
                all_run_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN)
                f.write(f"    失败    |     --    \n")
                print(f"Run {run_id + 1}: Failed to solve within {MAX_TRAIN_EPISODES_PER_RUN} episodes.")

            # if 'bar' in locals(): bar() # Uncomment if adding outer bar


        # Close environment after all runs
        env_main.close()

        f.write("\n===== 统计摘要 =====\n")
        f.write(f"总运行次数: {N_RUNS}\n")
        f.write(f"成功次数 (达到{maximum_episode_length}帧): {success_count}\n") # Clarify success criterion
        f.write(f"失败次数: {N_RUNS - success_count}\n")

        # Calculate and report OVERALL statistics including failures
        if all_run_episode_counts: # Check if any runs were attempted
            # Use statistics.mean for robustness
            avg_episode_overall = statistics.mean(all_run_episode_counts)
            worst_episode_overall = max(all_run_episode_counts) # Max value in the list, including MAX_EPISODES_PER_RUN
            f.write(f"\n平均达成episode (总体, 失败记为{MAX_TRAIN_EPISODES_PER_RUN}): {avg_episode_overall:.1f}\n") # Updated description
            f.write(f"最差成绩 (总体): {worst_episode_overall}\n") # Overall worst is max episode count


        # Calculate and report statistics ONLY for successful runs
        if success_count > 0:
            # Convert 0-indexed success_episodes to 1-based for reporting
            successful_episodes_1based = [ep + 1 for ep in success_episodes]
            avg_episode_successful = statistics.mean(successful_episodes_1based)
            min_ep_successful = min(successful_episodes_1based)
            max_ep_successful = max(successful_episodes_1based) # Added max for successful runs
            f.write(f"平均达成episode (成功运行): {avg_episode_successful:.1f}\n") # Average among successful runs
            f.write(f"最佳成绩 (成功运行达成episode): {min_ep_successful} (episode越小越好)\n")
            f.write(f"最差成绩 (成功运行达成episode): {max_ep_successful}\n") # Worst among successful runs

            print(f"\nSummary: Avg episodes to solve (Successful Runs): {avg_episode_successful:.1f}, Best: {min_ep_successful}, Worst: {max_ep_successful}")
        else:
            f.write("\n所有运行均未达到目标帧数，无成功运行统计\n")
            print(f"\nSummary: No run successfully solved the environment within {MAX_TRAIN_EPISODES_PER_RUN} episodes.")

        f.write("\n===== 报告结束 =====")

    # --- MODIFICATION END ---

    print(f"\nOptimized parameter run report saved to {output_file}")
    # env_main.close() # Moved env.close() inside the 'with open' block before final print