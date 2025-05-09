'''
在test3基础上网路加了一层隐藏层之后的版本
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

# 超参数保持不变
BATCH_SIZE = 256
LR = 0.0024303199955094507
GAMMA = 0.9556460180442318
TARGET_NETWORK_REPLACE_FREQ = 300
MEMORY_CAPACITY = 1000
EPSILON_START = 0.9212305525092727
EPSILON_END = 0.045039045287420844
EPSILON_DECAY_STEPS = 7000
HIDDEN_SIZE = 64  # 这个hidden_size现在会用于两个隐藏层
maximum_episode_length = 500


class Net(nn.Module):  # 网络结构修改在这里
    def __init__(self, n_states, n_actions, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 新增的隐藏层
        self.fc2.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(hidden_size, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # 通过新增的隐藏层
        return self.out(x)


class DDQN(object):
    def __init__(self, n_states, n_actions, env_a_shape):
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape

        self.eval_net = Net(self.n_states, self.n_actions, HIDDEN_SIZE)  # 会使用新的Net定义
        self.target_net = Net(self.n_states, self.n_actions, HIDDEN_SIZE)  # 会使用新的Net定义
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

        q_eval_next_actions_indices = self.eval_net(b_s_).detach().max(1)[1].view(BATCH_SIZE, 1)
        q_target_next_all_values = self.target_net(b_s_).detach()
        q_target_next_selected_value = q_target_next_all_values.gather(1, q_eval_next_actions_indices)

        q_target = b_r + GAMMA * q_target_next_selected_value

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_single_experiment_ddqn_deeper_net(env_instance, n_states, n_actions, env_a_shape):
    ddqn_agent = DDQN(n_states, n_actions, env_a_shape)  # DDQN将使用新的更深的网络
    MAX_EPISODES_PER_RUN = 200
    achieved_episode = None

    for i_episode in range(MAX_EPISODES_PER_RUN):
        s, _ = env_instance.reset()
        ep_r = 0
        step_count = 0

        while True:
            a = ddqn_agent.choose_action(s)
            s_, r, done, truncated, _ = env_instance.step(a)

            x, _, theta, _ = s_
            rx = -(x / env_instance.x_threshold) ** 2
            rtheta = -(theta / env_instance.theta_threshold_radians) ** 2
            modified_r = r + (rx + rtheta) * 0.1

            if done and step_count < maximum_episode_length - 1:
                modified_r = -100

            ddqn_agent.store_transition(s, a, modified_r, s_)
            ep_r += r

            if ddqn_agent.memory_counter > BATCH_SIZE:
                ddqn_agent.learn()

            if done or truncated or step_count >= maximum_episode_length - 1:
                if step_count >= maximum_episode_length - 1:
                    achieved_episode = i_episode
                break
            s = s_
            step_count += 1

        if achieved_episode is not None:
            break

    return achieved_episode


if __name__ == "__main__":
    env_main = gym.make("CartPole-v1")
    env_main = env_main.unwrapped
    N_ACTIONS_MAIN = env_main.action_space.n
    N_STATES_MAIN = env_main.observation_space.shape[0]
    ENV_A_SHAPE_MAIN = 0 if isinstance(env_main.action_space.sample(), int) else env_main.action_space.sample().shape

    results_filename = "dqn_test4.txt"  # 新的文件名
    with open(results_filename, "a") as f:
        f.write("===== DDQN (Deeper Net) 训练结果报告 (使用优化参数) =====\n")  # 标明是Deeper Net
        f.write(
            f"Parameters: BATCH_SIZE={BATCH_SIZE}, LR={LR:.6f}, GAMMA={GAMMA:.6f}, TARGET_NETWORK_REPLACE_FREQ={TARGET_NETWORK_REPLACE_FREQ}, MEMORY_CAPACITY={MEMORY_CAPACITY}, EPSILON_START={EPSILON_START:.6f}, EPSILON_END={EPSILON_END:.6f}, EPSILON_DECAY_STEPS={EPSILON_DECAY_STEPS}, HIDDEN_SIZE={HIDDEN_SIZE} (x2 layers), MAX_EP_LEN={maximum_episode_length}\n")  # 标明HIDDEN_SIZE用于两层
        f.write("运行次数 | 是否成功 | 达成episode\n")
        f.write("-" * 50 + "\n")

        success_count = 0
        success_episodes = []
        N_RUNS = 30

        for run_id in range(N_RUNS):
            print(f"\n--- Starting DDQN (Deeper Net) Run {run_id + 1}/{N_RUNS} ---")  # 标明
            f.write(f"  第{run_id + 1:02d}次  |")

            # 函数名也改一下以明确
            achieved_episode_in_run = run_single_experiment_ddqn_deeper_net(env_main, N_STATES_MAIN, N_ACTIONS_MAIN,
                                                                            ENV_A_SHAPE_MAIN)

            if achieved_episode_in_run is not None:
                success_count += 1
                success_episodes.append(achieved_episode_in_run)
                f.write(f"    成功    |   {achieved_episode_in_run:3d}\n")
                print(f"DDQN (Deeper Net) Run {run_id + 1}: Success, solved in {achieved_episode_in_run} episodes.")
            else:
                f.write(f"    失败    |     --    \n")
                print(f"DDQN (Deeper Net) Run {run_id + 1}: Failed to solve within episodes.")

        f.write("\n===== DDQN (Deeper Net) 统计摘要 =====\n")  # 标明
        f.write(f"总运行次数: {N_RUNS}\n")
        f.write(f"成功次数: {success_count}\n")
        f.write(f"失败次数: {N_RUNS - success_count}\n")

        if success_count > 0:
            avg_episode = sum(success_episodes) / success_count
            min_ep = min(success_episodes)
            max_ep = max(success_episodes)
            f.write(f"\n平均达成episode: {avg_episode:.1f}\n")
            f.write(f"最佳成绩 (达成episode): {min_ep}\n")
            f.write(f"最差成绩 (达成episode): {max_ep}\n")
            print(
                f"\nDDQN (Deeper Net) Summary: Avg episodes to solve: {avg_episode:.1f}, Best: {min_ep}, Worst: {max_ep}")
        else:
            f.write("\n所有DDQN (Deeper Net)运行均未达到目标帧数\n")  # 标明
            print("\nDDQN (Deeper Net) Summary: No run successfully solved the environment.")

        f.write("\n===== DDQN (Deeper Net) 报告结束 =====")  # 标明

    print(f"\nDDQN (Deeper Net) run report saved to {results_filename}")
    env_main.close()