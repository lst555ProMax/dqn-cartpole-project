'''
简化代码，去除可视化，同时一旦frame达到500就停止运行，记录当前episode值，以上重复30次计算episode均值作为模型的评价指标
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar

# 超参数
BATCH_SIZE = 100
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100
MEMORY_CAPACITY = 1000
maximum_episode_length = 500

# 创建环境
env = gym.make("CartPole-v1")
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
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
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

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
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

    with alive_bar() as bar:
        i_episode = 0
        while i_episode < 200 and not max_frames_achieved:
            s = env.reset()[0]
            ep_r = 0
            step_count = 0

            while True:
                bar()
                a = dqn.choose_action(s)
                s_, r, done, _, _ = env.step(a)

                # 自定义奖励
                x, _, theta, _ = s_
                rx = -(x / env.x_threshold) ** 2
                rtheta = -(theta / env.theta_threshold_radians) ** 2
                r += rx + rtheta

                dqn.store_transition(s, a, r, s_)
                ep_r += r

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if done or step_count >= maximum_episode_length:
                    if step_count >= maximum_episode_length:
                        print(f'\nReached 500 frames at episode {i_episode}!')
                        max_frames_achieved = True
                        achieved_episode = i_episode
                    print(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)}')
                    break

                s = s_
                step_count += 1

            if max_frames_achieved:
                break
            i_episode += 1

    return achieved_episode if max_frames_achieved else None


if __name__ == "__main__":
    with open("dqn_test1.txt", "a") as f:
        f.write("===== 训练结果报告 =====\n")
        f.write("运行次数 | 是否成功 | 达成episode\n")
        f.write("-" * 40 + "\n")

        success_count = 0
        success_episodes = []

        for run_id in range(30):
            f.write(f"  第{run_id + 1:02d}次  |")
            achieved_episode = demo()

            if achieved_episode is not None:
                success_count += 1
                success_episodes.append(achieved_episode)
                f.write(f"    成功    |   {achieved_episode}\n")
            else:
                f.write(f"    失败    |     --    \n")

        # 写入统计摘要
        f.write("\n===== 统计摘要 =====\n")
        f.write(f"总运行次数: 30\n")
        f.write(f"成功次数: {success_count}\n")
        f.write(f"失败次数: {30 - success_count}\n")

        if success_count > 0:
            avg_episode = sum(success_episodes) / success_count
            f.write(f"\n平均达成episode: {avg_episode:.1f}\n")
            f.write(f"最佳成绩: {min(success_episodes)} (episode越小越好)\n")
            f.write(f"最差成绩: {max(success_episodes)}\n")
        else:
            f.write("\n所有运行均未达到500帧\n")

        f.write("\n===== 报告结束 =====")

    print("完整报告已保存至 dqn_test1.txt")