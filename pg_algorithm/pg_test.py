'''
最初的Policy Gradient (REINFORCE) 版本，有可视化运行比较慢，使用自定义奖励。
'''

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
from alive_progress import alive_bar
matplotlib.use("QtAgg")

LR = 0.01           # learning rate
GAMMA = 0.9         # discount factor
maximum_episode_lenght = 500

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 个动作，左移 0，右移 1
N_STATES = env.observation_space.shape[0]  # 状态为 4 维，位置，夹角，速度，角速度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(10, 10)  # layer 2
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.out(x)
        actions_logporbs = F.log_softmax(x, dim=-1)
        return actions_logporbs


# 策略梯度代理
class PG(object):
    def __init__(self):
        self.pg_net = Net()  # pg 网络
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 实现中用的并不是这个loss，这里是遗留代码

        # 用于在每个回合中存储经历的（状态、动作、奖励）元组
        self.state = []
        self.action = []
        self.reward = []

    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x), 0)
        action_logporbs = self.pg_net(x)
        action = Dist.Categorical(logits=action_logporbs).sample()  # 这里是根据网络的输出概率分布来随机选择动作
        return action.numpy()[0]

    def clear_transition(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()

    def store_transition(self, s, a, r, s_):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)

    def learn(self):  # 执行策略网络的更新，在一个回合结束后调用
        state_array = torch.tensor(self.state)
        action_array = torch.tensor(
            self.action, dtype=torch.long).unsqueeze(-1)  # 将动作列表转换为长整型张量，并增加一个维度，使形状变成[回合长度。1]，这与后面gather函数的期望形状匹配
        reward_array = torch.tensor(self.reward)
        reward_array = reward_array-torch.mean(reward_array)
        reward_array /= torch.std(reward_array)  # 奖励归一化，一种常见的方差缩减技术，有助于稳定训练，正负奖励区分更清晰
        discount_reward_array = torch.zeros(len(self.reward))  # 计算折扣回报
        dr = 0
        for t in range(len(self.reward)-1, -1, -1):
            dr *= GAMMA
            dr += reward_array[t]
            discount_reward_array[t] = dr   # G_t 定义为 R_t + γ * R_{t+1} + γ² * R_{t+2} + ...。
        '''1. 将回合中的所有状态一次性输入网络，得到每个状态下所有动作的对数概率。形状是 [回合长度, 动作数量]。
           2. 根据 action_array 中存储的实际采取的动作索引，提取出每个时间步实际采取的那个动作的对数概率。结果形状是 [回合长度, 1]。'''
        action_logprobs = self.pg_net(state_array).gather(1, action_array)
        loss = -torch.mean(action_logprobs*discount_reward_array)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 模型训练
def demo():
    pg = PG()
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))  # 创建一个包含1行2列的Matplotlib图形的子图对象
    # 存储每个回合的数据（总步数，总奖励，回合编号），以便绘图
    steps = []
    reward = []
    episode = []
    plt.ion()  # 开启Matplotlib的交互模式，使得图可以动态更新

    print("\nCollecting experience...")
    with alive_bar(total=400, title="Training Episodes") as bar:  # 使用alive_progress 显示训练进度条
        for i_episode in range(400):
            episode.append(i_episode)
            s, s_info = env.reset() # 重置环境到初始状态，s是初始状态观测值， s_info是一些额外信息
            ep_r = 0  # 初始化本回合的总奖励
            step_count = 0  # 初始化本回合的步数计数
            pg.clear_transition()  #  清空PG代理中的存储的本回合之前的经验数据
            while True:
                image = env.render()

                ax[0].cla()
                ax[0].set_axis_off()
                ax[0].imshow(image)
                ax[0].set_title(f"Episode {i_episode} Frame {step_count}")
                plt.pause(0.01)

                step_count += 1
                a = pg.choose_action(s)
                s_, r, done, _, info = env.step(a)  # s_: 下一个状态   r:  在执行动作a后获得的奖励   done: 布尔值，指示回合是否结束
                # 重定义 reward
                x, x_dot, theta, theta_dot = s_
                rx = -(x/env.x_threshold)**2  # 越靠中间越好
                rtheta = -(theta/env.theta_threshold_radians)**2  # 角度越接近 0 越好
                r = rtheta + r+rx

                pg.store_transition(s, a, r, s_)

                ep_r += r
                if done or step_count >= maximum_episode_lenght:
                    print('Ep: ', i_episode, ' |',
                          'Ep_r: ', round(ep_r, 2))
                    break
                s = s_
            # 回合结束之后做的事情
            pg.learn()  # PG开始学习

            # 记录本回合的结果并更新绘图
            steps.append(step_count)
            reward.append(ep_r)
            ax[1].cla()
            ax[1].set_xlabel("episode")
            ax[1].set_ylabel("steps")
            ax[1].plot(episode, steps)
            ax[2].cla()
            ax[2].set_xlabel("episode")
            ax[2].set_ylabel("episode reward")
            ax[2].plot(episode, reward)
            plt.pause(0.1)

            bar()  # 更新进度条

    env.close()
    plt.ioff()

    savefig, saveax = plt.subplots(1, 2, figsize=(8, 4))
    saveax[0].set_xlabel("episode")
    saveax[0].set_ylabel("steps")
    saveax[0].plot(episode, steps)
    saveax[1].set_xlabel("episode")
    saveax[1].set_ylabel("episode reward")
    saveax[1].plot(episode, reward)
    savefig.savefig("Learning curve pg.png")
    visual_one_episode(pg)


def visual_one_episode(pg, max_length=1000):
    frames = []
    s, s_info = env.reset()
    while True:
        current_frame = env.render()
        frames.append(current_frame)
        a = pg.choose_action(s)
        s_, r, done, _, info = env.step(a)
        if done or len(frames) >= max_length:
            break
        s = s_
    imageio.mimsave("cartpole_rendering_pg.gif", frames)


if __name__ == "__main__":
    demo()
