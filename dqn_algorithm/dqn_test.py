'''
最初的版本，有可视化运行比较慢
'''


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
from alive_progress import alive_bar
matplotlib.use("QtAgg")

BATCH_SIZE = 100    # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100      # How frequently target netowrk updates
MEMORY_CAPACITY = 1000                  # The capacity of experience replay buffer
maximum_episode_length = 500

#环境初始化
env = gym.make("CartPole-v1", render_mode="rgb_array")  #小车平衡杆子问题    渲染模式为rgb数组
env = env.unwrapped   #移除Gym环境默认的时间限制等包装，让环境处于最原始状态。
N_ACTIONS = env.action_space.n  # 2 个动作，左移 0，右移 1
N_STATES = env.observation_space.shape[0]  # 状态为 4 维，位置，夹角，速度，角速度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1) # 随机初始化参数，高斯分布，均值，标准差
        self.out = nn.Linear(10, N_ACTIONS)  # layer 2
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)   #x  输入
        x = F.tanh(x)
        actions_value = self.out(x)
        return actions_value

# 3. Define the DQN network and its corresponding methods


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # target目标网络 和 eval评估网络
        self.learn_step_counter = 0  # 步数计数
        self.memory_counter = 0  # 回放缓冲计数

        # 每行为一个 transition，有两个状态，一个动作，一个奖励，所以是 N_STATES * 2 + 2 维向量。
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  #LR是学习率

        self.loss_func = nn.MSELoss()

    def choose_action(self, x):   # x是当前环境的状态输入
        x = torch.unsqueeze(torch.tensor(x), 0)    # 输入预处理，首先将输入地状态转换成tensor张量，然后unsqueeze将张量变成批处理（batch）格式
        if np.random.uniform() < EPSILON:   # epsilon greedy 方法    np.random.uniform(): 生成一个 0.0 到 1.0 之间的随机浮点数。
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()   # max（沿第一个维度寻找最大值）返回两个值，最大值本身和最大值的索引    data获取不带梯度信息的原始数据张量   再转化为numpy数组
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        else:   # （1 - EPSLION 的概率）
            action = np.random.randint(0, N_ACTIONS)    # 这里就是从0-2中随机选择一个整数，即0或者1，代表向左或者向右
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def deploy_choose_action(self, x):   # 在训练完成后，或者在需要评估智能体当前学习效果时，完全根据已学习到的策略来选择动作，不再进行随机探索
        x = torch.unsqueeze(torch.tensor(x), 0)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
            ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):   # 算法中经验回放的部分
        # 循环覆盖 buffer
        transition = np.hstack((s, [a, r], s_))    #将这几个部分水平堆叠起来，组合成一个新的一维Numpy数组
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 将 eval 权重赋值给 target
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            print("update target")
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽样本
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(
            b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        # 计算target
        #贝尔曼最优方程：Q*(s, a) = E[R_{t+1} + γ * max_a' Q*(s', a')]
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 评估网络对实际采取的动作 b_a 所预测的Q值
        q_next = self.target_net(b_s_).detach()  # 计算下一状态的所有 q 值
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # q_target = 即时奖励+折扣因子 * （从下一个状态s`出发能获得的最大未来Q值）
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()   #自动计算所有参数的梯度
        self.optimizer.step()


'''
--------------Procedures of DQN Algorithm------------------
'''


def demo():
    dqn = DQN()
    steps = []
    reward = []
    episode = []

    max_frames_achieved = False  # 新增终止标志
    achieved_episode = None  # 记录达到500帧的episode

    #可视化设置
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))    # 3个子图：环境渲染、训练步数、奖励曲线
    plt.ion()   # 开启交互模式，允许实时更新图表
    print("\nCollecting experience...")

    with alive_bar() as bar:
        # 将固定循环改为while循环以便提前终止
        i_episode = 0
        while i_episode < 200 and not max_frames_achieved:
            episode.append(i_episode)
            s, s_info = env.reset()    #重置环境，获得初始状态
            ep_r = 0     # 当前episode累计奖励
            step_count = 0    # 当前episode步数

            while True:
                bar()
                image = env.render()
                ax[0].cla()
                ax[0].set_axis_off()
                ax[0].imshow(image)
                step_count += 1
                ax[0].set_title(f"Episode {i_episode} Frame {step_count}")
                plt.pause(0.01)

                a = dqn.choose_action(s)   #选择动作
                s_, r, done, _, info = env.step(a)  # 执行动作

                # 重定义 reward
                x, x_dot, theta, theta_dot = s_
                rx = -(x/env.x_threshold)**2  # 越靠中间越好
                rtheta = -(theta/env.theta_threshold_radians)**2  # 角度越接近 0 越好
                r = rtheta + r+rx    # 综合奖励

                dqn.store_transition(s, a, r, s_)   #存储经验

                ep_r += r
                # 等 buffer 满了再开始训练
                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if done or step_count >= maximum_episode_length:
                    # 检测是否达到500帧
                    if step_count >= maximum_episode_length:
                        print(f'\nReached 500 frames at episode {i_episode}!')
                        max_frames_achieved = True
                        achieved_episode = i_episode
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
                    break
                s = s_

            # 如果达到500帧则终止训练
            if max_frames_achieved:
                env.close()  # 确保提前退出时环境被关闭
                plt.close()  # 关闭图表避免残留窗口
                break

            #记录训练数据
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
            plt.pause(0.2)

            i_episode += 1  # 手动递增episode计数器

    env.close()
    plt.ioff()

    # 写入结果到文件
    if max_frames_achieved:
        with open("../dqn_output/dqn_test.txt", "w") as f:
            f.write(f"Model achieved 500 frames at episode: {achieved_episode}")
        print(f"\nSuccess! Log saved to dqn_test.txt")
    else:
        print("\nTraining completed without reaching 500 frames")

    savefig, saveax = plt.subplots(1, 2, figsize=(8, 4))
    saveax[0].set_xlabel("episode")
    saveax[0].set_ylabel("steps")
    saveax[0].plot(episode, steps)
    saveax[1].set_xlabel("episode")
    saveax[1].set_ylabel("episode reward")
    saveax[1].plot(episode, reward)

    savefig.savefig("Learning curve.png")  #保存训练结果
    visual_one_episode(dqn)   #生成演示视频


def visual_one_episode(dqn, max_length=1000):
    frames = []
    s, s_info = env.reset()
    while True:
        current_frame = env.render()
        frames.append(current_frame)
        a = dqn.deploy_choose_action(s)
        s_, r, done, _, info = env.step(a)
        if done or len(frames) >= max_length:
            break
        s = s_
    imageio.mimsave("cartpole_rendering.gif", frames)


if __name__ == "__main__":
    demo()

