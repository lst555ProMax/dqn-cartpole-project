'''
在pg_test基础上，优化评估流程和指标：去除可视化，frame达到500即停止并记录episode，重复30次计算总体平均episode。使用自定义奖励和回合平均Baseline。
'''

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os
import statistics # For average calculation

LR = 0.01
GAMMA = 0.9
MAX_FRAMES_PER_EPISODE = 500 # Renamed for clarity

NUM_EVAL_RUNS = 30
MAX_TRAIN_EPISODES_PER_RUN = 200

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)

        self.fc2 = nn.Linear(10, 10)
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

        self.out = nn.Linear(10, N_ACTIONS)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.out(x)
        actions_logporbs = F.log_softmax(x, dim=-1)
        return actions_logporbs


class PG(object):
    def __init__(self):
        self.pg_net = Net()
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.state = []
        self.action = []
        self.reward = []

    def choose_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        action_logporbs = self.pg_net(x)
        action_prob_dist = Dist.Categorical(logits=action_logporbs)
        action = action_prob_dist.sample()
        return action.item()

    def clear_transition(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()

    def store_transition(self, s, a, r, s_):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)

    def learn(self):
        state_array = torch.tensor(np.array(self.state), dtype=torch.float32)
        action_array = torch.tensor(self.action, dtype=torch.long).unsqueeze(-1)
        reward_array = torch.tensor(self.reward, dtype=torch.float32)

        reward_array = reward_array - torch.mean(reward_array)
        std_dev = torch.std(reward_array)
        if std_dev > 1e-6:
             reward_array /= std_dev

        discound_reward_array = torch.zeros_like(reward_array)
        current_return = 0
        for t in range(len(self.reward)-1, -1, -1):
            current_return = reward_array[t] + GAMMA * current_return
            discound_reward_array[t] = current_return

        all_action_logprobs = self.pg_net(state_array)
        action_logprobs = all_action_logprobs.gather(1, action_array)

        loss = -torch.mean(action_logprobs * discound_reward_array.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_single_training_session(run_id, pg_agent, env, max_episodes, max_frames_per_episode):
    achieved_episode = None

    print(f"\n--- Starting Run {run_id + 1}/{NUM_EVAL_RUNS} ---")

    for i_episode in range(max_episodes):
        s, s_info = env.reset()
        ep_r = 0
        frame_count = 0
        pg_agent.clear_transition()

        while True:
            a = pg_agent.choose_action(s)
            s_, r, done, truncated, info = env.step(a)

            x, x_dot, theta, theta_dot = s_
            rx = -(x / env.x_threshold)**2
            rtheta = -(theta / env.theta_threshold_radians)**2
            r = rtheta + r + rx

            pg_agent.store_transition(s, a, r, s_)

            ep_r += r

            frame_count += 1

            if done or truncated:
                if frame_count >= max_frames_per_episode:
                    achieved_episode = i_episode
                    print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Reached {frame_count} frames (Success!)")
                # else: # Optional: print failed episodes within a run
                #     print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Achieved {frame_count} frames (Failed)")
                break

            s = s_

        if len(pg_agent.reward) > 0:
             pg_agent.learn()

        if achieved_episode is not None:
            break

    if achieved_episode is None:
         print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}: Did not reach {max_frames_per_episode} frames within {max_episodes} episodes.")

    return achieved_episode


if __name__ == "__main__":
    # Construct the output path relative to the script's potential location
    # Assuming script is in pg_algorithm/pg_test1.py and output is in pg_output/pg_test1.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Go up one level from pg_algorithm
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test1.txt")


    with open(output_file, "w") as f:
        f.write("===== Policy Gradient Training Results Report =====\n")
        f.write("Run ID | Success | Episode Achieved\n")
        f.write("-" * 40 + "\n")

        success_count = 0
        success_episodes = [] # Stores episode numbers ONLY for successful runs
        all_episode_counts = [] # Stores episode numbers for ALL runs (MAX_TRAIN_EPISODES_PER_RUN for failures)


        print(f"\nStarting {NUM_EVAL_RUNS} independent training runs...")
        # Using alive_bar to show progress of the 30 runs
        with alive_bar(total=NUM_EVAL_RUNS, title="Overall Progress") as bar:
             for run_id in range(NUM_EVAL_RUNS):
                pg = PG()
                achieved_episode = run_single_training_session(
                    run_id, pg, env,
                    MAX_TRAIN_EPISODES_PER_RUN, MAX_FRAMES_PER_EPISODE
                )

                f.write(f"  {run_id + 1:02d}   |")

                if achieved_episode is not None:
                    success_count += 1
                    success_episodes.append(achieved_episode)
                    all_episode_counts.append(achieved_episode)
                    f.write(f"    Yes    |   {achieved_episode}\n")
                else:
                    all_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN) # Count failure as max episodes
                    f.write(f"     No    |     --    \n")

                bar() # Update the overall progress bar after this run finishes

        env.close()

        f.write("\n===== Statistical Summary =====\n")
        f.write(f"Total Runs: {NUM_EVAL_RUNS}\n")
        f.write(f"Success Count (Reached {MAX_FRAMES_PER_EPISODE} frames): {success_count}\n")
        f.write(f"Failure Count: {NUM_EVAL_RUNS - success_count}\n")
        f.write(f"Success Rate: {success_count / NUM_EVAL_RUNS:.1%}\n") # Added success rate

        if all_episode_counts: # Check if any runs were performed
             avg_episode_overall = statistics.mean(all_episode_counts)
             worst_episode_overall = max(all_episode_counts)
             f.write(f"\nAverage Episode to Achieve {MAX_FRAMES_PER_EPISODE} Frames (Overall, failures counted as {MAX_TRAIN_EPISODES_PER_RUN}): {avg_episode_overall:.1f}\n")
             f.write(f"Worst Episode (Overall max): {worst_episode_overall}\n")

        if success_count > 0:
            best_episode_successful = min(success_episodes)
            f.write(f"Best Episode (min among successful runs): {best_episode_successful}\n")
        else:
            f.write(f"No runs achieved {MAX_FRAMES_PER_EPISODE} frames within {MAX_TRAIN_EPISODES_PER_RUN} episodes.\n")


        f.write("\n===== Report End =====\n")

    print(f"\nEvaluation complete. Report saved to {output_file}")