# pg_test4.py - Policy Gradient with Original Reward, Baseline, RELU Net, and Manually Tuned Hidden Size

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable # Retained Variable for consistency
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os
import statistics # For average calculation
import time # Import time for timestamp

# --- Optimized Hyperparameters from PG_test2's Optuna run (kept for LR, GAMMA) ---
LR = 0.02016594287454386 # Keep the best LR from the first PG optimization
GAMMA = 0.9860175118246439 # Keep the best GAMMA from the first PG optimization

# --- Manually Tunable Hidden Size ---
# !!! MODIFY THIS VALUE to test different hidden layer sizes !!!
BEST_HIDDEN_SIZE = 16 # Example: Using 4 based on your request. Modify as needed.
# -----------------------------------

# Network Architecture Configuration
NUM_HIDDEN_LAYERS = 3 # Fixed number of hidden layers in our Net class


MAX_FRAMES_PER_EPISODE = 500 # CartPole-v1 solved criterion
NUM_EVAL_RUNS = 30 # Number of independent runs to evaluate the performance for the current config
MAX_TRAIN_EPISODES_PER_RUN = 200 # Max episodes allowed for training within a single run


# Environment setup
env = gym.make("CartPole-v1", render_mode="rgb_array") # Use rgb_array if visualization needed
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


# --- Neural Network Definition (Using ReLU and Kaiming Init, uses BEST_HIDDEN_SIZE) ---
class Net(nn.Module):
    # 这个网络使用了两个隐藏层，激活函数为 ReLU，并接受 hidden_size 参数
    def __init__(self, n_states, n_actions, hidden_size): # hidden_size 是必须传入的参数
        super(Net, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(n_states, hidden_size)
        # 使用 Kaiming Normal 初始化，适用于 ReLU
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.1)

        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 使用 Kaiming Normal 初始化
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.1)

        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # 使用 Kaiming Normal 初始化
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc3.bias, 0.1)

        # Output Layer
        self.out = nn.Linear(hidden_size, n_actions)
        # 输出层通常使用 Normal 初始化
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(self, x):
        # 使用 ReLU 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        # 输出层使用 log_softmax 获取动作的对数概率
        actions_logprobs = F.log_softmax(x, dim=-1)
        return actions_logprobs


# --- PG Agent (Using BEST_HIDDEN_SIZE and Original Reward) ---
# The logic here remains the same as PG_test5, only the Net definition changed.
class PG(object):
    def __init__(self):
        # Instantiate Net using the BEST_HIDDEN_SIZE
        self.pg_net = Net(N_STATES, N_ACTIONS, hidden_size=BEST_HIDDEN_SIZE)
        # Use the global LR constant
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=LR)
        # self.loss_func = nn.MSELoss() # Retained from original, but not used in PG loss

        # Store transitions per episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [] # Stores immediate ORIGINAL rewards

    def choose_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        self.pg_net.eval() # Set actor to eval mode for action selection
        with torch.no_grad(): # No gradient calculation needed
            action_logprobs = self.pg_net(x)
        self.pg_net.train() # Set actor back to train mode

        action_prob_dist = Dist.Categorical(logits=action_logprobs)
        action = action_prob_dist.sample()
        return action.item()

    def clear_memory(self):
        """Clears memory buffer after each episode learning."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    # Stores s, a, r (ORIGINAL environment reward)
    def store_transition(self, s, a, r):
        """Stores state, action, and immediate ORIGINAL reward for the episode."""
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r) # Store the original 'r'


    def learn(self):
        """Performs the Policy Gradient update using discounted ORIGINAL rewards."""
        # Check if any transitions were stored (avoid learning from empty episode)
        if len(self.episode_rewards) == 0:
            return

        # Convert lists to tensors
        state_array = torch.tensor(np.array(self.episode_states), dtype=torch.float32)
        action_array = torch.tensor(self.episode_actions, dtype=torch.long).unsqueeze(-1) # Actions need to be shape (N, 1)


        # --- Calculate Discounted Future Returns (Go-to-Go) ---
        # Use the collected ORIGINAL rewards
        discounted_returns = []
        current_return = 0
        # Iterate through immediate rewards in reverse order
        for r in reversed(self.episode_rewards):
            current_return = r + GAMMA * current_return # Use the global GAMMA
            discounted_returns.append(current_return)
        # Reverse the list to get returns in chronological order for the episode
        discounted_returns.reverse()
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32) # Shape (N,)


        # --- Apply Baseline (Subtract Episode Mean) and Normalize (Divide by Std Dev) ---
        # This is the built-in variance reduction from PG_test1/PG_test2/PG_test5/PG_test6
        if len(discounted_returns) > 1:
            mean = torch.mean(discounted_returns)
            std_dev = torch.std(discounted_returns)
            if std_dev > 1e-6:
                 # Apply normalization: (G_t - mean(G)) / std(G)
                 # The mean(G) serves as the episode-specific baseline
                 advantage = (discounted_returns - mean) / std_dev
            else:
                 # If std_dev is zero, just subtract mean
                 advantage = discounted_returns - mean
        else:
             # If only one step, use the return as advantage
             advantage = discounted_returns


        # Get log probabilities of the actions taken for the states in the episode
        self.pg_net.train() # Set actor to train mode
        action_logprobs_all = self.pg_net(state_array) # Shape (N, N_ACTIONS)
        action_logprobs = action_logprobs_all.gather(1, action_array) # Shape (N, 1)

        # Calculate Policy Gradient loss
        # Loss = - mean (log_prob * advantage)
        # advantage has shape (N,), action_logprobs has shape (N, 1)
        # Unsqueeze advantage to match shape for element-wise multiplication
        loss = -torch.mean(action_logprobs * advantage.unsqueeze(-1))


        # Perform backpropagation and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after learning from the episode
        self.clear_memory()


# --- Training and Evaluation Function for a Single Run ---
def run_single_training_session(run_id, pg_agent, env, max_episodes, max_frames_per_episode):
    achieved_episode = None # To store the episode number if successful

    print(f"\n--- Starting Run {run_id + 1}/{NUM_EVAL_RUNS} ---")

    # Iterate through training episodes for this single run
    for i_episode in range(max_episodes):
        s, s_info = env.reset()
        ep_r_accumulated = 0 # Accumulate original reward to check success criteria
        frame_count = 0
        # Clear memory at the start of each episode before collection
        pg_agent.clear_memory()

        # --- Run a single episode ---
        while True:
            a = pg_agent.choose_action(s)
            # Get ORIGINAL reward 'r' from env.step()
            s_, r, done, truncated, info = env.step(a)

            # Store state, action, and the ORIGINAL reward for LEARNING
            pg_agent.store_transition(s, a, r) # Store the original 'r'

            # Accumulate ORIGINAL reward for success criterion check and reporting
            ep_r_accumulated += r
            frame_count += 1

            # Check if the target frame count is reached. If so, terminate immediately.
            if frame_count >= max_frames_per_episode:
                achieved_episode = i_episode # Record the episode number (0-indexed)
                print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Reached {frame_count} frames (Success!)")
                break # Terminate the episode immediately

            # Check for natural termination conditions (pole fell, went out of bounds)
            if done or truncated:
                 # If done/truncated occurred *before* reaching max_frames_per_episode, it's a failure for this episode.
                 break # Terminate the episode

            s = s_ # Move to the next state

        # --- Learning Step (after episode ends) ---
        # Only learn if the episode wasn't empty and transitions were stored
        # The learn function will use the stored ORIGINAL rewards
        if len(pg_agent.episode_rewards) > 0:
             pg_agent.learn()

        # --- Check if this run was successful (reached the target frame count) ---
        # Stop training for this run as soon as the goal is achieved
        if achieved_episode is not None:
            break # Break from the episode loop for this run

    # If the loop finishes without achieving the goal within max_episodes
    if achieved_episode is None:
         print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}: Did not reach {max_frames_per_episode} frames within {max_episodes} episodes.")

    # Return the episode number where success was achieved, or None if not successful
    return achieved_episode


# --- Main Execution Block ---
if __name__ == "__main__":
    # Construct the output path relative to the script's potential location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in pg_algorithm/pg_test4.py and output is in pg_output/pg_test4.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test4.txt") # Output file name changed to pg_test4.txt


    # --- MODIFICATION: Open file in append mode and write header for this run ---
    with open(output_file, "a") as f: # Use "a" for append mode
        # Add a separator and header for this specific configuration run
        f.write("\n" + "="*80 + "\n")
        f.write(f"===== Results for Network Config: {NUM_HIDDEN_LAYERS} Hidden Layers, Size {BEST_HIDDEN_SIZE} =====\n") # Describe network
        f.write(f"Timestamp: {time.ctime()}\n") # Add timestamp
        f.write(f"Hyperparameters: LR={LR}, GAMMA={GAMMA}\n")
        f.write(f"Reward Signal for Learning: Original Environment Reward (+1 per step)\n")
        f.write(f"Baseline Method: Episode Mean Return Subtraction + Normalization\n")
        f.write(f"Success Criterion: Reach {MAX_FRAMES_PER_EPISODE} frames (episode terminates immediately)\n")
        f.write(f"Max Episodes per Run: {MAX_TRAIN_EPISODES_PER_RUN}\n")
        f.write("-" * 80 + "\n")
        f.write("Run ID | Success | Episode Achieved | Frames Achieved\n")
        f.write("-" * 80 + "\n")

        print(f"Starting {NUM_EVAL_RUNS} independent training runs with config: {NUM_HIDDEN_LAYERS} Hidden Layers, Size {BEST_HIDDEN_SIZE}")
        print(f"Hyperparameters: LR={LR}, GAMMA={GAMMA}")
        print(f"Results will be appended to {output_file}")


        success_count = 0
        success_episodes = [] # Stores episode numbers ONLY for successful runs
        all_episode_counts = [] # Stores episode numbers for ALL runs (MAX_TRAIN_EPISODES_PER_RUN for failures)


        # Using alive_bar to show progress of the NUM_EVAL_RUNS runs
        with alive_bar(total=NUM_EVAL_RUNS, title="Overall Training Progress") as bar:
             for run_id in range(NUM_EVAL_RUNS):
                # Create a new agent for each independent run
                pg_agent = PG() # Using the modified PG class which uses the current BEST_HIDDEN_SIZE Net
                achieved_episode = run_single_training_session(
                    run_id, pg_agent, env,
                    MAX_TRAIN_EPISODES_PER_RUN, MAX_FRAMES_PER_EPISODE
                )

                # After a run finishes, log the result to the file inside the append block
                f.write(f"  {run_id + 1:02d}   |")

                if achieved_episode is not None:
                    success_count += 1
                    success_episodes.append(achieved_episode)
                    all_episode_counts.append(achieved_episode)
                    f.write(f"    Yes    |   {achieved_episode:03d}        |   {MAX_FRAMES_PER_EPISODE}\n")
                else:
                    all_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN) # Count failure as max episodes
                    f.write(f"     No    |     ---        |     ---    \n")

                bar() # Update the overall progress bar after this run finishes

        # --- MODIFICATION: Write statistical summary for this run ---
        f.write("-" * 80 + "\n")
        f.write("\n===== Statistical Summary for this Config =====\n") # Updated header
        f.write(f"Total Runs: {NUM_EVAL_RUNS}\n")
        f.write(f"Success Count (Reached {MAX_FRAMES_PER_EPISODE} frames): {success_count}\n")
        f.write(f"Failure Count: {NUM_EVAL_RUNS - success_count}\n")
        f.write(f"Success Rate: {success_count / NUM_EVAL_RUNS:.1%}\n")

        if all_episode_counts: # Check if any runs were performed
             avg_episode_overall = statistics.mean(all_episode_counts)
             worst_episode_overall = max(all_episode_counts)
             f.write(f"\nAverage Episode to Achieve {MAX_FRAMES_PER_EPISODE} Frames (Overall, failures counted as {MAX_TRAIN_EPISODES_PER_RUN}): {avg_episode_overall:.1f}\n")
             f.write(f"Worst Episode (Overall max, including failures): {worst_episode_overall}\n")


        if success_count > 0:
            avg_episode_successful = statistics.mean(success_episodes)
            best_episode_successful = min(success_episodes)
            f.write(f"Average Episode (min among successful runs): {avg_episode_successful:.1f}\n")
            f.write(f"Best Episode (min among successful runs): {best_episode_successful}\n")
            f.write(f"Frames achieved in successful runs: {MAX_FRAMES_PER_EPISODE}\n")
        else:
            f.write(f"No runs achieved {MAX_FRAMES_PER_EPISODE} frames within {MAX_TRAIN_EPISODES_PER_RUN} episodes.\n")

        f.write("\n" + "="*80 + "\n") # Add a clear separator after this summary


    # Close the environment after all evaluation runs for this config
    env.close()

    print(f"\nEvaluation for config ({NUM_HIDDEN_LAYERS} Layers, Size {BEST_HIDDEN_SIZE}) complete. Results appended to {output_file}")