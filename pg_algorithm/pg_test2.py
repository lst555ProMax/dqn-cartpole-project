'''
标准Policy Gradient，应用pg_test1_optuna找到的优化超参数，使用自定义奖励和Tanh网络。
'''

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable # Keep Variable for compatibility with original structure
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os
import statistics # For average calculation

# --- Best Hyperparameters found by Optuna ---
# Original values were: LR=0.01, GAMMA=0.9, HIDDEN_SIZE=10 (implicitly in Net definition)
LR = 0.02016594287454386
GAMMA = 0.9860175118246439
BEST_HIDDEN_SIZE = 16
# --------------------------------------------


MAX_FRAMES_PER_EPISODE = 500 # CartPole-v1 solved criterion
NUM_EVAL_RUNS = 30 # Number of independent runs to evaluate the performance of the fixed hyperparameters
MAX_TRAIN_EPISODES_PER_RUN = 200 # Maximum episodes allowed for training within a single run

# Environment setup
# Use render_mode="rgb_array" for potential visualization if needed, or None for faster execution if not visualizing
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


# --- Neural Network Definition (Modified to accept hidden_size) ---
class Net(nn.Module):
    # Added hidden_size parameter with a default value (though we'll override it)
    def __init__(self, n_states, n_actions, hidden_size=10):
        super(Net, self).__init__()
        # Use the provided hidden_size for layer dimensions
        self.fc1 = nn.Linear(n_states, hidden_size)
        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second hidden layer also uses hidden_size
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

        self.out = nn.Linear(hidden_size, n_actions)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(self, x):
        # Retained Tanh activation as in original PG_test1
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.out(x)
        actions_logprobs = F.log_softmax(x, dim=-1) # Use log_softmax for log probabilities
        return actions_logprobs


# --- PG Agent (Modified to use the BEST_HIDDEN_SIZE) ---
class PG(object):
    def __init__(self):
        # Instantiate Net using the BEST_HIDDEN_SIZE found by Optuna
        self.pg_net = Net(N_STATES, N_ACTIONS, hidden_size=BEST_HIDDEN_SIZE)
        # Use the global LR constant (which is now the best LR)
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss() # Retained from original, though not used in PG loss

        # Store transitions per episode
        # Renamed lists for clarity
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [] # Stores immediate rewards

    def choose_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        # Ensure network is in evaluation mode for action selection (important for some layers like Dropout, BatchNorm, though not strictly needed here)
        # self.pg_net.eval() # Optional, usually done for pure inference
        action_logprobs = self.pg_net(x)
        # self.pg_net.train() # Optional, switch back if eval was used

        action_prob_dist = Dist.Categorical(logits=action_logprobs)
        action = action_prob_dist.sample()
        return action.item()

    def clear_transition(self):
        """Clears memory buffer after each episode learning."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    # Modified to only store s, a, r as needed for PG
    def store_transition(self, s, a, r):
        """Stores state, action, and immediate reward for the episode."""
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)

    def learn(self):
        """Performs the policy gradient update based on episode memory."""
        # Convert lists to tensors
        state_array = torch.tensor(np.array(self.episode_states), dtype=torch.float32)
        action_array = torch.tensor(self.episode_actions, dtype=torch.long).unsqueeze(-1) # Actions need to be shape (N, 1)

        # Calculate discounted future returns (Go-to-Go)
        discounted_rewards = []
        current_return = 0
        # Iterate through immediate rewards in reverse order
        # Use the global GAMMA constant (which is now the best GAMMA)
        for r in reversed(self.episode_rewards):
            current_return = r + GAMMA * current_return
            discounted_rewards.append(current_return)
        # Reverse the list to get rewards in chronological order for the episode
        discounted_rewards.reverse()
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Optional: Normalize discounted rewards (recommended for PG stability)
        if len(discounted_rewards) > 1:
            mean = torch.mean(discounted_rewards)
            std_dev = torch.std(discounted_rewards)
            if std_dev > 1e-6:
                 discounted_rewards = (discounted_rewards - mean) / std_dev
            else:
                 # If std_dev is zero, just subtract mean
                 discounted_rewards = discounted_rewards - mean

        # Get log probabilities of the actions taken for the states in the episode
        # Ensure network is in training mode before forward pass for learning
        self.pg_net.train()
        all_action_logprobs = self.pg_net(state_array)
        # Select the log probability for the specific action taken at each state
        action_logprobs = all_action_logprobs.gather(1, action_array) # shape (N, 1)

        # Calculate the Policy Gradient loss
        # Loss = - mean (log_prob * discounted_return)
        loss = -torch.mean(action_logprobs * discounted_rewards.unsqueeze(-1))

        # Perform backpropagation and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after learning from the episode
        self.clear_transition()


# --- Training and Evaluation Function for a Single Run ---
def run_single_training_session(run_id, pg_agent, env, max_episodes, max_frames_per_episode):
    achieved_episode = None # To store the episode number if successful

    print(f"\n--- Starting Run {run_id + 1}/{NUM_EVAL_RUNS} ---")

    # Iterate through training episodes for this single run
    for i_episode in range(max_episodes):
        s, s_info = env.reset()
        ep_r = 0 # Accumulate original reward to check success criteria
        frame_count = 0
        # Clear memory at the start of each episode before collection
        pg_agent.clear_transition()

        # --- Run a single episode ---
        while True:
            a = pg_agent.choose_action(s)
            s_, r, done, truncated, info = env.step(a)

            # --- Custom Reward (copied from original PG_test1) ---
            # Used for LEARNING, not for success evaluation
            x, x_dot, theta, theta_dot = s_
            rx = -(x / env.x_threshold)**2
            rtheta = -(theta / env.theta_threshold_radians)**2
            custom_r = rtheta + r + rx # Use this for store_transition

            # Store state, action, and the custom reward for LEARNING
            pg_agent.store_transition(s, a, custom_r)

            # Accumulate ORIGINAL reward to check CartPole's success criterion (500 steps = 500 reward)
            ep_r += r
            frame_count += 1

            # --- MODIFICATION START ---
            # Check if the target frame count is reached. If so, terminate immediately.
            if frame_count >= max_frames_per_episode:
                achieved_episode = i_episode # Record the episode number (0-indexed)
                print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Reached {frame_count} frames (Success!)")
                break # Terminate the episode immediately
            # --- MODIFICATION END ---

            # Check for natural termination conditions (pole fell, went out of bounds)
            # This check now happens *after* the success check, so if frame_count hits 500
            # exactly, the success condition is met first.
            if done or truncated:
                # If done/truncated occurred *before* reaching max_frames_per_episode, it's a failure for this episode.
                # We don't need to explicitly set achieved_episode = None here, as it's initialized as None.
                # print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Achieved {frame_count} frames (Failed due to done/truncated).") # Optional detailed log
                break # Terminate the episode

            s = s_ # Move to the next state

        # --- Learning Step (after episode ends) ---
        # Only learn if the episode wasn't empty and transitions were stored
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
    # Assuming script is in pg_algorithm/pg_test2.py and output is in pg_output/pg_test2.txt
    # Adjust the number of os.path.dirname calls based on your actual directory structure
    # If pg_test2.py is directly in the project root, use script_dir.
    # If it's in a subdirectory like 'pg_algorithm', go up one level.
    # Let's assume it's in a subdir and we want output in a parallel 'pg_output' dir
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir (e.g., from 'pg_algorithm')
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test2.txt")

    print(f"Starting {NUM_EVAL_RUNS} independent training runs with optimized hyperparameters...")
    print(f"LR: {LR}, GAMMA: {GAMMA}, HIDDEN_SIZE: {BEST_HIDDEN_SIZE}")

    # Open the output file in write mode ('w') to clear any previous content
    with open(output_file, "w") as f:
        f.write("===== Policy Gradient Training Results Report (Optimized Hyperparams & Early Termination) =====\n")
        f.write(f"Hyperparameters: LR={LR}, GAMMA={GAMMA}, HIDDEN_SIZE={BEST_HIDDEN_SIZE}\n")
        f.write(f"Success Criterion: Reach {MAX_FRAMES_PER_EPISODE} frames (episode terminates immediately)\n") # Updated description
        f.write(f"Max Episodes per Run: {MAX_TRAIN_EPISODES_PER_RUN}\n")
        f.write("-" * 60 + "\n")
        f.write("Run ID | Success | Episode Achieved | Frames Achieved\n") # Updated column header
        f.write("-" * 60 + "\n")

        success_count = 0
        success_episodes = [] # Stores episode numbers ONLY for successful runs
        all_episode_counts = [] # Stores episode numbers for ALL runs (MAX_TRAIN_EPISODES_PER_RUN for failures)
        # List to store frame counts for successful runs, just for logging
        successful_frame_counts = []


        # Using alive_bar to show progress of the NUM_EVAL_RUNS runs
        with alive_bar(total=NUM_EVAL_RUNS, title="Overall Training Progress") as bar:
             for run_id in range(NUM_EVAL_RUNS):
                # Create a new agent for each independent run
                pg = PG()
                achieved_episode = run_single_training_session(
                    run_id, pg, env,
                    MAX_TRAIN_EPISODES_PER_RUN, MAX_FRAMES_PER_EPISODE
                )

                # After a run finishes, log the result
                f.write(f"  {run_id + 1:02d}   |")

                if achieved_episode is not None:
                    success_count += 1
                    success_episodes.append(achieved_episode)
                    all_episode_counts.append(achieved_episode)
                    # Since we terminate immediately at MAX_FRAMES_PER_EPISODE, the achieved frames IS MAX_FRAMES_PER_EPISODE
                    successful_frame_counts.append(MAX_FRAMES_PER_EPISODE)
                    f.write(f"    Yes    |   {achieved_episode:03d}        |   {MAX_FRAMES_PER_EPISODE}\n")
                else:
                    all_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN) # Count failure as max episodes
                    # If not successful, frames achieved is not 500. We don't log the partial frames in this report format.
                    f.write(f"     No    |     ---        |     ---    \n")

                bar() # Update the overall progress bar after this run finishes

        # Close the environment after all evaluation runs
        env.close()

        f.write("-" * 60 + "\n")
        f.write("\n===== Statistical Summary =====\n")
        f.write(f"Total Runs: {NUM_EVAL_RUNS}\n")
        f.write(f"Success Count (Reached {MAX_FRAMES_PER_EPISODE} frames): {success_count}\n")
        f.write(f"Failure Count: {NUM_EVAL_RUNS - success_count}\n")
        f.write(f"Success Rate: {success_count / NUM_EVAL_RUNS:.1%}\n") # Added success rate

        if all_episode_counts: # Check if any runs were performed
             # Average episode count considering failures as reaching MAX_TRAIN_EPISODES_PER_RUN
             avg_episode_overall = statistics.mean(all_episode_counts)
             # Max episode count (including failures as max)
             worst_episode_overall = max(all_episode_counts)
             f.write(f"\nAverage Episode to Achieve {MAX_FRAMES_PER_EPISODE} Frames (Overall, failures counted as {MAX_TRAIN_EPISODES_PER_RUN}): {avg_episode_overall:.1f}\n")
             f.write(f"Worst Episode (Overall max, including failures): {worst_episode_overall}\n")


        if success_count > 0:
            # Average and best episode counts ONLY among successful runs
            avg_episode_successful = statistics.mean(success_episodes)
            best_episode_successful = min(success_episodes)
            f.write(f"Average Episode (min among successful runs): {avg_episode_successful:.1f}\n")
            f.write(f"Best Episode (min among successful runs): {best_episode_successful}\n")
            # The achieved frames for all successful runs is MAX_FRAMES_PER_EPISODE by definition now.
            f.write(f"Frames achieved in successful runs: {MAX_FRAMES_PER_EPISODE}\n")
        else:
            f.write(f"No runs achieved {MAX_FRAMES_PER_EPISODE} frames within {MAX_TRAIN_EPISODES_PER_RUN} episodes.\n")


        f.write("\n===== Report End =====\n")

    print(f"\nEvaluation complete. Report saved to {output_file}")