# pg_test4.py - Actor-Critic with Newly Optimized Hyperparameters

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

# --- Newly Optimized Hyperparameters for Actor-Critic ---
# These values come from the last Optuna run (PG_test3_optuna_fixed_hidden)
LR = 0.02400603306343582
GAMMA = 0.9319160079248237
BEST_HIDDEN_SIZE = 16 # Explicitly setting to 16 based on the last result
CRITIC_LR_RATIO = 1.3849940236655809
CRITIC_LOSS_COEFF = 0.20856250806571486
ENTROPY_COEFF = 0.005097136133822689
# -----------------------------------------------------------


MAX_FRAMES_PER_EPISODE = 500 # CartPole-v1 solved criterion
NUM_EVAL_RUNS = 30 # Number of independent runs to evaluate the performance
MAX_TRAIN_EPISODES_PER_RUN = 200 # Max episodes allowed for training within a single run (kept from PG_test3)


# Environment setup
env = gym.make("CartPole-v1", render_mode="rgb_array") # Use rgb_array if visualization needed
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


# --- Neural Network Definition (Generalized for Actor/Critic, uses BEST_HIDDEN_SIZE) ---
class CommonNet(nn.Module):
    # This network is slightly simplified: it now requires hidden_size to be passed directly
    # It doesn't have a default, as we expect the fixed value.
    def __init__(self, n_states, hidden_size, output_size):
        super(CommonNet, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

        self.out = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # No final activation here; Actor will apply softmax/log_softmax, Critic's output is raw value
        return self.out(x)


# --- Actor-Critic Agent (Uses newly optimized hyperparameters) ---
class ActorCritic(object):
    def __init__(self):
        # Instantiate Actor Network (outputs action logits) using the BEST_HIDDEN_SIZE
        self.actor_net = CommonNet(N_STATES, BEST_HIDDEN_SIZE, N_ACTIONS)
        # Instantiate Critic Network (outputs state value V(s)) using the BEST_HIDDEN_SIZE
        self.critic_net = CommonNet(N_STATES, BEST_HIDDEN_SIZE, 1) # Output size 1 for value

        # Optimizers for both networks using the new LRs
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=LR) # Use global LR
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=LR * CRITIC_LR_RATIO) # Use global LR and Ratio

        # Store transitions per episode - need s, a, r, s', and done flag
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [] # Stores immediate rewards
        self.episode_next_states = []
        self.episode_dones = [] # Stores done flags


    def choose_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        # Use actor network to get action probabilities
        self.actor_net.eval() # Set actor to eval mode for action selection
        with torch.no_grad(): # No gradient calculation needed for action selection
            action_logits = self.actor_net(x)
        self.actor_net.train() # Set actor back to train mode

        # Create categorical distribution from logits
        action_prob_dist = Dist.Categorical(logits=action_logits)
        # Sample an action
        action = action_prob_dist.sample()
        # Return the action as a standard Python integer
        return action.item()

    def clear_memory(self):
        """Clears memory buffer after each episode learning."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_next_states.clear()
        self.episode_dones.clear()

    # Modified to store s, a, r, s_, done
    def store_transition(self, s, a, r, s_, done):
        """Stores state, action, reward, next_state, and done flag."""
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)
        self.episode_next_states.append(s_)
        self.episode_dones.append(done)


    def learn(self):
        """Performs the Actor-Critic update based on episode memory."""
        # Check if memory is empty (e.g., episode terminated immediately)
        if len(self.episode_rewards) == 0:
            return # Nothing to learn from

        # Convert lists to tensors
        state_array = torch.tensor(np.array(self.episode_states), dtype=torch.float32)
        action_array = torch.tensor(self.episode_actions, dtype=torch.long).unsqueeze(-1) # Actions need to be shape (N, 1)
        reward_array = torch.tensor(self.episode_rewards, dtype=torch.float32).unsqueeze(-1) # Shape (N, 1)
        next_state_array = torch.tensor(np.array(self.episode_next_states), dtype=torch.float32)
        done_array = torch.tensor(self.episode_dones, dtype=torch.float32).unsqueeze(-1) # Shape (N, 1)


        # --- Critic Update ---
        # Predict V(s) for all states in the episode
        self.critic_net.train() # Set critic to train mode
        predicted_values = self.critic_net(state_array) # Shape (N, 1)

        # Predict V(s') for all next states
        # Need to handle the terminal state where V(s') is 0
        self.critic_net.eval() # Set critic to eval mode for predicting next state values
        with torch.no_grad(): # No gradient calculation needed for target
             next_predicted_values = self.critic_net(next_state_array) # Shape (N, 1)
        self.critic_net.train() # Set critic back to train mode

        # Calculate TD Target: R + gamma * V(s')
        # If done, the target is just R (V(s') is effectively 0)
        td_targets = reward_array + GAMMA * next_predicted_values * (1 - done_array) # Use global GAMMA

        # Calculate Critic Loss (MSE between predicted V(s) and TD Target)
        critic_loss = F.mse_loss(predicted_values, td_targets.detach()) # Use detach() for target


        # --- Actor Update ---
        # Calculate Advantage using TD error: A = R + gamma * V(s') - V(s)
        # This is essentially td_targets - predicted_values
        # We need to detach the advantage so gradients from the actor loss don't flow back through the critic
        advantage = (td_targets - predicted_values).detach() # Shape (N, 1)

        # Get log probabilities of the actions taken
        self.actor_net.train() # Set actor to train mode
        action_logits = self.actor_net(state_array) # Shape (N, N_ACTIONS)
        action_logprobs_all = F.log_softmax(action_logits, dim=-1) # Shape (N, N_ACTIONS)
        action_logprobs = action_logprobs_all.gather(1, action_array) # Shape (N, 1)

        # Calculate Actor Loss: - mean (log_prob * advantage)
        actor_policy_loss = -torch.mean(action_logprobs * advantage)

        # Add Entropy Bonus for exploration
        # Calculate entropy for each state/action distribution
        # Recreate distribution from logits for entropy calculation
        action_prob_dist = Dist.Categorical(logits=action_logits)
        entropy = action_prob_dist.entropy().mean() # Mean entropy across the batch

        # Total Actor Loss (Policy Loss + Entropy Bonus) using global entropy coefficient
        actor_loss = actor_policy_loss - ENTROPY_COEFF * entropy


        # --- Total Loss and Optimization ---
        # Combine Actor and Critic losses using global critic loss coefficient
        total_loss = actor_loss + CRITIC_LOSS_COEFF * critic_loss

        # Perform backpropagation and update weights for BOTH networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Clear memory after learning from the episode
        self.clear_memory()


# --- Training and Evaluation Function for a Single Run ---
def run_single_training_session(run_id, ac_agent, env, max_episodes, max_frames_per_episode):
    achieved_episode = None # To store the episode number if successful

    print(f"\n--- Starting Run {run_id + 1}/{NUM_EVAL_RUNS} ---")

    # Iterate through training episodes for this single run
    for i_episode in range(max_episodes):
        s, s_info = env.reset()
        ep_r = 0 # Accumulate original reward to check success criteria
        frame_count = 0
        # Clear memory at the start of each episode before collection
        ac_agent.clear_memory()

        # --- Run a single episode ---
        while True:
            a = ac_agent.choose_action(s)
            # Use the custom reward for learning as in previous AC versions (PG_test3)
            s_, r, done, truncated, info = env.step(a)

            # --- Custom Reward (copied from PG_test3) ---
            # Used for LEARNING (value function and advantage)
            x, x_dot, theta, theta_dot = s_
            rx = -(x / env.x_threshold)**2
            rtheta = -(theta / env.theta_threshold_radians)**2
            custom_r = rtheta + r + rx # Use this for store_transition

            # Store transition: state, action, custom reward, next state, and done flag
            # Store done OR truncated as the signal for episode termination
            ac_agent.store_transition(s, a, custom_r, s_, done or truncated) # Pass custom_r, s_, and done/truncated

            # Accumulate ORIGINAL reward to check CartPole's success criterion (500 steps = 500 reward)
            ep_r += r
            frame_count += 1

            # --- Early Termination Logic ---
            # Check if the target frame count is reached. If so, terminate immediately.
            if frame_count >= max_frames_per_episode:
                achieved_episode = i_episode # Record the episode number (0-indexed)
                print(f"Run {run_id + 1}/{NUM_EVAL_RUNS}, Episode {i_episode}: Reached {frame_count} frames (Success!)")
                break # Terminate the episode immediately
            # --- END Early Termination Logic ---


            # Check for natural termination conditions (pole fell, went out of bounds, or truncated by time limit)
            # This check happens after the success check.
            if done or truncated:
                 # If done/truncated occurred *before* reaching max_frames_per_episode, it's a failure for this episode.
                 break # Terminate the episode

            s = s_ # Move to the next state

        # --- Learning Step (after episode ends) ---
        # Perform learning based on the episode's collected transitions
        # The learn function uses the custom rewards that were stored
        if len(ac_agent.episode_rewards) > 0:
             ac_agent.learn()

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
    # Assuming script is in pg_algorithm/pg_test4.py and output is in pg_output/pg_test3.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test3.txt") # Changed output filename

    print(f"Starting {NUM_EVAL_RUNS} independent training runs with Actor-Critic (Newly Optimized Hyperparameters)...")
    print(f"Hyperparameters: LR={LR:.5g}, GAMMA={GAMMA:.5g}, HIDDEN_SIZE={BEST_HIDDEN_SIZE}")
    print(f"AC Params: Critic LR Ratio={CRITIC_LR_RATIO:.5g}, Critic Loss Coeff={CRITIC_LOSS_COEFF:.5g}, Entropy Coeff={ENTROPY_COEFF:.5g}")


    # Open the output file in write mode ('w') to clear any previous content
    with open(output_file, "w") as f:
        f.write(f"===== Actor-Critic Training Results Report (Newly Optimized Hyperparams) =====\n")
        f.write(f"Hyperparameters: LR={LR:.5g}, GAMMA={GAMMA:.5g}, HIDDEN_SIZE={BEST_HIDDEN_SIZE}\n")
        f.write(f"AC Params: Critic LR Ratio={CRITIC_LR_RATIO:.5g}, Critic Loss Coeff={CRITIC_LOSS_COEFF:.5g}, Entropy Coeff={ENTROPY_COEFF:.5g}\n")
        f.write(f"Reward Signal for Learning: Custom Reward (rtheta + r + rx)\n") # Explicitly mention custom reward
        f.write(f"Value Estimation Method: Learned Value Function (Critic)\n") # Explicitly mention baseline
        f.write(f"Success Criterion: Reach {MAX_FRAMES_PER_EPISODE} frames (episode terminates immediately)\n")
        f.write(f"Max Episodes per Run: {MAX_TRAIN_EPISODES_PER_RUN}\n")
        f.write("-" * 80 + "\n")
        f.write("Run ID | Success | Episode Achieved | Frames Achieved\n")
        f.write("-" * 80 + "\n")

        success_count = 0
        success_episodes = [] # Stores episode numbers ONLY for successful runs
        all_episode_counts = [] # Stores episode numbers for ALL runs (MAX_TRAIN_EPISODES_PER_RUN for failures)


        # Using alive_bar to show progress of the NUM_EVAL_RUNS runs
        with alive_bar(total=NUM_EVAL_RUNS, title="Overall Training Progress") as bar:
             for run_id in range(NUM_EVAL_RUNS):
                # Create a new agent for each independent run
                ac_agent = ActorCritic() # Using the modified ActorCritic class
                achieved_episode = run_single_training_session(
                    run_id, ac_agent, env,
                    MAX_TRAIN_EPISODES_PER_RUN, MAX_FRAMES_PER_EPISODE
                )

                # After a run finishes, log the result
                f.write(f"  {run_id + 1:02d}   |")

                if achieved_episode is not None:
                    success_count += 1
                    success_episodes.append(achieved_episode)
                    all_episode_counts.append(achieved_episode)
                    # Since we terminate immediately at MAX_FRAMES_PER_EPISODE, the achieved frames IS MAX_FRAMES_PER_EPISODE
                    f.write(f"    Yes    |   {achieved_episode:03d}        |   {MAX_FRAMES_PER_EPISODE}\n")
                else:
                    all_episode_counts.append(MAX_TRAIN_EPISODES_PER_RUN) # Count failure as max episodes
                    # If not successful, frames achieved is not 500. We don't log the partial frames in this report format.
                    f.write(f"     No    |     ---        |     ---    \n")

                bar() # Update the overall progress bar after this run finishes

        # Close the environment after all evaluation runs
        env.close()

        f.write("-" * 80 + "\n")
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
            f.write(f"Frames achieved in successful runs: {MAX_FRAMES_PER_EPISODE}\n") # Always 500 now due to early termination
        else:
            f.write(f"No runs achieved {MAX_FRAMES_PER_EPISODE} frames within {MAX_TRAIN_EPISODES_PER_RUN} episodes.\n")


        f.write("\n===== Report End =====\n")

    print(f"\nEvaluation complete. Report saved to {output_file}")