'''
pg_test3_optuna.py - Optuna optimization for Actor-Critic Policy Gradient on CartPole
Fixed HIDDEN_SIZE to 128. Optimizing LR, GAMMA, CRITIC_LR_RATIO, CRITIC_LOSS_COEFF, ENTROPY_COEFF.
'''

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable # Retained Variable for consistency, though Tensor is standard now
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os
import statistics # For average calculation
import optuna  # Import Optuna
import time

# --- Environment Setup (remains mostly global for simplicity here) ---
# Use render_mode=None for faster training during hyperparameter search
env = gym.make("CartPole-v1")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MAX_FRAMES_PER_EPISODE = 500 # CartPole-v1 solved criterion is 500 steps/frames

# Max episodes for a *single trial* during Optuna optimization
MAX_TRAINING_EPISODES_PER_TRIAL = 300 # Adjust based on how long a reasonable trial should take

# --- Fixed Hyperparameters ---
# Using the best hidden size found in the previous optimization (PG_test2)
# This parameter is NOT tuned in THIS Optuna run.
FIXED_HIDDEN_SIZE = 16
# -----------------------------


# --- Neural Network Definition (Generalized for Actor/Critic, uses fixed hidden_size) ---
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


# --- Actor-Critic Agent (Modified to accept hyperparameters, uses fixed hidden_size) ---
class ActorCritic(object):
    # Modified to accept hyperparameters (excluding hidden_size now)
    def __init__(self, n_states, n_actions, hyperparams):
        self.n_states = n_states
        self.n_actions = n_actions

        # Unpack hyperparameters
        self.lr = hyperparams['LR']
        self.gamma = hyperparams['GAMMA']
        # self.hidden_size is NOT unpacked from hyperparams anymore
        self.critic_lr_ratio = hyperparams['CRITIC_LR_RATIO']
        self.critic_loss_coeff = hyperparams['CRITIC_LOSS_COEFF']
        self.entropy_coeff = hyperparams['ENTROPY_COEFF']


        # Instantiate Actor Network (outputs action logits) using the FIXED_HIDDEN_SIZE
        self.actor_net = CommonNet(self.n_states, FIXED_HIDDEN_SIZE, self.n_actions)
        # Instantiate Critic Network (outputs state value V(s)) using the FIXED_HIDDEN_SIZE
        self.critic_net = CommonNet(self.n_states, FIXED_HIDDEN_SIZE, 1) # Output size 1 for value

        # Optimizers for both networks using unpacked LRs
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr * self.critic_lr_ratio)

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
        td_targets = reward_array + self.gamma * next_predicted_values * (1 - done_array) # Shape (N, 1)

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

        # Total Actor Loss (Policy Loss + Entropy Bonus)
        actor_loss = actor_policy_loss - self.entropy_coeff * entropy


        # --- Total Loss and Optimization ---
        # Combine Actor and Critic losses using unpacked coefficient
        total_loss = actor_loss + self.critic_loss_coeff * critic_loss

        # Perform backpropagation and update weights for BOTH networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Clear memory after learning from the episode
        self.clear_memory()


# --- Training Function for a Single Optuna Trial ---
# This function will be called by Optuna's objective function
def run_training_iteration(hyperparams, trial_number=None, verbose=False):
    # Create a NEW environment instance for each trial for isolation
    # render_mode=None for speed during optimization
    local_env = gym.make("CartPole-v1", render_mode=None)
    local_env = local_env.unwrapped

    # Create a new Actor-Critic agent with the trial's hyperparameters
    # FIXED_HIDDEN_SIZE is used directly, NOT passed from hyperparams dict
    ac_agent = ActorCritic(N_STATES, N_ACTIONS, hyperparams)

    episodes_to_solve = MAX_TRAINING_EPISODES_PER_TRIAL + 1 # Initialize as failure (large value)

    # --- Helper function for the main loop content ---
    def _run_episode_logic(i_episode, current_ac_agent, current_env, current_bar=None):
        s, _ = current_env.reset()
        ep_r = 0 # Accumulate original reward for success criteria
        frame_count = 0 # Track frames for success criteria

        # Clear memory at the start of a new episode before collecting
        current_ac_agent.clear_memory()

        while True:
            a = current_ac_agent.choose_action(s)
            # Use the custom reward for learning as in PG_test3
            s_, r, done, truncated, info = current_env.step(a)

            # --- Custom Reward (copied from PG_test3) ---
            # Used for LEARNING (value function and advantage)
            x, x_dot, theta, theta_dot = s_
            rx = -(x / current_env.x_threshold)**2
            rtheta = -(theta / current_env.theta_threshold_radians)**2
            custom_r = rtheta + r + rx # Use this for store_transition

            # Store transition: state, action, custom reward, next state, and done flag
            # Store done OR truncated as the signal for episode termination
            current_ac_agent.store_transition(s, a, custom_r, s_, done or truncated)

            # Accumulate ORIGINAL reward to check CartPole's success criterion (500 steps = 500 reward)
            ep_r += r
            frame_count += 1

            # --- Early Termination Logic (same as PG_test2/3) ---
            # Check if the target frame count is reached. If so, terminate immediately.
            if frame_count >= MAX_FRAMES_PER_EPISODE:
                # Successfully reached goal
                if current_bar:
                    current_bar.text(f'Trial: {trial_number} | Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Frames: {frame_count} | SOLVED!')
                return i_episode # Return 0-indexed episode number
            # --- END Early Termination Logic ---


            # Check for natural termination conditions (pole fell, went out of bounds, or truncated by time limit)
            # This check happens after the success check.
            if done or truncated:
                 # If done/truncated occurred *before* reaching max_frames_per_episode, it's a failure for this episode.
                 if current_bar:
                    current_bar.text(f'Trial: {trial_number} | Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Frames: {frame_count} | Failed.')
                 break # Terminate the episode

            s = s_ # Move to the next state

        # --- Learning Step (after episode ends) ---
        # Perform learning based on the episode's collected transitions
        current_ac_agent.learn()

        return None # Not solved in this episode

    # --- Main execution flow for the trial ---
    if verbose:
        # Create a printable string of the hyperparameters for logging
        params_str = ", ".join([f"{k}={v:.5g}" for k, v in hyperparams.items()])
        print(f"\n--- Starting Trial {trial_number if trial_number is not None else 'N/A'} with params: {{ {params_str} }}, Fixed HIDDEN_SIZE={FIXED_HIDDEN_SIZE} ---")

        with alive_bar(MAX_TRAINING_EPISODES_PER_TRIAL, title=f"Trial {trial_number}") as bar:
            for i_episode in range(MAX_TRAINING_EPISODES_PER_TRIAL):
                solved_episode_num = _run_episode_logic(i_episode, ac_agent, local_env, current_bar=bar)
                if solved_episode_num is not None: # Check if solved (returns 0-indexed episode)
                    episodes_to_solve = solved_episode_num + 1 # Convert to 1-based count
                    if verbose: print(f"\nSolved in {episodes_to_solve} episodes for trial {trial_number}!")
                    break # Stop training this trial once solved
                # Update bar even if not solved
                bar()

    else: # Not verbose
        for i_episode in range(MAX_TRAINING_EPISODES_PER_TRIAL):
            solved_episode_num = _run_episode_logic(i_episode, ac_agent, local_env, current_bar=None)
            if solved_episode_num is not None: # Check if solved
                episodes_to_solve = solved_episode_num + 1 # Convert to 1-based count
                break # Stop training this trial once solved

    # Close the environment after the trial
    local_env.close()

    if verbose and episodes_to_solve > MAX_TRAINING_EPISODES_PER_TRIAL:
        print(f"\nTrial {trial_number} did not solve within {MAX_TRAINING_EPISODES_PER_TRIAL} episodes.")

    # Optuna minimizes the objective function. We want to minimize episodes to solve.
    # If it didn't solve, return a large value (MAX_TRAINING_EPISODES_PER_TRIAL + 1)
    return episodes_to_solve


# --- Optuna Objective Function ---
def objective(trial):
    # Define search space for hyperparameters
    # HIDDEN_SIZE is NOT included here, it's fixed.
    hyperparams = {
        'LR': trial.suggest_float('LR', 5e-4, 5e-2, log=True), # Learning rate range (slightly wider/adjusted)
        'GAMMA': trial.suggest_float('GAMMA', 0.9, 0.999, log=True), # Discount factor range

        # New Actor-Critic specific params
        'CRITIC_LR_RATIO': trial.suggest_float('CRITIC_LR_RATIO', 0.5, 2.0), # Ratio of Critic LR to Actor LR
        'CRITIC_LOSS_COEFF': trial.suggest_float('CRITIC_LOSS_COEFF', 0.1, 2.0), # Weight of Critic loss in total loss
        'ENTROPY_COEFF': trial.suggest_float('ENTROPY_COEFF', 1e-5, 0.05, log=True), # Weight of entropy bonus
    }

    # Run the training iteration for this trial
    score = run_training_iteration(hyperparams, trial_number=trial.number,
                                   verbose=False) # Set verbose=True for detailed trial logs

    return score  # Optuna will try to minimize this value (episodes to solve)


# --- Main Execution Block ---
if __name__ == "__main__":
    N_OPTUNA_TRIALS = 300 # Number of different hyperparameter sets to try

    # Construct the output path relative to the script's potential location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in pg_algorithm/pg_test3_optuna.py and output is in pg_output/pg_test3_optuna.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test3_optuna.txt") # Changed output filename

    study_name = f"pg-ac-cartpole-fixed-hidden-study-{int(time.time())}"
    # You can use a database for storage to resume studies:
    # storage_name = "sqlite:///pg_ac_fixed_hidden_optuna_study.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize")
    study = optuna.create_study(study_name=study_name, direction="minimize")

    print(f"Starting Optuna study: {study_name}. Optimizing for {N_OPTUNA_TRIALS} trials.")
    print(f"Note: HIDDEN_SIZE is fixed at {FIXED_HIDDEN_SIZE}.")
    # show_progress_bar=True requires 'tqdm' and 'colorama'
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print("\n===== Optuna Study Complete =====")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Fixed HIDDEN_SIZE: {FIXED_HIDDEN_SIZE}")


    best_trial = study.best_trial
    print("\n--- Best Trial ---")
    print(f"  Value (Episodes to solve): {best_trial.value}")
    print("  Best Hyperparameters (excluding fixed HIDDEN_SIZE):")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"  Fixed Hyperparameter: HIDDEN_SIZE={FIXED_HIDDEN_SIZE}")


    # Save results to a file
    with open(output_file, "a") as f:
        f.write(f"===== Optuna Study: {study.study_name} =====\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Number of trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"Objective: Minimize episodes to solve (max {MAX_TRAINING_EPISODES_PER_TRIAL} per trial)\n")
        f.write(f"Fixed Hyperparameter: HIDDEN_SIZE={FIXED_HIDDEN_SIZE}\n") # Report fixed parameter

        f.write("\n--- Best Trial ---\n")
        f.write(f"Value (Episodes to solve): {best_trial.value}\n")
        f.write("Best Tuned Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Fixed Hyperparameter: HIDDEN_SIZE={FIXED_HIDDEN_SIZE}\n")


        f.write("\n--- All Trials Summary ---\n")
        # Sort trials by value (episodes to solve)
        sorted_trials = sorted(study.trials, key=lambda t: t.value)
        for i, trial_item in enumerate(sorted_trials):
             # Format for readability, especially for params dict
             params_str = ", ".join([f"{k}={v:.5g}" for k, v in trial_item.params.items()])
             f.write(f"Trial {trial_item.number}: Value={trial_item.value}, Tuned Params={{ {params_str} }}\n")

        f.write("===== Report End =====\n\n")

    print(f"\nOptuna results saved to {output_file}")

    # Optional visualization code remains the same (requires plotly)