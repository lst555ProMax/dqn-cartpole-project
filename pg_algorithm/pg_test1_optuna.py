'''
一个特殊的版本，用optuna库和对应的操作寻找相对较优的超参数
Policy Gradient for CartPole
'''

import torch
import torch.nn as nn
import torch.distributions as Dist
from torch.autograd import Variable # Keep Variable for compatibility if needed, though Tensor is more common now
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import os
import statistics # For average calculation
import optuna  # Import Optuna
import time

# --- Environment Setup (remains mostly global for simplicity here) ---
# Note: Use render_mode=None for faster training during hyperparameter search
env = gym.make("CartPole-v1")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MAX_FRAMES_PER_EPISODE = 500 # CartPole-v1 solved criterion is 500 steps/frames

# --- Neural Network Definition ---
# Modified to accept hidden_size and use two hidden layers as in original PG code
class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        # Using kaiming_uniform is often better for ReLU, normal for Tanh
        # Since original used Tanh, let's stick to normal init as in original
        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second hidden layer
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.1)

        self.out = nn.Linear(hidden_size, n_actions)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)
        nn.init.constant_(self.out.bias, 0.1)

    def forward(self, x):
        # Original used Tanh, let's keep Tanh for consistency with PG_test1
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.out(x)
        actions_logprobs = F.log_softmax(x, dim=-1) # Use log_softmax directly
        return actions_logprobs


# --- PG Agent ---
class PG(object):
    # Modified to accept hyperparameters
    def __init__(self, n_states, n_actions, hyperparams):
        self.n_states = n_states
        self.n_actions = n_actions

        # Unpack hyperparameters
        self.lr = hyperparams['LR']
        self.gamma = hyperparams['GAMMA']
        hidden_size = hyperparams['HIDDEN_SIZE']

        # Pass hidden_size to the network constructor
        self.pg_net = Net(n_states, n_actions, hidden_size)
        self.optimizer = torch.optim.Adam(self.pg_net.parameters(), lr=self.lr)
        # Note: PG loss is derived, no standard nn.MSELoss used here

        # Store transitions per episode
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = [] # Stores immediate rewards

    def choose_action(self, x):
        # Ensure x is a tensor, add batch dimension
        x = torch.from_numpy(x).float().unsqueeze(0)
        # Get log probabilities for each action
        action_logprobs = self.pg_net(x)
        # Create categorical distribution from logits
        action_prob_dist = Dist.Categorical(logits=action_logprobs)
        # Sample an action
        action = action_prob_dist.sample()
        # Return the action as a standard Python integer
        return action.item()

    def clear_transition(self):
        """Clears memory buffer after each episode learning."""
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

    # Modified to only store s, a, r as needed for PG
    def store_transition(self, s, a, r):
        """Stores state, action, and immediate reward for the episode."""
        self.state_memory.append(s)
        self.action_memory.append(a)
        self.reward_memory.append(r)

    def learn(self):
        """Performs the policy gradient update based on episode memory."""
        # Convert lists to tensors
        state_array = torch.tensor(np.array(self.state_memory), dtype=torch.float32)
        action_array = torch.tensor(self.action_memory, dtype=torch.long).unsqueeze(-1) # Actions need to be shape (N, 1)

        # Calculate discounted future returns (Go-to-Go)
        # Start from the end and work backwards
        discounted_rewards = []
        current_return = 0
        # Iterate through immediate rewards in reverse order
        for r in reversed(self.reward_memory):
            current_return = r + self.gamma * current_return
            discounted_rewards.append(current_return)
        # Reverse the list to get rewards in chronological order for the episode
        discounted_rewards.reverse()
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Optional: Normalize discounted rewards (recommended for PG stability)
        # This acts as a form of baseline subtraction
        # Check if there's more than one reward to avoid division by zero/nan
        if len(discounted_rewards) > 1:
            mean = torch.mean(discounted_rewards)
            std_dev = torch.std(discounted_rewards)
            # Add a small epsilon to prevent division by zero if std_dev is 0
            if std_dev > 1e-6:
                 discounted_rewards = (discounted_rewards - mean) / std_dev
            else:
                 # If std_dev is zero (all rewards are the same), just subtract mean
                 discounted_rewards = discounted_rewards - mean
        # If only one reward, normalization doesn't change it except potentially subtracting 0

        # Get log probabilities of the actions taken
        all_action_logprobs = self.pg_net(state_array)
        # Select the log probability for the specific action taken at each state
        action_logprobs = all_action_logprobs.gather(1, action_array) # shape (N, 1)

        # Calculate the Policy Gradient loss
        # Loss = - sum (log_prob * advantage)
        # Here, discounted_rewards are used as the 'advantage' (or return as advantage)
        # Negative sign because we are maximizing the expected return, which is equivalent
        # to minimizing the negative expected return.
        # The discounted_rewards tensor needs to have the same shape as action_logprobs (N, 1)
        loss = -torch.mean(action_logprobs * discounted_rewards.unsqueeze(-1))

        # Perform backpropagation and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after learning from the episode
        self.clear_transition()


# --- Training Function for a Single Run (to be called by Optuna) ---
def run_training_iteration(hyperparams, trial_number=None, verbose=False):
    # Recreate environment for each trial to ensure independence
    # Note: render_mode is None during optimization for speed
    local_env = gym.make("CartPole-v1")
    local_env = local_env.unwrapped

    # Create a new PG agent with the trial's hyperparameters
    pg_agent = PG(N_STATES, N_ACTIONS, hyperparams)

    # Maximum episodes allowed for a trial
    MAX_TRAINING_EPISODES = 300 # Set a reasonable limit for a trial

    episodes_to_solve = MAX_TRAINING_EPISODES + 1 # Initialize as failure

    # --- Helper function for the main loop content to reduce duplication ---
    def _run_episode_logic(i_episode, current_pg_agent, current_env, current_bar=None):
        s, _ = current_env.reset()
        ep_r = 0 # Accumulate original reward to check success criteria
        frame_count = 0 # Track frames for success criteria

        # Clear memory at the start of a new episode before collecting
        current_pg_agent.clear_transition()

        while True:
            a = current_pg_agent.choose_action(s)
            s_, r, done, truncated, info = current_env.step(a)

            # --- Custom Reward (copied from original PG_test1) ---
            # Encourages balancing near the center and vertically
            x, x_dot, theta, theta_dot = s_
            rx = -(x / current_env.x_threshold)**2
            rtheta = -(theta / current_env.theta_threshold_radians)**2
            # Original reward is 1 for staying alive, 0 for termination
            # Let's keep the original reward calculation and use the custom reward ONLY for learning
            # Store the custom reward for policy update
            custom_r = rtheta + r + rx # Using rtheta + r + rx directly for learning

            # Store state, action, and the reward for LEARNING (can be custom or original)
            current_pg_agent.store_transition(s, a, custom_r) # Store custom reward

            ep_r += r # Accumulate ORIGINAL reward to check CartPole's success criterion (500 steps = 500 reward)
            frame_count += 1

            # Episode termination condition (done or truncated or hitting step limit)
            # CartPole-v1 solved when it reaches 500 steps (which gives 500 reward)
            # The 'done' flag includes hitting bounds or angle limit *before* 500.
            # 'truncated' means the time limit (500 steps) was reached.
            # We consider it "solved" if frame_count reaches MAX_FRAMES_PER_EPISODE
            solved_episode_flag = False
            if done or truncated:
                 if frame_count >= MAX_FRAMES_PER_EPISODE:
                     solved_episode_flag = True # Successfully reached goal
                     if current_bar:
                         current_bar.text(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Frames: {frame_count} | SOLVED!')
                 # else: # Optional: Log failure reason within episode
                 #    if current_bar:
                 #         current_bar.text(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Frames: {frame_count} | Failed.')
                 break # Episode ends

            s = s_ # Move to the next state

        # --- Learning Step (after episode ends) ---
        # Only learn if the episode wasn't empty (e.g., terminated immediately)
        if len(current_pg_agent.reward_memory) > 0:
             current_pg_agent.learn()

        # Return the episode number if solved, otherwise None
        if solved_episode_flag:
             return i_episode + 1 # Return 1-based episode number

        return None # Not solved in this episode

    # --- Main execution flow for the trial ---
    if verbose:
        print(f"\n--- Starting Trial {trial_number if trial_number is not None else 'N/A'} with params: {hyperparams} ---")
        with alive_bar(MAX_TRAINING_EPISODES, title=f"Trial {trial_number}") as bar:
            for i_episode in range(MAX_TRAINING_EPISODES):
                solved_episode_num = _run_episode_logic(i_episode, pg_agent, local_env, current_bar=bar)
                if solved_episode_num is not None:
                    episodes_to_solve = solved_episode_num
                    if verbose: print(f"\nSolved in {episodes_to_solve} episodes for trial {trial_number}!")
                    break # Stop training this trial once solved
                # Update bar even if not solved, to show progress through episodes
                # The text is updated inside _run_episode_logic
                bar()

    else: # Not verbose
        for i_episode in range(MAX_TRAINING_EPISODES):
            solved_episode_num = _run_episode_logic(i_episode, pg_agent, local_env, current_bar=None)
            if solved_episode_num is not None:
                episodes_to_solve = solved_episode_num
                break # Stop training this trial once solved


    # Close the environment after the trial
    local_env.close()

    if verbose and episodes_to_solve > MAX_TRAINING_EPISODES:
        print(f"\nTrial {trial_number} did not solve within {MAX_TRAINING_EPISODES} episodes.")

    # Optuna minimizes the objective function. We want to minimize episodes to solve.
    # If it didn't solve, return a large value (MAX_TRAINING_EPISODES + 1)
    return episodes_to_solve


# --- Optuna Objective Function ---
def objective(trial):
    # Define search space for hyperparameters
    # These ranges are suggestions, you might need to adjust based on initial runs
    hyperparams = {
        'LR': trial.suggest_float('LR', 1e-4, 5e-2, log=True), # Learning rate range
        'GAMMA': trial.suggest_float('GAMMA', 0.9, 0.999, log=True), # Discount factor range
        'HIDDEN_SIZE': trial.suggest_categorical('HIDDEN_SIZE', [16, 32, 64, 128, 256]) # Hidden layer size (applied to both layers)
        # You could add other params like network architecture variations if needed
    }

    # Run the training and get the number of episodes to solve
    # For more robust evaluation per trial, you could run 'run_training_iteration' multiple times and average
    # However, for CartPole and faster optimization, a single run per trial is often sufficient initially.
    # num_eval_runs_per_trial = 3 # Example: Run 3 times with same params
    # scores = []
    # for i in range(num_eval_runs_per_trial):
    #     # Pass trial number and eval run index if you want more detailed logs
    #     scores.append(run_training_iteration(hyperparams, trial_number=f"{trial.number}-{i}", verbose=False))
    # score = statistics.mean(scores) # Use the average episodes to solve

    # For simplicity, let's use a single run per trial first
    score = run_training_iteration(hyperparams, trial_number=trial.number,
                                   verbose=False)  # Set verbose=True for detailed trial logs

    return score  # Optuna will try to minimize this value (episodes to solve)


# --- Main Execution Block ---
if __name__ == "__main__":
    N_OPTUNA_TRIALS = 100 # Number of different hyperparameter sets to try

    # Construct the output path relative to the script's potential location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in pg_algorithm/PG_test_optuna.py and output is in pg_output/pg_test1_optuna.txt
    project_root = os.path.dirname(script_dir) # Go up one level from pg_algorithm (adjust if needed)
    output_dir = os.path.join(project_root, "pg_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pg_test1_optuna.txt")

    study_name = f"pg-cartpole-study-{int(time.time())}"
    # You can use a database for storage to resume studies:
    # storage_name = "sqlite:///pg_optuna_study.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize")
    study = optuna.create_study(study_name=study_name, direction="minimize")

    print(f"Starting Optuna study: {study_name}. Optimizing for {N_OPTUNA_TRIALS} trials.")
    # show_progress_bar=True requires 'tqdm' and 'colorama'
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print("\n===== Optuna Study Complete =====")
    print(f"Study name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print("\n--- Best Trial ---")
    print(f"  Value (Episodes to solve): {best_trial.value}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save results to a file
    with open(output_file, "a") as f:
        f.write(f"===== Optuna Study: {study.study_name} =====\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Number of trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"Objective: Minimize episodes to solve (max {run_training_iteration.__defaults__[0]} per trial)\n") # Accessing default MAX_TRAINING_EPISODES

        f.write("\n--- Best Trial ---\n")
        f.write(f"Value (Episodes to solve): {best_trial.value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n--- All Trials Summary ---\n")
        # Sort trials by value (episodes to solve)
        sorted_trials = sorted(study.trials, key=lambda t: t.value)
        for i, trial_item in enumerate(sorted_trials):
             # Format for readability, especially for params dict
             params_str = ", ".join([f"{k}={v}" for k, v in trial_item.params.items()])
             f.write(f"Trial {trial_item.number}: Value={trial_item.value}, Params={{ {params_str} }}\n")

        f.write("===== Report End =====\n\n")

    print(f"\nOptuna results saved to {output_file}")

    # Optional: You can also visualize the results using Optuna's plotting functions
    # Requires plotly: pip install plotly
    # import plotly.io as pio
    # pio.renderers.default = "browser" # or "vscode", "notebook" etc.
    # print("Generating plots (requires plotly)...")
    # try:
    #     optuna.visualization.plot_optimization_history(study).show()
    #     optuna.visualization.plot_param_importances(study).show()
    #     optuna.visualization.plot_slice(study).show()
    # except Exception as e:
    #     print(f"Could not generate plots. Make sure plotly is installed. Error: {e}")

    # Example of running with best params found after the study
    # print("\n--- Running one simulation with best parameters found ---")
    # best_params = study.best_params
    # # You might want to run it for more episodes or with different seeds here
    # # To see rendering, you'd need to set render_mode="human" when making the env
    # # for this specific final run, but not during the optimization trials.
    # print(f"Using parameters: {best_params}")
    # # Create a *new* env instance here if you want rendering
    # final_env = gym.make("CartPole-v1", render_mode="human")
    # final_env = final_env.unwrapped
    #
    # print("\nStarting final run...")
    # final_agent = PG(N_STATES, N_ACTIONS, best_params)
    # # You could run this for more episodes or specific tests
    # s, _ = final_env.reset()
    # ep_r = 0
    # frame_count = 0
    # while True:
    #     a = final_agent.choose_action(s)
    #     s_, r, done, truncated, info = final_env.step(a)
    #     ep_r += r
    #     frame_count += 1
    #     if done or truncated:
    #         print(f"Final Run Episode: Achieved {frame_count} frames. Total reward: {ep_r}")
    #         break # End of the final run episode
    #     s = s_
    # final_env.close()
    # print("Final run complete.")