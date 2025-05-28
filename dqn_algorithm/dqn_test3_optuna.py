'''
dqn_test3_optuna.py - Optuna optimization for Double DQN (DDQN) on CartPole
借鉴 dqn_test1_optuna.py 结构，优化 DDQN 模型超参数
'''

import torch
import torch.nn as nn
from torch.autograd import Variable # Retained Variable for consistency
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import optuna  # Import Optuna
import time
import os # Import os for file path handling

# --- Environment Setup (remains mostly global for simplicity here) ---
# Note: Use render_mode=None for faster training during hyperparameter search
env = gym.make("CartPole-v1")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
MAX_ENV_FRAMES = 500  # CartPole-v1 solved criterion is 500 steps/frames

# --- Constants for Training Trial ---
# These are used *within* each Optuna trial run
MAX_TRAINING_EPISODES_PER_TRIAL = 300 # Maximum episodes allowed for a single Optuna trial
CONSECUTIVE_SUCCESS_THRESHOLD = 5 # Criterion for solving within a trial (e.g., 5 consecutive successes > 475 reward)
# ------------------------------------


# --- Neural Network Definition (Same as dqn_test1_optuna) ---
class Net(nn.Module):
    # Accepts hidden_size as a parameter
    def __init__(self, n_states, n_actions, hidden_size=50):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        # Using ReLU now, so Kaiming init is usually preferred, but stick to original Normal init as in base code
        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        self.out = nn.Linear(hidden_size, n_actions)
        nn.init.normal_(self.out.weight, mean=0., std=0.1)

    def forward(self, x):
        # Using ReLU activation as in dqn_test1_optuna
        x = F.relu(self.fc1(x))
        return self.out(x)


# --- Double DQN (DDQN) Agent (Adapted from your DDQN class, now accepts hyperparams) ---
class DDQN(object):
    # Modified __init__ to accept hyperparams dictionary
    def __init__(self, n_states, n_actions, env_a_shape, hyperparams):
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape

        # Unpack hyperparameters from the dictionary
        self.lr = hyperparams['LR']
        self.gamma = hyperparams['GAMMA']
        self.target_network_replace_freq = hyperparams['TARGET_NETWORK_REPLACE_FREQ']
        self.memory_capacity = hyperparams['MEMORY_CAPACITY']
        self.batch_size = hyperparams['BATCH_SIZE']
        self.epsilon_start = hyperparams['EPSILON_START']
        self.epsilon_end = hyperparams['EPSILON_END']
        self.epsilon_decay_steps = hyperparams['EPSILON_DECAY_STEPS']
        hidden_size = hyperparams['HIDDEN_SIZE'] # Get hidden_size from hyperparams

        # Instantiate networks using the unpacked hidden_size
        self.eval_net = Net(n_states, n_actions, hidden_size)
        self.target_net = Net(n_states, n_actions, hidden_size)
        # Copy weights and set target net to evaluation mode
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval() # Target network is not for training

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.total_steps = 0 # For epsilon decay

        # Use unpacked LR for the optimizer
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss() # MSE loss for Q-value prediction error

        # Initialize memory buffer
        self.memory = np.zeros((self.memory_capacity, n_states * 2 + 2)) # s, a, r, s_


    def _get_current_epsilon(self):
        # Use unpacked epsilon parameters
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        # Decay function using unpacked epsilon parameters
        return self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.total_steps / (self.epsilon_decay_steps / 5)) # Keep decay shape


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        epsilon = self._get_current_epsilon()
        self.total_steps += 1 # Increment total steps for epsilon decay

        if np.random.uniform() < epsilon:
            # Explore: choose a random action
            action = np.random.randint(0, self.n_actions)
            # Reshape action if necessary (e.g., for environments with Box action space, though CartPole is Discrete)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else:
            # Exploit: choose action with highest Q-value from eval_net
            self.eval_net.eval() # Set to evaluation mode for inference
            with torch.no_grad(): # No need to track gradients here
                actions_value = self.eval_net(x) # Get Q-values for current state
            self.eval_net.train() # Set back to training mode
            # Select the action with the highest Q-value
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action


    def store_transition(self, s, a, r, s_):
        # Store a single transition (s, a, r, s')
        # Note: done/truncated flag is implicitly handled by modified reward or not stored in this memory format
        transition = np.hstack((s, [a, r], s_))
        # Store transition in the circular memory buffer
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1 # Increment memory counter


    def learn(self):
        # Perform a learning step (experience replay)
        # Check if memory is full enough for a batch
        current_memory_size = min(self.memory_counter, self.memory_capacity)
        if current_memory_size < self.batch_size:
            return # Not enough samples to learn

        # Soft update target network periodically
        if self.learn_step_counter % self.target_network_replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1 # Increment learn step counter


        # Sample a random batch from memory
        sample_index = np.random.choice(current_memory_size, self.batch_size, replace=False)
        b_memory = self.memory[sample_index, :]

        # Extract batch data into tensors
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states])) # Batch of states
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))) # Batch of actions
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])) # Batch of rewards
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:])) # Batch of next states

        # Calculate Q(s, a) for the current state-action pairs using eval_net
        # .gather(1, b_a) selects the Q-value for the taken action in each state of the batch
        q_eval = self.eval_net(b_s).gather(1, b_a) # Shape (BATCH_SIZE, 1)

        # --- Double DQN Target Calculation ---
        # The core difference from standard DQN
        # 1. Select the *action* in the next state s' that would be chosen by the EVAL net
        #    We detach the output of eval_net here because we only need the action index,
        #    we don't want gradients to flow back through this selection process.
        q_eval_next_actions_indices = self.eval_net(b_s_).detach().max(1)[1].view(self.batch_size, 1) # Shape (BATCH_SIZE, 1)

        # 2. Evaluate the Q-value of that *selected action* in the next state s' using the TARGET net
        #    The target_net's output is already detached, as it's only used for target calculation.
        #    .gather(1, ...) selects the Q-value corresponding to the action chosen by the eval_net
        q_target_next_all_values = self.target_net(b_s_).detach() # Shape (BATCH_SIZE, n_actions)
        q_target_next_selected_value = q_target_next_all_values.gather(1, q_eval_next_actions_indices) # Shape (BATCH_SIZE, 1)

        # Calculate the DDQN target Q-value: R + gamma * Q_target(s', argmax_a' Q_eval(s', a'))
        q_target = b_r + self.gamma * q_target_next_selected_value # Use unpacked gamma

        # --- End of Double DQN Modification ---


        # Calculate the loss (MSE between predicted Q_eval and calculated Q_target)
        loss = self.loss_func(q_eval, q_target)

        # Perform backpropagation and update eval_net weights
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        # Optional: Gradient clipping can be added here for stability
        # torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10.0)
        self.optimizer.step() # Update weights


# --- Training Function for a Single Optuna Trial ---
# This function will be called by Optuna's objective function
def run_training_iteration_ddqn(hyperparams, trial_number=None, verbose=False):
    # Create a NEW environment instance for each trial for isolation
    # render_mode=None for speed during optimization
    local_env = gym.make("CartPole-v1", render_mode=None)
    local_env = local_env.unwrapped

    # Create a new DDQN agent with the trial's hyperparameters
    # Pass N_STATES, N_ACTIONS, ENV_A_SHAPE and the hyperparams dictionary
    ddqn_agent = DDQN(N_STATES, N_ACTIONS, ENV_A_SHAPE, hyperparams)

    # Use constants defined at the top of the script for trial limits
    MAX_TRAINING_EPISODES = MAX_TRAINING_EPISODES_PER_TRIAL
    CONSECUTIVE_SUCCESSES_REQUIRED = CONSECUTIVE_SUCCESS_THRESHOLD

    episodes_to_solve = MAX_TRAINING_EPISODES + 1 # Initialize as failure (large value)
    consecutive_successes = 0
    # We are optimizing for episodes to solve, reward tracking is less critical here
    # all_episode_rewards = [] # Optional: if you wanted to log/analyze rewards per trial

    # --- Helper function for the main loop content ---
    # This logic is similar to dqn_test1_optuna but uses the DDQN agent
    def _run_episode_logic(i_episode, current_agent, current_env, current_bar=None):
        nonlocal consecutive_successes # Allow modification of outer scope variable

        s, _ = current_env.reset()
        ep_r_original = 0 # Accumulate ORIGINAL reward for success criteria
        step_count = 0
        # frames_this_episode = 0 # Equivalent to step_count + 1

        while True:
            a = current_agent.choose_action(s)
            # env.step returns s_, r, done, truncated, info
            s_, r, done, truncated, _ = current_env.step(a)

            # --- Custom Reward Logic (from dqn_test1_optuna & dqn_test2/3) ---
            # This modified reward is used for LEARNING
            x, _, theta, _ = s_
            rx = -(x / current_env.x_threshold) ** 2
            rtheta = -(theta / current_env.theta_threshold_radians) ** 2
            modified_r = r # Start with original reward
            if not done: # Only add custom penalty/bonus if not terminated by failure yet
                # Add small penalty for angle/position deviations
                modified_r += (rx + rtheta) * 0.1
            # Add a large penalty if the episode ends due to failure *before* reaching max frames
            # The condition done is True but step_count < MAX_ENV_FRAMES - 1
            elif step_count < MAX_ENV_FRAMES - 1:
                 modified_r = -100
            # Note: if done is True AND step_count is MAX_ENV_FRAMES - 1 (episode 500), it means truncated
            # In this case, done is False (usually), truncated is True. The modified_r would just be r (which is 1).
            # If done is True because pole fell, and step_count is < 499, modified_r = -100. This seems correct.
            # If using gym newer than 0.21, done is only for termination due to environment rules (pole falling, out of bounds), truncated is for time limits.
            # For CartPole-v1, at step 500, done is False, truncated is True.
            # So, the penalty should apply if done is True OR truncated is True AND step_count < 499?
            # Let's refine: if done is True (failure) and not truncated (not max steps), apply penalty.
            # if done and not truncated: # If done is True (failure before time limit)
            #     modified_r = -100 # This seems more precise based on newer gym
            # Let's stick to the original logic from your code for consistency:
            # if done and step_count < MAX_ENV_FRAMES - 1: modified_r = -100
            # This original logic implicitly handles 'done' as failure before time limit, and truncated as time limit success.
            # It seems `step_count < MAX_ENV_FRAMES - 1` is the key here.
            # Let's use the logic from the provided snippet which adds penalty only if 'done' is true AND it's not the last step.
            # This assumes 'done' implies failure before time limit, which is usually true in older gym versions or for this penalty logic.

            current_agent.store_transition(s, a, modified_r, s_) # Store s, a, modified_r, s_

            ep_r_original += r # Accumulate ORIGINAL reward for the success metric (CartPole gives +1 per step alive)
            # frames_this_episode += 1 # Equivalent to step_count + 1

            # --- Learning Condition ---
            # Learn if enough transitions are in memory (memory_counter > BATCH_SIZE)
            # and memory is at least BATCH_SIZE (memory_counter >= BATCH_SIZE)
            # The condition `current_agent.memory_counter > hyperparams['BATCH_SIZE']` is slightly odd,
            # typically you start learning *after* the memory_counter first exceeds BATCH_SIZE.
            # Let's use a more standard condition: `current_agent.memory_counter >= hyperparams['BATCH_SIZE']`
            # Or even better, only attempt to learn if memory_counter has reached BATCH_SIZE or more.
            # Sticking to the provided structure `current_agent.memory_counter > hyperparams['BATCH_SIZE']` for now.
            # Let's also add a check that memory_counter has at least BATCH_SIZE valid entries.
            if current_agent.memory_counter > hyperparams['BATCH_SIZE'] and min(current_agent.memory_counter, current_agent.memory_capacity) >= hyperparams['BATCH_SIZE']:
                current_agent.learn()

            # --- Episode Termination Conditions ---
            # Check for CartPole success criterion: reach MAX_ENV_FRAMES
            # The original reward ep_r_original accumulates +1 per step if not done by failure.
            # So ep_r_original reaching MAX_ENV_FRAMES - 1 corresponds to step_count reaching MAX_ENV_FRAMES - 1
            # Let's use the original reward accumulation for the success check as in dqn_test1_optuna.
            if ep_r_original >= MAX_ENV_FRAMES - 1: # Reached target total reward (500)
                 consecutive_successes += 1
            else:
                 consecutive_successes = 0 # Reset consecutive count if not a success episode

            # Check if consecutive successes threshold is met
            if consecutive_successes >= CONSECUTIVE_SUCCESSES_REQUIRED:
                # Solved the environment consistently!
                # Return the episode number (1-based) where the *last* success in the consecutive streak occurred.
                # Or just the current episode number if we want to track how many episodes were trained *until* solving.
                # dqn_test1_optuna returns i_episode + 1 upon meeting the consecutive criteria. Let's do that.
                if current_bar:
                    current_bar.text(f'Trial: {trial_number} | Ep: {i_episode} | Ep_r: {round(ep_r_original, 2)} | Steps: {step_count} | Consecutive: {consecutive_successes} / {CONSECUTIVE_SUCCESSES_REQUIRED} | SOLVED!')
                return i_episode + 1 # Return 1-based episode number as solving point

            # Check if episode ended naturally (failure or truncated by time limit)
            # This check happens *after* the consecutive success check
            if done or truncated:
                # If it ended by done/truncated, the consecutive count was already handled
                if current_bar: # Update bar text on episode end
                     status_text = f'Ep: {i_episode} | Ep_r: {round(ep_r_original, 2)} | Steps: {step_count} | Epsilon: {current_agent._get_current_epsilon():.3f}'
                     if done and not truncated: status_text += " | Failed"
                     elif truncated: status_text += " | Truncated (Max Steps)" # Should rarely happen if ep_r>=499 is checked first
                     current_bar.text(f'Trial: {trial_number} | {status_text}')

                break # End of episode

            s = s_ # Move to the next state
            step_count += 1 # Increment step counter


        return None # Not solved in this episode


    # --- Main execution flow for the trial ---
    # Use local_env within this function
    if verbose:
        # Create a printable string of the hyperparameters for logging
        params_str = ", ".join([f"{k}={v:.5g}" for k, v in hyperparams.items()])
        print(f"\n--- Starting Trial {trial_number if trial_number is not None else 'N/A'} with params: {{ {params_str} }} ---")
        # Use alive_bar based on MAX_TRAINING_EPISODES
        with alive_bar(MAX_TRAINING_EPISODES, title=f"Trial {trial_number}") as bar:
            for i_episode in range(MAX_TRAINING_EPISODES):
                # Run a single episode and check if the solving criteria was met
                solved_episode_num = _run_episode_logic(i_episode, ddqn_agent, local_env, current_bar=bar)
                if solved_episode_num is not None: # Check if solved (returns 1-based episode number)
                    episodes_to_solve = solved_episode_num # Store the 1-based episode number
                    if verbose: print(f"\nSolved in {episodes_to_solve} episodes for trial {trial_number}!")
                    break # Stop training this trial once solved
                # Update bar even if not solved
                bar()

    else: # Not verbose
        for i_episode in range(MAX_TRAINING_EPISODES):
            # Run a single episode without detailed bar updates
            solved_episode_num = _run_episode_logic(i_episode, ddqn_agent, local_env, current_bar=None)
            if solved_episode_num is not None: # Check if solved (returns 1-based episode number)
                episodes_to_solve = solved_episode_num # Store the 1-based episode number
                break # Stop training this trial once solved

    # Close the environment after the trial finishes or is stopped
    local_env.close()

    if verbose and episodes_to_solve > MAX_TRAINING_EPISODES:
        print(f"\nTrial {trial_number} did not solve within {MAX_TRAINING_EPISODES} episodes.")

    # Optuna minimizes the objective function. We want to minimize episodes to solve.
    # If it didn't solve within MAX_TRAINING_EPISODES, return a large value (MAX_TRAINING_EPISODES + 1)
    # This tells Optuna this parameter set was poor.
    return episodes_to_solve


# --- Optuna Objective Function ---
def objective(trial):
    # Define search space for hyperparameters
    # These ranges are suggestions, you might need to adjust based on initial runs
    # Based on the space from dqn_test1_optuna
    hyperparams = {
        'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [32, 64, 128, 256]),
        'LR': trial.suggest_float('LR', 1e-4, 1e-2, log=True),
        'GAMMA': trial.suggest_float('GAMMA', 0.9, 0.999, log=True),  # Often close to 1
        'TARGET_NETWORK_REPLACE_FREQ': trial.suggest_int('TARGET_NETWORK_REPLACE_FREQ', 50, 500, step=50),
        'MEMORY_CAPACITY': trial.suggest_int('MEMORY_CAPACITY', 1000, 20000, step=1000),
        'EPSILON_START': trial.suggest_float('EPSILON_START', 0.5, 1.0),
        'EPSILON_END': trial.suggest_float('EPSILON_END', 0.01, 0.1),
        'EPSILON_DECAY_STEPS': trial.suggest_int('EPSILON_DECAY_STEPS', 500, 10000, step=500),  # Total steps for decay
        'HIDDEN_SIZE': trial.suggest_categorical('HIDDEN_SIZE', [32, 64, 128])  # Hidden layer size
    }

    # Run the training and get the number of episodes to solve using the DDQN iteration
    # For more robust evaluation per trial, you could run this multiple times and average
    # However, for CartPole and faster optimization, a single run per trial is often sufficient initially.
    # num_eval_runs_per_trial = 3 # Example: Run 3 times with same params
    # scores = []
    # for i in range(num_eval_runs_per_trial):
    #     # Pass trial number and eval run index if you want more detailed logs
    #     scores.append(run_training_iteration_ddqn(hyperparams, trial_number=f"{trial.number}-{i}", verbose=False))
    # score = statistics.mean(scores) # Use the average episodes to solve

    # For simplicity, let's use a single run per trial first
    score = run_training_iteration_ddqn(hyperparams, trial_number=trial.number,
                                        verbose=False) # Set verbose=True for detailed trial logs

    return score  # Optuna will try to minimize this value (episodes to solve)


# --- Main Execution Block ---
if __name__ == "__main__":
    N_OPTUNA_TRIALS = 300  # Number of different hyperparameter sets to try

    # Construct the output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in dqn_algorithm/dqn_test3_optuna.py and output is in dqn_output/dqn_test3_optuna.txt
    project_root = os.path.dirname(script_dir) # Go up one level from script_dir (adjust if needed)
    output_dir = os.path.join(project_root, "dqn_output")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dqn_test3_optuna.txt") # Output file for DDQN Optuna results

    study_name = f"ddqn-cartpole-study-{int(time.time())}"
    # You can use a database for storage to resume studies:
    # storage_name = "sqlite:///ddqn_optuna_study.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize")
    study = optuna.create_study(study_name=study_name, direction="minimize") # Direction="minimize" episodes to solve

    print(f"Starting Optuna study: {study_name}. Optimizing DDQN for {N_OPTUNA_TRIALS} trials.")
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
    with open(output_file, "a") as f: # Use append mode for the optuna report file
        f.write(f"===== DDQN Optuna Study: {study.study_name} =====\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Number of trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"Objective: Minimize episodes to solve (max {MAX_TRAINING_EPISODES_PER_TRIAL} per trial, {CONSECUTIVE_SUCCESS_THRESHOLD} consecutive successes)\n")

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
             params_str = ", ".join([f"{k}={v:.5g}" for k, v in trial_item.params.items()]) # Use 5 significant digits for floats
             f.write(f"Trial {trial_item.number}: Value={trial_item.value}, Params={{ {params_str} }}\n")

        f.write("===== Report End =====\n\n")

    print(f"\nDDQN Optuna results saved to {output_file}")

    # Optional visualization code (requires plotly: pip install plotly)
    # import plotly.io as pio
    # pio.renderers.default = "browser" # or "vscode", "notebook" etc.
    # print("Generating plots (requires plotly)...")
    # try:
    #     optuna.visualization.plot_optimization_history(study).show()
    #     optuna.visualization.plot_param_importances(study).show()
    #     optuna.visualization.plot_slice(study).show()
    # except Exception as e:
    #     print(f"Could not generate plots. Make sure plotly is installed. Error: {e}")