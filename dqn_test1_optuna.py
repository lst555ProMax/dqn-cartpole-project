''''
一个特殊的版本，用optuna库和对应的操作寻找相对较优的超参数
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from alive_progress import alive_bar
import optuna  # Import Optuna
import time

# --- Environment Setup (remains mostly global for simplicity here) ---
env = gym.make("CartPole-v1")
env = env.unwrapped  # Use unwrapped for more control if needed, though not strictly necessary for CartPole hyperparams
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
MAX_ENV_FRAMES = 500  # Renamed from maximum_episode_length for clarity as it's a per-episode limit


# --- Neural Network Definition ---
class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=50):  # Added hidden_size as a tunable param
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Changed to ReLU, often works better
        return self.out(x)


# --- DQN Agent ---
class DQN(object):
    def __init__(self, n_states, n_actions, env_a_shape, hyperparams):
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape

        # Unpack hyperparameters
        self.lr = hyperparams['LR']
        self.gamma = hyperparams['GAMMA']
        self.target_network_replace_freq = hyperparams['TARGET_NETWORK_REPLACE_FREQ']
        self.memory_capacity = hyperparams['MEMORY_CAPACITY']
        self.batch_size = hyperparams['BATCH_SIZE']
        self.epsilon_start = hyperparams['EPSILON_START']
        self.epsilon_end = hyperparams['EPSILON_END']
        self.epsilon_decay_steps = hyperparams['EPSILON_DECAY_STEPS']
        hidden_size = hyperparams['HIDDEN_SIZE']

        self.eval_net = Net(n_states, n_actions, hidden_size)
        self.target_net = Net(n_states, n_actions, hidden_size)
        self.target_net.load_state_dict(self.eval_net.state_dict())  # Ensure they start identical
        self.target_net.eval()  # Target network is not for training

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.total_steps = 0  # For epsilon decay

    def _get_current_epsilon(self):
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        return self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.total_steps / (self.epsilon_decay_steps / 5))  # Faster decay at start

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        epsilon = self._get_current_epsilon()
        self.total_steps += 1

        if np.random.uniform() < epsilon:
            action = np.random.randint(0, self.n_actions)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        else:
            self.eval_net.eval()  # Set to evaluation mode for inference
            with torch.no_grad():  # No need to track gradients here
                actions_value = self.eval_net(x)
            self.eval_net.train()  # Set back to training mode
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_network_replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Ensure memory is full enough for batching
        current_memory_size = min(self.memory_counter, self.memory_capacity)
        if current_memory_size < self.batch_size:
            return  # Not enough samples to learn

        sample_index = np.random.choice(current_memory_size, self.batch_size, replace=False)
        b_memory = self.memory[sample_index, :]

        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but can help stability)
        # torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()


# --- Training Function for a Single Run (to be called by Optuna) ---
# --- Training Function for a Single Run (to be called by Optuna) ---
def run_training_iteration(hyperparams, trial_number=None, verbose=False):
    dqn = DQN(N_STATES, N_ACTIONS, ENV_A_SHAPE, hyperparams)

    MAX_TRAINING_EPISODES = 300
    CONSECUTIVE_SUCCESS_THRESHOLD = 5

    episodes_to_solve = MAX_TRAINING_EPISODES + 1
    consecutive_successes = 0
    all_episode_rewards = []

    # --- Helper function for the main loop content to reduce duplication ---
    def _run_episode_logic(i_episode, current_dqn, current_bar=None):
        nonlocal consecutive_successes # Allow modification of outer scope variable
        s, _ = env.reset()
        ep_r = 0
        step_count = 0
        frames_this_episode = 0

        while True:
            # if current_bar: # If you want per-step updates for alive_bar
            #     current_bar()

            a = current_dqn.choose_action(s)
            s_, r, done, truncated, _ = env.step(a)

            x, _, theta, _ = s_
            rx = -(x / env.x_threshold) ** 2
            rtheta = -(theta / env.theta_threshold_radians) ** 2
            modified_r = r
            if not done:
                modified_r += (rx + rtheta) * 0.1
            elif step_count < MAX_ENV_FRAMES -1:
                 modified_r = -100

            current_dqn.store_transition(s, a, modified_r, s_)
            ep_r += r # Original reward for success metric
            frames_this_episode +=1


            if current_dqn.memory_counter > hyperparams['BATCH_SIZE']:
                current_dqn.learn()

            if done or truncated or step_count >= MAX_ENV_FRAMES - 1:
                all_episode_rewards.append(ep_r)
                if current_bar:
                    current_bar.text(f'Ep: {i_episode} | Ep_r: {round(ep_r, 2)} | Steps: {step_count} | Epsilon: {current_dqn._get_current_epsilon():.3f}')

                if ep_r >= MAX_ENV_FRAMES -1: # frames_this_episode also works here if r is always 1
                    consecutive_successes += 1
                else:
                    consecutive_successes = 0

                if consecutive_successes >= CONSECUTIVE_SUCCESS_THRESHOLD:
                    return i_episode + 1 # Solved
                break
            s = s_
            step_count += 1
        return None # Not solved in this episode

    # --- Main execution flow for the trial ---
    if verbose:
        print(f"\n--- Starting Trial {trial_number if trial_number is not None else 'N/A'} with params: {hyperparams} ---")
        with alive_bar(MAX_TRAINING_EPISODES, title=f"Trial {trial_number}") as bar:
            for i_episode in range(MAX_TRAINING_EPISODES):
                solved_episode_num = _run_episode_logic(i_episode, dqn, current_bar=bar)
                if solved_episode_num is not None:
                    episodes_to_solve = solved_episode_num
                    if verbose: print(f"\nSolved in {episodes_to_solve} episodes for trial {trial_number}!")
                    return episodes_to_solve
                bar() # Manually update the bar for each episode completed
    else: # Not verbose
        for i_episode in range(MAX_TRAINING_EPISODES):
            solved_episode_num = _run_episode_logic(i_episode, dqn, current_bar=None)
            if solved_episode_num is not None:
                episodes_to_solve = solved_episode_num
                # No print here as it's not verbose
                return episodes_to_solve

    if verbose and episodes_to_solve > MAX_TRAINING_EPISODES:
        print(f"\nTrial {trial_number} did not solve within {MAX_TRAINING_EPISODES} episodes.")
    return episodes_to_solve

# --- Optuna Objective Function ---
def objective(trial):
    # Define search space for hyperparameters
    # Note: Some of these ranges might need adjustment based on initial runs
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

    # Run the training and get the number of episodes to solve
    # For more robust evaluation, you could run this multiple times and average
    # num_eval_runs = 3
    # scores = []
    # for i in range(num_eval_runs):
    #     scores.append(run_training_iteration(hyperparams, trial_number=f"{trial.number}-{i}", verbose=False))
    # score = np.mean(scores)

    score = run_training_iteration(hyperparams, trial_number=trial.number,
                                   verbose=False)  # Set verbose=True for detailed trial logs

    return score  # Optuna will try to minimize this value (episodes to solve)


# --- Main Execution Block ---
if __name__ == "__main__":
    N_OPTUNA_TRIALS = 50  # Number of different hyperparameter sets to try

    study_name = f"dqn-cartpole-study-{int(time.time())}"
    # You can use a database for storage to resume studies:
    # storage_name = "sqlite:///example.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize")
    study = optuna.create_study(study_name=study_name, direction="minimize")

    print(f"Starting Optuna study: {study_name}. Optimizing for {N_OPTUNA_TRIALS} trials.")
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
    with open("dqn_test1_optuna.txt", "a") as f:
        f.write(f"===== Optuna Study: {study.study_name} =====\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Number of trials: {N_OPTUNA_TRIALS}\n")
        f.write("\n--- Best Trial ---\n")
        f.write(f"Value (Episodes to solve): {best_trial.value}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n--- All Trials ---\n")
        for trial_item in study.trials:
            f.write(f"Trial {trial_item.number}: Value={trial_item.value}, Params={trial_item.params}\n")
        f.write("===== Report End =====\n\n")

    print("\nOptuna results saved to dqn_test1_optuna.txt")

    # You can also visualize:
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_slice(study).show()
    # Needs plotly: pip install plotly

    # Example of running with best params
    # print("\n--- Running with best parameters found ---")
    # best_params = study.best_params
    # # You might want to run it for more episodes or with different seeds here
    # final_score = run_training_iteration(best_params, trial_number="BEST", verbose=True)
    # print(f"Performance with best params: Solved in {final_score} episodes (or failed if > {MAX_TRAINING_EPISODES})")