import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

# --- Gymnasium Wrapper for Overcooked ---
class OvercookedGymnasiumEnv(gym.Env):
    def __init__(self, mdp, horizon=400):
        super().__init__()
        self.env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
        self.action_space = spaces.MultiDiscrete([6, 6])  # Two agents, each has 6 actions
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        ])

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs_agent0 = self.env.mdp.featurize_state(self.env.state, 0)
        obs_agent1 = self.env.mdp.featurize_state(self.env.state, 1)
        obs = (obs_agent0, obs_agent1)
        return obs, {}

    def step(self, actions):
        symbolic_actions = (Action.ALL_ACTIONS[actions[0]], Action.ALL_ACTIONS[actions[1]])
        next_state, reward, done, info = self.env.step(symbolic_actions)
        obs_agent0 = self.env.mdp.featurize_state(next_state, 0)
        obs_agent1 = self.env.mdp.featurize_state(next_state, 1)
        obs = (obs_agent0, obs_agent1)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

# --- Hyperparameters ---
HORIZON = 400
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
STEPS_PER_ROLLOUT = 2048
EPOCHS_PER_UPDATE = 10
MINIBATCH_SIZE = 256
TOTAL_TIMESTEPS = 100_000

# --- Environment Setup ---
mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedGymnasiumEnv(mdp, horizon=HORIZON)

ACTIONS = Action.ALL_ACTIONS
obs_dim = 5  # the Overcooked symbolic obs dictionary size
n_actions = len(ACTIONS)

# --- Policy and Value Networks ---
class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits

class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        value = self.fc(x)
        return value.squeeze(-1)

# --- Initialize Models for Two Agents ---
policy_agent0 = Policy(obs_dim, n_actions)
value_agent0 = Value(obs_dim)
policy_agent1 = Policy(obs_dim, n_actions)
value_agent1 = Value(obs_dim)

policy_optimizer_agent0 = optim.Adam(policy_agent0.parameters(), lr=POLICY_LR)
value_optimizer_agent0 = optim.Adam(value_agent0.parameters(), lr=VALUE_LR)
policy_optimizer_agent1 = optim.Adam(policy_agent1.parameters(), lr=POLICY_LR)
value_optimizer_agent1 = optim.Adam(value_agent1.parameters(), lr=VALUE_LR)

# --- Rollout Storage ---
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

buffer_agent0 = RolloutBuffer()
buffer_agent1 = RolloutBuffer()

# --- Helper Functions ---
def select_action(policy, value_fn, obs):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    logits = policy(obs_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    value = value_fn(obs_tensor)
    return action.item(), log_prob, value

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# --- Training Loop ---
obs, info = env.reset()
obs_agent0 = obs[0]
obs_agent1 = obs[1]

timesteps = 0

while timesteps < TOTAL_TIMESTEPS:
    buffer_agent0.clear()
    buffer_agent1.clear()

    for _ in range(STEPS_PER_ROLLOUT):
        action_idx0, log_prob0, value0 = select_action(policy_agent0, value_agent0, obs_agent0)
        action_idx1, log_prob1, value1 = select_action(policy_agent1, value_agent1, obs_agent1)

        joint_action = [action_idx0, action_idx1]

        obs, reward, terminated, truncated, info = env.step(joint_action)
        obs_agent0 = obs[0]
        obs_agent1 = obs[1]
        done = terminated or truncated

        buffer_agent0.obs.append(obs_agent0)
        buffer_agent0.actions.append(action_idx0)
        buffer_agent0.log_probs.append(log_prob0)
        buffer_agent0.rewards.append(reward)
        buffer_agent0.dones.append(done)
        buffer_agent0.values.append(value0)

        buffer_agent1.obs.append(obs_agent1)
        buffer_agent1.actions.append(action_idx1)
        buffer_agent1.log_probs.append(log_prob1)
        buffer_agent1.rewards.append(reward)
        buffer_agent1.dones.append(done)
        buffer_agent1.values.append(value1)

        timesteps += 1

        if done:
            obs, info = env.reset()
            obs_agent0 = obs[0]
            obs_agent1 = obs[1]

    # (Training updates unchanged from above)
