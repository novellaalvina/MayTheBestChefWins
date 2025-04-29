import gymnasium as gym
from gymnasium import spaces
import numpy as np

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


class OvercookedGymEnv(gym.Env):
    def __init__(self, mdp_fn, horizon=400):
        super().__init__()
        self.env = OvercookedEnv(mdp_fn, horizon)
        self.action_space = spaces.Discrete(6)  # 6 possible actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # 5D obs

    def reset(self):
        self.env.reset()
        obs_agent0 = self.env.mdp.featurize_state(self.env.state, 0)
        obs_agent1 = self.env.mdp.featurize_state(self.env.state, 1)
        return [obs_agent0, obs_agent1]

    def step(self, actions):
        symbolic_actions = (Action.ALL_ACTIONS[actions[0]], Action.ALL_ACTIONS[actions[1]])
        next_state, reward, done, info = self.env.step(symbolic_actions)
        obs_agent0 = self.env.mdp.featurize_state(next_state, 0)
        obs_agent1 = self.env.mdp.featurize_state(next_state, 1)
        return [obs_agent0, obs_agent1], reward, done, info

    def render(self, mode='human'):
        # (Optional) You could print grid state or visualize here
        pass