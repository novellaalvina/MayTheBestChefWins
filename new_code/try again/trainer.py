import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action


# ======================== ENV WRAPPER ========================
class OvercookedMultiAgentEnv:
    def __init__(self, env):
        self.env = env
        self.n_agents = 2
        self.action_size = 6  # 6 discrete actions

    def reset(self):
        self.env.reset()
        obs = self.env.mdp.get_standard_start_state().to_dict()
        # print(f'Observation keys: {obs.keys()}')
        # print(f'Observation values: {obs}')
        return self._process_obs(obs)

    def step(self, actions):
        mapped_actions = (Action.ALL_ACTIONS[actions[0]], Action.ALL_ACTIONS[actions[1]])
        _, reward, done, info = self.env.step(mapped_actions)
        obs = self.env.state.to_dict()
        return self._process_obs(obs), [reward] * self.n_agents, done, info


    def _process_obs(self, obs):
        # Extract player positions and orientations
        player_data = []
        for player in obs['players']:
            pos = player['position']
            orientation = player['orientation']
            player_data.extend([pos[0], pos[1], orientation[0], orientation[1]])
        
        # Ensure player_data has a fixed length (8 for 2 players with position and orientation)
        player_data_length = 8  # 2 players × (2 position + 2 orientation)
        if len(player_data) < player_data_length:
            player_data.extend([0] * (player_data_length - len(player_data)))
        
        # Add placeholder for objects with fixed length
        obj_data = []
        if 'objects' in obs and obs['objects']:
            for obj in obs['objects']:
                if 'position' in obj:
                    obj_data.extend([obj['position'][0], obj['position'][1]])
        
        # Ensure obj_data has a fixed length (padding with zeros)
        obj_data_length = 4  # For example, space for 2 objects with x,y coordinates
        if len(obj_data) < obj_data_length:
            obj_data.extend([0] * (obj_data_length - len(obj_data)))
        elif len(obj_data) > obj_data_length:
            obj_data = obj_data[:obj_data_length]  # Truncate if too many objects
        
        # Add simple encoding for orders with fixed length
        order_data = []
        for order in obs['all_orders']:
            if 'ingredients' in order:
                ingredients = order['ingredients']
                # Count number of each ingredient type
                onion_count = sum(1 for ing in ingredients if ing == 'onion')
                tomato_count = sum(1 for ing in ingredients if ing == 'tomato')
                order_data.extend([onion_count, tomato_count])
        
        # Ensure order_data has a fixed length
        order_data_length = 2  # For example, space for onion count and tomato count
        if len(order_data) < order_data_length:
            order_data.extend([0] * (order_data_length - len(order_data)))
        elif len(order_data) > order_data_length:
            order_data = order_data[:order_data_length]  # Truncate if too many orders
        
        # Combine all data
        state = np.array(player_data + obj_data + order_data, dtype=np.float32)
        return [state.copy(), state.copy()]

# ======================== REPLAY BUFFER ========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure consistent shapes before storing
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Make sure we have enough samples
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Separate the batch into components
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ======================== NETWORKS ========================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):
        return self.fc(obs)  # logits (for discrete action selection)


class Critic(nn.Module):
    def __init__(self, full_obs_dim, full_action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(full_obs_dim + full_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=1)
        return self.fc(x)


# ======================== MADDPG AGENT ========================
class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, full_obs_dim, full_action_dim, lr=1e-3):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(full_obs_dim, full_action_dim)

        self.target_actor = Actor(obs_dim, action_dim)
        self.target_critic = Critic(full_obs_dim, full_action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.update_targets(tau=1.0)

    def act(self, obs, noise=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.actor(obs)
        probs = torch.softmax(logits, dim=-1)
        action = probs.multinomial(num_samples=1).item()
        if random.random() < noise:
            action = random.randint(0, probs.shape[-1] - 1)
        return action

    def update_targets(self, tau=0.01):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ======================== MAIN TRAINING ========================

def train():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env_raw = OvercookedEnv.from_mdp(mdp, horizon=400)
    env = OvercookedMultiAgentEnv(env_raw)

    obs_dim = len(env.reset()[0])
    action_dim = env.action_size
    full_obs_dim = obs_dim * env.n_agents
    full_action_dim = action_dim * env.n_agents

    agents = [MADDPGAgent(obs_dim, action_dim, full_obs_dim, full_action_dim) for _ in range(env.n_agents)]
    buffer = ReplayBuffer(100000)

    episodes = 100
    batch_size = 64

    for ep in range(episodes):
        print(f'Episode {ep}')
        obs_n = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = [agent.act(obs) for agent, obs in zip(agents, obs_n)]
            next_obs_n, rewards, done, info = env.step(actions)

            buffer.push(np.concatenate(obs_n), actions, rewards, np.concatenate(next_obs_n), done)

            obs_n = next_obs_n
            episode_reward += sum(rewards)

            if len(buffer) >= batch_size:
                states, actions_batch, rewards_batch, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states)
                actions_batch = torch.LongTensor(actions_batch)
                rewards_batch = torch.FloatTensor(rewards_batch)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                all_agent_actions = []
                next_all_agent_actions = []
                for agent_id, agent in enumerate(agents):
                    next_logits = agent.target_actor(next_states[:, agent_id*obs_dim:(agent_id+1)*obs_dim])
                    next_probs = torch.softmax(next_logits, dim=-1)
                    next_action = next_probs.multinomial(num_samples=1)
                    next_all_agent_actions.append(torch.nn.functional.one_hot(next_action.squeeze(), action_dim))

                    curr_action = torch.nn.functional.one_hot(actions_batch[:, agent_id], action_dim)
                    all_agent_actions.append(curr_action)

                all_agent_actions = torch.cat(all_agent_actions, dim=1)
                next_all_agent_actions = torch.cat(next_all_agent_actions, dim=1)

                for agent_id, agent in enumerate(agents):
                    # Critic update
                    curr_Q = agent.critic(states, all_agent_actions)
                    next_Q = agent.target_critic(next_states, next_all_agent_actions)
                    target_Q = rewards_batch[:, agent_id].unsqueeze(1) + 0.99 * (1 - dones.unsqueeze(1)) * next_Q.detach()
                    critic_loss = nn.MSELoss()(curr_Q, target_Q)

                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic_optimizer.step()

                    # Actor update
                    logits = agent.actor(states[:, agent_id*obs_dim:(agent_id+1)*obs_dim])
                    probs = torch.softmax(logits, dim=-1)
                    actions_pi = probs.multinomial(num_samples=1)
                    all_actions_pi = []
                    for j in range(env.n_agents):
                        if j == agent_id:
                            all_actions_pi.append(torch.nn.functional.one_hot(actions_pi.squeeze(), action_dim))
                        else:
                            all_actions_pi.append(torch.nn.functional.one_hot(actions_batch[:, j], action_dim))
                    all_actions_pi = torch.cat(all_actions_pi, dim=1)

                    actor_loss = -agent.critic(states, all_actions_pi).mean()

                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                    agent.update_targets()
        print(f"Episode {ep}, Reward {episode_reward}")
        # if ep % 10 == 0:
        #     print(f"Episode {ep}, Reward {episode_reward}")


if __name__ == "__main__":
    train()
