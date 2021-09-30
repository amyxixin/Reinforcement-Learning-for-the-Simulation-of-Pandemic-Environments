import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

from copy import deepcopy

import pandemic_simulator as ps
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.tail = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.tail] = (state, action, reward, next_state, done)
        self.tail = (self.tail + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

class ValueNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        return x

    def sample_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)

        return action.detach().numpy()[0]

def soft_update(net, target, tau=1e-2):
    for target_param, param in zip(target.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class TD3:
    buffer_capacity = 10000
    training_batch_size = 64

    policy_lr = 1e-2
    value_lr = 1e-2
    gamma = 0.99
    tau = 1e-2
    alpha = 0.1

    def __init__(self, policy, value):
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.policy = policy
        self.target_policy = deepcopy(policy)
        self.value_1 = value
        self.value_2 = deepcopy(value)
        self.target_value_1 = deepcopy(value)
        self.target_value_2 = deepcopy(value)
        self.value_loss = nn.MSELoss()
        self.policy_optim = optim.Adam(self.policy.parameters(),
            lr=self.policy_lr)
        self.value_optim_1 = optim.Adam(self.value_1.parameters(),
            lr=self.value_lr)
        self.value_optim_2 = optim.Adam(self.value_2.parameters(),
            lr= self.value_lr)

    def training_step(self):
        state, action, reward, next_state, done = \
            self.replay_buffer.sample(self.training_batch_size)
        # Convert to torch tensors if needed
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        self.update_parameters(state, action, reward, next_state, done)

    def update_parameters(self, state, action, reward, next_state, done):
        torch.autograd.set_detect_anomaly(True)

        next_action = F.gumbel_softmax(self.target_policy(next_state), hard=True)
        #next_action = np.random.choice([next_action, np.random.choice([-1, 0, 1])], p=[1-self.alpha, self.alpha])

        target_val_1 = self.target_value_1(next_state, next_action)
        target_val_2 = self.target_value_2(next_state, next_action)
        target_val = torch.min(target_val_1, target_val_2)
        expected_val = reward + (1.0 - done) * self.gamma * target_val

        val_1 = self.value_1(state, action)
        val_2 = self.value_2(state, action)

        val_loss_1 = self.value_loss(val_1, expected_val.detach())
        val_loss_2 = self.value_loss(val_2, expected_val.detach())

        self.value_optim_1.zero_grad()
        self.value_optim_2.zero_grad()
        val_loss_1.backward()
        val_loss_2.backward()
        self.value_optim_1.step()
        self.value_optim_1.step()

        policy_loss = self.value_1(state, self.policy(state))
        policy_loss = -policy_loss.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.value_1, self.target_value_1)
        soft_update(self.value_2, self.target_value_2)
        soft_update(self.policy, self.target_policy)

def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def sanity_test():
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    ps.init_globals(seed=0)

    # select a simulator config
    sim_config = ps.sh.small_town_config

    # make env
    env = ps.env.PandemicGymEnv.from_config(sim_config, pandemic_regulations=ps.sh.austin_regulations)

    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)

    state_dim  = 12
    action_dim = 5#env.action_space.shape[0]
    hidden_dim = 128

    value = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    trainer = TD3(policy, value)

    frames = 12000
    frame = 0
    steps = 120

    rewards = []

    while frame < frames:
        state = env.reset()
        state = np.concatenate((
            np.reshape(state.global_infection_summary, (1,5)),
            np.reshape(state.global_testing_summary, (1,5)),
            np.reshape(state.stage, (1,1)),
            np.reshape(state.infection_above_threshold, (1,1))
        ), axis=1)
        episode_reward = 0
        
        for step in range(steps):
            action = F.gumbel_softmax(torch.tensor(trainer.policy.sample_action(state)), hard=True).detach().numpy().argmax()
            #action = np.random.choice([action, np.random.choice([-1, 0, 1])], p=[1-trainer.alpha, trainer.alpha])
            next_state, reward, done, _ = env.step(int(action))
            next_state_obs= next_state
            next_state = np.concatenate((
                np.reshape(next_state.global_infection_summary, (1,5)),
                np.reshape(next_state.global_testing_summary, (1,5)),
                np.reshape(next_state.stage, (1,1)),
                np.reshape(next_state.infection_above_threshold, (1,1))
        ),axis=1)
            
            action_one_hot = np.zeros((1, 5))
            action_one_hot[0,action] = 1
            trainer.replay_buffer.push(state, action_one_hot, reward, next_state, done)
            if len(trainer.replay_buffer) > trainer.training_batch_size:
                trainer.training_step()
            
            state = next_state
            episode_reward += reward
            frame += 1

            viz.record((next_state_obs, reward))

            print("Step {}, Reward: {}".format(step, reward))
            
            # if frame % 1000 == 0:
            #     plot(frame, rewards)
            
            if done:
                break

        rewards.append(episode_reward)

        if frame < frames:
            viz = ps.viz.GymViz.from_config(sim_config=sim_config)

    viz.plot()
    #viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    with open('results2.pickle', 'wb') as handle:
        pickle.dump(viz, handle, protocol=pickle.HIGHEST_PROTOCOL)

sanity_test()