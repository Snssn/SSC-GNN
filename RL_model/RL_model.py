import random
import time
from utils.utils import writeHeader, writeRow
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

device = T.device('cuda')

class PolicyNetwork(nn.Module):

    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)
        self.instancenorm = nn.LayerNorm(state_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.001)
        self.to(device)

    def forward(self, state, target):
        state = state.to(device)
        if isinstance(target, np.ndarray):
            target = T.from_numpy(target)
        target = target.to(device)  

        target = target.to(dtype=T.long)
        unique_values = T.unique(target)
        class_counts = T.bincount(target, minlength=len(unique_values)).float()

        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum()

        class_weights = class_weights.to(device)

        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        logits = self.prob(x)

        adjusted_logits = logits * class_weights
        prob = F.softmax(adjusted_logits, dim=-1)

        return prob

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class Reinforce:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, sample_rate, ckpt_dir, tolerance, gamma=0.99, tau=0.005, alpha_entropy=0.0000001): #tau=0.005
        self.gamma = gamma
        
        self.tau = tau  
        self.alpha_entropy = alpha_entropy 
        
        self.checkpoint_dir = ckpt_dir
        self.sample_rate = sample_rate
        self.reward_memory = []  
        self.state_memory = []   
        self.action_memory = []  
        self.log_prob_memory = [] 
        self.tolerance = tolerance

        self.num = 1
        self.rate = 0.6
        self.rate_f = 0
        self.rate_b = 0
        self.pre_reward = 0.5

        self.policy = PolicyNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=fc1_dim, fc2_dim=fc2_dim)

      
        self.update_target_networks()  

    def choose_action(self, observation,list,batch_labels):
        state = observation
        probabilities = self.policy.forward(state,batch_labels)
        dist = T.distributions.Categorical(probabilities)
        action=dist.sample()
        log_prob=dist.log_prob(action)
        self.log_prob_memory.append(log_prob[list])
        self.action_memory.append(action[list])

        return action

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def update_target_networks(self):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, batch_labels):
        self.rate = 0.6
        loss = 0
        reward = 0
        reward_pos = 0

        log_memory = torch.cat(self.log_prob_memory)
        action = torch.cat(self.action_memory)
        self.policy.optimizer.zero_grad()

        for g, log_prob, label, act in zip(self.reward_memory[0], log_memory, batch_labels, action):
            if isinstance(act, torch.Tensor) and act.numel() > 1:
                for elem in act:
                    if elem.item() == 1:
                        r = g[label] - (self.pre_reward - self.tolerance / self.rate)
                    else:
                        r = (self.pre_reward - self.tolerance / self.rate) - g[label]
                    reward += r
                    reward_pos += g[label]
                    loss += r * (-log_prob) + self.alpha_entropy * (-log_prob) 
            else:
                if act.item() == 1:
                    r = g[label] - (self.pre_reward - self.tolerance / self.rate)
                else:
                    r = (self.pre_reward - self.tolerance / self.rate) - g[label]
                reward += r
                reward_pos += g[label]
                loss += r * (-log_prob) + self.alpha_entropy * (-log_prob)  
        loss /= len(self.reward_memory[0])
        reward /= len(self.reward_memory[0])
        reward_pos /= len(self.reward_memory[0])

        self.pre_reward = (self.pre_reward * self.num + reward_pos.item()) / (self.num + 1)
        self.num += 1

        loss.backward(retain_graph=True)
        self.policy.optimizer.step()

        size = len(action)
        print(F'RL:{self.sample_rate},loss:{loss:.5f},rate:{self.rate:.5f},rate_f:{self.rate_f:.5f},rate_b:{self.rate_b:.5f},reward:{reward:.5f},reward_pos:{reward_pos:.5f},size:{size} ')
        self.reward_memory.clear()
        self.log_prob_memory.clear()
        self.action_memory.clear()


    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def memory_clear(self):
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
