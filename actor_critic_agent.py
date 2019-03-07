#!/usr/bin/env python
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MNISTNet(nn.Module):
    '''
    A CNN with ReLU activations and a three-headed output, two for the 
    actor and one for the critic
    
    y1 - movement direction distribution
    y2 - class probabilities
    y3 - critic's estimate of value
    
    Input shape:    (batch_size, D_in)
    Output shape:   (batch_size, 4), (batch_size, 10), (batch_size, 1)
    '''
    
    def __init__(self):
        
        super(MNISTNet, self).__init__()
        
        same_padding = (5 - 1) // 2
        
        self.conv1 = nn.Conv2d(1, 10, 5, padding=same_padding)
        self.conv2 = nn.Conv2d(10, 10, 5, padding=same_padding)
        self.lin1  = nn.Linear(10 * 7 * 7, 50)
        
        self.out_dir = nn.Linear(50, 4)
        self.out_digit = nn.Linear(50, 10)
        self.out_critic = nn.Linear(50, 1)
        
    def forward(self, x):
    
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2)(x)
        
        x = x.view(-1, 10 * 7 * 7)
        x = self.lin1(x)
        
        y1 = self.out_dir(x)
        y1 = F.softmax(y1, dim=-1)
        
        y2 = self.out_digit(x)
        y2 = F.softmax(y2, dim=-1)
        
        y3 = self.out_critic(x)
        
        return y1, y2, y3

def torch_to_numpy(tensor):
    return tensor.data.numpy()

def numpy_to_torch(array):
    return torch.tensor(array).float()

class ActorCriticNNAgent:
    '''
    Neural-net agent that trains using the actor-critic algorithm. The critic 
    is a value function that returns expected discounted reward given the
    state as input. We use advantage defined as
    
        A = r + g * V(s') - V(s)
        
    Notation:
        A - advantage
        V - value function
        r - current reward
        g - discount factor
        s - current state
        s' - next state
    '''
    
    def __init__(self, new_network, params=None, obs_to_input=lambda x: x, 
                 lr=1e-3, df=0.5, alpha=0.5):
    
        # model and parameters
        if params is not None:
            self.model = new_network(params)
        else:
            self.model = new_network()
        if isinstance(self.model, torch.nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.df = df # discount factor
        self.alpha = alpha # multiply critic updates by this factor
        
        # initialize replay history
        self.replay = []
        
        # function that converts observation into input of dimension D_in
        self.obs_to_input = obs_to_input
        
        # if trainable is changed to false, the model won't be updated
        self.trainable = True
        
    def act(self, o, env=None, display=False):
        
        # feed observation as input to net to get distribution as output
        x = self.obs_to_input(o)
        x = numpy_to_torch([x])
        y1, y2, y3 = self.model(x)
        
        pi1 = torch_to_numpy(y1).flatten()
        pi2 = torch_to_numpy(y2).flatten()
        v   = torch_to_numpy(y3).squeeze()
        
        # sample action from distribution
        direction = np.random.choice(np.arange(4), p=pi1)
        digit     = np.random.choice(np.arange(10), p=pi2)
        a = (direction, digit)
        
        if display: print("")
        if display: print("Sampled action:", a)
        if display: print("Value estimate:", v) 
        if display: print("Distributions:", pi1, pi2, sep='\n')
        
        # update current episode in replay with observation and chosen action
        if self.trainable:
            self.replay[-1]['observations'].append(o)
            self.replay[-1]['actions'].append(a)
        
        return np.array(a)
        
    def new_episode(self):
        # start a new episode in replay
        self.replay.append({'observations': [], 'actions': [], 'rewards': []})
        
    def store_reward(self, r):
        # insert 0s for actions that received no reward; end with reward r
        episode = self.replay[-1]
        T_no_reward = len(episode['actions']) - len(episode['rewards']) - 1
        episode['rewards'] += [0.0] * T_no_reward + [r]
        
    def _calculate_discounted_rewards(self):
        # calculate and store discounted rewards per episode
        
        for episode in self.replay:
            
            R = episode['rewards']
            R_disc = []
            R_sum = 0
            for r in R[::-1]:
                R_sum = r + self.df * R_sum
                R_disc.insert(0, R_sum)
                
            episode['rewards_disc'] = R_disc
        
    def update(self):
        
        assert(self.trainable)
        
        episode_losses = torch.tensor(0.0)
        N = len(self.replay)
        self._calculate_discounted_rewards()
        
        for episode in self.replay:

            O = episode['observations']
            A = episode['actions']
            R = numpy_to_torch(episode['rewards'])
            R_disc = numpy_to_torch(episode['rewards_disc'])
            T = len(R_disc)
            
            # forward pass, Y1 is pi(a | s), Y2 is V(s)
            X = numpy_to_torch([self.obs_to_input(o) for o in O])
            Y1, Y2, Y3 = self.model(X)
            pi1, pi2 = Y1, Y2
            Vs_curr = Y3.view(-1)
            
            # log probabilities of selected actions
            log_prob = torch.log(pi1[np.arange(T), [tup[0] for tup in A]]) \
                     + torch.log(pi2[np.arange(T), [tup[1] for tup in A]])
            
            # advantage of selected actions over expected reward given state
            Vs_next = torch.cat((Vs_curr[1:], torch.tensor([0.])))
            adv = R + self.df * Vs_next - Vs_curr
            
            # ignore gradients so the critic isn't affected by actor loss
            adv = adv.detach()
            
            # actor loss is -1 * advantage-weighted sum of log likelihood
            # critic loss is the SE between values and discounted rewards
            actor_loss = -torch.dot(log_prob, adv)
            critic_loss = torch.sum((R_disc - Vs_curr) ** 2)
            episode_losses += actor_loss + critic_loss * self.alpha
            
        # backward pass
        self.optimizer.zero_grad()
        loss = episode_losses / N
        loss.backward()
        self.optimizer.step()
        
        # reset the replay history
        self.replay = []

    def copy(self):
        
        # create a copy of this agent with frozen weights
        agent = ActorCriticNNAgent(lambda x: 0, 0, self.obs_to_input)
        agent.model = copy.deepcopy(self.model)
        agent.trainable = False
        for param in agent.model.parameters():
            param.requires_grad = False
            
        return agent
