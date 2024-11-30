import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNTarget():
    """Agent qui utilise l'algorithme DQN."""
    def __init__(self, state_size:int, action_size:int, teu:float,taille_buffer:int, taille_batch:int, gamma=0.99):
        """Constructeur.
        

        """
        self.state_size = state_size
        self.action_size = action_size
        self.taille_batch = taille_batch
        self.taille_buffer = taille_buffer
        self.rpBuffer = ReplayBuffer(taille_buffer,taille_batch)
        self.model = QNN(state_size,action_size)
        self.target_model = QNN(state_size,action_size)
        self.gamma = gamma
        self.tau = teu
        self.loss = nn.MSELoss()
        

    def sampling_step(self,state : np.ndarray ,action : np.ndarray ,reward: float,next_state: np.ndarray ,done: bool):      
        return self.rpBuffer.add(state, action, reward, next_state, done)
        
    def train_step(self):
        
        optimizer1 = optim.Adam(self.model.parameters(), lr=0.001)
        state,action,reward,next_state,done = self.rpBuffer.sample()

        prediction = self.model(state).gather(1, action)

        q_next_states = self.target_model(next_state).detach().max(1)[0].unsqueeze(1)

        
        with torch.no_grad():
            target_q_values = reward + (self.gamma*q_next_states)*(1-done)
            
        loss = self.loss(prediction, target_q_values)
        
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
             target_param.data.copy_(self.tau*online_param.data + (1.0-self.tau)*target_param.data)
                
    
    def act_egreedy(self, state : np.ndarray ,eps: float = 0.0) -> int:
            if np.random.rand() <= eps:
                return random.randint(0,self.action_size-1)
            
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(action_values).item()
        

