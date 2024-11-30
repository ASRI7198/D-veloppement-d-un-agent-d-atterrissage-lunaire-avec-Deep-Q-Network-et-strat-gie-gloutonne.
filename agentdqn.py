import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQN():
    """Agent qui utilise l'algorithme de deep QLearning avec replaybuffer."""

    def __init__(self, state_size:int, action_size:int,taille_buffer:int, taille_batch:int, gamma=0.99):
        """Constructeur.
        

        """
        self.state_size = state_size
        self.action_size = action_size
        self.taille_batch = taille_batch
        self.taille_buffer = taille_buffer
        self.rpBuffer = ReplayBuffer(taille_buffer, taille_batch)
        self.model = QNN(state_size,action_size)
        self.gamma = gamma
        self.loss = nn.MSELoss()
        

    def sampling_step(self,state : np.ndarray ,action : np.ndarray ,reward: float,next_state: np.ndarray ,done: bool):      
        return self.rpBuffer.add(state, action, reward, next_state, done)
        
    def train_step(self):
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        state,action,reward,next_state,done = self.rpBuffer.sample()
        
        # state = torch.FloatTensor(state)
        # action = torch.tensor(action, dtype=torch.float32).unsqueeze(1)
        # #action = torch.FloatTensor(action).unsqueeze(1)
        # reward = torch.tensor(reward)
        # next_state = torch.FloatTensor(next_state).unsqueeze(1)
        # done = torch.tensor(done)
        
        prediction = self.model(state)
       #S print(done)
        target = reward + self.gamma*torch.max(self.model(next_state)) * (1-done)

        
        loss = self.loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    def act_egreedy(self, state : np.ndarray ,eps: float = 0.0) -> int:
            if np.random.rand() <= eps:
                return random.randint(0,self.action_size-1)
            
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(action_values).item()
        