import numpy as np
import random
import torch

from QNN import QNN

class AgentGlouton():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self,state_dim,action_dim,epsilon_start):
        self.q_network = QNN(state_dim,action_dim)
        self.action_dim = action_dim
        self.epsilon_start = epsilon_start

    def act_egreedy(self, state : np.ndarray) -> int:

        if np.random.rand() < self.epsilon_start:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            return torch.argmax(q_values).item()
        

        
    
        



