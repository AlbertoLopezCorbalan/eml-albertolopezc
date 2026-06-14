"""
Module: Agent/sarsa_semi_gradient.py
Description: Implementación de Semi-gradient Sarsa.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/10
"""

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from agent import Agent

class QNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_space)
        )

    def forward(self, x):
        return self.network(x)


class SemiGradientSARSA(Agent):

    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float):

        super().__init__(env, discount_factor)
        self.network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state, training=True):

        # Con probabilidad epsilon seleccionamos una acción aleatoria
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()


        with torch.no_grad():
            q_values = self.network(torch.FloatTensor(state).unsqueeze(0))

        # Si no es aleatoria, la acción que maximice
        return int(torch.argmax(q_values).item())

    def update(self, state, next_state, action, next_action, reward, terminated, truncated):

        done = terminated or truncated

        state = (torch.FloatTensor(state).unsqueeze(0))
        next_state = (torch.FloatTensor(next_state).unsqueeze(0))

        # Q(S,A,w)
        current_q = self.network(state)[0, action]

        with torch.no_grad(): # Para que al next_q no se le calcule el gradiente cuando se haga loss.backwards() 

            # Si es terminal no se tiene en cuenta el next_q por que ya se habrá terminado
            if done:
                target = torch.tensor(reward, dtype=torch.float32)
            else:
                # R + gamma Q(S', A', w)
                next_q = self.network(next_state)[0, next_action]
                target = (reward + self.discount_factor * next_q)

        # Se calcula el error
        # delta = R + gamma * Q(S',A') - Q(S,A)
        loss = self.loss_fn(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Funciones auxiliares
    def decay_exploration(self):
        """Disminuye epsilon al final de cada episodio."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def end_episode(self):
        """No se requiere acción al final del episodio para Q-Learning."""
        pass


    def get_q_table(self):
        """
        No existe tabla Q en semi-gradient sarsa.
        """
        pass
