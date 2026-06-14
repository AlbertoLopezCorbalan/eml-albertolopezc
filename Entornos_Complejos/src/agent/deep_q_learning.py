"""
Module: Agent/DeepQLearning.py
Description: Implementación de Deep Q-Learning (DQN) para estados continuos.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/11
"""

from collections import deque
import random

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


class DeepQLearning(Agent):

    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float, replay_buffer_size=50000, batch_size=32):

        super().__init__(env, discount_factor)

        self.network = QNetwork(self.env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=replay_buffer_size)

        self.batch_size = batch_size

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state, training=True):
        # Con probabilidad epsilon seleccionamos una acción aleatoria
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Si no a_t = arg max a Q(phi(s_t), a ; theta)
        with torch.no_grad():
            q_values = self.network(torch.FloatTensor(state).unsqueeze(0))

        return int(torch.argmax(q_values).item())

    def update(self, state, next_state, action, reward, terminated, truncated):

        done = terminated or truncated

        # Guardamos la transición (phi_t, a_t, r_t, phi_t+1) en la memoria de repetición 
        self.memory.append((state, action, reward, next_state, done))

        # Esperar hasta tener suficientes experiencias para formar un lote
        if len(self.memory) < self.batch_size:
            return

        # Obtenemos un minilote de transiciones aleatorio
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)


        # yj = r_j + gamma max_a Q(s',a)  ó r_j si es nodo terminal (se controla mediante 1 - done)
        with torch.no_grad(): # Para que al next_q no se le calcule el gradiente cuando se haga loss.backwards() 
            next_q = self.network(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.discount_factor * next_q

        
        # Calculamos la pérdida (y_j − Q(phi_j, aj ; theta))^2
        # Q(phi_j, a_j ; theta)
        q_values = self.network(states)
        current_q = q_values.gather(1, actions)
        
        loss = self.loss_fn(current_q, targets)

        # Gradiente descendiente
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decay_exploration(self):
        """Disminuye epsilon al final de cada episodio."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def end_episode(self):
        pass

    def get_q_table(self):
        """
        No existe tabla Q en DQN.
        """
        pass