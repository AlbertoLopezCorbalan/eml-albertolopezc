"""
Module: Agent/MonteCarloOnPolicy.py
Description: Implementación Monte Carlo On-Policy First-Visit.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/08
"""

import gymnasium as gym
import numpy as np

from agent import Agent


class MonteCarloOnPolicy(Agent):

    def __init__(self, env: gym.Env, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float = 0.95):
                     
        super().__init__(env, discount_factor)

        # Exploración
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Para los promedios de las visitas
        self.n_visits = np.zeros((env.observation_space.n, self.env.action_space.n))

        # Trayectoria del episodio
        self.episode = []

    def get_action(self, state, training=True):
        """Política epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update(self, state, next_state, action, reward, terminated, truncated):
        """
        En Monte Carlo no se actualiza inmediatamente.
        Solo se almacena la transición.
        """
        self.episode.append((state, action, reward))
        
    def end_episode(self):
        """
        Actualización Monte Carlo On-Policy All-Visit.
        """
        G = 0.0
        for state, action, reward in reversed(self.episode):
            G = reward + self.discount_factor * G
            self.n_visits[state][action] += 1
            self.q_values[state][action] += ((G - self.q_values[state][action]) / self.n_visits[state][action])
        # Limpiar episodio
        self.episode.clear()

    def decay_exploration(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


