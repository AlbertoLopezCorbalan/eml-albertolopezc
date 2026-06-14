"""
Module: Agent/QLearning.py
Description: Implementación de diferencias temporales mediante Q-Learning.
Base on https://gymnasium.farama.org/introduction/train_agent/.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/08
"""


import gymnasium as gym
import numpy as np
from agent import Agent


class QLearning(Agent):

    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, 
                 epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):        
        super().__init__(env, discount_factor)
        
        self.learning_rate = learning_rate
        # Parámetros de exploración
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, state: np.ndarray, training: bool = True):
        """Política epsilon-greedy."""
        if training and np.random.random() < self.epsilon: # random
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float,
               terminated: bool, truncated: bool):
        
        done = terminated or truncated
        future_q = 0.0
        if not done:
            future_q = np.max(self.q_values[next_state])

        temporal_difference  = reward + self.discount_factor * future_q - self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * temporal_difference 

    # Funciones auxiliares

    def decay_exploration(self):
        """Disminuye epsilon al final de cada episodio."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def end_episode(self):
        """No se requiere acción al final del episodio para Q-Learning."""
        pass
