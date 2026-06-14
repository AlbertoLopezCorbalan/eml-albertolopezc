"""
Module: Agent/Agent.py
Description: Contiene la implementación abstracta de un Agente.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/05
"""

from abc import ABC, abstractmethod
import gymnasium as gym
from collections import defaultdict
import numpy as np

class Agent(ABC):

    def __init__(self, env: gym.Env, discount_factor: float):
        """
        Inicializa los parámetros del agente.

        Args:
            env: Entorno de entrenamiento.
            learning_rate: Tasa de aprendizaje.
            discount_factor: Factor de descuento.
        """
        self.env = env
        self.discount_factor = discount_factor
        # Tabla Q: estado -> vector de valores Q por acción 
        # (defaultdict crea automáticamente entradas con ceros para nuevas claves)
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

    @abstractmethod
    def get_action(self, state: np.ndarray, training: bool = True):
        """
        Selecciona una acción para un estado dado.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state: np.ndarray, next_state: np.ndarray, action: int, reward: float,
               terminated: bool, truncated: bool):
        """
        Actualiza el conocimiento del agente tras una transición.
        """
        raise NotImplementedError

    @abstractmethod
    def decay_exploration(self):
        """
        Actualiza los parámetros de exploración.
        Se invoca al final de cada episodio.
        """
        raise NotImplementedError

    @abstractmethod
    def end_episode(self):
        """
        Notifica al agente que el episodio ha finalizado.
        Este método permite realizar operaciones que requieren disponer de la trayectoria completa, como los algoritmos Monte Carlo.
        En algoritmos de Diferencias temporales (Q-Learning, SARSA, etc.) puede dejarse vacío.
        """
        raise NotImplementedError

    # devuelve la tabla Q completa
    def get_q_table(self):
        return dict(self.q_values)