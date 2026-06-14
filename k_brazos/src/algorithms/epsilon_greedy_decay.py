"""
Module: algorithms/epsilon_greedy_decay.py
Description: Implementación del algoritmo epsilon-greedy-decay para el problema de los k-brazos.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/09

"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonGreedyDecay(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1, epsilon_decay: float = 0.95, final_epsilon: float = 0.01):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :param epsilon_decay: Factor multiplicativo utilizado para reducir el valor de epsilon después de cada iteración.
        :param final_epsilon: Valor mínimo que puede alcanzar epsilon durante el proceso de decaimiento.
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon
        self.start_epsilon = epsilon # Guardamos el inicial para las etiquetas de las gráficas
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.

        :return: índice del brazo seleccionado.
        """

        # Seleccionar cada opción k una vez para inicializar la recompensa (opcional)
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            chosen_arm = np.argmax(self.values)

        return chosen_arm

    def decay_exploration(self):
        """Disminuye epsilon al final de cada episodio."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



