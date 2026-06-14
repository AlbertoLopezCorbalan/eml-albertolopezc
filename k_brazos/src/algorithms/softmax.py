"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/05
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo softmax.

        :param k: Número de brazos.
        :param tau: Temperatura. Un valor elevado se centrará en mayor exploración, mientras que un valor bajo será explotación.
        :raises ValueError: Si tau es 0.
        """
        assert 0 < tau, "El parámetro tau debe ser mayor que 0."

        super().__init__(k)
        self.tau = tau # Temperatura

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.

        :return: índice del brazo seleccionado.
        """

        exp_values = np.exp(self.values / self.tau) # exp_values =  e^(q_t(a) / tau)

        probabilities = exp_values / np.sum(exp_values) # exp_values / Suma todos los (exp_values)
    
        return np.random.choice(self.k, p=probabilities) # random choice entre las probabilidades calculadas




