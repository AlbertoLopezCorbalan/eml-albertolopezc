"""
Module: algorithms/UCB1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/05
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (usualmente se toma c = 1).
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.

        :return: índice del brazo seleccionado.
        """

        t = np.sum(self.counts) 

        if t < self.k: # Seleccionar cada opción k una vez para inicializar la recompensa (más compacta/eficiente que la usada en epsilon_greedy_exploration)
            return t
    
        ucb_values = self.values + self.c * np.sqrt( np.log(t) / self.counts )
    
        return np.argmax(ucb_values)

