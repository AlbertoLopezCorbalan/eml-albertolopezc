"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the normal distribution arm.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/04
"""


import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial.

        :param n: Número de ensayos.
        :param p: Probabilidad de éxito por ensayo.
        """
        assert n > 0, "El número de acciones n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial.

        :return: Recompensa obtenida del brazo (0 a n).
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.

        :return: Valor esperado de la distribución.
        """

        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo Binomial.

        :return: Descripción detallada del brazo Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max] y número de ensayos fijo n

        :param k: Número de brazos a generar.
        :param n: Número de ensayos por brazo.
        :param p_min: Valor mínimo de la probabilidad de éxito.
        :param p_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de mu_min debe ser menor que mu_max."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."

        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        arms = [ArmBinomial(n, p) for p in p_values]
        
        return arms


