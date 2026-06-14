"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the normal distribution arm.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/04
"""


import numpy as np

from .armbinomial import ArmBinomial

class ArmBernoulli(ArmBinomial):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli (caso especial de Binomial con n=1).

        :param p: Probabilidad de éxito.
        """
        super().__init__(n=1, p=p)

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad de éxito.
        :param p_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 2)
            p_values.add(p)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms


