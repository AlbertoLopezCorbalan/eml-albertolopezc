"""
Module: algorithms/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete Agents.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/08
"""

# Importación de módulos o clases
from .agent import Agent
from .monte_carlo_on_policy import MonteCarloOnPolicy
from .q_learning import QLearning
from .deep_q_learning import DeepQLearning
from .sarsa import SARSA
from .sarsa_semi_gradient import SemiGradientSARSA

# Lista de módulos o clases públicas
__all__ = ['Agent', 'MonteCarloOnPolicy', 'QLearning', 'DeepQLearning', 'SARSA', 'SemiGradientSARSA']

