"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, EpsilonGreedyExploration, EpsilonGreedyDecay, Softmax, UCB1


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy) or isinstance(algo, EpsilonGreedyExploration):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, EpsilonGreedyDecay):
        label += f" (start_epsilon={algo.start_epsilon})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Acción Óptima', fontsize=14)
    plt.title('Acción Óptima vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Rechazo Acumulado', fontsize=14)
    plt.title('Rechazo Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(avg_arm_rewards:np.ndarray, arm_selections:np.ndarray, optimal_arm:int , algorithms: List[Algorithm]):
    """
    Genera gráficas separadas de Selección de Arms:
    Ganancias vs Pérdidas para cada algoritmo.
    :param avg_arm_rewards:  Matriz con la recompensa promedio por cada brazo.
    :param arm_selections:  Matriz con el número de veces que cada brazo fue seleccionado
    :param optimal_arm: brazo óptimo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    
    for idx, algo in enumerate(algorithms):
        n_arms = len(avg_arm_rewards[idx])
    
        labels = [ f"Brazo {arm + 1}\nN={int(arm_selections[idx, arm])}"+ ("\nÓptimo" if arm == optimal_arm else "")  for arm in range(n_arms)]
        colors = ["tab:green" if arm == optimal_arm else "tab:blue" for arm in range(n_arms)]
    
        plt.figure(figsize=(14, 7))
    
        plt.bar(range(n_arms), avg_arm_rewards[idx], color=colors, edgecolor="black")
        plt.xticks(range(n_arms), labels)
        plt.xlabel("Brazos", fontsize=14)
        plt.ylabel("Ganancia Promedio", fontsize=14)
        plt.title(f"Estadísticas de Brazos - {get_algorithm_label(algo)}", fontsize=16)
        plt.tight_layout()
        plt.show()