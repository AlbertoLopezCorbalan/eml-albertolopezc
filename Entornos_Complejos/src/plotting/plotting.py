"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Alberto López Corbalán
Email: alberto.lopezc@um.es
Date: 2026/06/09
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_stats(list_stats, window=100):
    # Creamos una lista de índices para el eje x
    indices = list(range(len(list_stats)))

    # Creamos el gráfico
    plt.figure(figsize=(15, 5))
    plt.plot(indices, list_stats)

    # Curva de tendencia (media móvil)
    if len(list_stats) >= window:
      trend = np.convolve(list_stats, np.ones(window)/window, mode='valid')
      plt.plot(np.arange(window-1, len(list_stats)), trend,color='red', linewidth=2, label=f'Media móvil ({window})')

    # Añadimos título y etiquetas
    plt.title('Recompensas por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')

    # Mostramos el gráfico
    plt.grid(True)
    plt.show()

def plot_episode_lengths(episode_lengths, window=100):
    
    episodes = range(len(episode_lengths))

    # Longitud de cada episodio
    plt.figure(figsize=(15,5))
    plt.plot(episodes, episode_lengths)

    # Curva de tendencia (media móvil)
    if len(episode_lengths) >= window:
      trend = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
      plt.plot(np.arange(window-1, len(episode_lengths)), trend,color='red', linewidth=2, label=f'Media móvil ({window})')

    plt.title('Longitud de los episodios')
    plt.xlabel('Episodio')
    plt.ylabel('Número de pasos')
    plt.grid(True)
    plt.show()