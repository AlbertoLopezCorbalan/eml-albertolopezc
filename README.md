# Problema del Bandido, Entornos Tabulares y Técnicas Aproximadas

## Información
- **Alumno:** López Corbalán, Alberto
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** albertolopezc


## Descripción

Este repositorio contiene la implementación y experimentación de diferentes algoritmos de aprendizaje por refuerzo centrados en el problema del bandido multi-brazo (Multi-Armed Bandit), así como en entornos tabulares y métodos de aproximación de funciones utilizando `Gymnasium`.

El objetivo principal es analizar el comportamiento de distintas estrategias de exploración y explotación, evaluar algoritmos clásicos de aprendizaje por refuerzo y comparar el rendimiento de enfoques tabulares frente a técnicas aproximadas en diferentes entornos de simulación.

## Estructura del Repositorio

```text
albertolopezc/
├── informe.pdf                    # Informe técnico
├── main.ipynb                     # Notebook principal que enlaza y organiza todas las partes del proyecto
│
├── k_brazos/
│   ├── src/                       # Código fuente (.py)
│   ├── main.ipynb                 # Notebook principal de esta sección enlaces a los estudios realizados 
│   ├── epsilon-greedy.ipynb       
│   ├── epsilon-greedy-decay.ipynb 
│   ├── epsilon-greedy-exploration.ipynb
│   ├── UCB1.ipynb               
│   └── softmax.ipynb                 
│
└── Entornos_Complejos/
    ├── src/                       # Código fuente (.py)
    ├── main.ipynb                 # Notebook principal de esta sección con enlaces a los estudios realizados
    ├── deep_q_learning.ipynb            
    ├── gymnasium_test.ipynb         
    ├── monte_carlo.ipynb    
    ├── q_learning.ipynb     
    ├── sarsa.ipynb        	
    └── sarsa_semi_gradient.ipynb                  
```

## Instalación y Uso

El proyecto ha sido desarrollado para ejecutarse íntegramente en **Google Colab**, por lo que no es necesario realizar una instalación local de dependencias.


## Tecnologías Utilizadas

### Lenguaje de Programación

* Python 3

### Frameworks

* Gymnasium

### Librerías

* NumPy
* Pandas
* Matplotlib


