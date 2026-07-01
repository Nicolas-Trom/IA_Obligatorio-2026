"""Agente Dyna-Q para MountainCarContinuous-v0.

Dyna-Q (Sutton & Barto, cap. 8.1-8.2) combina aprendizaje directo con
planificacion sobre un modelo aprendido del ambiente:

1. Ejecutar una accion real y actualizar Q con la transicion real
   (identico a Q-Learning).
2. Guardar la transicion en un modelo ``model[(s, a)] = (r, s', terminal)``.
3. Repetir ``planning_steps`` veces: samplear una transicion ya observada
   y aplicar la misma actualizacion de Q, "simulando" experiencia.

Se reutiliza toda la maquinaria de ``QLearningAgent`` (politica, update,
train, evaluate, save/load); solo se agrega el modelo y el paso de
planificacion mediante el hook ``_after_real_step``. Con ``planning_steps=0``
Dyna-Q se comporta exactamente como Q-Learning.
"""

import numpy as np

from q_learning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    algo_name = "dyna_q"

    def __init__(self, *args, planning_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = planning_steps
        # Modelo determinista: (pos_idx, vel_idx, action) -> (r, npos, nvel, terminal)
        self.model = {}

    def _after_real_step(self, state, action, reward, next_state, terminal):
        """Guarda la transicion real y hace `planning_steps` updates simulados."""
        key = (state[0], state[1], action)
        self.model[key] = (reward, next_state[0], next_state[1], terminal)

        if self.planning_steps <= 0 or not self.model:
            return

        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            k = keys[self.rng.integers(len(keys))]
            r_sim, npos, nvel, terminal_sim = self.model[k]
            self._update((k[0], k[1]), k[2], r_sim, (npos, nvel), terminal_sim)

    def config(self) -> dict:
        cfg = super().config()
        cfg["planning_steps"] = self.planning_steps
        return cfg
