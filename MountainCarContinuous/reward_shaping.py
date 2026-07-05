"""Reward shaping basado en energia para MountainCarContinuous-v0.

Problema (ver docs/bitacora.md, Decision 3): con la recompensa original el
agente converge a "no hacer nada", porque el ambiente no penaliza el tiempo y
la exploracion casi nunca descubre la meta (+100). Falta una senal densa que
incentive el progreso, como el -1/paso de Cliff Walking.

Solucion: reward shaping *potential-based* (Ng, Harada & Russell, 1999):

    r_shaped = r + gamma * Phi(s') - Phi(s)

Con esta forma, la politica optima **no cambia** respecto al problema original
(la suma telescopica del termino de shaping solo depende de los estados inicial
y final). Se usa como potencial la **energia mecanica** del auto:

    Phi(x, v) = w_pos * altura(x) + w_vel * (v / v_max)^2
    altura(x) = sin(3x)     # forma de la colina de MountainCar

asi el agente recibe gradiente por ganar altura y velocidad, aprendiendo a
"bombear" energia (acelerar en contra para tomar impulso) antes de subir.

El wrapper preserva la recompensa original en info["original_reward"] para que
las metricas de entrenamiento/evaluacion reflejen el desempeno real y las
comparaciones sean justas.
"""

import numpy as np
import gymnasium as gym

VEL_MAX = 0.07  # |velocidad| maxima del ambiente


class EnergyShapingWrapper(gym.Wrapper):
    """Agrega reward shaping potential-based basado en energia mecanica."""

    def __init__(
        self,
        env,
        gamma: float = 0.99,
        w_pos: float = 1.0,
        w_vel: float = 1.0,
        scale: float = 1.0,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.scale = scale
        self._prev_phi = 0.0

    def potential(self, obs) -> float:
        x, v = float(obs[0]), float(obs[1])
        altura = np.sin(3.0 * x)
        cinetica = (v / VEL_MAX) ** 2
        return self.scale * (self.w_pos * altura + self.w_vel * cinetica)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self.potential(obs)
        # F = gamma * Phi(s') - Phi(s). En estado terminal Phi(s') no bootstrapea.
        shaping = (0.0 if terminated else self.gamma * phi) - self._prev_phi
        self._prev_phi = phi

        info["original_reward"] = reward
        return obs, reward + shaping, terminated, truncated, info
