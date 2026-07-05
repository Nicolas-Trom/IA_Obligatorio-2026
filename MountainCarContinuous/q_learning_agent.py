"""Agente de Q-Learning tabular para MountainCarContinuous-v0.

Diseno:
- Todos los hiperparametros viven en el constructor, asi cada experimento
  es "un agente con su configuracion". Esto facilita el barrido de
  hiperparametros y que Dyna-Q herede sin duplicar logica.
- ``train`` devuelve las metricas por episodio necesarias para los graficos.
- ``evaluate`` corre la politica greedy (sin exploracion).
- ``save`` / ``load`` guardan solo la tabla Q y la configuracion (no el env),
  para cumplir con el entregable obligatorio de al menos un modelo .pkl.

Regla de actualizacion:
    Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
    target = r                               si el estado es terminal (meta)
    target = r + gamma * max_a' Q(s',a')     en otro caso

Detalle importante de Gymnasium: un episodio termina con ``terminated`` o
``truncated``. Solo ``terminated`` (llegar a la meta) es un estado terminal
real; ``truncated`` (limite de pasos) NO lo es, por lo que en ese caso hay
que seguir haciendo bootstrap con ``gamma * max Q(s')``.
"""

import pickle
import time

import numpy as np

from discretization import Discretizer


class QLearningAgent:
    algo_name = "q_learning"

    def __init__(
        self,
        pos_bins: int = 20,
        vel_bins: int = 20,
        action_bins: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        optimistic_init: float = 0.0,
        seed: int = 0,
    ):
        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.action_bins = action_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.seed = seed

        self.disc = Discretizer(pos_bins, vel_bins, action_bins)
        self.Q = np.full(self.disc.q_shape, optimistic_init, dtype=np.float64)

        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon_start
        self.metrics = None  # se completa al entrenar

    # ------------------------------------------------------------------ #
    # Politica
    # ------------------------------------------------------------------ #
    def best_action(self, state) -> int:
        """argmax de Q en el estado, desempatando aleatoriamente."""
        q = self.Q[state[0], state[1]]
        best = np.flatnonzero(q == q.max())
        if len(best) == 1:
            return int(best[0])
        return int(self.rng.choice(best))

    def act(self, state, greedy: bool = False) -> int:
        """Politica epsilon-greedy (greedy=True fuerza explotacion)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_bins))
        return self.best_action(state)

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Actualizacion de Q
    # ------------------------------------------------------------------ #
    def _update(self, state, action, reward, next_state, terminal):
        best_next = 0.0 if terminal else self.Q[next_state[0], next_state[1]].max()
        target = reward + self.gamma * best_next
        idx = (state[0], state[1], action)
        self.Q[idx] += self.alpha * (target - self.Q[idx])

    def _after_real_step(self, state, action, reward, next_state, terminal):
        """Hook para planificacion (no-op en Q-Learning; usado por Dyna-Q)."""
        pass

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def train(self, env, episodes: int, log_every: int = 0) -> dict:
        """Entrena por ``episodes`` episodios y devuelve metricas por episodio."""
        metrics = {
            "reward": np.empty(episodes),
            "steps": np.empty(episodes, dtype=int),
            "success": np.empty(episodes, dtype=bool),
            "epsilon": np.empty(episodes),
        }

        # Sembrar el RNG del ambiente una sola vez para reproducibilidad;
        # los reset() posteriores avanzan ese mismo stream.
        env.reset(seed=self.seed)
        start = time.perf_counter()

        for ep in range(episodes):
            obs, _ = env.reset()
            state = self.disc.state(obs)
            total_reward, steps, success, done = 0.0, 0, False, False

            while not done:
                action = self.act(state)
                obs, reward, terminated, truncated, info = env.step(
                    self.disc.action_value(action)
                )
                next_state = self.disc.state(obs)
                done = terminated or truncated

                # Se aprende con `reward` (con shaping si el env esta envuelto),
                # pero la metrica usa el reward original del ambiente para que las
                # comparaciones y la tasa de exito reflejen el desempeno real.
                self._update(state, action, reward, next_state, terminated)
                self._after_real_step(state, action, reward, next_state, terminated)

                state = next_state
                total_reward += info.get("original_reward", reward)
                steps += 1
                if terminated:
                    success = True

            self._decay_epsilon()
            metrics["reward"][ep] = total_reward
            metrics["steps"][ep] = steps
            metrics["success"][ep] = success
            metrics["epsilon"][ep] = self.epsilon

            if log_every and (ep + 1) % log_every == 0:
                window = slice(max(0, ep - 99), ep + 1)
                print(
                    f"[{self.algo_name}] ep {ep + 1}/{episodes} "
                    f"| reward_avg100 {metrics['reward'][window].mean():8.2f} "
                    f"| exito100 {metrics['success'][window].mean():.2f} "
                    f"| eps {self.epsilon:.3f}"
                )

        metrics["train_time"] = time.perf_counter() - start
        metrics["episodes"] = episodes
        self.metrics = metrics
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluacion (sin exploracion)
    # ------------------------------------------------------------------ #
    def evaluate(self, env, episodes: int = 20, seed: int = None) -> dict:
        """Corre la politica greedy y resume el desempeno."""
        eval_seed = self.seed + 10_000 if seed is None else seed
        env.reset(seed=eval_seed)

        rewards = np.empty(episodes)
        steps_arr = np.empty(episodes, dtype=int)
        successes = np.empty(episodes, dtype=bool)

        for ep in range(episodes):
            obs, _ = env.reset()
            state = self.disc.state(obs)
            total_reward, steps, success, done = 0.0, 0, False, False

            while not done:
                action = self.act(state, greedy=True)
                obs, reward, terminated, truncated, _ = env.step(
                    self.disc.action_value(action)
                )
                state = self.disc.state(obs)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                if terminated:
                    success = True

            rewards[ep] = total_reward
            steps_arr[ep] = steps
            successes[ep] = success

        return {
            "episodes": episodes,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
            "success_rate": float(successes.mean()),
            "mean_steps": float(steps_arr.mean()),
            "rewards": rewards.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def config(self) -> dict:
        """Kwargs del constructor (para reconstruir el agente al cargar)."""
        return {
            "pos_bins": self.pos_bins,
            "vel_bins": self.vel_bins,
            "action_bins": self.action_bins,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "optimistic_init": self.optimistic_init,
            "seed": self.seed,
        }

    def save(self, path):
        """Guarda tabla Q + configuracion (no el env) con pickle."""
        payload = {
            "algo": self.algo_name,
            "config": self.config(),
            "Q": self.Q,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        agent = cls(**payload["config"])
        agent.Q = payload["Q"]
        agent.epsilon = agent.epsilon_min  # cargado -> modo explotacion
        return agent
