"""Configuraciones de experimentos para LOST.

Centraliza los hiperparametros de cada corrida para que ``train.py`` /
``evaluate.py`` y el notebook trabajen con las mismas definiciones y sea
facil documentar en el informe "que se probo y por que".

Las variantes siguen el plan:
- Discretizacion: chico -> intermedio -> fino -> muy fino.
- Hiperparametros: grilla chica alrededor de la discretizacion intermedia.
- Planning steps: Q-Learning puro (0) vs Dyna-Q (5, 10, 20).
"""

from dataclasses import dataclass

import gymnasium as gym

from q_learning_agent import QLearningAgent
from dyna_q_agent import DynaQAgent
from reward_shaping import EnergyShapingWrapper

ENV_ID = "MountainCarContinuous-v0"


@dataclass
class Config:
    """Descripcion completa y reproducible de un experimento."""

    name: str
    algo: str = "q_learning"  # "q_learning" | "dyna_q"
    episodes: int = 5000
    # discretizacion
    pos_bins: int = 20
    vel_bins: int = 20
    action_bins: int = 5
    # hiperparametros de aprendizaje
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.999
    optimistic_init: float = 0.0
    # solo Dyna-Q
    planning_steps: int = 10
    # reward shaping (opcional, potential-based basado en energia)
    reward_shaping: bool = False
    shaping_scale: float = 1.0
    shaping_w_pos: float = 1.0
    shaping_w_vel: float = 1.0
    seed: int = 0

    def agent_kwargs(self) -> dict:
        keys = [
            "pos_bins", "vel_bins", "action_bins", "alpha", "gamma",
            "epsilon_start", "epsilon_min", "epsilon_decay",
            "optimistic_init", "seed",
        ]
        kwargs = {k: getattr(self, k) for k in keys}
        if self.algo == "dyna_q":
            kwargs["planning_steps"] = self.planning_steps
        return kwargs


def build_agent(config: Config):
    """Instancia el agente correspondiente a la configuracion."""
    kwargs = config.agent_kwargs()
    if config.algo == "dyna_q":
        return DynaQAgent(**kwargs)
    if config.algo == "q_learning":
        return QLearningAgent(**kwargs)
    raise ValueError(f"algo desconocido: {config.algo!r}")


def make_env(config: Config, render: bool = False, shaping: bool = None):
    """Crea el ambiente, opcionalmente con reward shaping.

    ``shaping=None`` usa la config; ``shaping=False`` fuerza el env sin shaping
    (util para evaluar siempre sobre la recompensa original).
    """
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    use_shaping = config.reward_shaping if shaping is None else shaping
    if use_shaping:
        env = EnergyShapingWrapper(
            env,
            gamma=config.gamma,
            w_pos=config.shaping_w_pos,
            w_vel=config.shaping_w_vel,
            scale=config.shaping_scale,
        )
    return env


# --------------------------------------------------------------------- #
# Corrida corta para validar el flujo y dejar el primer .pkl entregable.
# --------------------------------------------------------------------- #
SMOKE = Config(
    name="smoke",
    algo="q_learning",
    episodes=10,
    pos_bins=12, vel_bins=12, action_bins=3,
    epsilon_decay=0.99,
)

# Baseline SIN shaping: documenta el fracaso (converge a "no hacer nada").
BASELINE_SIN_SHAPING = Config(
    "baseline_sin_shaping",
    pos_bins=20, vel_bins=20, action_bins=5, episodes=3000, reward_shaping=False,
)

# --------------------------------------------------------------------- #
# Variantes de discretizacion. Todo igual (shaping incluido) salvo la grilla,
# para aislar el impacto de la resolucion.
# --------------------------------------------------------------------- #
_DISC_COMMON = dict(episodes=3000, reward_shaping=True, shaping_scale=5.0)
DISCRETIZATION_VARIANTS = [
    Config("disc_chico",      pos_bins=12, vel_bins=12, action_bins=3, **_DISC_COMMON),
    Config("disc_intermedio", pos_bins=20, vel_bins=20, action_bins=5, **_DISC_COMMON),
    Config("disc_fino",       pos_bins=30, vel_bins=30, action_bins=7, **_DISC_COMMON),
    Config("disc_muy_fino",   pos_bins=40, vel_bins=40, action_bins=9, **_DISC_COMMON),
]

# --------------------------------------------------------------------- #
# Grilla chica de hiperparametros sobre la discretizacion intermedia.
# --------------------------------------------------------------------- #
# Grilla chica: se parte de la base (alpha=0.1, gamma=0.99, decay=0.999) y se
# cambia una perilla por vez. Todas con misma grilla y shaping para aislar el efecto.
_HP_COMMON = dict(episodes=2000, reward_shaping=True, shaping_scale=5.0)
HYPERPARAM_GRID = [
    Config("hp_a05_g99", alpha=0.05, gamma=0.99, epsilon_decay=0.999, **_HP_COMMON),
    Config("hp_a10_g99", alpha=0.10, gamma=0.99, epsilon_decay=0.999, **_HP_COMMON),
    Config("hp_a20_g99", alpha=0.20, gamma=0.99, epsilon_decay=0.999, **_HP_COMMON),
    Config("hp_a10_g95", alpha=0.10, gamma=0.95, epsilon_decay=0.999, **_HP_COMMON),
    Config("hp_a10_g99_slow", alpha=0.10, gamma=0.99, epsilon_decay=0.9995, **_HP_COMMON),
]

# --------------------------------------------------------------------- #
# Q-Learning vs Dyna-Q (misma discretizacion e hiperparametros base).
# --------------------------------------------------------------------- #
# Todas con la misma discretizacion, hiperparametros y shaping: solo cambia
# planning_steps, para aislar el efecto de Dyna-Q sobre la velocidad de aprendizaje.
_PLAN_COMMON = dict(episodes=2000, reward_shaping=True, shaping_scale=5.0)
PLANNING_VARIANTS = [
    Config("plan00_qlearning", algo="q_learning", **_PLAN_COMMON),
    Config("plan05_dyna", algo="dyna_q", planning_steps=5, **_PLAN_COMMON),
    Config("plan10_dyna", algo="dyna_q", planning_steps=10, **_PLAN_COMMON),
    Config("plan20_dyna", algo="dyna_q", planning_steps=20, **_PLAN_COMMON),
]


# --------------------------------------------------------------------- #
# Reward shaping por energia (discretizacion intermedia). Varias escalas
# para elegir cuanto pesa la senal densa frente al reward original.
# --------------------------------------------------------------------- #
SHAPING_VARIANTS = [
    Config("shaped_s1",  reward_shaping=True, shaping_scale=1.0),
    Config("shaped_s5",  reward_shaping=True, shaping_scale=5.0),
    Config("shaped_s10", reward_shaping=True, shaping_scale=10.0),
]


def _registry() -> dict:
    """Mapa nombre -> Config para seleccionar por linea de comandos."""
    reg = {SMOKE.name: SMOKE, BASELINE_SIN_SHAPING.name: BASELINE_SIN_SHAPING}
    for group in (DISCRETIZATION_VARIANTS, HYPERPARAM_GRID,
                  PLANNING_VARIANTS, SHAPING_VARIANTS):
        for cfg in group:
            reg[cfg.name] = cfg
    return reg


CONFIGS = _registry()


def get_config(name: str) -> Config:
    if name not in CONFIGS:
        disponibles = ", ".join(sorted(CONFIGS))
        raise KeyError(f"config {name!r} no existe. Disponibles: {disponibles}")
    return CONFIGS[name]
