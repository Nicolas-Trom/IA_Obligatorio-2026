"""Discretizacion de observaciones y acciones para MountainCarContinuous-v0.

El ambiente es continuo:
- Observacion: [posicion, velocidad] con posicion en [-1.2, 0.6] y
  velocidad en [-0.07, 0.07].
- Accion: fuerza continua en [-1.0, 1.0].

Q-Learning tabular necesita estados y acciones discretos, asi que
aca convertimos ambos espacios a indices enteros.
"""

import numpy as np

# Rangos por defecto del ambiente MountainCarContinuous-v0 (verificados
# contra env.observation_space / env.action_space).
POS_LOW, POS_HIGH = -1.2, 0.6
VEL_LOW, VEL_HIGH = -0.07, 0.07
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


class Discretizer:
    """Convierte observaciones continuas en indices y acciones discretas
    en valores continuos.

    La tabla Q asociada tiene forma ``(pos_bins, vel_bins, action_bins)``.
    """

    def __init__(
        self,
        pos_bins: int = 20,
        vel_bins: int = 20,
        action_bins: int = 5,
        pos_low: float = POS_LOW,
        pos_high: float = POS_HIGH,
        vel_low: float = VEL_LOW,
        vel_high: float = VEL_HIGH,
        action_low: float = ACTION_LOW,
        action_high: float = ACTION_HIGH,
    ):
        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.action_bins = action_bins

        # bins + 1 cortes -> exactamente `bins` intervalos.
        self.pos_edges = np.linspace(pos_low, pos_high, pos_bins + 1)
        self.vel_edges = np.linspace(vel_low, vel_high, vel_bins + 1)

        # Valores continuos de accion. Con action_bins impar se incluye el 0,
        # cubriendo empuje negativo, nulo y positivo (p.ej. 3 -> [-1, 0, 1]).
        self.actions = np.linspace(action_low, action_high, action_bins).astype(
            np.float32
        )

    @property
    def q_shape(self) -> tuple:
        """Forma de la tabla Q correspondiente a esta discretizacion."""
        return (self.pos_bins, self.vel_bins, self.action_bins)

    def state(self, obs) -> tuple:
        """Observacion continua -> (pos_idx, vel_idx).

        Se usa ``np.digitize`` sobre los cortes internos y se hace ``clip``
        para que valores en el borde no se salgan del rango de indices.
        """
        pos, vel = float(obs[0]), float(obs[1])
        pos_idx = np.clip(np.digitize(pos, self.pos_edges) - 1, 0, self.pos_bins - 1)
        vel_idx = np.clip(np.digitize(vel, self.vel_edges) - 1, 0, self.vel_bins - 1)
        return int(pos_idx), int(vel_idx)

    def action_value(self, action_idx: int) -> np.ndarray:
        """Indice de accion discreta -> array que espera ``env.step``."""
        return np.array([self.actions[action_idx]], dtype=np.float32)

    def config(self) -> dict:
        """Parametros para reconstruir este discretizador (para guardar el .pkl)."""
        return {
            "pos_bins": self.pos_bins,
            "vel_bins": self.vel_bins,
            "action_bins": self.action_bins,
            "pos_low": float(self.pos_edges[0]),
            "pos_high": float(self.pos_edges[-1]),
            "vel_low": float(self.vel_edges[0]),
            "vel_high": float(self.vel_edges[-1]),
            "action_low": float(self.actions[0]),
            "action_high": float(self.actions[-1]),
        }
