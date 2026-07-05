"""
Clase base para agentes de búsqueda en árbol (Minimax / Expectimax) sobre Isolation.

Provee:
- Combinación de heurísticas ponderadas (ver `heuristics.py`).
- Instrumentación de nodos expandidos y tiempo por jugada, para poder
  analizar el impacto de Alpha-Beta Pruning y comparar configuraciones.
"""
import time

from agent import Agent
from board import Board
from heuristics import weighted_heuristic

DEFAULT_WEIGHTS = {
    "relative_mobility": 1.0,
    "center_control": 0.25,
    "opponent_distance": 0.0,
    "corner_rival": 0.5,
}


class SearchAgent(Agent):
    def __init__(self, player=1, depth=3, weights=None):
        super().__init__(player)
        self.depth = depth
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        self.nodes_expanded = 0
        self.last_move_time = 0.0

    def heuristic_utility(self, board: Board):
        return weighted_heuristic(board, self.player, self.weights)

    def next_action(self, obs):
        self.nodes_expanded = 0
        start = time.perf_counter()
        action = self._search(obs)
        self.last_move_time = time.perf_counter() - start
        return action

    def _search(self, obs):
        raise NotImplementedError

    def terminal_value(self, board: Board, current_player: int):
        """Valor terminal desde la perspectiva de self.player, o None si no terminó."""
        done, winner = board.is_end(current_player)
        if not done:
            return None
        if winner == self.player:
            return float("inf")
        return float("-inf")
