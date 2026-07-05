"""
Agente Minimax con Alpha-Beta Pruning (opcional, para poder comparar el
impacto de la poda a igual profundidad).
"""
from search_agent import SearchAgent, DEFAULT_WEIGHTS


class MinimaxAgent(SearchAgent):
    def __init__(self, player=1, depth=3, weights=None, use_alpha_beta=True):
        super().__init__(player, depth, weights if weights is not None else DEFAULT_WEIGHTS)
        self.use_alpha_beta = use_alpha_beta

    def _search(self, obs):
        _, action = self._minimax(obs, self.depth, float("-inf"), float("inf"), self.player)
        return action

    def _minimax(self, board, depth, alpha, beta, current_player):
        self.nodes_expanded += 1

        terminal = self.terminal_value(board, current_player)
        if terminal is not None:
            return terminal, None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(current_player)
        rival = current_player % 2 + 1
        best_action = None

        if current_player == self.player:  # nodo MAX
            value = float("-inf")
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                child_value, _ = self._minimax(child, depth - 1, alpha, beta, rival)
                if best_action is None or child_value > value:
                    value, best_action = child_value, action
                if self.use_alpha_beta:
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            return value, best_action
        else:  # nodo MIN
            value = float("inf")
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                child_value, _ = self._minimax(child, depth - 1, alpha, beta, rival)
                if best_action is None or child_value < value:
                    value, best_action = child_value, action
                if self.use_alpha_beta:
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value, best_action
