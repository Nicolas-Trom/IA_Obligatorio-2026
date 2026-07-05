"""
Agente Expectimax: en el nodo del rival, en lugar de minimizar, se promedia
(esperanza uniforme) sobre sus jugadas posibles. Conviene cuando el rival
no juega de forma óptima/adversarial (p. ej. RandomAgent).
"""
from search_agent import SearchAgent, DEFAULT_WEIGHTS


class ExpectimaxAgent(SearchAgent):
    def __init__(self, player=1, depth=3, weights=None):
        super().__init__(player, depth, weights if weights is not None else DEFAULT_WEIGHTS)

    def _search(self, obs):
        _, action = self._expectimax(obs, self.depth, self.player)
        return action

    def _expectimax(self, board, depth, current_player):
        self.nodes_expanded += 1

        terminal = self.terminal_value(board, current_player)
        if terminal is not None:
            return terminal, None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(current_player)
        rival = current_player % 2 + 1

        if current_player == self.player:  # nodo MAX
            value = float("-inf")
            best_action = None
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                child_value, _ = self._expectimax(child, depth - 1, rival)
                if best_action is None or child_value > value:
                    value, best_action = child_value, action
            return value, best_action
        else:  # nodo CHANCE (rival)
            total = 0.0
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                child_value, _ = self._expectimax(child, depth - 1, rival)
                total += child_value
            return total / len(actions), None
