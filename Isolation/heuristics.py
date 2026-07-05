"""
Catálogo de funciones de evaluación (heurísticas) para Isolation.

Cada heurística es una función `h(board, player) -> float` que evalúa
el estado `board` desde el punto de vista de `player` (mayor es mejor
para `player`). No dependen de ningún agente concreto para poder
combinarse y testearse de forma aislada.
"""
from board import Board


def opponent_of(player: int) -> int:
    return player % 2 + 1


def own_mobility(board: Board, player: int) -> float:
    """Cantidad de movimientos legales propios."""
    return float(len(board.get_possible_actions(player)))


def relative_mobility(board: Board, player: int) -> float:
    """Diferencia de movilidad: movimientos propios - movimientos del rival."""
    rival = opponent_of(player)
    own = len(board.get_possible_actions(player))
    opp = len(board.get_possible_actions(rival))
    return float(own - opp)


def center_control(board: Board, player: int) -> float:
    """Cercanía al centro del tablero (más cerca = mejor)."""
    row, col = board.find_player_position(player)
    center_row = (board.board_size[0] - 1) / 2
    center_col = (board.board_size[1] - 1) / 2
    distance = abs(row - center_row) + abs(col - center_col)
    return -distance


def opponent_distance(board: Board, player: int, away: bool = True) -> float:
    """Distancia Manhattan al rival. `away=True`: alejarse es mejor."""
    rival = opponent_of(player)
    own_pos = board.find_player_position(player)
    rival_pos = board.find_player_position(rival)
    distance = abs(own_pos[0] - rival_pos[0]) + abs(own_pos[1] - rival_pos[1])
    return float(distance if away else -distance)


def corner_rival(board: Board, player: int) -> float:
    """Cantidad de celdas eliminadas/bloqueadas alrededor del rival (acorralamiento)."""
    rival = opponent_of(player)
    row, col = board.find_player_position(rival)
    blocked = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < board.board_size[0] and 0 <= nc < board.board_size[1]:
                if board.grid[nr, nc] != 0:
                    blocked += 1
            else:
                blocked += 1
    return float(blocked)


HEURISTICS = {
    "own_mobility": own_mobility,
    "relative_mobility": relative_mobility,
    "center_control": center_control,
    "opponent_distance": opponent_distance,
    "corner_rival": corner_rival,
}


def weighted_heuristic(board: Board, player: int, weights: dict) -> float:
    """
    Combina heurísticas del catálogo según `weights`, p. ej.:
        {"relative_mobility": 1.0, "center_control": 0.5, "corner_rival": 0.3}
    """
    total = 0.0
    for name, weight in weights.items():
        if weight == 0:
            continue
        total += weight * HEURISTICS[name](board, player)
    return total
