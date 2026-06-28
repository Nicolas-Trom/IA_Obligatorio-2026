"""
Barrido de profundidad: mide nodos expandidos y tiempo por jugada para
Minimax (con/sin Alpha-Beta) y Expectimax, a distintas profundidades,
sobre el mismo conjunto de tableros iniciales (misma semilla) para que la
comparación sea justa.
"""
import csv
import random

from board import Board
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent

DEPTHS = [1, 2, 3, 4]
N_BOARDS = 10


def make_boards(n, seed):
    random.seed(seed)
    return [Board() for _ in range(n)]


def measure(agent_factory, boards):
    nodes, times = [], []
    for board in boards:
        agent = agent_factory()
        agent.next_action(board.clone())
        nodes.append(agent.nodes_expanded)
        times.append(agent.last_move_time)
    return sum(nodes) / len(nodes), sum(times) / len(times)


if __name__ == "__main__":
    boards = make_boards(N_BOARDS, seed=7)
    rows = []

    for depth in DEPTHS:
        avg_nodes, avg_time = measure(lambda: MinimaxAgent(1, depth=depth, use_alpha_beta=True), boards)
        rows.append({"agent": "Minimax_AB", "depth": depth, "avg_nodes": avg_nodes, "avg_time": avg_time})
        print(f"Minimax AB depth={depth}: nodes={avg_nodes:.0f} time={avg_time:.3f}s")

        avg_nodes, avg_time = measure(lambda: MinimaxAgent(1, depth=depth, use_alpha_beta=False), boards)
        rows.append({"agent": "Minimax_NoAB", "depth": depth, "avg_nodes": avg_nodes, "avg_time": avg_time})
        print(f"Minimax NoAB depth={depth}: nodes={avg_nodes:.0f} time={avg_time:.3f}s")

        avg_nodes, avg_time = measure(lambda: ExpectimaxAgent(1, depth=depth), boards)
        rows.append({"agent": "Expectimax", "depth": depth, "avg_nodes": avg_nodes, "avg_time": avg_time})
        print(f"Expectimax depth={depth}: nodes={avg_nodes:.0f} time={avg_time:.3f}s")

    with open("depth_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "depth", "avg_nodes", "avg_time"])
        writer.writeheader()
        writer.writerows(rows)
    print("Guardado en depth_sweep_results.csv")
