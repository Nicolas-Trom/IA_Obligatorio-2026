"""
Harness de experimentación para Tarea 3 (MATE): enfrenta agentes entre sí,
registrando win-rate, tiempo medio por jugada y nodos expandidos.
"""
import csv
import random
import time

from isolation_env import IsolationEnv


def play_match(agent1_factory, agent2_factory, render=False):
    """
    Juega una partida completa. Los `factory` son callables que reciben
    `player` (1 o 2) y devuelven una instancia de agente nueva (necesario
    porque los agentes acumulan estado de instrumentación por jugada).
    Devuelve un dict con el resultado y métricas agregadas.
    """
    env = IsolationEnv()
    obs = env.reset()
    agent1 = agent1_factory(1)
    agent2 = agent2_factory(2)
    agents = {1: agent1, 2: agent2}

    done = False
    winner = 0
    move_times = {1: [], 2: []}
    nodes_expanded = {1: 0, 2: 0}
    current = 1

    while not done:
        if render:
            env.render()
        agent = agents[current]
        action = agent.next_action(obs)
        if hasattr(agent, "last_move_time"):
            move_times[current].append(agent.last_move_time)
        if hasattr(agent, "nodes_expanded"):
            nodes_expanded[current] += agent.nodes_expanded
        obs, _, done, winner, _ = env.step(action)
        current = current % 2 + 1

    if render:
        env.render()

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "winner": winner,
        "avg_time_p1": avg(move_times[1]),
        "avg_time_p2": avg(move_times[2]),
        "nodes_p1": nodes_expanded[1],
        "nodes_p2": nodes_expanded[2],
    }


def run_tournament(agent_a_factory, agent_b_factory, n_games=50, seed=None, label_a="A", label_b="B"):
    """
    Enfrenta `agent_a_factory` vs `agent_b_factory` durante `n_games`,
    alternando quién juega como jugador 1 / jugador 2 (mitad y mitad).
    Devuelve una lista de filas de resultados (una por partida).
    """
    if seed is not None:
        random.seed(seed)

    rows = []
    for i in range(n_games):
        a_starts = i % 2 == 0  # alterna quién es player 1
        if a_starts:
            factory1, factory2 = agent_a_factory, agent_b_factory
        else:
            factory1, factory2 = agent_b_factory, agent_a_factory

        result = play_match(factory1, factory2)
        winner_label = (
            label_a if (a_starts and result["winner"] == 1) or (not a_starts and result["winner"] == 2)
            else label_b
        )

        rows.append({
            "game": i,
            "agent_a": label_a,
            "agent_b": label_b,
            "a_starts": a_starts,
            "winner": winner_label,
            "avg_time_p1": result["avg_time_p1"],
            "avg_time_p2": result["avg_time_p2"],
            "nodes_p1": result["nodes_p1"],
            "nodes_p2": result["nodes_p2"],
        })
    return rows


def win_rate(rows, label):
    wins = sum(1 for r in rows if r["winner"] == label)
    return wins / len(rows) if rows else 0.0


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    from random_agent import RandomAgent
    from stratagem import Stratagem
    from minimax_agent import MinimaxAgent
    from expectimax_agent import ExpectimaxAgent

    configs = [
        ("Minimax_AB", lambda p: MinimaxAgent(p, depth=3, use_alpha_beta=True), "RandomAgent", lambda p: RandomAgent(p)),
        ("Minimax_AB", lambda p: MinimaxAgent(p, depth=3, use_alpha_beta=True), "Stratagem", lambda p: Stratagem(p)),
        ("Expectimax", lambda p: ExpectimaxAgent(p, depth=3), "RandomAgent", lambda p: RandomAgent(p)),
        ("Expectimax", lambda p: ExpectimaxAgent(p, depth=3), "Stratagem", lambda p: Stratagem(p)),
        ("Minimax_AB", lambda p: MinimaxAgent(p, depth=3, use_alpha_beta=True), "Minimax_NoAB",
         lambda p: MinimaxAgent(p, depth=3, use_alpha_beta=False)),
    ]

    all_rows = []
    for label_a, factory_a, label_b, factory_b in configs:
        t0 = time.time()
        rows = run_tournament(factory_a, factory_b, n_games=50, seed=42, label_a=label_a, label_b=label_b)
        elapsed = time.time() - t0
        rate_a = win_rate(rows, label_a)
        print(f"{label_a} vs {label_b}: win-rate {label_a}={rate_a:.2%} ({elapsed:.1f}s total)")
        all_rows.extend(rows)

    save_csv(all_rows, "tournament_results.csv")
    print("Resultados guardados en tournament_results.csv")
