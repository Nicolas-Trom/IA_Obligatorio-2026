"""
Barrido de configuraciones de heurísticas (Tarea 2): compara distintas
combinaciones de pesos enfrentando cada configuración contra RandomAgent
y Stratagem, para analizar el impacto de cada heurística y sus pesos.
"""
import csv
import time

from experiments import run_tournament, win_rate, save_csv
from minimax_agent import MinimaxAgent
from random_agent import RandomAgent
from stratagem import Stratagem

DEPTH = 3
N_GAMES = 30

CONFIGS = {
    "balanceado":     {"relative_mobility": 1.0, "center_control": 0.25, "corner_rival": 0.5},
    "agresivo":       {"relative_mobility": 0.5, "center_control": 0.0,  "corner_rival": 2.0},
    "posicional":     {"relative_mobility": 0.5, "center_control": 2.0,  "corner_rival": 0.0},
    "movilidad_pura": {"relative_mobility": 2.0, "center_control": 0.0,  "corner_rival": 0.0},
    "mixto":          {"relative_mobility": 1.0, "center_control": 1.0,  "corner_rival": 1.0},
}

if __name__ == "__main__":
    all_rows = []
    summary = []

    for config_name, weights in CONFIGS.items():
        factory = lambda p, w=weights: MinimaxAgent(p, depth=DEPTH, use_alpha_beta=True, weights=w)

        # vs Random
        t0 = time.time()
        rows_rand = run_tournament(
            factory, lambda p: RandomAgent(p),
            n_games=N_GAMES, seed=42,
            label_a=config_name, label_b="RandomAgent"
        )
        wr_rand = win_rate(rows_rand, config_name)
        print(f"{config_name} vs Random:   {wr_rand:.2%} ({time.time()-t0:.1f}s)")

        # vs Stratagem
        t0 = time.time()
        rows_strat = run_tournament(
            factory, lambda p: Stratagem(p),
            n_games=N_GAMES, seed=42,
            label_a=config_name, label_b="Stratagem"
        )
        wr_strat = win_rate(rows_strat, config_name)
        print(f"{config_name} vs Stratagem: {wr_strat:.2%} ({time.time()-t0:.1f}s)")

        all_rows.extend(rows_rand)
        all_rows.extend(rows_strat)
        summary.append({
            "config": config_name,
            "weights": str(weights),
            "wr_vs_random": wr_rand,
            "wr_vs_stratagem": wr_strat,
        })

    save_csv(all_rows, "heuristic_sweep_results.csv")

    with open("heuristic_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "weights", "wr_vs_random", "wr_vs_stratagem"])
        writer.writeheader()
        writer.writerows(summary)

    print("\n=== Resumen ===")
    for row in summary:
        print(f"{row['config']:20s}  vs Random: {row['wr_vs_random']:.2%}  vs Stratagem: {row['wr_vs_stratagem']:.2%}")

    print("\nGuardado en heuristic_sweep_results.csv y heuristic_summary.csv")
