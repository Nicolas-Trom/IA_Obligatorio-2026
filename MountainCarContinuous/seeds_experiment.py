"""Robustez con varias semillas: repite una config con distintas semillas y
agrega media +/- desvio (para mostrar que las conclusiones no fueron suerte).

Una semilla fija el azar del entrenamiento (exploracion + posicion inicial). Con
una sola corrida no sabemos si un resultado fue tipico o casualidad; repitiendo
con varias semillas y promediando, la conclusion es robusta.

Guarda, por config, `results/seeds_<name>.json` con:
- curvas por episodio promediadas entre semillas (media y desvio) de exito y reward,
- resumen de evaluacion (media +/- desvio entre semillas).

Uso:
    poetry run python seeds_experiment.py                 # 4 configs de planning, 5 semillas
    poetry run python seeds_experiment.py --seeds 0 1 2   # semillas a medida
"""

import argparse
import json
from dataclasses import replace

import numpy as np

import train
from experiment_configs import get_config, build_agent, make_env

# Comparacion Q-Learning vs Dyna-Q (misma discretizacion, shaping e hiperparametros).
DEFAULT_CONFIGS = ["plan00_qlearning", "plan05_dyna", "plan10_dyna", "plan20_dyna"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_seeds(config_name: str, seeds) -> dict:
    base = get_config(config_name)
    succ_runs, rew_runs = [], []
    eval_reward, eval_success, eval_steps = [], [], []

    print(f"\n=== {config_name} ({base.algo}) | {len(seeds)} semillas ===")
    for s in seeds:
        cfg = replace(base, seed=s)
        agent = build_agent(cfg)
        train_env = make_env(cfg)
        eval_env = make_env(cfg, shaping=False)
        tm = agent.train(train_env, episodes=cfg.episodes)
        em = agent.evaluate(eval_env, episodes=20)
        train_env.close()
        eval_env.close()

        succ_runs.append(np.asarray(tm["success"], dtype=float))
        rew_runs.append(np.asarray(tm["reward"], dtype=float))
        eval_reward.append(em["mean_reward"])
        eval_success.append(em["success_rate"])
        eval_steps.append(em["mean_steps"])
        print(f"  seed {s}: eval reward {em['mean_reward']:6.2f} | "
              f"exito {em['success_rate']:.2f} | {tm['train_time']:.0f}s")

    succ = np.stack(succ_runs)  # (n_semillas, n_episodios)
    rew = np.stack(rew_runs)
    agg = {
        "name": config_name,
        "algo": base.algo,
        "seeds": list(seeds),
        "episodes": int(base.episodes),
        "success_mean": succ.mean(0).tolist(),
        "success_std": succ.std(0).tolist(),
        "reward_mean": rew.mean(0).tolist(),
        "reward_std": rew.std(0).tolist(),
        "eval": {
            "reward_mean": float(np.mean(eval_reward)),
            "reward_std": float(np.std(eval_reward)),
            "success_mean": float(np.mean(eval_success)),
            "success_std": float(np.std(eval_success)),
            "steps_mean": float(np.mean(eval_steps)),
            "steps_std": float(np.std(eval_steps)),
        },
    }

    train.RESULTS_DIR.mkdir(exist_ok=True)
    out = train.RESULTS_DIR / f"seeds_{config_name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, ensure_ascii=False)
    print(f"  -> guardado {out.name} | "
          f"eval reward {agg['eval']['reward_mean']:.2f} +/- {agg['eval']['reward_std']:.2f} | "
          f"exito {agg['eval']['success_mean']:.2f} +/- {agg['eval']['success_std']:.2f}")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Experimento de robustez con semillas.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    for name in args.configs:
        run_seeds(name, args.seeds)
    print("\nlisto: experimento de semillas completo.")


if __name__ == "__main__":
    main()
