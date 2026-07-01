"""Entrenamiento de un experimento LOST de punta a punta.

Uso:
    poetry run python train.py smoke              # corrida corta -> primer .pkl
    poetry run python train.py disc_intermedio --episodes 5000
    poetry run python train.py plan10_dyna --seed 1

Guarda:
    models/<name>.pkl     tabla Q + configuracion (modelo entregable)
    results/<name>.json   configuracion + metricas de entrenamiento y evaluacion
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiment_configs import get_config, build_agent, make_env, Config

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def _jsonable(metrics: dict) -> dict:
    """Convierte arrays de numpy a listas para poder serializar a JSON."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def run_experiment(
    config: Config,
    episodes: int = None,
    eval_episodes: int = 20,
    save: bool = True,
    log_every: int = None,
    render: bool = False,
) -> dict:
    """Entrena y evalua una configuracion; opcionalmente guarda modelo y metricas."""
    episodes = episodes or config.episodes
    if log_every is None:
        log_every = max(1, episodes // 10)

    # Entrenamiento con shaping si la config lo pide; evaluacion SIEMPRE sobre
    # la recompensa original para medir el desempeno real.
    train_env = make_env(config, render=render)
    eval_env = make_env(config, render=render, shaping=False)

    agent = build_agent(config)
    shaping_txt = (
        f"shaping x{config.shaping_scale}" if config.reward_shaping else "reward original"
    )
    print(
        f"== {config.name} ({config.algo}) | "
        f"disc {config.pos_bins}x{config.vel_bins}x{config.action_bins} | "
        f"alpha {config.alpha} gamma {config.gamma} | {shaping_txt} | "
        f"episodios {episodes} =="
    )

    train_metrics = agent.train(train_env, episodes=episodes, log_every=log_every)
    eval_metrics = agent.evaluate(eval_env, episodes=eval_episodes)
    train_env.close()
    eval_env.close()

    print(
        f"-- entrenamiento: {train_metrics['train_time']:.1f}s | "
        f"exito(ult.100) {train_metrics['success'][-100:].mean():.2f}"
    )
    print(
        f"-- evaluacion greedy ({eval_episodes} ep): "
        f"reward {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f} | "
        f"exito {eval_metrics['success_rate']:.2f} | "
        f"pasos {eval_metrics['mean_steps']:.0f}"
    )

    summary = {
        "name": config.name,
        "algo": config.algo,
        "config": config.__dict__,
        "episodes": episodes,
        "train": _jsonable(train_metrics),
        "eval": eval_metrics,
    }

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{config.name}.pkl"
        results_path = RESULTS_DIR / f"{config.name}.json"
        agent.save(model_path)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"-- guardado: {os.path.relpath(model_path, BASE_DIR)} , "
              f"{os.path.relpath(results_path, BASE_DIR)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Entrenar un experimento LOST.")
    parser.add_argument("config", nargs="?", default="smoke",
                        help="nombre de la configuracion (ver experiment_configs.py)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override de la cantidad de episodios")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="override de la semilla")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="mostrar el simulador (mas lento)")
    args = parser.parse_args()

    config = get_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    run_experiment(
        config,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        save=not args.no_save,
        render=args.render,
    )


if __name__ == "__main__":
    main()
