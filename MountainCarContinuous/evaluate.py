"""Evaluacion de un modelo LOST guardado (politica greedy, sin exploracion).

Uso:
    poetry run python evaluate.py models/smoke.pkl
    poetry run python evaluate.py models/plan10_dyna.pkl --episodes 50 --render
"""

import argparse
import pickle
from pathlib import Path

import gymnasium as gym

from q_learning_agent import QLearningAgent
from dyna_q_agent import DynaQAgent

ENV_ID = "MountainCarContinuous-v0"

# Mapea el nombre de algoritmo guardado en el .pkl a su clase.
_AGENT_CLASSES = {
    QLearningAgent.algo_name: QLearningAgent,
    DynaQAgent.algo_name: DynaQAgent,
}


def load_agent(model_path):
    """Carga un agente eligiendo la clase segun el campo 'algo' del .pkl."""
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    agent_cls = _AGENT_CLASSES.get(payload.get("algo"), QLearningAgent)
    return agent_cls.load(model_path)


def evaluate_model(model_path, episodes: int = 50, render: bool = False) -> dict:
    agent = load_agent(model_path)
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    metrics = agent.evaluate(env, episodes=episodes)
    env.close()

    print(f"== evaluacion de {Path(model_path).name} ({episodes} episodios) ==")
    print(f"reward medio : {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"reward min/max: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    print(f"tasa de exito: {metrics['success_rate']:.2f}")
    print(f"pasos medios : {metrics['mean_steps']:.0f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluar un modelo LOST guardado.")
    parser.add_argument("model_path", help="ruta al .pkl del modelo")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate_model(args.model_path, episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
