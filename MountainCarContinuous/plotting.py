"""Gráficos para LOST a partir de los resultados guardados en results/*.json.

Genera:
- Curva de aprendizaje de una corrida (recompensa + promedio móvil, y tasa de éxito).
- Comparación de varias corridas (p.ej. Q-Learning vs Dyna-Q).

Uso:
    poetry run python plotting.py curva shaped_s5
    poetry run python plotting.py comparar plan00_qlearning plan05_dyna plan10_dyna plan20_dyna \
        --out plots/qlearning_vs_dynaq.png --titulo "Q-Learning vs Dyna-Q"

Las métricas se calculan sobre la **recompensa real** del ambiente (el shaping
solo guía el entrenamiento; ver docs/bitacora.md).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda PNG
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"


def load_result(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def moving_average(x, window: int = 100):
    """Promedio móvil; suaviza el ruido episodio a episodio."""
    x = np.asarray(x, dtype=float)
    window = min(window, len(x)) or 1
    kernel = np.ones(window) / window
    ma = np.convolve(x, kernel, mode="valid")
    # eje x alineado al final de cada ventana
    xs = np.arange(window - 1, len(x))
    return xs, ma


def plot_curva(name: str, out=None):
    r = load_result(name)
    reward = r["train"]["reward"]
    success = np.asarray(r["train"]["success"], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(reward, color="0.8", lw=0.6, label="por episodio")
    xs, ma = moving_average(reward, 100)
    ax[0].plot(xs, ma, color="C0", lw=2, label="promedio móvil (100)")
    ax[0].set_title(f"Recompensa por episodio — {name}")
    ax[0].set_xlabel("episodio")
    ax[0].set_ylabel("recompensa real")
    ax[0].legend(loc="lower right", fontsize=9)

    xs, sma = moving_average(success * 100, 100)
    ax[1].plot(xs, sma, color="C2", lw=2)
    ax[1].set_title(f"Tasa de éxito — {name}")
    ax[1].set_xlabel("episodio")
    ax[1].set_ylabel("% de éxito (ventana 100)")
    ax[1].set_ylim(-5, 105)

    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = out or (PLOTS_DIR / f"curva_{name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"guardado: {out}")
    return out


def plot_comparar(names, out, titulo, labels=None):
    labels = labels or names
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, label in zip(names, labels):
        r = load_result(name)
        reward = r["train"]["reward"]
        success = np.asarray(r["train"]["success"], dtype=float)
        xs, ma = moving_average(reward, 100)
        ax[0].plot(xs, ma, lw=2, label=label)
        xs, sma = moving_average(success * 100, 100)
        ax[1].plot(xs, sma, lw=2, label=label)

    ax[0].set_title("Recompensa (promedio móvil 100)")
    ax[0].set_xlabel("episodio")
    ax[0].set_ylabel("recompensa real")
    ax[0].legend(fontsize=9)

    ax[1].set_title("Tasa de éxito (ventana 100)")
    ax[1].set_xlabel("episodio")
    ax[1].set_ylabel("% de éxito")
    ax[1].set_ylim(-5, 105)
    ax[1].legend(fontsize=9)

    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"guardado: {out}")
    return out


def plot_semillas(names, out, titulo, labels=None, window=100):
    """Compara configs entrenadas con varias semillas: linea = media entre
    semillas, banda sombreada = +/- desvio (variabilidad por la suerte)."""
    labels = labels or names
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, label in zip(names, labels):
        with open(RESULTS_DIR / f"seeds_{name}.json", encoding="utf-8") as fh:
            r = json.load(fh)

        mean = np.asarray(r["reward_mean"])
        std = np.asarray(r["reward_std"])
        xs, m = moving_average(mean, window)
        _, sd = moving_average(std, window)
        line, = ax[0].plot(xs, m, lw=2, label=label)
        ax[0].fill_between(xs, m - sd, m + sd, alpha=0.15, color=line.get_color())

        mean = np.asarray(r["success_mean"]) * 100
        std = np.asarray(r["success_std"]) * 100
        xs, m = moving_average(mean, window)
        _, sd = moving_average(std, window)
        line, = ax[1].plot(xs, m, lw=2, label=label)
        ax[1].fill_between(xs, m - sd, m + sd, alpha=0.15, color=line.get_color())

    ax[0].set_title("Recompensa (media de semillas; banda = ±desvío)")
    ax[0].set_xlabel("episodio")
    ax[0].set_ylabel("recompensa real")
    ax[0].legend(fontsize=9)

    ax[1].set_title("Tasa de éxito (media de semillas; banda = ±desvío)")
    ax[1].set_xlabel("episodio")
    ax[1].set_ylabel("% de éxito")
    ax[1].set_ylim(-5, 105)
    ax[1].legend(fontsize=9)

    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"guardado: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Gráficos de resultados LOST.")
    sub = parser.add_subparsers(dest="modo", required=True)

    p_curva = sub.add_parser("curva", help="curva de aprendizaje de una corrida")
    p_curva.add_argument("name")

    p_cmp = sub.add_parser("comparar", help="comparar varias corridas")
    p_cmp.add_argument("names", nargs="+")
    p_cmp.add_argument("--out", default=str(PLOTS_DIR / "comparacion.png"))
    p_cmp.add_argument("--titulo", default="Comparación")
    p_cmp.add_argument("--labels", nargs="+", default=None)

    p_sem = sub.add_parser("semillas", help="comparar configs con varias semillas")
    p_sem.add_argument("names", nargs="+")
    p_sem.add_argument("--out", default=str(PLOTS_DIR / "comparacion_semillas.png"))
    p_sem.add_argument("--titulo", default="Comparación (varias semillas)")
    p_sem.add_argument("--labels", nargs="+", default=None)

    args = parser.parse_args()
    if args.modo == "curva":
        plot_curva(args.name)
    elif args.modo == "semillas":
        plot_semillas(args.names, args.out, args.titulo, args.labels)
    else:
        plot_comparar(args.names, args.out, args.titulo, args.labels)


if __name__ == "__main__":
    main()
