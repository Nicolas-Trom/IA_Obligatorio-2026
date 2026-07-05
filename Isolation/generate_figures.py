"""
Genera las figuras del ANEXO del informe MATE a partir de los CSVs de
resultados. Salida: PNGs en la carpeta figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figures", exist_ok=True)

# ── Figura 1: win-rate del torneo principal ────────────────────────────────
df = pd.read_csv("tournament_results.csv")
win_rates = (
    df.assign(win_a=lambda d: d["winner"] == d["agent_a"])
      .groupby(["agent_a", "agent_b"])["win_a"]
      .mean()
      .reset_index()
)
labels = win_rates["agent_a"] + "\nvs " + win_rates["agent_b"]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, win_rates["win_a"] * 100, color="#2E4057")
plt.axhline(50, color="#8B0000", linestyle="--", linewidth=1, label="50% (paridad)")
for bar, value in zip(bars, win_rates["win_a"] * 100):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
             ha="center", fontsize=9)
plt.ylabel("Win-rate del agente A (%)")
plt.ylim(0, 110)
plt.title("Torneo principal: win-rate por enfrentamiento (50 partidas c/u)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig1_winrate_torneo.png", dpi=150)
plt.close()

# ── Figura 2: nodos expandidos vs profundidad (escala log) ─────────────────
depth_df = pd.read_csv("depth_sweep_results.csv")

plt.figure(figsize=(8, 4.5))
styles = {"Minimax_AB": ("o-", "#2E4057"), "Minimax_NoAB": ("s--", "#8B0000"),
          "Expectimax": ("^:", "#B8860B")}
for agent_name, group in depth_df.groupby("agent"):
    marker, color = styles.get(agent_name, ("o-", None))
    plt.plot(group["depth"], group["avg_nodes"], marker, color=color, label=agent_name)
plt.yscale("log")
plt.xlabel("Profundidad de busqueda")
plt.ylabel("Nodos expandidos promedio (escala log)")
plt.title("Impacto de Alpha-Beta Pruning: nodos expandidos vs profundidad")
plt.xticks(depth_df["depth"].unique())
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig2_nodos_profundidad.png", dpi=150)
plt.close()

# ── Figura 3: win-rate por configuracion de heuristicas ────────────────────
summary = pd.read_csv("heuristic_summary.csv")

x = range(len(summary))
width = 0.38
plt.figure(figsize=(8, 4.5))
plt.bar([i - width / 2 for i in x], summary["wr_vs_random"] * 100, width,
        label="vs RandomAgent (n=30)", color="#B0C4DE")
bars = plt.bar([i + width / 2 for i in x], summary["wr_vs_stratagem"] * 100, width,
               label="vs Stratagem (n=100)", color="#2E4057")
for bar, value in zip(bars, summary["wr_vs_stratagem"] * 100):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
             ha="center", fontsize=9)
plt.axhline(50, color="#8B0000", linestyle="--", linewidth=1)
plt.xticks(list(x), summary["config"], rotation=15, ha="right")
plt.ylabel("Win-rate (%)")
plt.ylim(0, 115)
plt.title("Win-rate por configuracion de heuristicas (Minimax AB, profundidad 3)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig3_heuristicas.png", dpi=150)
plt.close()

print("ok: figuras generadas en figures/")
