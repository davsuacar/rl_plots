import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------
# Estilo tipo paper
# ---------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "figure.dpi": 140
})

# ---------------------------------------------------
# Datos simulados
# ---------------------------------------------------
np.random.seed(42)

titles = ["Case 1", "Case 2", "Case 3"]
labels = ["Agent 1", "Agent 2", "Agent 3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

def comfort_data(case_id, row_id, agent_id, n=60):
    base = 1.6 - 0.25 * agent_id + 0.18 * case_id + 0.10 * row_id
    spread = 0.22 + 0.03 * case_id
    return np.clip(np.random.normal(base, spread, n), 0, None)

# ---------------------------------------------------
# FIGURA
# ---------------------------------------------------
fig, axes = plt.subplots(
    2, 3,
    figsize=(9.5, 5.2),
    sharey=True
)

for r in range(2):
    for c in range(3):

        ax = axes[r, c]

        data = [
            comfort_data(c, r, 0),
            comfort_data(c, r, 1),
            comfort_data(c, r, 2)
        ]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            boxprops=dict(linewidth=1.1, color="black")
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels)

        if r == 0:
            ax.set_title(titles[c], pad=6)

        ax.grid(True, axis="y", alpha=0.35)

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
plt.subplots_adjust(
    left=0.1,
    right=0.98,
    bottom=0.10,
    top=0.86,
    wspace=0.25,
    hspace=0.35
)

# ---------------------------------------------------
# Y labels (MULTILINE)
# ---------------------------------------------------
axes[0, 0].set_ylabel("Mean episodic\ncomfort violation (°C)")
axes[1, 0].set_ylabel("Mean episodic\npower demand (W)")

# ---------------------------------------------------
# Leyenda
# ---------------------------------------------------
# legend_handles = [
#     Patch(facecolor=colors[i], edgecolor="black", label=labels[i])
#     for i in range(3)
# ]

# fig.legend(
#     handles=legend_handles,
#     loc="upper center",
#     ncol=3,
#     frameon=False,
#     bbox_to_anchor=(0.5, 1.16)
# )

# ---------------------------------------------------
# Guardar
# ---------------------------------------------------
plt.savefig(
    "comfort_power_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
