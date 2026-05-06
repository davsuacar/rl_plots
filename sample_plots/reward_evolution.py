import numpy as np
import matplotlib.pyplot as plt

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
x = np.arange(0, 101)

def reward_curve(speed=1.0, noise=0.03):
    y = -3 * np.exp(-speed * x / 25.0)
    y += np.random.normal(0, noise, len(x))
    return np.clip(y, -3, 0)

# ---------------------------------------------------
# FIGURA
# ---------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(11.5, 3.2),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

titles = ["Case 1", "Case 2", "Case 3"]

styles = [
    {"color": "tab:blue",   "linestyle": "-",  "marker": "o"},
    {"color": "tab:orange", "linestyle": "--", "marker": "s"},
    {"color": "tab:green",  "linestyle": ":",  "marker": "^"},
]

labels = ["Agent 1", "Agent 2", "Agent 3"]

for i, ax in enumerate(axes):

    y1 = reward_curve(speed=0.7 + i*0.15)
    y2 = reward_curve(speed=0.9 + i*0.12)
    y3 = reward_curve(speed=1.1 + i*0.10)

    for y, st, label in zip([y1, y2, y3], styles, labels):
        ax.plot(
            x, y,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markevery=8,
            markersize=4,
            label=label
        )

    ax.set_title(titles[i], pad=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3.05, 0.05)
    ax.grid(True, alpha=0.35)

axes[0].set_ylabel("Average reward")
axes[1].set_xlabel("Training episode")

# ---------------------------------------------------
# Leyenda
# ---------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.16)
)

# ---------------------------------------------------
# Guardar PNG
# ---------------------------------------------------
plt.savefig(
    "rl_reward_cases.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
