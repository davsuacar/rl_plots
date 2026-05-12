import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Estilo tipo paper
# ---------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "figure.dpi": 140
})

np.random.seed(42)

# ---------------------------------------------------
# Casos
# ---------------------------------------------------
cases = ["Default", "Deg. 1", "Deg. 2", "Deg. 3"]
colors = ["tab:blue", "0.6", "0.6", "0.6"]

# ---------------------------------------------------
# Datos simulados
# ---------------------------------------------------
def comfort_data(case_id, n=80):
    base = 0.8 + 0.4 * case_id
    return np.random.normal(base, 0.25 + 0.05 * case_id, n)

def power_data(case_id, n=80):
    base = 120 + 40 * case_id
    return np.random.normal(base, 15 + 5 * case_id, n)

# ---------------------------------------------------
# FIGURA
# ---------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(9.2, 3.8),
    sharex=True
)

# ===================================================
# Comfort
# ===================================================
ax = axes[0]

data = [comfort_data(i) for i in range(4)]

bp = ax.boxplot(
    data,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.0),
    capprops=dict(color="black", linewidth=1.0),
    boxprops=dict(linewidth=1.0, color="black")
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_title("Mean episodic comfort violations", pad=4)
ax.set_ylabel("Temperature (°C)")
ax.set_xticks(range(1, 5))
ax.set_xticklabels(cases)
ax.grid(True, axis="y", alpha=0.35)

# ===================================================
# Power
# ===================================================
ax = axes[1]

data = [power_data(i) for i in range(4)]

bp = ax.boxplot(
    data,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.0),
    capprops=dict(color="black", linewidth=1.0),
    boxprops=dict(linewidth=1.0, color="black")
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_title("Mean episodic power demand", pad=4)
ax.set_ylabel("Power (W)")
ax.set_xticks(range(1, 5))
ax.set_xticklabels(cases)
ax.grid(True, axis="y", alpha=0.35)

# ---------------------------------------------------
# Layout compacto horizontal
# ---------------------------------------------------
plt.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.88,
    bottom=0.18,
    wspace=0.30
)

# ---------------------------------------------------
# Guardar
# ---------------------------------------------------
plt.savefig(
    "boxplots_degradations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()