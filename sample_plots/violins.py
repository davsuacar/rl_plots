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

np.random.seed(42)

# ---------------------------------------------------
# Figura 1x2
# ---------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(10.5, 4.2)
)

# ===================================================
# 1) Supply water temperature
# ===================================================
ax = axes[0]

temp = np.random.normal(loc=35, scale=3.5, size=400)
temp = np.clip(temp, 25, 45)

vp = ax.violinplot(temp, showmeans=True, showextrema=False)

for pc in vp["bodies"]:
    pc.set_facecolor("tab:blue")   # color A
    pc.set_alpha(0.75)

vp["cmeans"].set_color("black")
vp["cmeans"].set_linewidth(2)

ax.set_title("Supply water temperature")
ax.set_ylabel("Temperature (°C)")
ax.set_xticks([])
ax.set_ylim(25, 45)
ax.grid(True, axis="y", alpha=0.35)

# ===================================================
# 2) Flow rates per zone
# ===================================================
ax = axes[1]

flow_data = [
    np.random.normal(1.2, 0.2, 300),
    np.random.normal(1.5, 0.25, 300),
    np.random.normal(1.1, 0.15, 300),
    np.random.normal(1.8, 0.3, 300),
    np.random.normal(1.4, 0.2, 300),
]

vp = ax.violinplot(flow_data, showmeans=True, showextrema=False)

for pc in vp["bodies"]:
    pc.set_facecolor("tab:orange")  # color B (mismo para todos)
    pc.set_alpha(0.75)

vp["cmeans"].set_color("black")
vp["cmeans"].set_linewidth(2)

ax.set_title("Flow rates per zone")
ax.set_ylabel("Flow rate (m³/h)")
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Zone {i}" for i in range(1, 6)])
ax.grid(True, axis="y", alpha=0.35)

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
plt.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.12,
    top=0.85,
    wspace=0.30
)

# ---------------------------------------------------
# Guardar
# ---------------------------------------------------
plt.savefig(
    "violin.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()