import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Estilo tipo paper
# ---------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "figure.dpi": 140
})

np.random.seed(42)

# ---------------------------------------------------
# Tiempo
# ---------------------------------------------------
dates = pd.date_range("2025-10-01", "2026-04-30", freq="D")
n = len(dates)
t = np.arange(n)

# ---------------------------------------------------
# Banda de confort
# ---------------------------------------------------
comfort_center = 21 + 0.5 * np.sin(2 * np.pi * t / 365)
comfort_low = comfort_center - 1
comfort_high = comfort_center + 1

# ---------------------------------------------------
# Figura
# ---------------------------------------------------
fig, axes = plt.subplots(
    6, 1,
    figsize=(12, 11),
    sharex=True
)

# ===================================================
# 5 ZONAS INTERIORES
# ===================================================
for i in range(5):

    ax = axes[i]

    base = 21.3 + 0.2 * np.sin(np.linspace(0, 6*np.pi, n))
    noise = np.random.normal(0, 0.25, n)
    temp = base + noise + 0.15 * i

    # ---------------------------------------------------
    # Banda de confort variable
    # ---------------------------------------------------
    ax.fill_between(
        dates,
        comfort_low,
        comfort_high,
        color="green",
        alpha=0.12
    )

    # ---------------------------------------------------
    # Línea con violaciones en rojo
    # ---------------------------------------------------
    for k in range(n - 1):

        x = dates[k:k+2]
        y = temp[k:k+2]

        c_low = comfort_low[k:k+2].mean()
        c_high = comfort_high[k:k+2].mean()

        color = "red" if (y.mean() < c_low or y.mean() > c_high) else "tab:blue"

        ax.plot(x, y, color=color)

    ax.set_title(f"Zone {i+1}", loc="left", pad=4)
    ax.grid(True, alpha=0.3)

# ===================================================
# OUTDOOR
# ===================================================
ax = axes[5]

days = np.arange(n)
outdoor = 12 + 8 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 1.5, n)

ax.plot(dates, outdoor, color="black", linewidth=1.5)

ax.set_title("Outdoor Temperature", loc="left", pad=4)
ax.grid(True, alpha=0.3)

# ---------------------------------------------------
# Label compartido
# ---------------------------------------------------
fig.supylabel("Temperature (°C)")

# ---------------------------------------------------
# Layout
# ---------------------------------------------------
plt.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.96,
    bottom=0.08,
    hspace=0.25
)

# ---------------------------------------------------
# Guardar
# ---------------------------------------------------
plt.savefig(
    "zone_temperature_with_outdoor.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()