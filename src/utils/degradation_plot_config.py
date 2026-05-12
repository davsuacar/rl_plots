"""Configuración externa para ``plot_degradations_html.py``.

Copia ``DEFAULT_DEGRADATION_PLOT_CONFIG``, ajusta rutas y pásala con
``--config /ruta/a/mi_config.py`` donde el archivo define
``DEGRADATION_PLOT_CONFIG`` (instancia de :class:`DegradationPlotConfig`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from utils.study_plot_config import TAB10

# matplotlib ``colors.to_rgb(0.6)`` en escala de grises → (0.6, 0.6, 0.6)
DEGRADATION_GRAY_MATPLOTLIB_06 = "#999999"


def degradation_series_colors(
    n_total: int,
    *,
    baseline_color: str,
    degradation_color: str,
) -> List[str]:
    """Primera serie = baseline (paleta); el resto = gris uniforme."""
    if n_total <= 0:
        return []
    colors = [baseline_color]
    for _ in range(1, n_total):
        colors.append(degradation_color)
    return colors


@dataclass
class DegradationPlotConfig:
    """Rutas, etiquetas y variables para generar figuras de degradación."""

    study_label: str
    original_label: str
    original_eval_dir: Path
    data_dir: Path
    experiments: Dict[str, str]
    output_base: Path
    episode_dir: str = "episode-20"
    general_plots_dirname: str = "general"
    summary_plots_dirname: str = "summary"
    # Color Tab10[0] (misma paleta que ``study_plot_config`` / scripts piloto).
    baseline_box_color: str = TAB10[0]
    degradation_box_color: str = DEGRADATION_GRAY_MATPLOTLIB_06
    # Umbral banda confort / trazas de temperatura (como ``StudyPlotConfig`` / Uponor).
    temperature_threshold: float = 1.0
    # Color de la línea interior en ``plot_case_temperatures`` (Tab10[0] por defecto).
    temp_zone_line_color: str = TAB10[0]
    datetime_base_year: int = 2026
    simulation_window_start: str = "2026-11-15"
    simulation_window_end: str = "2027-03-15 23:55"
    png_export_scale: int = 2
    # Boxplots: mismas dimensiones que ``plot_uponor_case_study_by_algorithm`` (plot_dfs_boxplot).
    boxplot_width: int = 1200
    boxplot_height: int = 600
    temp_multi_width: int = 1200
    temp_single_height: int = 500
    # Etiquetas eje Y: alineadas con ``plot_uponor_case_study_by_algorithm`` donde existe
    # equivalente (progress / boxplots evaluation).
    progress_metrics: List[Tuple[str, str, str, str]] = field(
        default_factory=lambda: [
            ("mean_reward", "", "box_mean_reward.html", "Average reward"),
            (
                "mean_temperature_violation",
                "",
                "box_mean_temperature_violation.html",
                "Mean episodic comfort violation (ºC)",
            ),
            ("mean_power_demand", "", "box_mean_power_demand.html", "Mean episodic power demand (W)"),
            (
                "mean_compressor_starts_per_day",
                "",
                "box_mean_compressor_starts_per_day.html",
                "Mean episodic compressor starts per day",
            ),
        ]
    )
    temperature_variables: List[str] = field(
        default_factory=lambda: [
            "air_temperature_living",
            "air_temperature_kitchen",
            "air_temperature_bed1",
            "air_temperature_bed2",
            "air_temperature_bed3",
        ]
    )
    setpoint_variables: List[str] = field(
        default_factory=lambda: [
            "heating_setpoint_living",
            "heating_setpoint_kitchen",
            "heating_setpoint_bed1",
            "heating_setpoint_bed2",
            "heating_setpoint_bed3",
        ]
    )
    flow_variables: List[str] = field(
        default_factory=lambda: [
            "flow_rate_living",
            "flow_rate_kitchen",
            "flow_rate_bed1",
            "flow_rate_bed2",
            "flow_rate_bed3",
        ]
    )
    flow_room_labels: List[str] = field(
        default_factory=lambda: [
            "Living room",
            "Kitchen",
            "Bedroom 1",
            "Bedroom 2",
            "Bedroom 3",
        ]
    )
    subplot_zone_titles: List[str] = field(
        default_factory=lambda: [
            "Living room",
            "Kitchen",
            "Bedroom 1",
            "Bedroom 2",
            "Bedroom 3",
        ]
    )


DEFAULT_DEGRADATION_PLOT_CONFIG = DegradationPlotConfig(
    study_label="pilot_degradation",
    original_label="Original (TQC)",
    original_eval_dir=Path(
        "/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso/"
        "Eval-DRL-Baseline-2026-cases/caso2/"
        "Eval-DRL-Baseline-2026-case-2_2025-12-17_10:31-res1"
    ),
    data_dir=Path("/home/jovyan/work/data/paper/data/pilot_study/eval_por_robustez"),
    experiments={
        "Initial": "evaluaciones-iniciales",
        "1 Episode": "entrenamiento-1ep",
        "5 Episodes": "entrenamiento-5ep",
    },
    output_base=Path("/home/jovyan/work/data/paper/plots/pilot_study/degradation_study"),
)


def load_degradation_plot_config(path: Path | None) -> DegradationPlotConfig:
    """Carga configuración desde un ``.py`` externo o devuelve el preset por defecto."""
    if path is None:
        return DEFAULT_DEGRADATION_PLOT_CONFIG

    import importlib.util

    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    spec = importlib.util.spec_from_file_location("_degradation_plot_user_cfg", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {p}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = getattr(module, "DEGRADATION_PLOT_CONFIG", None)
    if cfg is None:
        raise ValueError(
            f"{p} must define DEGRADATION_PLOT_CONFIG "
            "(instance of DegradationPlotConfig)."
        )
    if not isinstance(cfg, DegradationPlotConfig):
        raise TypeError(
            f"DEGRADATION_PLOT_CONFIG must be DegradationPlotConfig, got {type(cfg)}"
        )
    return cfg
