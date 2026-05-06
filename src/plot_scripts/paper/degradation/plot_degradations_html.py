import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, cast
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from utils.plot_functions.plot_functions import (
    PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
    PLOTLY_SUBPLOT_VERTICAL_SPACING,
)

ROOT = Path(__file__).resolve().parent

# Baseline evaluation (case 2 TQC) — not under DATA_DIR / EXPERIMENTS
DEFAULT_ORIGINAL_DIR = (
    "/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso/"
    "Eval-DRL-Baseline-2026-cases/caso2/"
    "Eval-DRL-Baseline-2026-case-2_2025-12-17_10:31-res1"
)
ORIGINAL_LABEL = "Original (TQC)"

# Parent folder for robustness evaluations (each value is a subfolder name)
DATA_DIR = "/home/jovyan/work/data/paper/data/pilot_study/eval_por_robustez"
EXPERIMENTS = {
    "Initial": "evaluaciones-iniciales",
    "1 Episode": "entrenamiento-1ep",
    "5 Episodes": "entrenamiento-5ep",
}

# Root output: layout matches reference
# .../pilot_study/degradation_2/<degradation_slug>/box_*.html
# Here: <DEFAULT_OUTPUT_DIR>/<experiment_slug>/<degradation_slug>/...
# plus <DEFAULT_OUTPUT_DIR>/<experiment_slug>/general/ (all degradations)
DEFAULT_OUTPUT_DIR = "/home/jovyan/work/data/paper/plots/pilot_study/try"

# Better quality for PNG export
pio.defaults.default_scale = 2
pio.templates.default = "plotly_white"

_PNG_EXPORT_WARNING_SHOWN = False

# Evaluation episode folder (monitor data)
EPISODE_DIR = "episode-20"

# Subfolder under each experiment: combined boxplots (baseline + all degradations)
GENERAL_PLOTS_DIRNAME = "general"

# Under output root: per degradation, Original vs each training regime (same colors as general/)
SUMMARY_PLOTS_DIRNAME = "summary"

# (column_name, unused legacy label, filename, y_axis_label)
PROGRESS_METRICS_SPEC = [
    ("mean_reward", "Episode mean reward", "box_mean_reward.html", "Reward"),
    (
        "mean_temperature_violation",
        "Episode mean temperature violation",
        "box_mean_temperature_violation.html",
        "Temperature violation (°C)",
    ),
    (
        "mean_power_demand",
        "Episode mean power demand",
        "box_mean_power_demand.html",
        "Power demand (kW)",
    ),
    (
        "mean_compressor_starts_per_day",
        "Episode mean compressor starts per day",
        "box_mean_compressor_starts_per_day.html",
        "Starts per day",
    ),
]


def progress_boxplot_colors(n_total: int) -> List[str]:
    """Colors for stacked progress boxplots: blue baseline + reddish gradient (matches general/)."""
    colors = ["#2E86AB"]
    for i in range(1, n_total):
        intensity = 0.6 + (i / max(n_total, 2)) * 0.4
        r = int(255 * intensity)
        g = int(100 * (1 - intensity * 0.5))
        b = int(50 * (1 - intensity * 0.3))
        colors.append(f"rgb({r},{g},{b})")
    return colors


def slugify_experiment(name: str) -> str:
    """Folder name for an EXPERIMENTS key (e.g. '1 Episode' -> '1_episode')."""
    t = name.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_") or "experiment"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input/output directories."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate degradation comparison plots (HTML + PNG). "
            "Uses DATA_DIR/EXPERIMENTS for runs and DEFAULT_ORIGINAL_DIR "
            "for the baseline (outside EXPERIMENTS)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DATA_DIR),
        help=(
            "Parent directory that contains each EXPERIMENTS subfolder "
            f"(default: {DATA_DIR})."
        ),
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path(DEFAULT_ORIGINAL_DIR),
        help=(
            "Baseline evaluation directory (progress.csv, episode monitor). "
            "Default is the case-2 TQC eval path; absolute paths are used as-is."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=(
            "Root output directory. Per experiment: "
            "<output-dir>/<experiment_slug>/<degradation_slug>/..."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "If set, only these EXPERIMENTS keys are run "
            "(e.g. 'Initial' '5 Episodes'). Default: all."
        ),
    )
    return parser.parse_args()


def resolve_original_dir(path: Path) -> Path:
    """Baseline path: absolute paths as-is; relative paths resolved from cwd."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return expanded.resolve()


def resolve_data_subdir(data_dir: Path, relative: Path) -> Path:
    """Resolve a run folder under data-dir."""
    expanded = relative.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (data_dir / expanded).resolve()


def save_figure(fig: go.Figure, html_path: Path, png_scale: int = 2) -> None:
    """Save a Plotly figure as both HTML and PNG."""
    global _PNG_EXPORT_WARNING_SHOWN

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.write_html(html_path)

    png_path = html_path.with_suffix(".png")
    try:
        fig.write_image(png_path, format="png", scale=png_scale)
    except Exception as exc:
        if not _PNG_EXPORT_WARNING_SHOWN:
            print(
                "Warning: PNG export failed. Install/upgrade 'kaleido' "
                "to enable static image export."
            )
            print(f"Details: {exc}")
            _PNG_EXPORT_WARNING_SHOWN = True


def get_degradation_evaluations(
    degradations_dir: Path,
) -> List[Tuple[str, str, Path, str]]:
    """Scan degradations directory and return run entries.

    Returns:
        List of (dir_name, display_label, directory_path, degradation_slug).
        ``degradation_slug`` matches reference layout, e.g. ``infiltration_3``.
    """
    evaluations = []

    if not degradations_dir.exists():
        return evaluations

    for dir_path in sorted(degradations_dir.iterdir()):
        if not dir_path.is_dir():
            continue

        dir_name = dir_path.name
        # Extract degradation type and number from directory name
        # Format: Degradation-1ep-case2-{type}_{num}_... or
        # Eval-...-case-2-{type}_{num}_... (note "case-2-" vs "case2-")
        match = re.search(r"case2?-(?:2-)?([^_]+)_(\d+)_", dir_name)
        if match:
            deg_type = match.group(1)
            deg_num = match.group(2)
            deg_slug = f"{deg_type}_{deg_num}"
            label = f"{deg_type.capitalize()} {deg_num}"
            evaluations.append((dir_name, label, dir_path, deg_slug))

    evaluations.sort(
        key=lambda x: (
            x[1].split()[0],
            int(x[1].split()[1]) if x[1].split()[1].isdigit() else 0,
        )
    )

    return evaluations


def load_progress_metrics(progress_path: Path) -> pd.DataFrame:
    """Load raw evaluation metrics from progress.csv."""
    df = pd.read_csv(progress_path)
    return df


def plot_progress_boxplots_original_vs_one(
    output_dir: Path,
    original_dir: Path,
    deg_label: str,
    deg_path: Path,
) -> None:
    """Boxplots per metric: baseline vs one degradation (cf. degradation_2/<slug>/)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / "progress.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original progress.csv not found: {original_path}")

    original_df = load_progress_metrics(original_path)
    progress_path = deg_path / "progress.csv"
    if not progress_path.exists():
        print(f"Warning: progress.csv not found in {deg_path}")
        return

    deg_df = load_progress_metrics(progress_path)

    all_data = {ORIGINAL_LABEL: original_df, deg_label: deg_df}
    all_labels = [ORIGINAL_LABEL, deg_label]
    colors = ["#2E86AB", "rgb(214,85,65)"]

    for col, _unused_title, filename, y_label in PROGRESS_METRICS_SPEC:
        fig = go.Figure()
        for idx, label in enumerate(all_labels):
            if col in all_data[label].columns:
                fig.add_trace(
                    go.Box(
                        y=all_data[label][col].to_numpy(),
                        name=label,
                        marker_color=colors[idx],
                        boxmean=True,
                    )
                )

        fig.update_layout(
            yaxis_title=y_label,
            showlegend=True,
            template="plotly_white",
            height=500,
            width=1000,
        )

        save_figure(fig, output_dir / filename)


def plot_progress_boxplots_original_vs_all(
    output_dir: Path,
    original_dir: Path,
    degradations: List[Tuple[str, str, Path, str]],
) -> None:
    """One figure per metric: baseline vs every degradation (same filenames as per-slug)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / "progress.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original progress.csv not found: {original_path}")

    original_df = load_progress_metrics(original_path)
    all_data: Dict[str, pd.DataFrame] = {ORIGINAL_LABEL: original_df}
    all_labels = [ORIGINAL_LABEL]

    for _dir_name, label, dir_path, _deg_slug in degradations:
        progress_path = dir_path / "progress.csv"
        if progress_path.exists():
            all_data[label] = load_progress_metrics(progress_path)
            all_labels.append(label)
        else:
            print(f"Warning: progress.csv not found in {dir_path}")

    colors = progress_boxplot_colors(len(all_labels))

    for col, _unused_title, filename, y_label in PROGRESS_METRICS_SPEC:
        fig = go.Figure()
        for idx, label in enumerate(all_labels):
            if label in all_data and col in all_data[label].columns:
                fig.add_trace(
                    go.Box(
                        y=all_data[label][col].to_numpy(),
                        name=label,
                        marker_color=colors[idx],
                        boxmean=True,
                    )
                )

        fig.update_layout(
            yaxis_title=y_label,
            showlegend=True,
            template="plotly_white",
            height=500,
            width=1000,
        )

        save_figure(fig, output_dir / filename)


def plot_progress_boxplots_summary_by_degradation(
    output_dir: Path,
    original_dir: Path,
    experiment_order: List[str],
    degradations_by_experiment: Dict[str, List[Tuple[str, str, Path, str]]],
) -> None:
    """Per degradation slug: Original + each experiment (same colors as general/)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / "progress.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original progress.csv not found: {original_path}")
    original_df = load_progress_metrics(original_path)

    by_slug: Dict[str, Dict[str, Path]] = {}
    for exp_name, degradations in degradations_by_experiment.items():
        for _dn, _label, dir_path, deg_slug in degradations:
            by_slug.setdefault(deg_slug, {})[exp_name] = dir_path

    def _slug_sort_key(slug: str) -> Tuple[str, int]:
        m = re.match(r"(.+)_(\d+)$", slug)
        if m:
            return m.group(1), int(m.group(2))
        return slug, 0

    for deg_slug in sorted(by_slug.keys(), key=_slug_sort_key):
        deg_out = output_dir / deg_slug
        deg_out.mkdir(parents=True, exist_ok=True)

        paths_for_slug = by_slug[deg_slug]

        all_data: Dict[str, pd.DataFrame] = {ORIGINAL_LABEL: original_df}
        ordered_labels: List[str] = [ORIGINAL_LABEL]

        for exp_name in experiment_order:
            if exp_name not in paths_for_slug:
                continue
            pp = paths_for_slug[exp_name] / "progress.csv"
            if not pp.exists():
                print(
                    f"Warning: summary missing progress.csv for "
                    f"{exp_name} / {deg_slug}: {pp}"
                )
                continue
            all_data[exp_name] = load_progress_metrics(pp)
            ordered_labels.append(exp_name)

        if len(ordered_labels) <= 1:
            continue

        colors = progress_boxplot_colors(len(ordered_labels))

        for col, _unused_title, filename, y_label in PROGRESS_METRICS_SPEC:
            fig = go.Figure()
            for idx, label in enumerate(ordered_labels):
                if label in all_data and col in all_data[label].columns:
                    fig.add_trace(
                        go.Box(
                            y=all_data[label][col].to_numpy(),
                            name=label,
                            marker_color=colors[idx],
                            boxmean=True,
                        )
                    )

            fig.update_layout(
                yaxis_title=y_label,
                showlegend=True,
                template="plotly_white",
                height=500,
                width=1000,
            )

            save_figure(fig, deg_out / filename)


def plot_flow_and_water_violin_original_vs_one(
    output_dir: Path,
    original_dir: Path,
    deg_label: str,
    deg_path: Path,
) -> None:
    """Violin plots for flow and water: baseline vs one degradation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_monitor_dir = original_dir / EPISODE_DIR / "monitor"
    deg_monitor = deg_path / EPISODE_DIR / "monitor"
    if not deg_monitor.exists():
        print(f"Warning: monitor dir missing, skip violins: {deg_monitor}")
        return

    all_labels = [ORIGINAL_LABEL, deg_label]
    all_monitor_dirs = [original_monitor_dir, deg_monitor]

    colors = ["#2E86AB", "rgb(214,85,65)"]

    flow_cols = [
        "flow_rate_living",
        "flow_rate_kitchen",
        "flow_rate_bed1",
        "flow_rate_bed2",
        "flow_rate_bed3",
    ]

    # Water temperature violin plot
    fig_water = go.Figure()

    for idx, (label, monitor_dir) in enumerate(
            zip(all_labels, all_monitor_dirs)):
        obs_path = monitor_dir / "observations.csv"
        infos_path = monitor_dir / "infos.csv"

        if not obs_path.exists() or not infos_path.exists():
            print(f"Warning: Missing files in {monitor_dir}")
            continue

        obs = pd.read_csv(obs_path)
        infos = pd.read_csv(infos_path)

        min_len = min(len(obs), len(infos))
        obs = obs.iloc[:min_len].reset_index(drop=True)
        infos = infos.iloc[:min_len].reset_index(drop=True)

        dt_index = build_datetime_index(infos, base_year=2026)
        obs.index = dt_index

        start = datetime(2026, 11, 15)
        end = datetime(2027, 3, 15, 23, 55)
        mask = (obs.index >= start) & (obs.index <= end)
        obs = obs.loc[mask]

        if "water_temperature" in obs.columns:
            water_vals = obs["water_temperature"].to_numpy().astype(float)
            # Convert color to rgba with transparency for softer shading
            color = colors[idx]
            if color.startswith('#'):
                # Hex color
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            elif color.startswith('rgb'):
                # Extract RGB values from rgb(r, g, b) format
                rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
                if rgb_match:
                    r, g, b = map(int, rgb_match.groups())
                else:
                    r, g, b = 128, 128, 128  # Default gray
            else:
                r, g, b = 128, 128, 128  # Default gray

            color_rgba = f"rgba({r}, {g}, {b}, 0.3)"
            mean_val = np.mean(water_vals)
            fig_water.add_trace(
                go.Violin(
                    y=water_vals,
                    name=label,
                    fillcolor=color_rgba,
                    line_color=color,
                    box_visible=False,
                    meanline_visible=False,
                )
            )
            # Add custom mean line with fixed horizontal length
            fig_water.add_shape(
                type="line",
                x0=idx - 0.15,
                x1=idx + 0.15,
                y0=mean_val,
                y1=mean_val,
                line=dict(color='blue', width=2),
                xref='x',
                yref='y',
            )

    fig_water.update_layout(
        yaxis_title="Water temperature (°C)",
        showlegend=False,
        template="plotly_white",
        height=500,
        width=1000,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )

    save_figure(fig_water, output_dir / "violin_water_temperature.html")

    # Flow-rate violins per room (baseline vs this degradation)
    room_labels = [
        "Living room",
        "Kitchen",
        "Bedroom 1",
        "Bedroom 2",
        "Bedroom 3"]

    # Load all data first
    all_obs_data = {}
    for idx, (label, monitor_dir) in enumerate(
            zip(all_labels, all_monitor_dirs)):
        obs_path = monitor_dir / "observations.csv"
        infos_path = monitor_dir / "infos.csv"

        if not obs_path.exists() or not infos_path.exists():
            continue

        obs = pd.read_csv(obs_path)
        infos = pd.read_csv(infos_path)

        min_len = min(len(obs), len(infos))
        obs = obs.iloc[:min_len].reset_index(drop=True)
        infos = infos.iloc[:min_len].reset_index(drop=True)

        dt_index = build_datetime_index(infos, base_year=2026)
        obs.index = dt_index

        start = datetime(2026, 11, 15)
        end = datetime(2027, 3, 15, 23, 55)
        mask = (obs.index >= start) & (obs.index <= end)
        obs = obs.loc[mask]

        all_obs_data[label] = obs

    # Create one plot per room
    for room_idx, (col, room_name) in enumerate(
            zip(flow_cols, room_labels)):
        fig_flow = go.Figure()

        for idx, label in enumerate(all_labels):
            if label not in all_obs_data:
                continue

            obs = all_obs_data[label]
            if col not in obs.columns:
                continue

            room_flows = obs[col].to_numpy().astype(float)
            mean_val = np.mean(room_flows)

            # Get color for this degradation
            color = colors[idx]
            if color.startswith('#'):
                # Hex color
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            elif color.startswith('rgb'):
                # Extract RGB values from rgb(r, g, b) format
                rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
                if rgb_match:
                    r, g, b = map(int, rgb_match.groups())
                else:
                    r, g, b = 128, 128, 128  # Default gray
            else:
                r, g, b = 128, 128, 128  # Default gray

            color_rgba = f"rgba({r}, {g}, {b}, 0.3)"
            color_line = f"rgba({r}, {g}, {b}, 0.6)"

            fig_flow.add_trace(
                go.Violin(
                    y=room_flows,
                    name=label,
                    fillcolor=color_rgba,
                    line_color=color_line,
                    box_visible=False,
                    meanline_visible=False,
                )
            )
            # Add custom mean line with fixed horizontal length
            fig_flow.add_shape(
                type="line",
                x0=idx - 0.15,
                x1=idx + 0.15,
                y0=mean_val,
                y1=mean_val,
                line=dict(color='blue', width=2),
                xref='x',
                yref='y',
            )

        room_slug = room_name.lower().replace(" ", "_")
        fig_flow.update_layout(
            yaxis_title="Flow rate",
            showlegend=False,
            template="plotly_white",
            height=500,
            width=1000,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
        )

        save_figure(fig_flow, output_dir / f'violin_flow_{room_slug}.html')


def build_datetime_index(
    infos: pd.DataFrame, base_year: int = 2026
) -> pd.DatetimeIndex:
    """Build a continuous datetime index from November to March."""
    months = infos["month"].astype(int).to_numpy()
    days = infos["day"].astype(int).to_numpy()
    hours = infos["hour"].astype(int).to_numpy()

    years = np.where(months >= 11, base_year, base_year + 1)

    datetimes = pd.to_datetime(
        {
            "year": years,
            "month": months,
            "day": days,
            "hour": hours,
        }
    )

    return pd.DatetimeIndex(datetimes)


TEMP_COLS = [
    "air_temperature_living",
    "air_temperature_kitchen",
    "air_temperature_bed1",
    "air_temperature_bed2",
    "air_temperature_bed3",
]

SETPOINT_COLS = [
    "heating_setpoint_living",
    "heating_setpoint_kitchen",
    "heating_setpoint_bed1",
    "heating_setpoint_bed2",
    "heating_setpoint_bed3",
]

# Indoor temperature line when inside setpoint ±1 °C band
TEMP_IN_COMFORT_COLOR = "#1ABC9C"

# Legend above the plot area (temperature figures)
TEMP_LEGEND_LAYOUT = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5,
)


def add_temperature_traces(
    fig, obs_data, temp_col, sp_col, show_legend=True, row=None, col=None
):
    """Helper function to add temperature traces with comfort band coloring."""
    temp = obs_data[temp_col].to_numpy()
    sp = obs_data[sp_col].to_numpy()
    index = obs_data.index

    sp_upper = sp + 1.0
    sp_lower = sp - 1.0
    in_comfort = (temp >= sp_lower) & (temp <= sp_upper)

    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}

    # Add setpoint comfort band
    fig.add_trace(
        go.Scatter(
            x=index,
            y=sp_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ),
        **trace_kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=index,
            y=sp_lower,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 165, 0, 0.2)',
            fill='tonexty',
            name='Setpoint ±1°C' if show_legend else None,
            showlegend=show_legend,
            hovertemplate='Setpoint band<extra></extra>',
        ),
        **trace_kwargs,
    )

    # Split into segments for color coding
    segments_in = []
    segments_out = []
    current_segment_in = {"x": [], "y": []}
    current_segment_out = {"x": [], "y": []}

    for i in range(len(temp)):
        if in_comfort[i]:
            current_segment_in["x"].append(index[i])
            current_segment_in["y"].append(temp[i])
            if len(current_segment_out["x"]) > 0:
                segments_out.append(current_segment_out.copy())
                current_segment_out = {"x": [], "y": []}
                current_segment_out["x"].append(index[i])
                current_segment_out["y"].append(temp[i])
        else:
            current_segment_out["x"].append(index[i])
            current_segment_out["y"].append(temp[i])
            if len(current_segment_in["x"]) > 0:
                segments_in.append(current_segment_in.copy())
                current_segment_in = {"x": [], "y": []}
                current_segment_in["x"].append(index[i])
                current_segment_in["y"].append(temp[i])

    if len(current_segment_in["x"]) > 0:
        segments_in.append(current_segment_in)
    if len(current_segment_out["x"]) > 0:
        segments_out.append(current_segment_out)

    for i, seg in enumerate(segments_in):
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode='lines',
                name='Indoor temp (in comfort)' if (
                    show_legend and i == 0) else None,
                showlegend=(
                    show_legend and i == 0),
                line=dict(
                    color=TEMP_IN_COMFORT_COLOR,
                    width=1.5),
                hovertemplate='Indoor: %{y:.2f}°C<extra></extra>',
                legendgroup='indoor_in',
            ),
            **trace_kwargs,
        )

    for i, seg in enumerate(segments_out):
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode='lines',
                name='Indoor temp (out of comfort)' if (
                    show_legend and i == 0) else None,
                showlegend=(
                    show_legend and i == 0),
                line=dict(
                    color='#d62728',
                    width=1.5),
                hovertemplate='Indoor: %{y:.2f}°C (OUT OF COMFORT)<extra></extra>',
                legendgroup='indoor_out',
            ),
            **trace_kwargs,
        )


def plot_degradation_temperatures(
    label: str, monitor_dir: Path, output_dir: Path, daily_date: pd.Timestamp
) -> None:
    """Plot interactive indoor air temperatures and setpoints for a given degradation."""
    obs_path = monitor_dir / "observations.csv"
    infos_path = monitor_dir / "infos.csv"

    if not obs_path.exists() or not infos_path.exists():
        print(f"Warning: Missing files in {monitor_dir}")
        return

    obs = pd.read_csv(obs_path)
    infos = pd.read_csv(infos_path)

    min_len = min(len(obs), len(infos))
    obs = obs.iloc[:min_len].reset_index(drop=True)
    infos = infos.iloc[:min_len].reset_index(drop=True)

    dt_index = build_datetime_index(infos, base_year=2026)
    obs.index = dt_index

    start = datetime(2026, 11, 15)
    end = datetime(2027, 3, 15, 23, 55)
    mask = (obs.index >= start) & (obs.index <= end)
    obs = obs.loc[mask]

    daily_date_norm = daily_date.normalize()
    daily_mask = obs.index.normalize() == daily_date_norm
    obs_daily = obs.loc[daily_mask]

    week_start = daily_date_norm
    week_end = week_start + pd.Timedelta(days=6)
    week_mask = (obs.index >= week_start) & (obs.index <= week_end)
    obs_week = obs.loc[week_mask]

    month_start = daily_date_norm.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    month_mask = (obs.index >= month_start) & (obs.index <= month_end)
    obs_month = obs.loc[month_mask]

    # Multi-panel figure for all rooms
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Living room",
            "Kitchen",
            "Bedroom 1",
            "Bedroom 2",
            "Bedroom 3",
        ],
        vertical_spacing=PLOTLY_SUBPLOT_VERTICAL_SPACING,
        horizontal_spacing=PLOTLY_SUBPLOT_HORIZONTAL_SPACING,
    )

    for i, (temp_col, sp_col) in enumerate(zip(TEMP_COLS, SETPOINT_COLS)):
        row = (i // 2) + 1
        col = (i % 2) + 1

        add_temperature_traces(
            fig, obs, temp_col, sp_col, show_legend=(i == 0), row=row, col=col
        )

    # width: mismo ancho que el PNG (Kaleido usa layout; sin width el PNG ~700px y el
    # margen/anotación se ven distintos que en HTML a pantalla ancha).
    fig.update_layout(
        width=1200,
        height=760,
        template="plotly_white",
        hovermode=False,
        legend=TEMP_LEGEND_LAYOUT,
        margin=dict(l=102, t=72, r=22, b=32),
    )
    # Single shared Y-axis label for all zones (avoid repeating on each subplot)
    fig.add_annotation(
        text="Temperature (°C)",
        xref="paper",
        yref="paper",
        x=-0.068,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        showarrow=False,
        font=dict(size=13),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")
    save_figure(fig, output_dir / f'{safe_label}_temperatures.html')

    # Individual room plots with rangeslider
    for i, temp_col in enumerate(TEMP_COLS):
        base_name = temp_col.replace("air_temperature_", "")
        if base_name == "living":
            room_slug = "living_room"
            room_title = "Living room"
        elif base_name == "kitchen":
            room_slug = "kitchen"
            room_title = "Kitchen"
        elif base_name.startswith("bed"):
            idx = base_name.replace("bed", "")
            room_slug = f"bedroom_{idx}"
            room_title = f"Bedroom {idx}"
        else:
            room_slug = base_name
            room_title = base_name.capitalize()

        # output_dir is already .../<experiment>/<degradation_slug>/
        room_dir = output_dir / room_slug
        room_dir.mkdir(parents=True, exist_ok=True)

        # Full period plot
        fig_r = go.Figure()
        add_temperature_traces(
            fig_r,
            obs,
            temp_col,
            SETPOINT_COLS[i],
            show_legend=True)

        fig_r.update_layout(
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            height=500,
            hovermode=False,
            legend=TEMP_LEGEND_LAYOUT,
            margin=dict(t=60),
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )

        save_figure(fig_r, room_dir / "temperature.html")

        # Daily zoom
        if not obs_daily.empty:
            fig_d = go.Figure()
            add_temperature_traces(
                fig_d, obs_daily, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            daily_fn = (
                f"daily_temperature_{daily_date_norm.strftime('%Y-%m-%d')}.html"
            )
            fig_d.update_layout(
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                legend=TEMP_LEGEND_LAYOUT,
                margin=dict(t=60),
            )

            save_figure(fig_d, room_dir / daily_fn)

        # Weekly zoom
        if not obs_week.empty:
            fig_w = go.Figure()
            add_temperature_traces(
                fig_w, obs_week, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            weekly_fn = (
                f"weekly_temperature_{week_start.strftime('%Y-%m-%d')}_"
                f"to_{week_end.strftime('%Y-%m-%d')}.html"
            )
            fig_w.update_layout(
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                legend=TEMP_LEGEND_LAYOUT,
                margin=dict(t=60),
            )

            save_figure(fig_w, room_dir / weekly_fn)

        # Monthly zoom
        if not obs_month.empty:
            fig_m = go.Figure()
            add_temperature_traces(
                fig_m, obs_month, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            monthly_fn = (
                f"monthly_temperature_{month_start.strftime('%Y-%m-%d')}_"
                f"to_{month_end.strftime('%Y-%m-%d')}.html"
            )
            fig_m.update_layout(
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                legend=TEMP_LEGEND_LAYOUT,
                margin=dict(t=60),
                xaxis=dict(
                    rangeslider=dict(
                        visible=True),
                    type="date"),
            )

            save_figure(fig_m, room_dir / monthly_fn)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    original_dir = resolve_original_dir(args.original_dir)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = list(EXPERIMENTS.items())
    if args.experiments:
        wanted = set(args.experiments)
        experiments = [(k, v) for k, v in experiments if k in wanted]
        unknown = wanted - {k for k, _ in EXPERIMENTS.items()}
        if unknown:
            print(f"Warning: unknown experiment keys ignored: {sorted(unknown)}")

    print(f"Data dir: {data_dir}")
    print(f"Baseline (original) eval dir: {original_dir}")
    print(f"Output root: {output_root}")

    original_monitor_dir = original_dir / EPISODE_DIR / "monitor"
    original_infos_path = original_monitor_dir / "infos.csv"
    if not original_infos_path.exists():
        print(f"Warning: Original infos.csv not found: {original_infos_path}")
        return

    original_infos = pd.read_csv(original_infos_path)
    original_dt_index = build_datetime_index(original_infos, base_year=2026)
    first_mask = (original_dt_index >= datetime(2026, 11, 15)) & (
        original_dt_index <= datetime(2027, 3, 15, 23, 55)
    )
    masked_dates = original_dt_index[first_mask]
    valid_dates = pd.Series(masked_dates).dt.normalize().unique()
    if len(valid_dates) == 0:
        raise RuntimeError(
            "No valid dates found in the selected Nov–Mar window."
        )

    rng = np.random.default_rng(seed=0)
    idx = int(rng.integers(low=0, high=len(valid_dates)))
    daily_raw = valid_dates[idx]
    if isinstance(daily_raw, pd.Timestamp):
        daily_core = daily_raw.to_pydatetime().date()
    else:
        daily_core = pd.to_datetime(daily_raw).date()
    daily_date = pd.Timestamp(daily_core)

    degradations_by_experiment: Dict[
        str, List[Tuple[str, str, Path, str]]
    ] = {}

    for exp_name, exp_subdir in experiments:
        exp_slug = slugify_experiment(exp_name)
        degradations_dir = resolve_data_subdir(data_dir, Path(exp_subdir))
        output_exp = output_root / exp_slug

        print(f"\n=== Experiment: {exp_name} -> {exp_slug}")
        print(f"    Degradations dir: {degradations_dir}")
        print(f"    Output: {output_exp}")

        baseline_out = output_exp / "original"
        plot_degradation_temperatures(
            ORIGINAL_LABEL,
            original_monitor_dir,
            baseline_out,
            cast(pd.Timestamp, daily_date),
        )

        degradations = get_degradation_evaluations(degradations_dir)
        if not degradations:
            print(f"    Warning: no degradation runs in {degradations_dir}")
            continue

        degradations_by_experiment[exp_name] = degradations

        general_dir = output_exp / GENERAL_PLOTS_DIRNAME
        print(f"    -> {GENERAL_PLOTS_DIRNAME}/ (all degradations boxplots)")
        plot_progress_boxplots_original_vs_all(
            general_dir, original_dir, degradations
        )

        for _dir_name, label, dir_path, deg_slug in degradations:
            out_deg = output_exp / deg_slug
            print(f"    -> {deg_slug}")

            plot_progress_boxplots_original_vs_one(
                out_deg, original_dir, label, dir_path
            )
            plot_flow_and_water_violin_original_vs_one(
                out_deg, original_dir, label, dir_path
            )

            monitor_dir = dir_path / EPISODE_DIR / "monitor"
            if monitor_dir.exists():
                plot_degradation_temperatures(
                    label,
                    monitor_dir,
                    out_deg,
                    cast(pd.Timestamp, daily_date),
                )

    if degradations_by_experiment:
        summary_dir = output_root / SUMMARY_PLOTS_DIRNAME
        print(
            f"\n=== {SUMMARY_PLOTS_DIRNAME}/ "
            "(Original vs training regimes per degradation)"
        )
        plot_progress_boxplots_summary_by_degradation(
            summary_dir,
            original_dir,
            [name for name, _ in experiments],
            degradations_by_experiment,
        )

    print(f"\nPlots saved under: {output_root} (HTML + PNG per folder)")


if __name__ == "__main__":
    main()
