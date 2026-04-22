from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, cast
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent

# Original Case 2 baseline
ORIGINAL_DIR = "Eval-DRL-Baseline-2026-case-2_2025-12-17_10:31-res1"
ORIGINAL_LABEL = "Original (TQC)"

# Directory with degradation evaluations
DEGRADATIONS_DIR = ROOT / "evaluaciones-5ep-entrenado"


def get_degradation_evaluations() -> List[Tuple[str, str, Path]]:
    """Scan degradations directory and return list of (name, label, path) tuples.

    Returns:
        List of tuples: (degradation_name, display_label, directory_path)
        Sorted by degradation type and number.
    """
    evaluations = []

    if not DEGRADATIONS_DIR.exists():
        return evaluations

    for dir_path in sorted(DEGRADATIONS_DIR.iterdir()):
        if not dir_path.is_dir():
            continue

        dir_name = dir_path.name
        # Extract degradation type and number from directory name
        # Format: Degradation-1ep-case2-{type}_{num}_... or
        # Eval-DRL-Baseline-2026-case-2-{type}_{num}_...
        match = re.search(r'case2?-([^_]+)_(\d+)_', dir_name)
        if match:
            # e.g., "window", "infiltration", "material", "all"
            deg_type = match.group(1)
            deg_num = match.group(2)   # e.g., "1", "2", "3"

            # Create display label
            label = f"{deg_type.capitalize()} {deg_num}"
            evaluations.append((dir_name, label, dir_path))

    # Sort by type, then by number
    evaluations.sort(
        key=lambda x: (
            x[1].split()[0], int(
                x[1].split()[1]) if x[1].split()[1].isdigit() else 0))

    return evaluations


def load_progress_metrics(progress_path: Path) -> pd.DataFrame:
    """Load raw evaluation metrics from progress.csv."""
    df = pd.read_csv(progress_path)
    return df


def plot_progress_comparison(output_dir: Path) -> None:
    """Create one interactive boxplot per metric to compare original vs degradations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original
    original_path = ROOT / ORIGINAL_DIR / "progress.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original progress.csv not found: {original_path}")

    original_df = load_progress_metrics(original_path)

    # Load degradations
    degradations = get_degradation_evaluations()

    if not degradations:
        print("Warning: No degradation evaluations found!")
        return

    # Prepare data: original first, then degradations
    all_data = {}
    all_labels = [ORIGINAL_LABEL]
    all_data[ORIGINAL_LABEL] = original_df

    for dir_name, label, dir_path in degradations:
        progress_path = dir_path / "progress.csv"
        if progress_path.exists():
            all_data[label] = load_progress_metrics(progress_path)
            all_labels.append(label)
        else:
            print(f"Warning: progress.csv not found in {dir_path}")

    # (column_name, label, filename, y_label)
    metrics_spec = [
        ("mean_reward",
         "Episode mean reward",
         "box_mean_reward.html",
         "Reward"),
        ("mean_temperature_violation",
            "Episode mean temperature violation",
            "box_mean_temperature_violation.html",
            "Temperature violation (°C)",
         ),
        ("mean_power_demand",
            "Episode mean power demand",
            "box_mean_power_demand.html",
            "Power demand (kW)",
         ),
        ("mean_compressor_starts_per_day",
            "Episode mean compressor starts per day",
            "box_mean_compressor_starts_per_day.html",
            "Starts per day",
         ),
    ]

    # Generate colors: original gets a distinct color, degradations get
    # gradient
    n_total = len(all_labels)
    colors = ["#2E86AB"]  # Blue for original
    # Generate colors for degradations (reddish gradient)
    for i in range(1, n_total):
        # Gradient from orange to red
        intensity = 0.6 + (i / n_total) * 0.4
        r = int(255 * intensity)
        g = int(100 * (1 - intensity * 0.5))
        b = int(50 * (1 - intensity * 0.3))
        colors.append(f"rgb({r},{g},{b})")

    for col, title, filename, y_label in metrics_spec:
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
            title=title,
            yaxis_title=y_label,
            showlegend=True,
            template="plotly_white",
            height=500,
            width=1000,
        )

        fig.write_html(output_dir / filename)


def plot_flow_and_water_violin(output_dir: Path) -> None:
    """Create interactive violin plots for flow rates and water temperature."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original
    original_monitor_dir = ROOT / ORIGINAL_DIR / "episode-20" / "monitor"

    # Load degradations
    degradations = get_degradation_evaluations()

    all_labels = [ORIGINAL_LABEL]
    all_monitor_dirs = [original_monitor_dir]

    for dir_name, label, dir_path in degradations:
        monitor_dir = dir_path / "episode-20" / "monitor"
        if monitor_dir.exists():
            all_labels.append(label)
            all_monitor_dirs.append(monitor_dir)

    colors = ["#2E86AB"]  # Blue for original
    n_total = len(all_labels)
    for i in range(1, n_total):
        intensity = 0.6 + (i / n_total) * 0.4
        r = int(255 * intensity)
        g = int(100 * (1 - intensity * 0.5))
        b = int(50 * (1 - intensity * 0.3))
        colors.append(f"rgb({r},{g},{b})")

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
        title="Water temperature distribution",
        yaxis_title="Water temperature (°C)",
        showlegend=False,
        template="plotly_white",
        height=500,
        width=1000,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )

    fig_water.write_html(output_dir / "violin_water_temperature.html")

    # Flow-rate violins per room (one plot per room showing all degradations)
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
            title=f'Flow rate: {room_name}',
            yaxis_title="Flow rate",
            showlegend=False,
            template="plotly_white",
            height=500,
            width=1000,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
        )

        fig_flow.write_html(output_dir / f'violin_flow_{room_slug}.html')


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
                    color='#1f77b4',
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

    n_rooms = len(TEMP_COLS)

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
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for i, (temp_col, sp_col) in enumerate(zip(TEMP_COLS, SETPOINT_COLS)):
        row = (i // 2) + 1
        col = (i % 2) + 1

        add_temperature_traces(
            fig, obs, temp_col, sp_col, show_legend=(i == 0), row=row, col=col
        )

        fig.update_yaxes(title_text="Temperature (°C)", row=row, col=col)

    fig.update_layout(
        title=f'{label} – All Rooms',
        height=900,
        template="plotly_white",
        hovermode=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")
    fig.write_html(output_dir / f'{safe_label}_temperatures.html')

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

        room_dir = output_dir / safe_label / room_slug
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
            title=f'{label} – {room_title}',
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            height=500,
            hovermode=False,
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )

        fig_r.write_html(room_dir / "temperature.html")

        # Daily zoom
        if not obs_daily.empty:
            fig_d = go.Figure()
            add_temperature_traces(
                fig_d, obs_daily, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            fig_d.update_layout(
                title=f'{label} – {room_title} (daily: {daily_date.date()})',
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
            )

            fig_d.write_html(room_dir / "daily_temperature.html")

        # Weekly zoom
        if not obs_week.empty:
            fig_w = go.Figure()
            add_temperature_traces(
                fig_w, obs_week, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            fig_w.update_layout(
                title=f'{label} – {room_title} (weekly: {
                    week_start.date()} to {
                    week_end.date()})',
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
            )

            fig_w.write_html(room_dir / "weekly_temperature.html")

        # Monthly zoom
        if not obs_month.empty:
            fig_m = go.Figure()
            add_temperature_traces(
                fig_m, obs_month, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            fig_m.update_layout(
                title=f'{label} – {room_title} (monthly: {
                    month_start.date()} to {
                    month_end.date()})',
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                xaxis=dict(
                    rangeslider=dict(
                        visible=True),
                    type="date"),
            )

            fig_m.write_html(room_dir / "monthly_temperature.html")


def main() -> None:
    """Main entry point."""
    output_dir = ROOT / "degradation_plots_interactive"

    # 1) Comparison of evaluation metrics between original and degradations
    print("Creating progress comparison plots...")
    plot_progress_comparison(output_dir)

    # 1b) Violin plots for flow rate and water temperature
    print("Creating violin plots...")
    plot_flow_and_water_violin(output_dir)

    # 2) Time series of temperatures and setpoints per room
    print("Creating temperature plots...")

    # Get a valid date from original
    original_monitor_dir = ROOT / ORIGINAL_DIR / "episode-20" / "monitor"
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
            "No valid dates found in the selected Nov–Mar window.")

    rng = np.random.default_rng(seed=0)
    idx = int(rng.integers(low=0, high=len(valid_dates)))
    daily_raw = valid_dates[idx]

    if isinstance(daily_raw, pd.Timestamp):
        daily_core = daily_raw.to_pydatetime().date()
    else:
        daily_core = pd.to_datetime(daily_raw).date()

    daily_date = pd.Timestamp(daily_core)

    # Plot original
    plot_degradation_temperatures(
        ORIGINAL_LABEL,
        original_monitor_dir,
        output_dir,
        cast(pd.Timestamp, daily_date))

    # Plot degradations
    degradations = get_degradation_evaluations()
    for dir_name, label, dir_path in degradations:
        monitor_dir = dir_path / "episode-20" / "monitor"
        if monitor_dir.exists():
            plot_degradation_temperatures(
                label, monitor_dir, output_dir, cast(pd.Timestamp, daily_date)
            )

    print(f"Interactive plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
