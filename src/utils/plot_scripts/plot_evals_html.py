from datetime import datetime
from pathlib import Path
from typing import Dict, cast
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

ROOT = Path(__file__).resolve().parents[0]

CASE_CONFIG = {1: {"name": "Case 1",
                   "agent": "SAC",
                   "dir": "/workspaces/python-plot-env/75_study/SAC-best-case1_2025-12-23_12:22-res1",
                   },
               2: {"name": "Case 2",
                   "agent": "SAC",
                   "dir": "/workspaces/python-plot-env/75_study/SAC-best-case2_2025-12-23_12:49-res1",
                   },
               3: {"name": "Case 3",
                   "agent": "TQC",
                   "dir": "/workspaces/python-plot-env/75_study/SAC-best-case3_2025-12-23_13:03-res1",
                   },
               }


def load_progress_metrics(case_cfg: Dict) -> pd.DataFrame:
    """Load raw evaluation metrics from progress. csv for a given case."""
    progress_path = ROOT / case_cfg["dir"] / "progress.csv"
    df = pd.read_csv(progress_path)
    return df


def plot_progress_comparison(output_dir: Path) -> None:
    """Create one interactive boxplot per metric to compare cases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    case_labels = [
        f'{cfg["name"]} ({cfg["agent"]})' for cfg in CASE_CONFIG.values()]

    # Load dataframes per case
    dfs = {case_id: load_progress_metrics(cfg)
           for case_id, cfg in CASE_CONFIG.items()}

    # (column_name, label, filename, y_label)
    metrics_spec = [
        ("mean_reward",
         "Episode mean reward",
         "box_mean_reward. html",
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

    colors = ["#4c72b0", "#55a868", "#c44e52"]

    for col, title, filename, y_label in metrics_spec:
        fig = go.Figure()

        for idx, (case_id, label) in enumerate(
                zip(CASE_CONFIG.keys(), case_labels)):
            fig.add_trace(
                go.Box(
                    y=dfs[case_id][col].to_numpy(),
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
            width=800,
        )

        fig.write_html(output_dir / filename)


def plot_flow_and_water_violin(output_dir: Path) -> None:
    """Create interactive violin plots for flow rates and water temperature."""
    output_dir.mkdir(parents=True, exist_ok=True)

    case_labels = [
        f'{cfg["name"]} ({cfg["agent"]})' for cfg in CASE_CONFIG.values()]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    flow_cols = [
        "flow_rate_living",
        "flow_rate_kitchen",
        "flow_rate_bed1",
        "flow_rate_bed2",
        "flow_rate_bed3",
    ]

    case_items = list(CASE_CONFIG.items())

    # Water temperature violin plot
    fig_water = go.Figure()

    for idx, (_, cfg) in enumerate(case_items):
        monitor_dir = ROOT / cfg["dir"] / "episode-20" / "monitor"
        obs_path = monitor_dir / "observations.csv"
        infos_path = monitor_dir / "infos.csv"

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
                name=case_labels[idx],
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

    # Flow-rate violins per room and per case
    room_labels = [
        "Living room",
        "Kitchen",
        "Bedroom 1",
        "Bedroom 2",
        "Bedroom 3"]

    for idx, (_, cfg) in enumerate(case_items):
        monitor_dir = ROOT / cfg["dir"] / "episode-20" / "monitor"
        obs_path = monitor_dir / "observations.csv"
        infos_path = monitor_dir / "infos.csv"

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

        fig_flow = go.Figure()

        for room_idx, (col, room_name) in enumerate(
                zip(flow_cols, room_labels)):
            room_flows = obs[col].to_numpy().astype(float)
            mean_val = np.mean(room_flows)

            # Use a default color scheme for rooms (can be customized)
            # Using a simple color palette for rooms
            room_colors = [
                "#4c72b0",
                "#55a868",
                "#c44e52",
                "#8172b2",
                "#ccb974"]
            color = room_colors[room_idx % len(room_colors)]

            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 128, 128, 128  # Default gray

            color_rgba = f"rgba({r}, {g}, {b}, 0.3)"
            color_line = f"rgba({r}, {g}, {b}, 0.6)"

            fig_flow.add_trace(
                go.Violin(
                    y=room_flows,
                    name=room_name,
                    fillcolor=color_rgba,
                    line_color=color_line,
                    box_visible=False,
                    meanline_visible=False,
                )
            )
            # Add custom mean line with fixed horizontal length
            fig_flow.add_shape(
                type="line",
                x0=room_idx - 0.15,
                x1=room_idx + 0.15,
                y0=mean_val,
                y1=mean_val,
                line=dict(color='blue', width=2),
                xref='x',
                yref='y',
            )

        fig_flow.update_layout(
            title=f'{cfg["name"]} – Flow rate per room',
            yaxis_title="Flow rate",
            showlegend=False,
            template="plotly_white",
            height=500,
            width=1000,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
        )

        fig_flow.write_html(output_dir / f'violin_flow_case{idx + 1}.html')


def build_datetime_index(
    infos: pd.DataFrame, base_year: int = 2026
) -> pd.DatetimeIndex:
    """Build a continuous datetime index from November to March.
    Months 11 and 12 belong to the base year, and months 1–3 to base_year + 1.
    """
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
    """Helper function to add temperature traces with comfort band coloring.

    Adds:
    - Setpoint comfort band (orange, semi-transparent)
    - Indoor temperature (blue when in comfort, red when out of comfort)
    """
    # Get data
    temp = obs_data[temp_col].to_numpy()
    sp = obs_data[sp_col].to_numpy()
    index = obs_data.index

    # Calculate comfort band
    sp_upper = sp + 1.0
    sp_lower = sp - 1.0

    # Determine if temperature is within comfort band
    in_comfort = (temp >= sp_lower) & (temp <= sp_upper)

    # Trace kwargs
    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}

    # 1. Add setpoint comfort band
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

    # 2. Add indoor temperature with color based on comfort
    # Split into continuous segments for smooth rendering
    segments_in = []
    segments_out = []

    current_segment_in = {"x": [], "y": []}
    current_segment_out = {"x": [], "y": []}

    for i in range(len(temp)):
        if in_comfort[i]:
            # Add to "in comfort" segment
            current_segment_in["x"].append(index[i])
            current_segment_in["y"].append(temp[i])

            # If we were building an "out" segment, save it and start new
            if len(current_segment_out["x"]) > 0:
                segments_out.append(current_segment_out.copy())
                current_segment_out = {"x": [], "y": []}
                # Add this point to start the new "out" segment (for
                # continuity)
                current_segment_out["x"].append(index[i])
                current_segment_out["y"].append(temp[i])
        else:
            # Add to "out of comfort" segment
            current_segment_out["x"].append(index[i])
            current_segment_out["y"].append(temp[i])

            # If we were building an "in" segment, save it and start new
            if len(current_segment_in["x"]) > 0:
                segments_in.append(current_segment_in.copy())
                current_segment_in = {"x": [], "y": []}
                # Add this point to start the new "in" segment (for continuity)
                current_segment_in["x"].append(index[i])
                current_segment_in["y"].append(temp[i])

    # Save last segments
    if len(current_segment_in["x"]) > 0:
        segments_in.append(current_segment_in)
    if len(current_segment_out["x"]) > 0:
        segments_out.append(current_segment_out)

    # Add "in comfort" segments (blue)
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
                hovertemplate='Indoor:  %{y:. 2f}°C<extra></extra>',
                legendgroup='indoor_in',
            ),
            **trace_kwargs,
        )

    # Add "out of comfort" segments (red)
    for i, seg in enumerate(segments_out):
        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode='lines',
                name=(
                    'Indoor temp (out of comfort)' if (show_legend and i == 0) else None
                ),
                showlegend=(show_legend and i == 0),
                line=dict(color='#d62728', width=1.5),
                hovertemplate='Indoor: %{y:.2f}°C (OUT OF COMFORT)<extra></extra>',
                legendgroup='indoor_out',
            ),
            **trace_kwargs,
        )


def plot_case_temperatures(
    case_id: int, case_cfg: Dict, output_dir: Path, daily_date: pd.Timestamp
) -> None:
    """Plot interactive indoor air temperatures and setpoints for a given case."""
    monitor_dir = ROOT / case_cfg["dir"] / "episode-20" / "monitor"
    obs_path = monitor_dir / "observations.csv"
    infos_path = monitor_dir / "infos.csv"

    obs = pd.read_csv(obs_path)
    infos = pd.read_csv(infos_path)

    min_len = min(len(obs), len(infos))
    obs = obs.iloc[:min_len].reset_index(drop=True)
    infos = infos.iloc[:min_len].reset_index(drop=True)

    dt_index = build_datetime_index(infos, base_year=2026)
    obs.index = dt_index

    # Temporal crop:  from mid-November to mid-March
    start = datetime(2026, 11, 15)
    end = datetime(2027, 3, 15, 23, 55)
    mask = (obs.index >= start) & (obs.index <= end)
    obs = obs.loc[mask]

    # Masks for daily, weekly and monthly windows
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

    # Create multi-panel figure for all rooms
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
        title=f'{case_cfg["name"]} – {case_cfg["agent"]} – All Rooms',
        height=900,
        template="plotly_white",
        hovermode=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / f'case{case_id}_temperatures.html')

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

        room_dir = output_dir / f"case{case_id}" / room_slug
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
            title=f'{case_cfg["name"]} – {room_title}',
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
                title=(
                    f'{case_cfg["name"]} – {room_title} '
                    f'(daily: {daily_date.date()})'
                ),
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )

            fig_d.write_html(room_dir / "daily_temperature.html")

        # Weekly zoom
        if not obs_week.empty:
            fig_w = go.Figure()
            add_temperature_traces(
                fig_w, obs_week, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            fig_w.update_layout(
                title=(
                    f'{case_cfg["name"]} – {room_title} (weekly: '
                    f'{week_start.date()} to {week_end.date()})'
                ),
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            )

            fig_w.write_html(room_dir / "weekly_temperature.html")

        # Monthly zoom
        if not obs_month.empty:
            fig_m = go.Figure()
            add_temperature_traces(
                fig_m, obs_month, temp_col, SETPOINT_COLS[i], show_legend=True
            )

            fig_m.update_layout(
                title=(
                    f'{case_cfg["name"]} – {room_title} (monthly: '
                    f'{month_start.date()} to {month_end.date()})'
                ),
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                hovermode=False,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis=dict(
                    rangeslider=dict(
                        visible=True),
                    type="date"),
            )

            fig_m.write_html(room_dir / "monthly_temperature.html")


def main() -> None:
    """Main entry point."""
    output_dir = ROOT / "eval_plots_interactive"

    # 1) Comparison of evaluation metrics between agents
    plot_progress_comparison(output_dir)

    # 1b) Violin plots for flow rate and water temperature per case
    plot_flow_and_water_violin(output_dir)

    # 2) Time series of temperatures and setpoints per room for each case
    first_case = next(iter(CASE_CONFIG.values()))
    first_infos_path = ROOT / \
        first_case["dir"] / "episode-20" / "monitor" / "infos.csv"
    first_infos = pd.read_csv(first_infos_path)
    first_dt_index = build_datetime_index(first_infos, base_year=2026)

    first_mask = (first_dt_index >= datetime(2026, 11, 15)) & (
        first_dt_index <= datetime(2027, 3, 15, 23, 55)
    )
    valid_dates = pd.to_datetime(
        first_dt_index[first_mask]).normalize().unique()

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

    for case_id, cfg in CASE_CONFIG.items():
        plot_case_temperatures(
            case_id, cfg, output_dir, cast(
                pd.Timestamp, daily_date))

    print(f"Interactive plots saved in:  {output_dir}")


if __name__ == "__main__":
    main()
