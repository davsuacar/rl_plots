#!/usr/bin/env python3
"""
Comparativa Weather (Grafana limpio) vs agente (historial sim2real), mismo estilo
que plot_uponor_pilot_case_by_algorithm.py (plot_functions + save_figure).

Ventanas:
  - Weather: 19–20 de febrero (año 2026, calendario local del timestamp).
  - Agente: 29–30 de marzo (2026).

Salida: data/paper/plots/case_study/deployment_agent_vs_weather/
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from utils.plot_functions.plot_functions import (
    plot_action_distribution,
    plot_bar,
    plot_bar_means_by_zones,
    plot_case_temperatures,
    plot_control,
    plot_smoothed_signal,
    safe_read_csv,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
WEATHER_CSV = REPO_ROOT / 'data/paper/data/case_study/sim2real/weather-2026-04-20 10_48_28_cleaned.csv'
AGENT_HISTORY_CSV = REPO_ROOT / (
    'data/paper/data/case_study/sim2real/ai-uponor_smatrix-alcorcon-lab_y07e51pj_history.csv'
)
OUTPUT_BASE = REPO_ROOT / 'data/paper/plots/case_study/deployment_agent_vs_weather'

WEATHER_LABEL = 'Weather (19–20 Feb 2026)'
AGENT_LABEL = 'Agent (29–30 Mar 2026)'

# Filtros calendario (año fijo 2026; día/mes en hora local del datetime)
WEATHER_FILTER = dict(year=2026, month=2, days=(19, 20))
AGENT_FILTER = dict(year=2026, month=3, days=(29, 30))

LOCAL_TZ = 'Europe/Madrid'

TEMPERATURE_THRESHOLD = 1.0
SMOOTH_WINDOW = 1

zone_names = [
    'Living-kitchen',
    'Bathroom-lobby',
    'Bedroom 1',
    'Bedroom 2',
    'Bedroom 3',
    'Bathroom-corridor',
    'Bathroom-dressing',
]
temperature_variables = [
    'air_temperature_living_kitchen',
    'air_temperature_bathroom_lobby',
    'air_temperature_bed1',
    'air_temperature_bed2',
    'air_temperature_bed3',
    'air_temperature_bathroom_corridor',
    'air_temperature_bathroom_dressing',
]
setpoint_variables = [
    'heating_setpoint_living_kitchen',
    'heating_setpoint_bathroom_lobby',
    'heating_setpoint_bed1',
    'heating_setpoint_bed2',
    'heating_setpoint_bed3',
    'heating_setpoint_bathroom_corridor',
    'heating_setpoint_bathroom_dressing',
]
flow_variables = [
    'flow_rate_living_kitchen',
    'flow_rate_bathroom_lobby',
    'flow_rate_bed1',
    'flow_rate_bed2',
    'flow_rate_bed3',
    'flow_rate_bathroom_corridor',
    'flow_rate_bathroom_dressing',
]

colors = [
    '#1ABC9C',
    '#3498DB',
    '#9B59B6',
    '#E74C3C',
    '#F1C40F',
    '#2ECC71',
    '#E67E22',
    '#95A5A6',
    '#34495E',
    '#D35400',
    '#16A085',
    '#8E44AD',
    '#C0392B',
    '#2980B9',
]

energy_variable = 'heat_source_electricity_rate'
water_temperature_variable = 'water_temperature'


def _slugify(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^a-z0-9]+', '_', t)
    return t.strip('_')


# --- Carga weather (misma lógica que plot_uponor_pilot_case_weather_cleaned_sim2real) ---
_WEATHER_ZONE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ('living-kitchen', 'air_temperature_living_kitchen', 'heating_setpoint_living_kitchen', 'flow_rate_living_kitchen'),
    ('bathroom_lobby', 'air_temperature_bathroom_lobby', 'heating_setpoint_bathroom_lobby', 'flow_rate_bathroom_lobby'),
    ('bedroom_1', 'air_temperature_bed1', 'heating_setpoint_bed1', 'flow_rate_bed1'),
    ('bedroom_2', 'air_temperature_bed2', 'heating_setpoint_bed2', 'flow_rate_bed2'),
    ('bedroom_3', 'air_temperature_bed3', 'heating_setpoint_bed3', 'flow_rate_bed3'),
    ('bathroom_corridor', 'air_temperature_bathroom_corridor', 'heating_setpoint_bathroom_corridor', 'flow_rate_bathroom_corridor'),
    ('bathroom_dressing', 'air_temperature_bathroom_dressing', 'heating_setpoint_bathroom_dressing', 'flow_rate_bathroom_dressing'),
)


def _load_weather_cleaned(path: Path) -> pd.DataFrame:
    df = safe_read_csv(str(path))
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    for csv_prefix, tcol, scol, fcol in _WEATHER_ZONE_SPECS:
        amb = f'{csv_prefix}_ambient_temperature'
        sp = f'{csv_prefix}_setpoint'
        act = f'{csv_prefix}_actuator_status'
        if amb in df.columns:
            out[tcol] = pd.to_numeric(df[amb], errors='coerce')
        if sp in df.columns:
            out[scol] = pd.to_numeric(df[sp], errors='coerce')
        if act in df.columns:
            out[fcol] = pd.to_numeric(df[act], errors='coerce')
    if 'outdoor_dry_bulb_temp_celsius' in df.columns:
        out['outdoor_temperature'] = pd.to_numeric(df['outdoor_dry_bulb_temp_celsius'], errors='coerce')
    if 'heatpump_supply_temperature' in df.columns:
        out['water_temperature'] = pd.to_numeric(df['heatpump_supply_temperature'], errors='coerce')
    if 'heatpump_power' in df.columns:
        out['heat_source_electricity_rate'] = pd.to_numeric(df['heatpump_power'], errors='coerce')
    if 'valid_time_local' in df.columns:
        out['datetime'] = pd.to_datetime(df['valid_time_local'], errors='coerce')
    elif 'time' in df.columns:
        out['datetime'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='s', errors='coerce')
    elif {'year', 'month', 'day'}.issubset(df.columns):
        h = df['hour'] if 'hour' in df.columns else 0
        out['datetime'] = pd.to_datetime(
            dict(year=df['year'], month=df['month'], day=df['day'], hour=h),
            errors='coerce',
        )
    else:
        raise ValueError('Weather CSV: sin valid_time_local / time / year-month-day')
    out = out.loc[out['datetime'].notna()].sort_values('datetime').reset_index(drop=True)
    return out


# --- Carga agente (historial Ray / monitor) ---
_SIM2REAL_COLUMN_MAP: dict[str, str] = {
    'observation/air_temperature_f0_livroom-kitchen': 'air_temperature_living_kitchen',
    'observation/setpoint_f0_livroom-kitchen': 'heating_setpoint_living_kitchen',
    'observation/air_temperature_f0_bathroom': 'air_temperature_bathroom_lobby',
    'observation/setpoint_f0_bathroom': 'heating_setpoint_bathroom_lobby',
    'observation/air_temperature_f1_bed1': 'air_temperature_bed1',
    'observation/setpoint_f1_bed1': 'heating_setpoint_bed1',
    'observation/air_temperature_f1_bed2': 'air_temperature_bed2',
    'observation/setpoint_f1_bed2': 'heating_setpoint_bed2',
    'observation/air_temperature_f1_bed3': 'air_temperature_bed3',
    'observation/setpoint_f1_bed3': 'heating_setpoint_bed3',
    'observation/air_temperature_f1_secondary_bathroom': 'air_temperature_bathroom_corridor',
    'observation/setpoint_f1_secondary_bathroom': 'heating_setpoint_bathroom_corridor',
    'observation/air_temperature_f1_main_bathroom': 'air_temperature_bathroom_dressing',
    'observation/setpoint_f1_main_bathroom': 'heating_setpoint_bathroom_dressing',
    'info/valve_f0_livroom-kitchen': 'flow_rate_living_kitchen',
    'info/valve_f0_bathroom': 'flow_rate_bathroom_lobby',
    'info/valve_f1_bed1': 'flow_rate_bed1',
    'info/valve_f1_bed2': 'flow_rate_bed2',
    'info/valve_f1_bed3': 'flow_rate_bed3',
    'info/valve_f1_secondary_bathroom': 'flow_rate_bathroom_corridor',
    'info/valve_f1_main_bathroom': 'flow_rate_bathroom_dressing',
    'observation/heat_pump_electricity_rate': 'heat_source_electricity_rate',
    'info/t_supply': 'water_temperature',
    'observation/heat_pump_load_side_outlet_temp': 'heat_source_load_side_outlet_temp',
    'observation/outdoor_temperature': 'outdoor_temperature',
    'info/total_power_demand': 'total_power_demand',
    'info/total_temperature_violation': 'total_temperature_violation',
    'info/comfort_term': 'comfort_term',
    'info/energy_term': 'energy_term',
}


def _load_agent_history(path: Path) -> pd.DataFrame:
    df = safe_read_csv(str(path))
    if df.empty:
        return df
    rename = {k: v for k, v in _SIM2REAL_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    if '_timestamp' not in df.columns:
        raise ValueError(f'Se esperaba _timestamp en {path}')
    df['datetime'] = pd.to_datetime(df['_timestamp'], unit='s', utc=True)
    if 'water_temperature' not in df.columns and 'info/t_supply' in df.columns:
        df['water_temperature'] = df['info/t_supply']
    return df


def _to_local_series(ser: pd.Series) -> pd.Series:
    s = pd.to_datetime(ser, errors='coerce')
    if getattr(s.dt, 'tz', None) is None:
        return s.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(LOCAL_TZ)
    return s.dt.tz_convert(LOCAL_TZ)


def filter_calendar(df: pd.DataFrame, year: int, month: int, days: tuple[int, int]) -> pd.DataFrame:
    if df.empty or 'datetime' not in df.columns:
        return df.iloc[0:0].copy()
    loc = _to_local_series(df['datetime'])
    lo, hi = days
    mask = (loc.dt.year == year) & (loc.dt.month == month) & (loc.dt.day >= lo) & (loc.dt.day <= hi)
    return df.loc[mask].copy().reset_index(drop=True)


def add_hours_from_start(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty or 'datetime' not in out.columns:
        out['hours_from_start'] = pd.Series(dtype=float)
        return out
    t0 = out['datetime'].min()
    out['hours_from_start'] = (out['datetime'] - t0).dt.total_seconds() / 3600.0
    return out


def plot_compare_timeseries(
    unified: dict[str, pd.DataFrame],
    column: str,
    yaxis_title: str,
    output_stem: Path,
    *,
    x_col: str = 'hours_from_start',
    x_title: str = 'Hours from window start',
) -> None:
    fig = go.Figure()
    for i, (name, df) in enumerate(unified.items()):
        if column not in df.columns or x_col not in df.columns or df.empty:
            continue
        sub = df[[x_col, column]].dropna()
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[column],
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.85,
            )
        )
    fig.update_layout(
        title=None,
        xaxis_title=x_title,
        yaxis_title=yaxis_title,
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=20, color='black'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
    )
    save_figure(fig, output_stem, width=1200, height=600, scale=2)


def main() -> None:
    pio.defaults.default_scale = 2

    out_dirs = (
        OUTPUT_BASE,
        OUTPUT_BASE / 'zone_temperatures',
        OUTPUT_BASE / 'temp_vs_flow',
        OUTPUT_BASE / 'compare_timeseries',
        OUTPUT_BASE / 'action_distribution',
        OUTPUT_BASE / 'means' / 'general',
        OUTPUT_BASE / 'heat_pump_and_control',
    )
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    df_w = _load_weather_cleaned(WEATHER_CSV)
    df_a = _load_agent_history(AGENT_HISTORY_CSV)
    if df_w.empty:
        raise SystemExit(f'Sin datos weather: {WEATHER_CSV}')
    if df_a.empty:
        raise SystemExit(f'Sin datos agente: {AGENT_HISTORY_CSV}')

    df_w = filter_calendar(df_w, **WEATHER_FILTER)
    df_a = filter_calendar(df_a, **AGENT_FILTER)

    if df_w.empty:
        raise SystemExit(
            f'Weather: 0 filas en {WEATHER_FILTER["month"]}/{WEATHER_FILTER["days"]} '
            f'{WEATHER_FILTER["year"]} (local {LOCAL_TZ}). Revisa fechas en el CSV.'
        )
    if df_a.empty:
        raise SystemExit(
            f'Agente: 0 filas en {AGENT_FILTER["month"]}/{AGENT_FILTER["days"]} '
            f'{AGENT_FILTER["year"]} (local {LOCAL_TZ}). Revisa fechas en el historial.'
        )

    unified = {
        WEATHER_LABEL: add_hours_from_start(df_w),
        AGENT_LABEL: add_hours_from_start(df_a),
    }

    _zones = list(zip(temperature_variables, setpoint_variables, zone_names, strict=True))

    # --- Temperaturas por zona (mismo flujo que by_algorithm: una carpeta por serie) ---
    for key, df in unified.items():
        model_dir = OUTPUT_BASE / 'zone_temperatures' / _slugify(key)
        d_med = pd.to_datetime(df['datetime']).median().normalize()
        p_start = pd.to_datetime(df['datetime'].min()).to_pydatetime()
        p_end = pd.to_datetime(df['datetime'].max()).to_pydatetime()
        plot_case_temperatures(
            df=df,
            zones=_zones,
            output_dir=model_dir,
            daily_date=d_med,
            case_id=0,
            summary_title=key,
            threshold=TEMPERATURE_THRESHOLD,
            outdoor_temp_var=None,
            png_width=1200,
            png_height_single=500,
            png_scale=2,
            temp_colors=list(colors[: len(zone_names)]),
            period_start=p_start,
            period_end=p_end,
        )

    # --- Temp vs flow (por fuente) ---
    for key, df in unified.items():
        _temps = [c for c in temperature_variables if c in df.columns]
        _flows = [c for c in flow_variables if c in df.columns]
        fig = plot_control(
            df=df,
            temperature_variables=_temps,
            flow_variables=_flows,
            names=[f'Temp {z}' for z, c in zip(zone_names, temperature_variables) if c in df.columns]
            + [f'Flow {z}' for z, c in zip(zone_names, flow_variables) if c in df.columns],
            colors=colors[: len(_temps) + len(_flows)],
            outdoor_temp_var='outdoor_temperature' if 'outdoor_temperature' in df.columns else None,
        )
        save_figure(fig, OUTPUT_BASE / 'temp_vs_flow' / _slugify(key), width=1200, height=700, scale=2)

    # --- Comparativa temporal alineada por horas desde el inicio de cada ventana ---
    compare_dir = OUTPUT_BASE / 'compare_timeseries'
    if 'outdoor_temperature' in df_w.columns or 'outdoor_temperature' in df_a.columns:
        plot_compare_timeseries(
            unified,
            'outdoor_temperature',
            'Outdoor temperature (°C)',
            compare_dir / 'compare_outdoor_temperature',
        )
    plot_compare_timeseries(
        unified,
        'water_temperature',
        'Supply / water temperature (°C)',
        compare_dir / 'compare_water_temperature',
    )
    if all(energy_variable in df.columns for df in unified.values()):
        plot_compare_timeseries(
            unified,
            energy_variable,
            'Power / heat rate (W)',
            compare_dir / f'compare_{_slugify(energy_variable)}',
        )
    for tcol, zname in zip(temperature_variables, zone_names, strict=True):
        plot_compare_timeseries(
            unified,
            tcol,
            f'Air temperature — {zname} (°C)',
            compare_dir / f'compare_{_slugify(tcol)}',
        )

    # --- Violines (misma función que by_algorithm) ---
    for var in list(temperature_variables) + list(flow_variables) + [water_temperature_variable, energy_variable]:
        if not any(var in df.columns for df in unified.values()):
            continue
        fig = plot_action_distribution(unified, var, colors=colors[: len(unified)])
        fig.update_layout(title=None, showlegend=False)
        save_figure(
            fig,
            OUTPUT_BASE / 'action_distribution' / f'distribution_{_slugify(var)}',
            width=1200,
            height=600,
            scale=2,
        )

    # --- Medias por zona (barras agrupadas Weather vs Agent) ---
    fig = plot_bar_means_by_zones(
        unified,
        temperature_variables,
        zone_names,
        colors=colors[: len(unified)],
    )
    fig.update_layout(title=None, yaxis_title='Mean air temperature (°C)')
    save_figure(fig, OUTPUT_BASE / 'means' / 'general' / 'mean_air_temperature_by_zone', width=1200, height=600, scale=2)

    fig_sp = plot_bar_means_by_zones(
        unified,
        setpoint_variables,
        zone_names,
        colors=colors[: len(unified)],
    )
    fig_sp.update_layout(title=None, yaxis_title='Mean heating setpoint (°C)')
    save_figure(fig_sp, OUTPUT_BASE / 'means' / 'general' / 'mean_setpoint_by_zone', width=1200, height=600, scale=2)

    # --- Media global de potencia (barra simple) ---
    pmeans = {}
    for name, df in unified.items():
        if energy_variable in df.columns and df[energy_variable].notna().any():
            pmeans[name] = float(df[energy_variable].mean())
    if pmeans:
        fig_p = plot_bar(pmeans, bar_colors=colors[: len(pmeans)])
        fig_p.update_layout(
            title=None,
            xaxis_title='Source',
            yaxis_title='Mean power / heat rate (W)',
        )
        save_figure(fig_p, OUTPUT_BASE / 'means' / 'general' / 'mean_power', width=1200, height=600, scale=2)

    # --- Señales suavizadas (por fuente, estilo pilot) ---
    hp_dir = OUTPUT_BASE / 'heat_pump_and_control' / 'by_source'
    hp_dir.mkdir(parents=True, exist_ok=True)
    for key, df in unified.items():
        sub = hp_dir / _slugify(key)
        sub.mkdir(parents=True, exist_ok=True)
        for var in (energy_variable, water_temperature_variable):
            if var not in df.columns:
                continue
            fig = plot_smoothed_signal(
                df=df,
                variable=var,
                datetime_col='datetime',
                window=SMOOTH_WINDOW,
                color=colors[0],
                title=None,
                yaxis_title=var.replace('_', ' '),
            )
            save_figure(fig, sub / f'control_{_slugify(var)}', width=1200, height=500, scale=2)

    print(f'Gráficas guardadas en: {OUTPUT_BASE}')


if __name__ == '__main__':
    main()
