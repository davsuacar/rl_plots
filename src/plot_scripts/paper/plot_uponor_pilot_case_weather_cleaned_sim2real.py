from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import plotly.io as pio
from utils.plot_functions.plot_functions import (
    compute_crf_daily_stats,
    filer_interval,
    mean_variable,
    plot_action_distribution,
    plot_bar,
    plot_bar_means_by_zones,
    plot_bar_with_std,
    plot_comfort_energy_balance,
    plot_control,
    plot_dfs_boxplot,
    plot_dfs_line,
    plot_energy_savings,
    plot_episode_reward_terms_timestep,
    plot_heat_work,
    plot_mean_energy_savings,
    plot_case_temperatures,
    plot_smoothed_signal,
    plot_training_reward_terms_progression,
    plot_dfs_bar_grouped_by_month,
    safe_read_csv,
    save_figure,
)

# =============================================================================
# CONFIG — CSV Grafana limpio (weather)
# =============================================================================
# Misma tubería que plot_uponor_pilot_case_sim2real_alcorcon_history.py; la entrada
# es el export weather limpio (clean_grafana_data.py): zonas living-kitchen, etc.

REPO_ROOT = Path(__file__).resolve().parents[4]
WEATHER_CSV = REPO_ROOT / (
    'data/paper/data/case_study/sim2real/'
    'weather-2026-04-20 10_48_28_cleaned.csv'
)
EXPERIMENT_LABEL = 'weather_grafana_cleaned'

# Progress del entrenamiento / evaluation progress: no aplican a este CSV → vacío.
TRAINING_PROGRESS_PATHS: dict[str, str] = {}

names_reference: list[str] = []
names_comparison: list[str] = []
combination_size = len(names_reference) * len(names_comparison)

# Sin filtro fijo: el rango temporal sale del propio historial (timestamps reales).
FILTER_INTERVAL: tuple[str, str] | None = None
# Se fija tras cargar datos (mediana temporal).
ZONE_TEMP_PLOT_DAILY_DATE = pd.Timestamp('2000-01-01')
TEMPERATURE_THRESHOLD = 1.0
SMOOTH_WINDOW = 1

# =============================================================================
# CONFIG — OUTPUT DIRECTORIES (subcarpetas por tipo de gráfico)
# =============================================================================

OUTPUT_BASE = REPO_ROOT / 'data/paper/plots/case_study/deployment_weather_efficiency/'
OUTPUT_PROGRESS = OUTPUT_BASE / 'progress'
OUTPUT_ZONE_TEMPERATURES = OUTPUT_BASE / 'zone_temperatures'
OUTPUT_TEMP_VS_FLOW = OUTPUT_BASE / 'temp_vs_flow'
OUTPUT_HEAT_PUMP_AND_CONTROL = OUTPUT_BASE / 'heat_pump_and_control'
OUTPUT_HEAT_WORK = OUTPUT_BASE / 'heat_work'
OUTPUT_MEANS = OUTPUT_BASE / 'means'
OUTPUT_MEANS_GENERAL = OUTPUT_MEANS / 'general'
OUTPUT_MEANS_MONTH = OUTPUT_MEANS / 'month'
OUTPUT_SAVINGS = OUTPUT_BASE / 'savings'
OUTPUT_ACTION_DISTRIBUTION = OUTPUT_BASE / 'action_distribution'
OUTPUT_BOXPLOTS = OUTPUT_BASE / 'boxplots'
OUTPUT_REWARD_BALANCE = OUTPUT_BASE / 'reward_balance'
# Medias/resumen por episodio o por entrenamiento (progress.csv)
OUTPUT_REWARD_BALANCE_SUMMARY = OUTPUT_REWARD_BALANCE / 'summary'
# Evolución timestep a timestep dentro de un episodio (infos)
OUTPUT_REWARD_BALANCE_PER_TIMESTEP = OUTPUT_REWARD_BALANCE / 'per_timestep'

for _d in (
    OUTPUT_BASE,
    OUTPUT_PROGRESS,
    OUTPUT_ZONE_TEMPERATURES,
    OUTPUT_TEMP_VS_FLOW,
    OUTPUT_HEAT_PUMP_AND_CONTROL,
    OUTPUT_HEAT_WORK,
    OUTPUT_MEANS,
    OUTPUT_MEANS_GENERAL,
    OUTPUT_MEANS_MONTH,
    OUTPUT_SAVINGS,
    OUTPUT_ACTION_DISTRIBUTION,
    OUTPUT_BOXPLOTS,
    OUTPUT_REWARD_BALANCE,
    OUTPUT_REWARD_BALANCE_SUMMARY,
    OUTPUT_REWARD_BALANCE_PER_TIMESTEP,
):
    _d.mkdir(parents=True, exist_ok=True)

# Mejor calidad al exportar PNG (scale=2 en save_figure)
pio.defaults.default_scale = 2

# =============================================================================
# CONFIG — VARIABLES (zonas, columnas, umbrales, colores)
# =============================================================================
# Orden alineado con el modelo Uponor / IDs físicos en el CSV:
# f0_livroom-kitchen, f0_bathroom, f1_bed1..3, f1_secondary_bathroom, f1_main_bathroom.
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
# En el CSV Grafana, el estado del actuador por zona equivale al caudal/válvula usado en los plots.
flow_variables = [
    'flow_rate_living_kitchen',
    'flow_rate_bathroom_lobby',
    'flow_rate_bed1',
    'flow_rate_bed2',
    'flow_rate_bed3',
    'flow_rate_bathroom_corridor',
    'flow_rate_bathroom_dressing',
]
inlet_temperature_variables = [
    'radiant_hvac_inlet_temperature_living_kitchen',
    'radiant_hvac_inlet_temperature_bathroom_lobby',
    'radiant_hvac_inlet_temperature_bed1',
    'radiant_hvac_inlet_temperature_bed2',
    'radiant_hvac_inlet_temperature_bed3',
    'radiant_hvac_inlet_temperature_bathroom_corridor',
    'radiant_hvac_inlet_temperature_bathroom_dressing',
]
outlet_temperature_variables = [
    'radiant_hvac_outlet_temperature_living_kitchen',
    'radiant_hvac_outlet_temperature_bathroom_lobby',
    'radiant_hvac_outlet_temperature_bed1',
    'radiant_hvac_outlet_temperature_bed2',
    'radiant_hvac_outlet_temperature_bed3',
    'radiant_hvac_outlet_temperature_bathroom_corridor',
    'radiant_hvac_outlet_temperature_bathroom_dressing',
]

heat_pump_variables = ['crf', 'plr_current']

action_distribution_variables = list(flow_variables) + ['water_temperature']

energy_variable = 'heat_source_electricity_rate'
water_temperature_variable = 'water_temperature'
heat_pump_outlet_variable = 'heat_source_load_side_outlet_temp'

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


def _slugify(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^a-z0-9]+', '_', t)
    return t.strip('_')


# =============================================================================
# DATA LOADING — CSV Grafana weather (cleaned)
# =============================================================================

# (prefijo CSV, temp interna, setpoint interno, flow interno).
# flow_* se rellena desde {prefijo}_actuator_status (estado actuador ≈ válvula/caudal).
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
    if 'heatpump_return_temperature' in df.columns:
        out['heat_source_load_side_outlet_temp'] = pd.to_numeric(
            df['heatpump_return_temperature'], errors='coerce'
        )
    if 'valid_time_local' in df.columns:
        out['datetime'] = pd.to_datetime(df['valid_time_local'], utc=True, errors='coerce')
    elif 'time' in df.columns:
        out['datetime'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='s', errors='coerce')
    elif {'year', 'month', 'day'}.issubset(df.columns):
        h = df['hour'] if 'hour' in df.columns else 0
        out['datetime'] = pd.to_datetime(
            dict(year=df['year'], month=df['month'], day=df['day'], hour=h),
            errors='coerce',
        )
    else:
        raise ValueError(
            f"No se pudo inferir datetime en {path}: faltan valid_time_local, time o year/month/day."
        )
    out = out.loc[out['datetime'].notna()].sort_values('datetime').reset_index(drop=True)
    return out


_raw = _load_weather_cleaned(WEATHER_CSV)
if _raw.empty:
    raise SystemExit(f"No hay datos en {WEATHER_CSV}")

unified = {EXPERIMENT_LABEL: _raw}

training_progress = {
    key: safe_read_csv(path) for key, path in TRAINING_PROGRESS_PATHS.items()
}
training_progress = {
    k: v for k, v in training_progress.items() if v is not None and not v.empty
}
evaluation_progress: dict[str, pd.DataFrame] = {}

# =============================================================================
# PREPROCESS — filtro opcional y rango temporal real
# =============================================================================

for key in unified:
    if FILTER_INTERVAL is not None:
        unified[key] = filer_interval(  # type: ignore[assignment]
            unified[key], FILTER_INTERVAL[0], FILTER_INTERVAL[1]
        )

_df0 = unified[EXPERIMENT_LABEL]
ZONE_TEMP_PLOT_DAILY_DATE = pd.to_datetime(_df0['datetime']).median().normalize()
PERIOD_START = pd.to_datetime(_df0['datetime'].min()).to_pydatetime()
PERIOD_END = pd.to_datetime(_df0['datetime'].max()).to_pydatetime()

df_num = len(unified)

mean_temp_violation_dict = {
    key: mean_variable(df, variable='total_temperature_violation')
    for key, df in unified.items()
    if 'total_temperature_violation' in df.columns
}
mean_energy_consumption_dict = {
    key: mean_variable(df, variable=energy_variable)
    for key, df in unified.items()
    if energy_variable in df.columns
}

# CRF: media (y std) de transiciones encendido/apagado y de encendidos estrictos por día
crf_trans_mean_dict = {}
crf_trans_std_dict = {}
crf_ons_mean_dict = {}
crf_ons_std_dict = {}
for key, df in unified.items():
    stats = compute_crf_daily_stats(df, crf_col='crf', datetime_col='datetime')
    if stats is not None:
        mean_trans, std_trans, mean_ons, std_ons = stats
        crf_trans_mean_dict[key] = mean_trans
        crf_trans_std_dict[key] = std_trans
        crf_ons_mean_dict[key] = mean_ons
        crf_ons_std_dict[key] = std_ons

# =============================================================================
# FIGURES — Training progress (solo si TRAINING_PROGRESS_PATHS tiene rutas)
# =============================================================================

if training_progress:
    _n = len(training_progress)
    fig = plot_dfs_line(
        df_dict=training_progress,
        variable_name='mean_reward',
        colors=colors[:_n],
    )
    fig.update_layout(title=None, xaxis_title='Episode', yaxis_title='Mean reward')
    save_figure(
        fig, OUTPUT_PROGRESS / 'training_progress', width=1200, height=700, scale=2
    )

# =============================================================================
# FIGURES — Temperaturas (plot_case_temperatures: rejilla + por habitación y recortes)
# =============================================================================

_zones = list(
    zip(temperature_variables, setpoint_variables, zone_names, strict=True)
)
for key, df in unified.items():
    model_dir = OUTPUT_ZONE_TEMPERATURES / _slugify(key)
    _kwargs = dict(
        df=df,
        zones=_zones,
        output_dir=model_dir,
        daily_date=ZONE_TEMP_PLOT_DAILY_DATE,
        case_id=0,
        summary_title=key,
        threshold=TEMPERATURE_THRESHOLD,
        outdoor_temp_var=None,
        png_width=1200,
        png_height_single=500,
        png_scale=2,
        temp_colors=list(colors[: len(zone_names)]),
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    plot_case_temperatures(**_kwargs)

# =============================================================================
# FIGURES — Temperature vs flow (control)
# =============================================================================

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
    save_figure(
        fig,
        OUTPUT_TEMP_VS_FLOW / _slugify(key),
        width=1200,
        height=700,
        scale=2,
    )

# =============================================================================
# FIGURES — Heat pump and control signals (per model)
# =============================================================================

for key, df in unified.items():
    model_dir = OUTPUT_HEAT_PUMP_AND_CONTROL / _slugify(key)
    model_dir.mkdir(parents=True, exist_ok=True)

    vars_to_plot = (
        [energy_variable] + list(flow_variables) + [water_temperature_variable]
    )

    for i, var in enumerate(vars_to_plot):
        if var not in df.columns:
            print(
                f'⚠️ {key}: columna "{var}" no disponible; se omite gráfica suavizada.'
            )
            continue

        _y_title = _slugify(var).replace('_', ' ').capitalize()
        if var == water_temperature_variable:
            _y_title = f'{_y_title} ( ºC)'
        fig = plot_smoothed_signal(
            df=df,
            variable=var,
            datetime_col='datetime',
            window=SMOOTH_WINDOW,
            color=colors[i % len(colors)],
            title=None,
            yaxis_title=_y_title
        )
        save_figure(
            fig,
            model_dir / f'heat_pump_control_{_slugify(var)}',
            width=1200,
            height=500,
            scale=2,
        )

# =============================================================================
# FIGURES — Heat work (requested vs real outlet temperature)
# =============================================================================

for key, df in unified.items():
    model_dir = OUTPUT_HEAT_WORK / _slugify(key)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prefer direct variable name; fallback to action-suffixed naming if needed.
    water_col = (
        water_temperature_variable
        if water_temperature_variable in df.columns
        else f'{water_temperature_variable}_action'
    )
    outlet_col = heat_pump_outlet_variable

    if water_col not in df.columns or outlet_col not in df.columns:
        print(
            f'⚠️ {key}: faltan columnas para heat_work '
            f'({water_col}, {outlet_col}); se omite gráfica.'
        )
        continue

    fig = plot_heat_work(
        df=df,
        requested_var=water_col,
        outlet_var=outlet_col,
        datetime_col='datetime',
        requested_name='Water temperature setpoint',
        outlet_name='Heat source outlet temperature',
        title=None,
    )
    save_figure(
        fig,
        model_dir / 'heat_work_requested_vs_outlet_temperature',
        width=1200,
        height=500,
        scale=2,
    )

# =============================================================================
# FIGURES — Medias (barras) y power demand por mes
# =============================================================================

if mean_temp_violation_dict:
    fig = plot_bar(mean_temp_violation_dict, bar_colors=colors[: len(mean_temp_violation_dict)])
    fig.update_layout(
        title=None,
        xaxis_title='Model',
        yaxis_title='Mean episodic temperature violation (ºC)',
    )
    save_figure(
        fig, OUTPUT_MEANS_GENERAL / 'mean_temp_violations', width=1200, height=600, scale=2
    )
else:
    print('⚠️ Sin columna total_temperature_violation; se omite mean_temp_violations.')

if mean_energy_consumption_dict:
    fig = plot_bar(mean_energy_consumption_dict, bar_colors=colors[: len(mean_energy_consumption_dict)])
    fig.update_layout(
        title=None,
        xaxis_title='Model',
        yaxis_title='Mean episodic power consumption (W)',
    )
    save_figure(
        fig, OUTPUT_MEANS_GENERAL / 'mean_power_demand', width=1200, height=600, scale=2
    )
else:
    print(f'⚠️ Sin columna {energy_variable}; se omite mean_power_demand.')

if all(energy_variable in df.columns for df in unified.values()):
    fig = plot_dfs_bar_grouped_by_month(
        unified,
        energy_variable,
        colors=colors[:df_num],
    )
    fig.update_layout(title=None, xaxis_title='Date', yaxis_title='Mean episodic power demand (W)')
    save_figure(
        fig, OUTPUT_MEANS_MONTH / 'month_power_demand', width=1200, height=600, scale=2
    )
else:
    print(f'⚠️ Sin {energy_variable} en todos los runs; se omite month_power_demand.')

if all('total_temperature_violation' in df.columns for df in unified.values()):
    fig = plot_dfs_bar_grouped_by_month(
        unified,
        'total_temperature_violation',
        colors=colors[:df_num],
    )
    fig.update_layout(title=None, xaxis_title='Date', yaxis_title='Mean episodic temperature violation (°C)')
    save_figure(
        fig,
        OUTPUT_MEANS_MONTH / 'month_temperature_violation',
        width=1200,
        height=600,
        scale=2,
    )
else:
    print('⚠️ Sin total_temperature_violation; se omite month_temperature_violation.')

# Inlet/outlet: medias por zona por experimento (barras agrupadas)
if any(
    any(v in df.columns for v in inlet_temperature_variables) for df in unified.values()
):
    fig = plot_bar_means_by_zones(
        unified,
        inlet_temperature_variables,
        zone_names,
        colors=colors[:df_num],
    )
    fig.update_layout(
        title=None, yaxis_title='Mean inlet (°C)'
    )
    save_figure(
        fig,
        OUTPUT_MEANS_GENERAL / 'mean_inlet_temperature_by_zone',
        width=1200,
        height=600,
        scale=2,
    )
if any(
    any(v in df.columns for v in outlet_temperature_variables)
    for df in unified.values()
):
    fig = plot_bar_means_by_zones(
        unified,
        outlet_temperature_variables,
        zone_names,
        colors=colors[:df_num],
    )
    fig.update_layout(
        title=None, yaxis_title='Mean outlet (°C)'
    )
    save_figure(
        fig,
        OUTPUT_MEANS_GENERAL / 'mean_outlet_temperature_by_zone',
        width=1200,
        height=600,
        scale=2,
    )

# CRF: media ± std de transiciones/día y de encendidos estrictos/día
if crf_trans_mean_dict:
    fig = plot_bar_with_std(
        crf_trans_mean_dict,
        crf_trans_std_dict,
        bar_colors=colors[: len(crf_trans_mean_dict)],
    )
    fig.update_layout(
        title=None,
        yaxis_title='Transitions per day',
        xaxis_title='Model',
    )
    save_figure(
        fig,
        OUTPUT_MEANS_GENERAL / 'crf_mean_transitions_per_day',
        width=1200,
        height=600,
        scale=2,
    )
if crf_ons_mean_dict:
    fig = plot_bar_with_std(
        crf_ons_mean_dict,
        crf_ons_std_dict,
        bar_colors=colors[: len(crf_ons_mean_dict)],
    )
    fig.update_layout(
        title=None,
        yaxis_title='Ons per day',
        xaxis_title='Model',
    )
    save_figure(
        fig,
        OUTPUT_MEANS_GENERAL / 'crf_mean_ons_per_day',
        width=1200,
        height=600,
        scale=2,
    )

# =============================================================================
# FIGURES — Energy savings (media global y por mes; solo si hay referencia y comparación)
# =============================================================================

if names_reference and names_comparison:
    fig = plot_mean_energy_savings(
        data=unified,
        names_reference=names_reference,
        names_comparison=names_comparison,
        variable=energy_variable,
        colors=colors[1 : combination_size + 1],
    )
    fig.update_layout(title=None)
    save_figure(fig, OUTPUT_SAVINGS / 'mean_savings', width=1200, height=600, scale=2)

    fig = plot_energy_savings(
        data=unified,
        names_reference=names_reference,
        names_comparison=names_comparison,
        variable=energy_variable,
        colors=colors[1 : combination_size + 1],
    )
    save_figure(
        fig, OUTPUT_SAVINGS / 'month_energy_savings', width=1200, height=600, scale=2
    )
else:
    print('⚠️ Sin referencia o comparación en unified; se omiten gráficas de ahorro.')

# =============================================================================
# FIGURES — Action distribution (violines)
# =============================================================================

for var in action_distribution_variables:
    if not any(var in df.columns for df in unified.values()):
        print(f'⚠️ Variable "{var}" ausente en datos; se omite action_distribution.')
        continue
    fig = plot_action_distribution(
        unified,
        var,
        colors=colors[:df_num],
    )
    fig.update_layout(
        title=None,
        showlegend=False,
    )
    save_figure(
        fig,
        OUTPUT_ACTION_DISTRIBUTION / f'distribution_{_slugify(var)}',
        width=1200,
        height=600,
        scale=2,
    )

# =============================================================================
# FIGURES — Boxplots (evaluation progress)
# =============================================================================

if not evaluation_progress:
    print('⚠️ No hay evaluation_progress (progress.csv en monitor); se omiten boxplots.')
else:
    fig = plot_dfs_boxplot(
        evaluation_progress,
        'comfort_violation_time(%)',
        colors=colors[:df_num],
        yaxis_title='Episodic comfort violation time (%)',
        xaxis_title='Model'
    )
    save_figure(
        fig,
        OUTPUT_BOXPLOTS / 'comfort_violation_time',
        width=1200,
        height=600,
        scale=2,
    )

    fig = plot_dfs_boxplot(
        evaluation_progress,
        'mean_temperature_violation',
        colors=colors[:df_num],
    )
    fig.update_layout(
        title=None,
        yaxis_title='Episodic temperature violation (ºC)',
        xaxis_title='Model'
    )
    save_figure(
        fig,
        OUTPUT_BOXPLOTS / 'temperature_violation',
        width=1200,
        height=600,
        scale=2,
    )

    fig = plot_dfs_boxplot(
        evaluation_progress,
        'mean_power_demand',
        colors=colors[:df_num],
        yaxis_title='Episodic power demand (W)',
        xaxis_title='Model'
    )
    fig.update_layout(title=None, yaxis_title='Power demand (W)')
    save_figure(
        fig,
        OUTPUT_BOXPLOTS / 'power_demand',
        width=1200,
        height=600,
        scale=2,
    )

# =============================================================================
# FIGURES — Reward balance (comfort vs energy term from progress.csv)
# =============================================================================

REWARD_COMFORT_COL = 'mean_reward_comfort_term'
REWARD_ENERGY_COL = 'mean_reward_energy_term'

if not evaluation_progress:
    print('⚠️ No hay evaluation_progress; se omite gráfica de equilibrio reward.')
else:
    progress_with_terms = {
        k: v
        for k, v in evaluation_progress.items()
        if REWARD_COMFORT_COL in v.columns and REWARD_ENERGY_COL in v.columns
    }
    if not progress_with_terms:
        print(
            f'⚠️ Ningún progress.csv tiene "{REWARD_COMFORT_COL}" y "{REWARD_ENERGY_COL}"; '
            'se omite gráfica de equilibrio reward.'
        )
    else:
        # Media y desviación típica sobre episodios (filas del progress) por experimento
        balance_comfort_means = {
            k: v[REWARD_COMFORT_COL].mean() for k, v in progress_with_terms.items()
        }
        balance_comfort_stds = {
            k: v[REWARD_COMFORT_COL].std() for k, v in progress_with_terms.items()
        }
        balance_energy_means = {
            k: v[REWARD_ENERGY_COL].mean() for k, v in progress_with_terms.items()
        }
        balance_energy_stds = {
            k: v[REWARD_ENERGY_COL].std() for k, v in progress_with_terms.items()
        }
        # NaN std cuando hay un solo episodio
        for d in (balance_comfort_stds, balance_energy_stds):
            for k in list(d.keys()):
                val = d[k]
                if not (isinstance(val, (int, float)) and math.isfinite(val)):
                    d[k] = 0.0

        fig = plot_comfort_energy_balance(
            balance_comfort_means,
            balance_comfort_stds,
            balance_energy_means,
            balance_energy_stds,
            experiment_names=list(progress_with_terms.keys()),
            color_comfort=colors[0],
            color_energy=colors[1],
        )
        fig.update_layout(
            title=None,
            xaxis_tickangle=-25,
        )
        save_figure(
            fig,
            OUTPUT_REWARD_BALANCE_SUMMARY / 'comfort_vs_energy_term',
            width=1200,
            height=600,
            scale=2,
        )

# =============================================================================
# FIGURES — Training reward terms progression (comfort vs energy durante entrenamiento)
# =============================================================================

if training_progress:
    for run_name, df_progress in training_progress.items():
        if (
            REWARD_COMFORT_COL not in df_progress.columns
            or REWARD_ENERGY_COL not in df_progress.columns
        ):
            print(
                f'⚠️ {run_name}: progress sin "{REWARD_COMFORT_COL}" o "{REWARD_ENERGY_COL}"; '
                'se omite gráfica de progresión de términos.'
            )
            continue
        fig = plot_training_reward_terms_progression(
            df_progress,
            episode_col='episode_num',
            comfort_col=REWARD_COMFORT_COL,
            energy_col=REWARD_ENERGY_COL,
            std_comfort_col='std_reward_comfort_term',
            std_energy_col='std_reward_energy_term',
            title=None,
            color_comfort=colors[0],
            color_energy=colors[1],
            show_std_band=True,
        )
        save_figure(
            fig,
            OUTPUT_REWARD_BALANCE_SUMMARY
            / f'training_progression_{_slugify(run_name)}',
            width=1200,
            height=600,
            scale=2,
        )

# =============================================================================
# FIGURES — Reward terms per timestep (evolución dentro de un episodio, infos)
# =============================================================================

INFO_COMFORT_COL = 'comfort_term'
INFO_ENERGY_COL = 'energy_term'
# Suavizado opcional para series muy largas (ej. ventana en número de timesteps)
REWARD_TERMS_SMOOTH_WINDOW = 12

for key, df in unified.items():
    if INFO_COMFORT_COL not in df.columns or INFO_ENERGY_COL not in df.columns:
        print(
            f'⚠️ {key}: infos sin "{INFO_COMFORT_COL}" o "{INFO_ENERGY_COL}"; '
            'se omite gráfica por timestep.'
        )
        continue
    fig = plot_episode_reward_terms_timestep(
        df,
        datetime_col='datetime',
        comfort_col=INFO_COMFORT_COL,
        energy_col=INFO_ENERGY_COL,
        title=None,
        color_comfort=colors[0],
        color_energy=colors[1],
        smooth_window=REWARD_TERMS_SMOOTH_WINDOW,
    )
    save_figure(
        fig,
        OUTPUT_REWARD_BALANCE_PER_TIMESTEP
        / f'{_slugify(key)}_history_comfort_vs_energy',
        width=1200,
        height=600,
        scale=2,
    )