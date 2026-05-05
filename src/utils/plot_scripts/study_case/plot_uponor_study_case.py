from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pandas as pd
import plotly.io as pio
from plot_functions import (
    add_datetime_column,
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
    plot_dfs_line_grouped_by_month,
    plot_energy_savings,
    plot_episode_reward_terms_timestep,
    plot_heat_work,
    plot_mean_energy_savings,
    plot_smoothed_signal,
    plot_temperature_one_zone,
    plot_training_reward_terms_progression,
    safe_read_csv,
    save_figure,
)

pio.templates.default = "plotly_white"

# =============================================================================
# CONFIG — DATA PATHS & EXPERIMENTS
# =============================================================================

DATA_DIR = '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1'
EPISODE = 2

# Clave -> nombre de carpeta bajo DATA_DIR (cada una con progress.csv y episode-N/monitor/)
EXPERIMENTS = {
    'ppo': 'Eplus-PPO-radiant_case1_heating-Example_2026-03-16_13:32-res1',
    'tqc': 'Eplus-TQC-radiant_case1_heating-Example_2026-03-19_08:28-res1',
    'sac': 'Eplus-SAC-radiant_case1_heating-Example_2026-03-16_13:42-res1',
    'rpo': 'Eplus-RecurrentPPO-radiant_case1_heating-Example_2026-03-16_13:41-res1',
}

# Progress del ENTRENAMIENTO (mean_reward por episodio). Rutas externas a DATA_DIR.
# Solo se pinta la gráfica de training progress si aquí hay al menos una entrada.
# Ej.: si solo SAC tiene log de entrenamiento en otra ruta: {'SAC': '/ruta/al/SAC/training/progress.csv'}
TRAINING_PROGRESS_PATHS: dict[str, str] = {
    # 'TQC': '/workspaces/sinergym/artifacts/TQC_humidity_fix/Sinergym_output/progress.csv',
}

names_reference = ['ppo']
names_comparison = [
    'tqc',
    'sac',
    'rpo',
]
combination_size = len(names_reference) * len(names_comparison)

FILTER_INTERVAL = ('2006-11-01 00:00:00', '2007-03-31 23:55:00')
TEMPERATURE_THRESHOLD = 1.0
SMOOTH_WINDOW = 1

# =============================================================================
# CONFIG — OUTPUT DIRECTORIES (subcarpetas por tipo de gráfico)
# =============================================================================

OUTPUT_BASE = Path('/home/jovyan/work/data/paper/plots/pilot_study/training_and_evaluation/caso_1')
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
zone_names = [
    'F0_Living-Kitchen',
    'F0_Bathroom-Lobby',
    'F1_Bedroom1',
    'F1_Bedroom2',
    'F1_Bedroom3',
    'F1_Bathroom-Corridor',
    'F1_Bathroom-Dressing',
]
temperature_variables = [
    'air_temperature_f0_living-kitchen',
    'air_temperature_f0_bathroom-lobby',
    'air_temperature_f1_bedroom1',
    'air_temperature_f1_bedroom2',
    'air_temperature_f1_bedroom3',
    'air_temperature_f1_bathroom-corridor',
    'air_temperature_f1_bathroom-dressing',
]
setpoint_variables = [
    'heating_setpoint_f0_living-kitchen',
    'heating_setpoint_f0_bathroom-lobby',
    'heating_setpoint_f1_bedroom1',
    'heating_setpoint_f1_bedroom2',
    'heating_setpoint_f1_bedroom3',
    'heating_setpoint_f1_bathroom-corridor',
    'heating_setpoint_f1_bathroom-dressing',
]
flow_variables = [
    'flow_rate_f0_living-kitchen',
    'flow_rate_f0_bathroom-lobby',
    'flow_rate_f1_bedroom1',
    'flow_rate_f1_bedroom2',
    'flow_rate_f1_bedroom3',
    'flow_rate_f1_bathroom-corridor',
    'flow_rate_f1_bathroom-dressing',
]
inlet_temperature_variables = [
    'radiant_hvac_inlet_temperature_f0_living-kitchen',
    'radiant_hvac_inlet_temperature_f0_bathroom-lobby',
    'radiant_hvac_inlet_temperature_f1_bedroom1',
    'radiant_hvac_inlet_temperature_f1_bedroom2',
    'radiant_hvac_inlet_temperature_f1_bedroom3',
    'radiant_hvac_inlet_temperature_f1_bathroom-corridor',
    'radiant_hvac_inlet_temperature_f1_bathroom-dressing',
]
outlet_temperature_variables = [
    'radiant_hvac_outlet_temperature_f0_living-kitchen',
    'radiant_hvac_outlet_temperature_f0_bathroom-lobby',
    'radiant_hvac_outlet_temperature_f1_bedroom1',
    'radiant_hvac_outlet_temperature_f1_bedroom2',
    'radiant_hvac_outlet_temperature_f1_bedroom3',
    'radiant_hvac_outlet_temperature_f1_bathroom-corridor',
    'radiant_hvac_outlet_temperature_f1_bathroom-dressing',
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
# DATA LOADING — rutas y construcción de diccionarios
# =============================================================================


def _eval_monitor_dir(key: str, run_folder: str) -> str:
    """Monitor de evaluación: DATA_DIR / run_folder / episode-{EPISODE} / monitor."""
    return os.path.join(DATA_DIR, run_folder, f'episode-{EPISODE}', 'monitor')


_monitor_dirs = {
    key: _eval_monitor_dir(key, run_folder) for key, run_folder in EXPERIMENTS.items()
}

# Training progress: solo si TRAINING_PROGRESS_PATHS tiene rutas (entrenamiento, puede ser otra ruta)
training_progress = {
    key: safe_read_csv(path) for key, path in TRAINING_PROGRESS_PATHS.items()
}
training_progress = {
    k: v for k, v in training_progress.items() if v is not None and not v.empty
}

evaluation_obs = {
    key: safe_read_csv(os.path.join(monitor_dir, 'observations.csv'))
    for key, monitor_dir in _monitor_dirs.items()
}
evaluation_actions = {
    key: safe_read_csv(os.path.join(monitor_dir, 'simulated_actions.csv')).add_suffix(
        '_action'
    )
    for key, monitor_dir in _monitor_dirs.items()
}
evaluation_infos = {
    key: safe_read_csv(os.path.join(monitor_dir, 'infos.csv'))
    for key, monitor_dir in _monitor_dirs.items()
}

# Unificar obs + actions + infos. Si algún CSV está vacío (ej. simulated_actions), ese bloque
# aporta 0 columnas y no se excluye el experimento; las columnas faltantes quedarán ausentes.
unified = {
    key: pd.concat(
        [evaluation_obs[key], evaluation_infos[key]],
        axis=1,
    )
    for key in evaluation_obs.keys()
}

# Progress por experimento: DATA_DIR / run_folder / progress.csv (el de cada run en Eplus-study)
evaluation_progress = {
    key: safe_read_csv(os.path.join(DATA_DIR, run_folder, 'progress.csv'))
    for key, run_folder in EXPERIMENTS.items()
}
evaluation_progress = {
    k: v for k, v in evaluation_progress.items() if v is not None and not v.empty
}

# =============================================================================
# PREPROCESS — datetime, filtro de intervalo, medias
# =============================================================================

for key in unified:
    unified[key] = add_datetime_column(unified[key])
    if FILTER_INTERVAL is not None:
        unified[key] = filer_interval(  # type: ignore[assignment]
            unified[key], FILTER_INTERVAL[0], FILTER_INTERVAL[1]
        )

df_num = len(unified)

mean_temp_violation_dict = {
    key: mean_variable(df, variable='total_temperature_violation')
    for key, df in unified.items()
}
mean_energy_consumption_dict = {
    key: mean_variable(df, variable='total_power_demand') for key, df in unified.items()
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
    fig.update_layout(title='', xaxis_title='Episode', yaxis_title='Mean Reward')
    save_figure(
        fig, OUTPUT_PROGRESS / 'training_progress', width=1200, height=700, scale=2
    )

# =============================================================================
# FIGURES — Temperaturas por zona (una gráfica por zona y por experimento)
# =============================================================================

for key, df in unified.items():
    model_dir = OUTPUT_ZONE_TEMPERATURES / _slugify(key)
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, zone_name in enumerate(zone_names):
        fig = plot_temperature_one_zone(
            df=df,
            temp_var=temperature_variables[i],
            setpoint_var=setpoint_variables[i],
            zone_name=zone_name,
            threshold=TEMPERATURE_THRESHOLD,
            temp_color=colors[i % len(colors)],
        )
        zone_slug = (
            temperature_variables[i].replace('air_temperature_', '').replace('-', '_')
        )
        fig.update_layout(
            title=f'{key} — {zone_name}',
            xaxis_title='',
            yaxis_title='Temperature (°C)',
        )
        save_figure(
            fig,
            model_dir / f'zone_{zone_slug}',
            width=1200,
            height=500,
            scale=2,
        )

# =============================================================================
# FIGURES — Temperature vs flow (control)
# =============================================================================

for key, df in unified.items():
    fig = plot_control(
        df=df,
        temperature_variables=temperature_variables,
        flow_variables=flow_variables,
        names=[f'Temp {z}' for z in zone_names] + [f'Flow {z}' for z in zone_names],
        colors=colors[: len(temperature_variables) + len(flow_variables)],
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

        fig = plot_smoothed_signal(
            df=df,
            variable=var,
            datetime_col='datetime',
            window=SMOOTH_WINDOW,
            color=colors[i % len(colors)],
            title=f'{key} — Heat pump and control: {var}',
            yaxis_title=var,
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
        title=f'{key} — Heat work: requested vs real outlet temperature',
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

fig = plot_bar(mean_temp_violation_dict, bar_colors=colors[:df_num])
fig.update_layout(
    title='Comparative mean temperature violation',
    xaxis_title='',
    yaxis_title='Mean Temperature Violation (ºC)',
)
save_figure(
    fig, OUTPUT_MEANS_GENERAL / 'mean_temp_violations', width=1200, height=600, scale=2
)

fig = plot_bar(mean_energy_consumption_dict, bar_colors=colors[:df_num])
fig.update_layout(
    title='Comparative power demand',
    xaxis_title='',
    yaxis_title='Mean power consumption (W)',
)
save_figure(
    fig, OUTPUT_MEANS_GENERAL / 'mean_power_demand', width=1200, height=600, scale=2
)

fig = plot_dfs_line_grouped_by_month(
    unified,
    energy_variable,
    colors=colors[:df_num],
)
fig.update_layout(title='', xaxis_title=None, yaxis_title='Power demand (W)')
save_figure(
    fig, OUTPUT_MEANS_MONTH / 'month_power_demand', width=1200, height=600, scale=2
)

fig = plot_dfs_line_grouped_by_month(
    unified,
    'total_temperature_violation',
    colors=colors[:df_num],
)
fig.update_layout(title='', xaxis_title=None, yaxis_title='Temperature violation (°C)')
save_figure(
    fig,
    OUTPUT_MEANS_MONTH / 'month_temperature_violation',
    width=1200,
    height=600,
    scale=2,
)

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
        title='Mean inlet temperature by zone', yaxis_title='Mean inlet (°C)'
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
        title='Mean outlet temperature by zone', yaxis_title='Mean outlet (°C)'
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
        title='CRF: mean on/off transitions per day',
        yaxis_title='Transitions per day',
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
        title='CRF: mean strict ons per day',
        yaxis_title='Ons per day',
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
    fig = plot_action_distribution(
        unified,
        var,
        colors=colors[:df_num],
    )
    fig.update_layout(
        title=f'{var} — Distribution',
        yaxis_title='',
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
    )
    fig.update_layout(title='Comfort violation time (% of episode)', yaxis_title='')
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
        title='Mean temperature violation',
        yaxis_title='Temperature violation (ºC)',
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
    )
    fig.update_layout(title='Mean power demand', yaxis_title='Power demand (W)')
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
            title='Reward balance: comfort term vs energy term (mean ± std across episodes)',
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
            title=f'{run_name} — Comfort vs energy term during training',
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
        title=f'{key} — Episode {EPISODE}: comfort vs energy term (per timestep)',
        color_comfort=colors[0],
        color_energy=colors[1],
        smooth_window=REWARD_TERMS_SMOOTH_WINDOW,
    )
    save_figure(
        fig,
        OUTPUT_REWARD_BALANCE_PER_TIMESTEP
        / f'{_slugify(key)}_episode_{EPISODE}_comfort_vs_energy',
        width=1200,
        height=600,
        scale=2,
    )
