from __future__ import annotations

import argparse
import functools
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import AbstractSet, Dict, List, Optional, Tuple

import pandas as pd
import plotly.io as pio
from utils.plot_functions.plot_functions import (
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
    plot_energy_savings,
    plot_episode_reward_terms_timestep,
    plot_heat_work,
    plot_mean_energy_savings,
    plot_case_temperatures,
    plot_smoothed_signal,
    plot_training_reward_terms_progression,
    plot_dfs_bar_grouped_by_month,
    safe_read_csv,
    save_figure as _save_figure,
)

# Todas las exportaciones usan el estilo paper (sample_plots / seaborn whitegrid + serif).
save_figure = functools.partial(_save_figure, paper_style=True)

# =============================================================================
# SHARED — export tuning & reward column names (iguales para todos los estudios)
# =============================================================================

# plot_case_temperatures alinea layout con el PNG (Kaleido): mismo ancho/alto evita que
# etiquetas de ejes respecto al HTML redimensionado.
ZONE_TEMP_EXPORT_WIDTH_PX = 1200
ZONE_TEMP_EXPORT_SINGLE_HEIGHT_PX = 500

REWARD_COMFORT_COL = 'mean_reward_comfort_term'
REWARD_ENERGY_COL = 'mean_reward_energy_term'
INFO_COMFORT_COL = 'comfort_term'
INFO_ENERGY_COL = 'energy_term'
REWARD_TERMS_SMOOTH_WINDOW = 12

# Paleta alineada con ``sample_plots`` (Tab10 + 5 tonos Tab20 emparejados / matplotlib).
_TAB10 = (
    '#1f77b4',  # tab:blue
    '#ff7f0e',  # tab:orange
    '#2ca02c',  # tab:green
    '#d62728',  # tab:red
    '#9467bd',  # tab:purple
    '#8c564b',  # tab:brown
    '#e377c2',  # tab:pink
    '#7f7f7f',  # tab:gray
    '#bcbd22',  # tab:olive
    '#17becf',  # tab:cyan
    '#aec7e8',  # tab20 light blue
    '#ffbb78',  # tab20 light orange
    '#98df8a',  # tab20 light green
    '#ff9896',  # tab20 light red
    '#c5b0d5',  # tab20 light purple
)
DEFAULT_COLORS = list(_TAB10)


@dataclass
class StudyPlotConfig:
    """Un conjunto de rutas de datos + variables + carpeta de salida para generar todas las figuras."""

    study_label: str
    output_base: Path
    data_dir: str
    episode: int
    experiments: Dict[str, str]
    training_progress_paths: Dict[str, str]
    zone_names: List[str]
    temperature_variables: List[str]
    setpoint_variables: List[str]
    flow_variables: List[str]
    inlet_temperature_variables: List[str]
    outlet_temperature_variables: List[str]
    names_reference: List[str] = field(default_factory=list)
    names_comparison: Optional[List[str]] = None
    filter_interval: Optional[Tuple[str, str]] = (
        '2006-11-01 00:00:00',
        '2007-03-31 23:55:00',
    )
    zone_temp_plot_daily_date: Optional[pd.Timestamp] = None
    temperature_threshold: float = 1.0
    smooth_window: int = 1
    energy_variable: str = 'heat_source_electricity_rate'
    water_temperature_variable: str = 'water_temperature'
    heat_pump_outlet_variable: str = 'heat_source_load_side_outlet_temp'
    colors: List[str] = field(default_factory=lambda: list(DEFAULT_COLORS))
    temp_zone_line_color: str = _TAB10[0]


def _slugify(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r'[^a-z0-9]+', '_', t)
    return t.strip('_')


def _heat_pump_control_y_axis_title(var: str, cfg: StudyPlotConfig) -> str:
    """Etiqueta del eje Y para ``heat_pump_and_control`` (flujo, energía, temperatura de agua)."""
    if var == cfg.energy_variable:
        return 'Electric power (W)'
    if var in cfg.flow_variables:
        return 'Flow rate (m³/s)'
    if var == cfg.water_temperature_variable:
        return 'Water temperature (°C)'
    label = var.replace('_', ' ')
    return label[:1].upper() + label[1:] if label else var


# Carpetas lógicas bajo ``output_base`` (argumento ``--outputs``). Coinciden con nombres de subcarpeta.
OUTPUT_SECTION_IDS = frozenset(
    {
        'progress',
        'zone_temperatures',
        'temp_vs_flow',
        'heat_pump_and_control',
        'heat_work',
        'means',
        'savings',
        'action_distribution',
        'boxplots',
        'reward_balance',
    }
)

# Atributos de ``out_dirs`` a crear por sección (``means`` y ``reward_balance`` agrupan varias rutas).
_SECTION_TO_OUTDIR_ATTRS: Dict[str, Tuple[str, ...]] = {
    'progress': ('progress',),
    'zone_temperatures': ('zone_temperatures',),
    'temp_vs_flow': ('temp_vs_flow',),
    'heat_pump_and_control': ('heat_pump_and_control',),
    'heat_work': ('heat_work',),
    'means': ('means', 'means_general', 'means_month'),
    'savings': ('savings',),
    'action_distribution': ('action_distribution',),
    'boxplots': ('boxplots',),
    'reward_balance': ('reward_balance', 'reward_balance_summary', 'reward_balance_per_timestep'),
}


def _study_output_dirs(base: Path) -> SimpleNamespace:
    """Subcarpetas por tipo de gráfico bajo ``base``."""
    reward_balance = base / 'reward_balance'
    means = base / 'means'
    return SimpleNamespace(
        base=base,
        progress=base / 'progress',
        zone_temperatures=base / 'zone_temperatures',
        temp_vs_flow=base / 'temp_vs_flow',
        heat_pump_and_control=base / 'heat_pump_and_control',
        heat_work=base / 'heat_work',
        means=means,
        means_general=means / 'general',
        means_month=means / 'month',
        savings=base / 'savings',
        action_distribution=base / 'action_distribution',
        boxplots=base / 'boxplots',
        reward_balance=reward_balance,
        reward_balance_summary=reward_balance / 'summary',
        reward_balance_per_timestep=reward_balance / 'per_timestep',
    )


def _ensure_dirs(
    ns: SimpleNamespace, output_sections: Optional[AbstractSet[str]] = None
) -> None:
    """Crea carpetas de salida. Si ``output_sections`` es ``None``, todas; si no, solo las indicadas."""
    if output_sections is None:
        for name in (
            'base',
            'progress',
            'zone_temperatures',
            'temp_vs_flow',
            'heat_pump_and_control',
            'heat_work',
            'means',
            'means_general',
            'means_month',
            'savings',
            'action_distribution',
            'boxplots',
            'reward_balance',
            'reward_balance_summary',
            'reward_balance_per_timestep',
        ):
            getattr(ns, name).mkdir(parents=True, exist_ok=True)
        return

    ns.base.mkdir(parents=True, exist_ok=True)
    for sec in output_sections:
        for attr in _SECTION_TO_OUTDIR_ATTRS.get(sec, ()):
            getattr(ns, attr).mkdir(parents=True, exist_ok=True)


def _validate_zone_lists(cfg: StudyPlotConfig) -> None:
    n = len(cfg.zone_names)
    for label, seq in (
        ('temperature_variables', cfg.temperature_variables),
        ('setpoint_variables', cfg.setpoint_variables),
        ('flow_variables', cfg.flow_variables),
        ('inlet_temperature_variables', cfg.inlet_temperature_variables),
        ('outlet_temperature_variables', cfg.outlet_temperature_variables),
    ):
        if len(seq) != n:
            raise ValueError(
                f'{cfg.study_label}: {label} tiene longitud {len(seq)}; '
                f'se esperaba {n} (como zone_names).'
            )


# =============================================================================
# PRESETS — añade entradas aquí para nuevas carpetas de salida
# =============================================================================

CASE_STUDY_CONFIG = StudyPlotConfig(
    study_label='case_study',
    output_base=Path('/home/jovyan/work/data/paper/plots/case_study/training_and_evaluation'),
    data_dir='/home/jovyan/work/data/paper/data/case_study/simulation',
    episode=10,
    experiments={
        'PPO': 'PPO/Eplus-PPO-evaluation',
        'TQC': 'TQC/Eplus-TQC-evaluation',
        'SAC': 'SAC/Eplus-SAC-evaluation',
        'RecPPO': 'RPO/Eplus-RecurrentPPO-evaluation',
    },
    training_progress_paths={
        'PPO': '/home/jovyan/work/data/paper/data/case_study/simulation/PPO/training/progress.csv',
        'TQC': '/home/jovyan/work/data/paper/data/case_study/simulation/TQC/training/progress.csv',
        'SAC': '/home/jovyan/work/data/paper/data/case_study/simulation/SAC/training/progress.csv',
        'RecPPO': '/home/jovyan/work/data/paper/data/case_study/simulation/RPO/training/progress.csv',
    },
    zone_names=[
        'Living-Kitchen',
        'Bathroom-Lobby',
        'Bedroom 1',
        'Bedroom 2',
        'Bedroom 3',
        'Bathroom-Corridor',
        'Bathroom-Dressing',
    ],
    temperature_variables=[
        'air_temperature_f0_living-kitchen',
        'air_temperature_f0_bathroom-lobby',
        'air_temperature_f1_bedroom1',
        'air_temperature_f1_bedroom2',
        'air_temperature_f1_bedroom3',
        'air_temperature_f1_bathroom-corridor',
        'air_temperature_f1_bathroom-dressing',
    ],
    setpoint_variables=[
        'heating_setpoint_f0_living-kitchen',
        'heating_setpoint_f0_bathroom-lobby',
        'heating_setpoint_f1_bedroom1',
        'heating_setpoint_f1_bedroom2',
        'heating_setpoint_f1_bedroom3',
        'heating_setpoint_f1_bathroom-corridor',
        'heating_setpoint_f1_bathroom-dressing',
    ],
    flow_variables=[
        'flow_rate_f0_living-kitchen',
        'flow_rate_f0_bathroom-lobby',
        'flow_rate_f1_bedroom1',
        'flow_rate_f1_bedroom2',
        'flow_rate_f1_bedroom3',
        'flow_rate_f1_bathroom-corridor',
        'flow_rate_f1_bathroom-dressing',
    ],
    inlet_temperature_variables=[
        'radiant_hvac_inlet_temperature_f0_living-kitchen',
        'radiant_hvac_inlet_temperature_f0_bathroom-lobby',
        'radiant_hvac_inlet_temperature_f1_bedroom1',
        'radiant_hvac_inlet_temperature_f1_bedroom2',
        'radiant_hvac_inlet_temperature_f1_bedroom3',
        'radiant_hvac_inlet_temperature_f1_bathroom-corridor',
        'radiant_hvac_inlet_temperature_f1_bathroom-dressing',
    ],
    outlet_temperature_variables=[
        'radiant_hvac_outlet_temperature_f0_living-kitchen',
        'radiant_hvac_outlet_temperature_f0_bathroom-lobby',
        'radiant_hvac_outlet_temperature_f1_bedroom1',
        'radiant_hvac_outlet_temperature_f1_bedroom2',
        'radiant_hvac_outlet_temperature_f1_bedroom3',
        'radiant_hvac_outlet_temperature_f1_bathroom-corridor',
        'radiant_hvac_outlet_temperature_f1_bathroom-dressing',
    ],
    names_reference=[],
    names_comparison=['PPO', 'TQC', 'SAC', 'RecPPO'],
)

PILOT_CASO_1_CONFIG = StudyPlotConfig(
    study_label='pilot_caso_1',
    output_base=Path(
        '/home/jovyan/work/data/paper/plots/pilot_study/training_and_evaluation/caso_1'
    ),
    data_dir='/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1',
    episode=2,
    experiments={
        'PPO': 'PPO/Eplus-PPO-radiant_case1_heating-Example_2026-03-16_13:32-res1',
        'TQC': 'TQC/Eplus-TQC-radiant_case1_heating-Example_2026-03-19_08:28-res1',
        'SAC': 'SAC/Eplus-SAC-radiant_case1_heating-Example_2026-03-16_13:42-res1',
        'RecPPO': 'RPO/Eplus-RecurrentPPO-radiant_case1_heating-Example_2026-03-16_13:41-res1',
    },
    training_progress_paths={
        'PPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1/PPO/training/progress.csv',
        'TQC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1/TQC/training/progress.csv',
        'SAC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1/SAC/training/progress.csv',
        'RecPPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso1/RPO/training/progress.csv',
    },
    zone_names=[
        'Living Room',
        'Bathroom',
        'Bedroom 1',
        'Bedroom 2',
        'Bedroom 3',
    ],
    temperature_variables=[
        'air_temperature_living',
        'air_temperature_kitchen',
        'air_temperature_bed1',
        'air_temperature_bed2',
        'air_temperature_bed3',
    ],
    setpoint_variables=[
        'heating_setpoint_living',
        'heating_setpoint_kitchen',
        'heating_setpoint_bed1',
        'heating_setpoint_bed2',
        'heating_setpoint_bed3',
    ],
    flow_variables=[
        'flow_rate_living',
        'flow_rate_kitchen',
        'flow_rate_bed1',
        'flow_rate_bed2',
        'flow_rate_bed3',
    ],
    inlet_temperature_variables=[
        'radiant_hvac_inlet_temperature_living',
        'radiant_hvac_inlet_temperature_kitchen',
        'radiant_hvac_inlet_temperature_bed1',
        'radiant_hvac_inlet_temperature_bed2',
        'radiant_hvac_inlet_temperature_bed3',
    ],
    outlet_temperature_variables=[
        'radiant_hvac_outlet_temperature_living',
        'radiant_hvac_outlet_temperature_kitchen',
        'radiant_hvac_outlet_temperature_bed1',
        'radiant_hvac_outlet_temperature_bed2',
        'radiant_hvac_outlet_temperature_bed3',
    ],
    names_reference=[],
    names_comparison=['PPO', 'TQC', 'SAC', 'RecPPO'],
)

# Todas las configuraciones que debe procesar este script (una carpeta ``output_base`` por entrada).
STUDY_CONFIGS: List[StudyPlotConfig] = [
    CASE_STUDY_CONFIG,
    #PILOT_CASO_1_CONFIG,
]


def run_study_plots(
    cfg: StudyPlotConfig,
    *,
    output_sections: Optional[AbstractSet[str]] = None,
) -> None:
    """Carga datos según ``cfg`` y escribe figuras bajo ``cfg.output_base``.

    Si ``output_sections`` es ``None``, se genera todo. Si es un conjunto de ids
    (véase ``OUTPUT_SECTION_IDS``), solo esas carpetas lógicas.
    """
    if output_sections is not None:
        unknown = set(output_sections) - OUTPUT_SECTION_IDS
        if unknown:
            raise ValueError(
                f'{cfg.study_label}: sección(es) desconocida(s): {sorted(unknown)}. '
                f'Válidas: {sorted(OUTPUT_SECTION_IDS)}'
            )

    _validate_zone_lists(cfg)

    def want(sec: str) -> bool:
        return output_sections is None or sec in output_sections
    names_comparison = (
        cfg.names_comparison
        if cfg.names_comparison is not None
        else list(cfg.experiments.keys())
    )
    combination_size = len(cfg.names_reference) * len(names_comparison)
    colors = cfg.colors
    filter_interval = cfg.filter_interval

    if cfg.zone_temp_plot_daily_date is not None:
        zone_daily = pd.Timestamp(cfg.zone_temp_plot_daily_date).normalize()
    elif filter_interval is not None:
        zone_daily = (
            pd.Timestamp(filter_interval[0])
            + (pd.Timestamp(filter_interval[1]) - pd.Timestamp(filter_interval[0])) / 2
        ).normalize()
    else:
        raise ValueError(
            f'{cfg.study_label}: indica zone_temp_plot_daily_date o filter_interval.'
        )

    action_distribution_variables = list(cfg.flow_variables) + [cfg.water_temperature_variable]
    temp_colors = [cfg.temp_zone_line_color] * len(cfg.zone_names)

    out_dirs = _study_output_dirs(cfg.output_base)
    _ensure_dirs(out_dirs, output_sections)

    # =============================================================================
    # DATA LOADING — rutas monitor, CSVs y ``unified``
    # =============================================================================

    def _eval_monitor_dir(run_folder: str) -> str:
        return os.path.join(
            cfg.data_dir, run_folder, f'episode-{cfg.episode}', 'monitor'
        )

    _monitor_dirs = {
        key: _eval_monitor_dir(run_folder)
        for key, run_folder in cfg.experiments.items()
    }

    training_progress = {
        key: safe_read_csv(path) for key, path in cfg.training_progress_paths.items()
    }
    training_progress = {
        k: v for k, v in training_progress.items() if v is not None and not v.empty
    }

    evaluation_obs = {
        key: safe_read_csv(os.path.join(monitor_dir, 'observations.csv'))
        for key, monitor_dir in _monitor_dirs.items()
    }
    evaluation_infos = {
        key: safe_read_csv(os.path.join(monitor_dir, 'infos.csv'))
        for key, monitor_dir in _monitor_dirs.items()
    }

    unified = {
        key: pd.concat(
            [evaluation_obs[key], evaluation_infos[key]],
            axis=1,
        )
        for key in evaluation_obs.keys()
    }

    evaluation_progress = {
        key: safe_read_csv(os.path.join(cfg.data_dir, run_folder, 'progress.csv'))
        for key, run_folder in cfg.experiments.items()
    }
    evaluation_progress = {
        k: v for k, v in evaluation_progress.items() if v is not None and not v.empty
    }

    # =============================================================================
    # PREPROCESS — datetime, filtro de intervalo, medias y CRF
    # =============================================================================

    for key in unified:
        unified[key] = add_datetime_column(unified[key])
        if filter_interval is not None:
            unified[key] = filer_interval(
                unified[key], filter_interval[0], filter_interval[1]
            )

    df_num = len(unified)

    mean_temp_violation_dict = {
        key: mean_variable(df, variable='total_temperature_violation')
        for key, df in unified.items()
    }
    mean_energy_consumption_dict = {
        key: mean_variable(df, variable='total_power_demand') for key, df in unified.items()
    }

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

    if want('progress') and training_progress:
        _n = len(training_progress)
        fig = plot_dfs_line(
            df_dict=training_progress,
            variable_name='mean_reward',
            colors=colors[:_n],
        )
        fig.update_layout(title=None, xaxis_title='Episode', yaxis_title='Mean reward')
        save_figure(
            fig, out_dirs.progress / 'training_progress', width=1200, height=700, scale=2
        )

    # =============================================================================
    # FIGURES — Temperaturas (plot_case_temperatures: rejilla + por habitación y recortes)
    # =============================================================================

    if want('zone_temperatures'):
        _zones = list(
            zip(
                cfg.temperature_variables,
                cfg.setpoint_variables,
                cfg.zone_names,
                strict=True,
            )
        )
        for key, df in unified.items():
            model_dir = out_dirs.zone_temperatures / _slugify(key)
            _kwargs = dict(
                df=df,
                zones=_zones,
                output_dir=model_dir,
                daily_date=zone_daily,
                case_id=0,
                summary_title=key,
                threshold=cfg.temperature_threshold,
                outdoor_temp_var=None,
                png_width=ZONE_TEMP_EXPORT_WIDTH_PX,
                png_height_single=ZONE_TEMP_EXPORT_SINGLE_HEIGHT_PX,
                png_scale=2,
                temp_colors=temp_colors,
                paper_style=True,
            )
            if filter_interval is not None:
                _kwargs['period_start'] = pd.Timestamp(filter_interval[0]).to_pydatetime()
                _kwargs['period_end'] = pd.Timestamp(filter_interval[1]).to_pydatetime()
            plot_case_temperatures(**_kwargs)

    # =============================================================================
    # FIGURES — Temperature vs flow (control)
    # =============================================================================

    if want('temp_vs_flow'):
        for key, df in unified.items():
            fig = plot_control(
                df=df,
                temperature_variables=cfg.temperature_variables,
                flow_variables=cfg.flow_variables,
                names=[f'Temp {z}' for z in cfg.zone_names]
                + [f'Flow {z}' for z in cfg.zone_names],
                colors=colors[
                    : len(cfg.temperature_variables) + len(cfg.flow_variables)
                ],
            )
            save_figure(
                fig,
                out_dirs.temp_vs_flow / _slugify(key),
                width=1200,
                height=700,
                scale=2,
            )

    # =============================================================================
    # FIGURES — Heat pump and control signals (per model)
    # =============================================================================

    if want('heat_pump_and_control'):
        for key, df in unified.items():
            model_dir = out_dirs.heat_pump_and_control / _slugify(key)
            model_dir.mkdir(parents=True, exist_ok=True)

            vars_to_plot = (
                [cfg.energy_variable]
                + list(cfg.flow_variables)
                + [cfg.water_temperature_variable]
            )

            for i, var in enumerate(vars_to_plot):
                if var not in df.columns:
                    print(
                        f'⚠️ [{cfg.study_label}] {key}: columna "{var}" no disponible; '
                        'se omite gráfica suavizada.'
                    )
                    continue

                fig = plot_smoothed_signal(
                    df=df,
                    variable=var,
                    datetime_col='datetime',
                    window=cfg.smooth_window,
                    color=colors[i % len(colors)],
                    title=None,
                    yaxis_title=_heat_pump_control_y_axis_title(var, cfg),
                    show_legend=False,
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

    if want('heat_work'):
        for key, df in unified.items():
            model_dir = out_dirs.heat_work / _slugify(key)
            model_dir.mkdir(parents=True, exist_ok=True)

            water_col = (
                cfg.water_temperature_variable
                if cfg.water_temperature_variable in df.columns
                else f'{cfg.water_temperature_variable}_action'
            )
            outlet_col = cfg.heat_pump_outlet_variable

            if water_col not in df.columns or outlet_col not in df.columns:
                print(
                    f'⚠️ [{cfg.study_label}] {key}: faltan columnas para heat_work '
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
            fig.update_layout(
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                ),
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

    if want('means'):
        fig = plot_bar(mean_temp_violation_dict, bar_colors=colors[:df_num])
        fig.update_layout(
            title=None,
            xaxis_title='',
            yaxis_title='Mean episodic temperature violation (ºC)',
        )
        save_figure(
            fig,
            out_dirs.means_general / 'mean_temp_violations',
            width=1200,
            height=600,
            scale=2,
        )

        fig = plot_bar(mean_energy_consumption_dict, bar_colors=colors[:df_num])
        fig.update_layout(
            title=None,
            xaxis_title='',
            yaxis_title='Mean episodic power consumption (W)',
        )
        save_figure(
            fig,
            out_dirs.means_general / 'mean_power_demand',
            width=1200,
            height=600,
            scale=2,
        )

        fig = plot_dfs_bar_grouped_by_month(
            unified,
            cfg.energy_variable,
            colors=colors[:df_num],
        )
        fig.update_layout(
            title=None, xaxis_title='', yaxis_title='Mean episodic power demand (W)'
        )
        save_figure(
            fig,
            out_dirs.means_month / 'month_power_demand',
            width=1200,
            height=600,
            scale=2,
        )

        fig = plot_dfs_bar_grouped_by_month(
            unified,
            'total_temperature_violation',
            colors=colors[:df_num],
        )
        fig.update_layout(
            title=None,
            xaxis_title='',
            yaxis_title='Mean episodic temperature violation (°C)',
        )
        save_figure(
            fig,
            out_dirs.means_month / 'month_temperature_violation',
            width=1200,
            height=600,
            scale=2,
        )

        # --- Inlet/outlet: medias por zona por experimento (barras agrupadas)

        if any(
            any(v in df.columns for v in cfg.inlet_temperature_variables)
            for df in unified.values()
        ):
            fig = plot_bar_means_by_zones(
                unified,
                cfg.inlet_temperature_variables,
                cfg.zone_names,
                colors=colors[:df_num],
            )
            fig.update_layout(title=None, yaxis_title='Mean inlet (°C)')
            save_figure(
                fig,
                out_dirs.means_general / 'mean_inlet_temperature_by_zone',
                width=1200,
                height=600,
                scale=2,
            )
        if any(
            any(v in df.columns for v in cfg.outlet_temperature_variables)
            for df in unified.values()
        ):
            fig = plot_bar_means_by_zones(
                unified,
                cfg.outlet_temperature_variables,
                cfg.zone_names,
                colors=colors[:df_num],
            )
            fig.update_layout(title=None, yaxis_title='Mean outlet (°C)')
            save_figure(
                fig,
                out_dirs.means_general / 'mean_outlet_temperature_by_zone',
                width=1200,
                height=600,
                scale=2,
            )

        # --- CRF: media ± std de transiciones/día y de encendidos estrictos/día

        if crf_trans_mean_dict:
            fig = plot_bar_with_std(
                crf_trans_mean_dict,
                crf_trans_std_dict,
                bar_colors=colors[: len(crf_trans_mean_dict)],
            )
            fig.update_layout(
                title=None,
                yaxis_title='Transitions per day',
                xaxis_title='',
            )
            save_figure(
                fig,
                out_dirs.means_general / 'crf_mean_transitions_per_day',
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
                xaxis_title='',
            )
            save_figure(
                fig,
                out_dirs.means_general / 'crf_mean_ons_per_day',
                width=1200,
                height=600,
                scale=2,
            )

    # =============================================================================
    # FIGURES — Energy savings (media global y por mes; solo si hay referencia y comparación)
    # =============================================================================

    if want('savings'):
        if cfg.names_reference and names_comparison:
            fig = plot_mean_energy_savings(
                data=unified,
                names_reference=cfg.names_reference,
                names_comparison=names_comparison,
                variable=cfg.energy_variable,
                colors=colors[1 : combination_size + 1],
            )
            fig.update_layout(title=None)
            save_figure(
                fig, out_dirs.savings / 'mean_savings', width=1200, height=600, scale=2
            )

            fig = plot_energy_savings(
                data=unified,
                names_reference=cfg.names_reference,
                names_comparison=names_comparison,
                variable=cfg.energy_variable,
                colors=colors[1 : combination_size + 1],
            )
            save_figure(
                fig,
                out_dirs.savings / 'month_energy_savings',
                width=1200,
                height=600,
                scale=2,
            )
        else:
            print(
                f'⚠️ [{cfg.study_label}] Sin referencia o comparación en unified; '
                'se omiten gráficas de ahorro.'
            )

    # =============================================================================
    # FIGURES — Action distribution (violines)
    # =============================================================================

    if want('action_distribution'):
        for var in action_distribution_variables:
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
                out_dirs.action_distribution / f'distribution_{_slugify(var)}',
                width=1200,
                height=600,
                scale=2,
            )

    # =============================================================================
    # FIGURES — Boxplots (evaluation progress)
    # =============================================================================

    if want('boxplots'):
        if not evaluation_progress:
            print(
                f'⚠️ [{cfg.study_label}] No hay evaluation_progress (progress.csv en monitor); '
                'se omiten boxplots.'
            )
        else:
            fig = plot_dfs_boxplot(
                evaluation_progress,
                'comfort_violation_time(%)',
                colors=colors[:df_num],
                yaxis_title='Episodic comfort violation time (%)',
                xaxis_title='',
            )
            save_figure(
                fig,
                out_dirs.boxplots / 'comfort_violation_time',
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
                xaxis_title='',
            )
            save_figure(
                fig,
                out_dirs.boxplots / 'temperature_violation',
                width=1200,
                height=600,
                scale=2,
            )

            fig = plot_dfs_boxplot(
                evaluation_progress,
                'mean_power_demand',
                colors=colors[:df_num],
                yaxis_title='Episodic power demand (W)',
                xaxis_title='',
            )
            fig.update_layout(title=None, yaxis_title='Power demand (W)')
            save_figure(
                fig,
                out_dirs.boxplots / 'power_demand',
                width=1200,
                height=600,
                scale=2,
            )

    # =============================================================================
    # FIGURES — Reward balance (comfort vs energy term from progress.csv)
    # =============================================================================

    if want('reward_balance'):
        if not evaluation_progress:
            print(
                f'⚠️ [{cfg.study_label}] No hay evaluation_progress; '
                'se omite gráfica de equilibrio reward.'
            )
        else:
            progress_with_terms = {
                k: v
                for k, v in evaluation_progress.items()
                if REWARD_COMFORT_COL in v.columns and REWARD_ENERGY_COL in v.columns
            }
            if not progress_with_terms:
                print(
                    f'⚠️ [{cfg.study_label}] Ningún progress.csv tiene '
                    f'"{REWARD_COMFORT_COL}" y "{REWARD_ENERGY_COL}"; '
                    'se omite gráfica de equilibrio reward.'
                )
            else:
                balance_comfort_means = {
                    k: v[REWARD_COMFORT_COL].mean()
                    for k, v in progress_with_terms.items()
                }
                balance_comfort_stds = {
                    k: v[REWARD_COMFORT_COL].std()
                    for k, v in progress_with_terms.items()
                }
                balance_energy_means = {
                    k: v[REWARD_ENERGY_COL].mean()
                    for k, v in progress_with_terms.items()
                }
                balance_energy_stds = {
                    k: v[REWARD_ENERGY_COL].std()
                    for k, v in progress_with_terms.items()
                }
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
                fig.update_layout(title=None)
                save_figure(
                    fig,
                    out_dirs.reward_balance_summary / 'comfort_vs_energy_term',
                    width=1200,
                    height=600,
                    scale=2,
                )

    # =============================================================================
    # FIGURES — Training reward terms progression (comfort vs energy durante entrenamiento)
    # =============================================================================

    if want('reward_balance') and training_progress:
        for run_name, df_progress in training_progress.items():
            if (
                REWARD_COMFORT_COL not in df_progress.columns
                or REWARD_ENERGY_COL not in df_progress.columns
            ):
                print(
                    f'⚠️ [{cfg.study_label}] {run_name}: progress sin '
                    f'"{REWARD_COMFORT_COL}" o "{REWARD_ENERGY_COL}"; '
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
                out_dirs.reward_balance_summary
                / f'training_progression_{_slugify(run_name)}',
                width=1200,
                height=600,
                scale=2,
            )

    # =============================================================================
    # FIGURES — Reward terms per timestep (evolución dentro de un episodio, infos)
    # =============================================================================

    if want('reward_balance'):
        for key, df in unified.items():
            if INFO_COMFORT_COL not in df.columns or INFO_ENERGY_COL not in df.columns:
                print(
                    f'⚠️ [{cfg.study_label}] {key}: infos sin "{INFO_COMFORT_COL}" o '
                    f'"{INFO_ENERGY_COL}"; se omite gráfica por timestep.'
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
                out_dirs.reward_balance_per_timestep
                / f'{_slugify(key)}_episode_{cfg.episode}_comfort_vs_energy',
                width=1200,
                height=600,
                scale=2,
            )


def main(
    study_labels: Optional[List[str]] = None,
    output_sections: Optional[AbstractSet[str]] = None,
) -> None:
    """Genera figuras para cada entrada en ``STUDY_CONFIGS`` (o solo las indicadas).

    ``output_sections``: ``None`` = todas las carpetas lógicas; si se pasa un conjunto,
    solo esas (véase ``OUTPUT_SECTION_IDS``).
    """
    pio.defaults.default_scale = 2
    configs = STUDY_CONFIGS
    if study_labels is not None:
        label_set = set(study_labels)
        configs = [c for c in STUDY_CONFIGS if c.study_label in label_set]
        missing = label_set - {c.study_label for c in configs}
        if missing:
            raise ValueError(f'Estudio(s) desconocido(s): {sorted(missing)}')

    for cfg in configs:
        print(f'=== [{cfg.study_label}] -> {cfg.output_base} ===')
        run_study_plots(cfg, output_sections=output_sections)


def _parse_cli():
    p = argparse.ArgumentParser(
        description=(
            'Genera figuras del case study. Sin --outputs se crean todas las '
            'subcarpetas bajo output_base.'
        )
    )
    p.add_argument(
        '--study',
        '-s',
        action='append',
        metavar='LABEL',
        dest='studies',
        help='Solo este ``study_label`` de ``STUDY_CONFIGS``. Repetible.',
    )
    p.add_argument(
        '--outputs',
        '-o',
        nargs='+',
        metavar='SECTION',
        default=None,
        help=(
            'Solo estas salidas (nombres de carpeta lógica). Ej.: boxplots '
            'zone_temperatures. Omitir para todas. Opciones: '
            + ', '.join(sorted(OUTPUT_SECTION_IDS))
        ),
    )
    return p.parse_args()


if __name__ == '__main__':
    _args = _parse_cli()
    _sections = frozenset(_args.outputs) if _args.outputs else None
    main(study_labels=_args.studies, output_sections=_sections)
