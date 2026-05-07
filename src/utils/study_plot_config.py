"""Configuración reutilizable para scripts de gráficos (case study, piloto, etc.).

Importa :class:`StudyPlotConfig` y los presets (``CASE_STUDY_CONFIG``, …) o define
nuevas instancias con rutas/variables distintas para otros pipelines de plots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Paleta alineada con ``sample_plots`` (Tab10 + 5 tonos Tab20 emparejados / matplotlib).
TAB10: Tuple[str, ...] = (
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
DEFAULT_COLORS: List[str] = list(TAB10)


@dataclass
class StudyPlotConfig:
    """Rutas de datos, variables de zona y carpeta de salida para generar las figuras."""

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
    temp_zone_line_color: str = TAB10[0]


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

PILOT_CASE_1_CONFIG = StudyPlotConfig(
    study_label='Case 1',
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

PILOT_CASE_2_CONFIG = StudyPlotConfig(
    study_label='Case 2',
    output_base=Path(
        '/home/jovyan/work/data/paper/plots/pilot_study/training_and_evaluation/caso_2'
    ),
    data_dir='/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso2',
    episode=2,
    experiments={
        'PPO': 'PPO/Eplus-PPO-radiant_case2_heating-Example_2026-03-16_14:35-res1',
        'TQC': 'TQC/Eplus-TQC-radiant_case2_heating-Example_2026-03-16_14:34-res1',
        'SAC': 'SAC/Eplus-SAC-radiant_case2_heating-Example_2026-03-16_14:35-res1',
        'RecPPO': 'RPO/Eplus-RecurrentPPO-radiant_case2_heating-Example_2026-03-16_14:34-res1',
    },
    training_progress_paths={
        'PPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso2/PPO/training/progress.csv',
        'TQC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso2/TQC/training/progress.csv',
        'SAC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso2/SAC/training/progress.csv',
        'RecPPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso2/RPO/training/progress.csv',
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

PILOT_CASE_3_CONFIG = StudyPlotConfig(
    study_label='Case 3',
    output_base=Path(
        '/home/jovyan/work/data/paper/plots/pilot_study/training_and_evaluation/caso_3'
    ),
    data_dir='/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3',
    episode=2,
    experiments={
        'PPO': 'PPO/Eplus-PPO-radiant_case3_heating-Example_2026-03-19_09:00-res1',
        'TQC': 'TQC/Eplus-TQC-radiant_case3_heating-Example_2026-03-16_14:39-res1',
        'SAC': 'SAC/Eplus-SAC-radiant_case3_heating-Example_2026-03-16_14:38-res1',
        'RecPPO': 'RPO/Eplus-RecurrentPPO-radiant_case3_heating-Example_2026-03-16_14:40-res1',
    },
    training_progress_paths={
        'PPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/PPO/training/progress.csv',
        'TQC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/TQC/training/progress.csv',
        'SAC': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/SAC/training/progress.csv',
        'RecPPO': '/home/jovyan/work/data/paper/data/pilot_study/eval_por_caso_y_model/caso3/RPO/training/progress.csv',
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
