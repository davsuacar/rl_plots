import argparse
import re
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import plotly.io as pio

from utils.degradation_plot_config import (
    DegradationPlotConfig,
    degradation_series_colors,
    load_degradation_plot_config,
)
from utils.plot_functions.plot_functions import (
    plot_action_distribution,
    plot_case_temperatures,
    plot_dfs_boxplot,
    save_figure,
)

pio.templates.default = "plotly_white"


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
            "Paths and experiments load from ``utils.degradation_plot_config`` "
            "unless ``--config`` points to a Python file defining "
            "``DEGRADATION_PLOT_CONFIG``."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Python file that defines DEGRADATION_PLOT_CONFIG "
            "(DegradationPlotConfig). Omisión: preset por defecto del proyecto."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Override config: parent directory that contains each experiment "
            "subfolder."
        ),
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=None,
        help="Override config: baseline evaluation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override config: root output directory.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "If set, only these experiment keys from config are run "
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


def _load_observations_in_simulation_window(
    monitor_dir: Path,
    cfg: DegradationPlotConfig,
) -> Optional[pd.DataFrame]:
    """``observations.csv`` recortado al intervalo de simulación (índice temporal)."""
    obs_path = monitor_dir / "observations.csv"
    infos_path = monitor_dir / "infos.csv"
    if not obs_path.exists() or not infos_path.exists():
        return None
    obs = pd.read_csv(obs_path)
    infos = pd.read_csv(infos_path)
    n = min(len(obs), len(infos))
    if n == 0:
        return None
    obs = obs.iloc[:n].reset_index(drop=True)
    infos = infos.iloc[:n].reset_index(drop=True)
    dt_index = build_datetime_index(infos, base_year=cfg.datetime_base_year)
    obs = obs.copy()
    obs.index = dt_index
    win_start = pd.Timestamp(cfg.simulation_window_start)
    win_end = pd.Timestamp(cfg.simulation_window_end)
    mask = (obs.index >= win_start) & (obs.index <= win_end)
    return obs.loc[mask].reset_index(drop=True)


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


def _progress_df_dict_for_column(
    all_data: Dict[str, pd.DataFrame],
    ordered_labels: List[str],
    col: str,
) -> Dict[str, pd.DataFrame]:
    """Orden fijo; solo entradas con la columna ``col``."""
    out: Dict[str, pd.DataFrame] = {}
    for label in ordered_labels:
        if label in all_data and col in all_data[label].columns:
            out[label] = all_data[label]
    return out


def _save_metric_boxplot_uponor_style(
    df_dict: Dict[str, pd.DataFrame],
    col: str,
    y_label: str,
    path_stem: Path,
    colors: List[str],
    cfg: DegradationPlotConfig,
) -> None:
    """Mismo estilo que ``plot_dfs_boxplot`` en el script Uponor (relleno + líneas negras)."""
    if not df_dict:
        return
    names = list(df_dict.keys())
    fig = plot_dfs_boxplot(
        {k: df_dict[k] for k in names},
        col,
        colors=colors[: len(names)],
        yaxis_title=y_label,
        xaxis_title="",
    )
    fig.update_layout(title=None)
    save_figure(
        fig,
        path_stem,
        width=cfg.boxplot_width,
        height=cfg.boxplot_height,
        scale=cfg.png_export_scale,
        paper_style=True,
    )


def plot_progress_boxplots_original_vs_one(
    output_dir: Path,
    original_dir: Path,
    deg_label: str,
    deg_path: Path,
    cfg: DegradationPlotConfig,
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

    all_data = {cfg.original_label: original_df, deg_label: deg_df}
    all_labels = [cfg.original_label, deg_label]
    for col, _unused_title, filename, y_label in cfg.progress_metrics:
        df_dict = _progress_df_dict_for_column(all_data, all_labels, col)
        if len(df_dict) < 2:
            continue
        colors = degradation_series_colors(
            len(df_dict),
            baseline_color=cfg.baseline_box_color,
            degradation_color=cfg.degradation_box_color,
        )
        _save_metric_boxplot_uponor_style(
            df_dict,
            col,
            y_label,
            output_dir / Path(filename).stem,
            colors,
            cfg,
        )


def plot_progress_boxplots_original_vs_all(
    output_dir: Path,
    original_dir: Path,
    degradations: List[Tuple[str, str, Path, str]],
    cfg: DegradationPlotConfig,
) -> None:
    """One figure per metric: baseline vs every degradation (same filenames as per-slug)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / "progress.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original progress.csv not found: {original_path}")

    original_df = load_progress_metrics(original_path)
    all_data: Dict[str, pd.DataFrame] = {cfg.original_label: original_df}
    all_labels = [cfg.original_label]

    for _dir_name, label, dir_path, _deg_slug in degradations:
        progress_path = dir_path / "progress.csv"
        if progress_path.exists():
            all_data[label] = load_progress_metrics(progress_path)
            all_labels.append(label)
        else:
            print(f"Warning: progress.csv not found in {dir_path}")

    for col, _unused_title, filename, y_label in cfg.progress_metrics:
        df_dict = _progress_df_dict_for_column(all_data, all_labels, col)
        if len(df_dict) < 2:
            continue
        colors = degradation_series_colors(
            len(df_dict),
            baseline_color=cfg.baseline_box_color,
            degradation_color=cfg.degradation_box_color,
        )
        _save_metric_boxplot_uponor_style(
            df_dict,
            col,
            y_label,
            output_dir / Path(filename).stem,
            colors,
            cfg,
        )


def plot_progress_boxplots_summary_by_degradation(
    output_dir: Path,
    original_dir: Path,
    experiment_order: List[str],
    degradations_by_experiment: Dict[str, List[Tuple[str, str, Path, str]]],
    cfg: DegradationPlotConfig,
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

        all_data: Dict[str, pd.DataFrame] = {cfg.original_label: original_df}
        ordered_labels: List[str] = [cfg.original_label]

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

        for col, _unused_title, filename, y_label in cfg.progress_metrics:
            df_dict = _progress_df_dict_for_column(all_data, ordered_labels, col)
            if len(df_dict) < 2:
                continue
            colors = degradation_series_colors(
                len(df_dict),
                baseline_color=cfg.baseline_box_color,
                degradation_color=cfg.degradation_box_color,
            )
            _save_metric_boxplot_uponor_style(
                df_dict,
                col,
                y_label,
                deg_out / Path(filename).stem,
                colors,
                cfg,
            )


def plot_flow_and_water_violin_original_vs_one(
    output_dir: Path,
    original_dir: Path,
    deg_label: str,
    deg_path: Path,
    cfg: DegradationPlotConfig,
) -> None:
    """Violines baseline vs degradación: mismo pipeline que Uponor (``plot_action_distribution``)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    original_monitor_dir = original_dir / cfg.episode_dir / "monitor"
    deg_monitor = deg_path / cfg.episode_dir / "monitor"
    if not deg_monitor.exists():
        print(f"Warning: monitor dir missing, skip violins: {deg_monitor}")
        return

    all_labels = [cfg.original_label, deg_label]
    all_monitor_dirs = [original_monitor_dir, deg_monitor]

    df_dict: Dict[str, pd.DataFrame] = {}
    for label, monitor_dir in zip(all_labels, all_monitor_dirs):
        df_run = _load_observations_in_simulation_window(monitor_dir, cfg)
        if df_run is not None:
            df_dict[label] = df_run

    ordered_keys = [lab for lab in all_labels if lab in df_dict]
    if not ordered_keys:
        print("Warning: no observation data for violins")
        return

    df_ordered: Dict[str, pd.DataFrame] = {k: df_dict[k] for k in ordered_keys}

    def _dfs_for_variable(variable: str) -> Dict[str, pd.DataFrame]:
        keys_v = [k for k in ordered_keys if variable in df_ordered[k].columns]
        return {k: df_ordered[k] for k in keys_v}

    def _colors_for_keys(keys_v: List[str]) -> List[str]:
        return degradation_series_colors(
            len(keys_v),
            baseline_color=cfg.baseline_box_color,
            degradation_color=cfg.degradation_box_color,
        )

    water_dfs = _dfs_for_variable("water_temperature")
    if water_dfs:
        fig_w = plot_action_distribution(
            water_dfs,
            "water_temperature",
            colors=_colors_for_keys(list(water_dfs.keys())),
        )
        save_figure(
            fig_w,
            output_dir / "violin_water_temperature",
            width=cfg.boxplot_width,
            height=cfg.boxplot_height,
            scale=cfg.png_export_scale,
            paper_style=True,
        )

    flow_cols = cfg.flow_variables
    room_labels = cfg.flow_room_labels
    for col, room_name in zip(flow_cols, room_labels):
        flow_dfs = _dfs_for_variable(col)
        if not flow_dfs:
            continue
        fig_f = plot_action_distribution(
            flow_dfs,
            col,
            colors=_colors_for_keys(list(flow_dfs.keys())),
        )
        room_slug = room_name.lower().replace(" ", "_")
        save_figure(
            fig_f,
            output_dir / f"violin_flow_{room_slug}",
            width=cfg.boxplot_width,
            height=cfg.boxplot_height,
            scale=cfg.png_export_scale,
            paper_style=True,
        )


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


def load_monitor_observations_with_datetime(
    monitor_dir: Path,
    cfg: DegradationPlotConfig,
) -> Optional[pd.DataFrame]:
    """Observaciones monitor + columna ``datetime`` (convención ``plot_case_temperatures``)."""
    obs_path = monitor_dir / "observations.csv"
    infos_path = monitor_dir / "infos.csv"
    if not obs_path.exists() or not infos_path.exists():
        return None
    obs = pd.read_csv(obs_path)
    infos = pd.read_csv(infos_path)
    n = min(len(obs), len(infos))
    if n == 0:
        return None
    obs = obs.iloc[:n].reset_index(drop=True)
    infos = infos.iloc[:n].reset_index(drop=True)
    dt = build_datetime_index(infos, base_year=cfg.datetime_base_year)
    out = obs.copy()
    out["datetime"] = dt
    return out


def plot_degradation_temperatures(
    label: str,
    monitor_dir: Path,
    output_dir: Path,
    daily_date: pd.Timestamp,
    cfg: DegradationPlotConfig,
) -> None:
    """Misma cadena que Uponor: ``plot_case_temperatures`` (rejilla 1 columna, bandas confort)."""
    df = load_monitor_observations_with_datetime(monitor_dir, cfg)
    if df is None:
        print(f"Warning: Missing observations/infos in {monitor_dir}")
        return

    zones = list(
        zip(
            cfg.temperature_variables,
            cfg.setpoint_variables,
            cfg.subplot_zone_titles,
            strict=True,
        )
    )
    temp_colors = [cfg.temp_zone_line_color] * len(cfg.temperature_variables)

    safe_label = label.lower().replace(" ", "_")
    period_end_str = cfg.simulation_window_end.strip().replace(" ", "T", 1)
    plot_case_temperatures(
        df=df,
        zones=zones,
        output_dir=output_dir,
        daily_date=daily_date,
        case_id=0,
        summary_title=label,
        threshold=cfg.temperature_threshold,
        temp_colors=temp_colors,
        outdoor_temp_var=None,
        period_start=datetime.fromisoformat(cfg.simulation_window_start.strip()),
        period_end=datetime.fromisoformat(period_end_str),
        export_format="both",
        png_width=cfg.temp_multi_width,
        png_height_single=cfg.temp_single_height,
        png_scale=cfg.png_export_scale,
        paper_style=True,
        export_zone_subfolders=True,
        grid_filename_stem=safe_label,
        nest_zone_dirs_under_case=False,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    cfg_path = (
        args.config.expanduser().resolve() if args.config is not None else None
    )
    cfg = load_degradation_plot_config(cfg_path)
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir.expanduser().resolve())
    if args.output_dir is not None:
        cfg = replace(cfg, output_base=args.output_dir.expanduser().resolve())

    if args.original_dir is not None:
        original_dir = resolve_original_dir(args.original_dir)
    else:
        original_dir = resolve_original_dir(cfg.original_eval_dir)

    data_dir = cfg.data_dir
    output_root = cfg.output_base
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = list(cfg.experiments.items())
    if args.experiments:
        wanted = set(args.experiments)
        experiments = [(k, v) for k, v in experiments if k in wanted]
        unknown = wanted - set(cfg.experiments.keys())
        if unknown:
            print(f"Warning: unknown experiment keys ignored: {sorted(unknown)}")

    print(f"Config: {cfg.study_label}" + (f" ({cfg_path})" if cfg_path else " (default)"))
    print(f"Data dir: {data_dir}")
    print(f"Baseline (original) eval dir: {original_dir}")
    print(f"Output root: {output_root}")

    original_monitor_dir = original_dir / cfg.episode_dir / "monitor"
    original_infos_path = original_monitor_dir / "infos.csv"
    if not original_infos_path.exists():
        print(f"Warning: Original infos.csv not found: {original_infos_path}")
        return

    win_start = pd.Timestamp(cfg.simulation_window_start)
    win_end = pd.Timestamp(cfg.simulation_window_end)

    original_infos = pd.read_csv(original_infos_path)
    original_dt_index = build_datetime_index(
        original_infos, base_year=cfg.datetime_base_year
    )
    first_mask = (original_dt_index >= win_start) & (original_dt_index <= win_end)
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
            cfg.original_label,
            original_monitor_dir,
            baseline_out,
            cast(pd.Timestamp, daily_date),
            cfg,
        )

        degradations = get_degradation_evaluations(degradations_dir)
        if not degradations:
            print(f"    Warning: no degradation runs in {degradations_dir}")
            continue

        degradations_by_experiment[exp_name] = degradations

        general_dir = output_exp / cfg.general_plots_dirname
        print(f"    -> {cfg.general_plots_dirname}/ (all degradations boxplots)")
        plot_progress_boxplots_original_vs_all(
            general_dir, original_dir, degradations, cfg
        )

        for _dir_name, label, dir_path, deg_slug in degradations:
            out_deg = output_exp / deg_slug
            print(f"    -> {deg_slug}")

            plot_progress_boxplots_original_vs_one(
                out_deg, original_dir, label, dir_path, cfg
            )
            plot_flow_and_water_violin_original_vs_one(
                out_deg, original_dir, label, dir_path, cfg
            )

            monitor_dir = dir_path / cfg.episode_dir / "monitor"
            if monitor_dir.exists():
                plot_degradation_temperatures(
                    label,
                    monitor_dir,
                    out_deg,
                    cast(pd.Timestamp, daily_date),
                    cfg,
                )

    if degradations_by_experiment:
        summary_dir = output_root / cfg.summary_plots_dirname
        print(
            f"\n=== {cfg.summary_plots_dirname}/ "
            "(Original vs training regimes per degradation)"
        )
        plot_progress_boxplots_summary_by_degradation(
            summary_dir,
            original_dir,
            [name for name, _ in experiments],
            degradations_by_experiment,
            cfg,
        )

    print(f"\nPlots saved under: {output_root} (HTML + PNG per folder)")


if __name__ == "__main__":
    main()
