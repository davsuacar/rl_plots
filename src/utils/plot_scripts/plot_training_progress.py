from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent

# Original Case 2 baseline
ORIGINAL_DIR = "Eval-DRL-Baseline-2026-case-2_2025-12-17_10:31-res1"
ORIGINAL_LABEL = "Original (TQC)"
ORIGINAL_COLOR = "#2E86AB"  # Blue

# Directories with evaluations
EVAL_DIRS = {
    "initial": ROOT / "evaluaciones-iniciales",
    "1ep": ROOT / "evaluaciones-1ep-entrenado",
    "5ep": ROOT / "evaluaciones-5ep-entrenado",
}

# Evaluation stage labels
STAGE_LABELS = {
    "initial": "Initial",
    "1ep": "1 Ep",
    "5ep": "5 Ep",
}

# Colors for each stage - using same scheme as plot_degradations_html.py
# The stages will use variations of the model's base color
STAGE_COLORS = {
    "initial": "#2E86AB",  # Blue (same as original in plot_degradations_html)
    "1ep": "#A23B72",      # Purple
    "5ep": "#F18F01",      # Orange
}


def get_model_color(model_index: int, total_models: int) -> str:
    """Get color for a model based on its position, using same scheme as plot_degradations_html.py.

    Args:
        model_index: Index of the model (0-based, excluding original)
        total_models: Total number of models

    Returns:
        Color string in rgb format
    """
    # Same gradient scheme as plot_degradations_html.py
    # Gradient from orange to red (starting from index 1, since 0 would be original)
    # For training progress, we start from index 0 and use the same gradient
    intensity = 0.6 + ((model_index + 1) / (total_models + 1)) * 0.4
    r = int(255 * intensity)
    g = int(100 * (1 - intensity * 0.5))
    b = int(50 * (1 - intensity * 0.3))
    return f"rgb({r},{g},{b})"


def extract_model_info(dir_name: str) -> Optional[Tuple[str, str]]:
    """Extract model type and number from directory name.

    Args:
        dir_name: Directory name like 'Eval-DRL-Baseline-2026-case-2-window_1_...'
                  or 'Degradation-1ep-case2-window_1_...'

    Returns:
        Tuple of (type, num) like ('window', '1'), or None if pattern doesn't match
    """
    # Pattern to match: case-2-{type}_{num}_ or case2-{type}_{num}_
    # Match either "case-2-" or "case2-" followed by type and number
    patterns = [
        r'case-2-([^_]+)_(\d+)_',  # For "case-2-window_1_"
        r'case2-([^_]+)_(\d+)_',    # For "case2-window_1_"
    ]

    for pattern in patterns:
        match = re.search(pattern, dir_name)
        if match:
            deg_type = match.group(1)
            deg_num = match.group(2)
            return (deg_type, deg_num)
    return None


def scan_evaluations() -> Dict[str, Dict[str, Path]]:
    """Scan all evaluation directories and group by model.

    Returns:
        Dictionary mapping model_key (e.g., 'window_1') to a dict of
        {stage: path} like {'initial': Path(...), '1ep': Path(...), '5ep': Path(...)}
    """
    models = {}

    for stage, eval_dir in EVAL_DIRS.items():
        if not eval_dir.exists():
            print(f"Warning: Directory not found: {eval_dir}")
            continue

        for dir_path in sorted(eval_dir.iterdir()):
            if not dir_path.is_dir():
                continue

            model_info = extract_model_info(dir_path.name)
            if model_info is None:
                continue

            type_name, num = model_info
            model_key = f"{type_name}_{num}"

            if model_key not in models:
                models[model_key] = {}

            # Check if progress.csv exists
            progress_path = dir_path / "progress.csv"
            if progress_path.exists():
                models[model_key][stage] = progress_path
            else:
                print(f"Warning: progress.csv not found in {dir_path}")

    return models


def load_progress_metrics(progress_path: Path) -> pd.DataFrame:
    """Load raw evaluation metrics from progress.csv."""
    df = pd.read_csv(progress_path)
    return df


def plot_model_progress(
    model_key: str,
    model_data: Dict[str, Path],
    output_dir: Path,
    metrics_spec: List[Tuple[str, str, str]],
    model_index: int,
    total_models: int,
    original_df: Optional[pd.DataFrame] = None
) -> None:
    """Create boxplots for a single model comparing progress across stages.

    Args:
        model_key: Model identifier like 'window_1'
        model_data: Dictionary mapping stage to progress.csv path
        output_dir: Output directory for plots
        metrics_spec: List of (column_name, title, filename) tuples
        model_index: Index of this model for color assignment
        total_models: Total number of models
        original_df: DataFrame with original model data (baseline)
    """
    # Load data for each stage
    stage_data = {}
    for stage, path in model_data.items():
        try:
            df = load_progress_metrics(path)
            stage_data[stage] = df
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not stage_data:
        print(f"Warning: No data available for model {model_key}")
        return

    # Get model color (reddish gradient for degradations)
    model_color = get_model_color(model_index, total_models)

    # Use the same color for all stages of this model
    stage_colors = {
        "initial": model_color,
        "1ep": model_color,
        "5ep": model_color,
    }

    # Create one boxplot per metric
    for col, title, filename in metrics_spec:
        fig = go.Figure()

        # Add original baseline first (if available)
        if original_df is not None and col in original_df.columns:
            original_values = original_df[col].to_numpy()
            if len(original_values) > 0:
                fig.add_trace(
                    go.Box(
                        y=original_values,
                        name=ORIGINAL_LABEL,
                        marker_color=ORIGINAL_COLOR,
                        boxmean=True,
                    )
                )

        # Add boxplot for each stage
        stages_added = 0
        for stage in ["initial", "1ep", "5ep"]:
            if stage not in stage_data:
                continue

            df = stage_data[stage]
            if col not in df.columns:
                print(
                    f"Warning: Column {col} not found in {model_key} {stage}")
                continue

            values = df[col].to_numpy()
            if len(values) == 0:
                continue

            label = STAGE_LABELS[stage]
            color = stage_colors[stage]

            fig.add_trace(
                go.Box(
                    y=values,
                    name=label,
                    marker_color=color,
                    boxmean=True,
                )
            )
            stages_added += 1

        if stages_added == 0 and (
                original_df is None or col not in original_df.columns):
            print(f"Warning: No stages added for {model_key} {col}")
            continue

        # Format model name for display
        type_name, num = model_key.split('_')
        display_name = f"{type_name.capitalize()} {num}"

        fig.update_layout(
            title=f"{display_name} – {title}",
            yaxis_title=title,
            showlegend=True,
            template="plotly_white",
            height=500,
            width=800,
        )

        # Create output directory structure: {model_key}/{filename}
        model_output_dir = output_dir / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        fig.write_html(model_output_dir / filename)


def main() -> None:
    """Main entry point."""
    output_dir = ROOT / "training_progress_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to plot
    metrics_spec = [
        ("mean_reward",
         "Episode mean reward",
         "box_mean_reward.html"),
        ("mean_temperature_violation",
         "Episode mean temperature violation",
         "box_mean_temperature_violation.html"),
        ("mean_power_demand",
         "Episode mean power demand",
         "box_mean_power_demand.html"),
        ("mean_compressor_starts_per_day",
         "Episode mean compressor starts per day",
         "box_mean_compressor_starts_per_day.html"),
    ]

    print("Scanning evaluation directories...")
    models = scan_evaluations()

    if not models:
        print("Error: No models found in evaluation directories!")
        return

    print(f"Found {len(models)} models")

    # Load original baseline data
    original_path = ROOT / ORIGINAL_DIR / "progress.csv"
    original_df = None
    if original_path.exists():
        try:
            original_df = load_progress_metrics(original_path)
            print(f"Loaded original baseline data from {original_path}")
        except Exception as e:
            print(f"Warning: Could not load original data: {e}")
    else:
        print(f"Warning: Original baseline not found at {original_path}")

    # Sort models by type, then by number
    sorted_models = sorted(
        models.items(),
        key=lambda x: (
            x[0].split('_')[0],
            int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else 0
        )
    )

    # Generate plots for each model
    print("Generating progress plots...")
    for idx, (model_key, model_data) in enumerate(sorted_models):
        print(f"  Processing {model_key}...")
        plot_model_progress(
            model_key,
            model_data,
            output_dir,
            metrics_spec,
            idx,
            len(sorted_models),
            original_df)

    print(f"\nInteractive plots saved in: {output_dir}")
    print(f"\nGenerated plots for {len(sorted_models)} models:")
    for model_key, _ in sorted_models:
        print(f"  - {model_key}")


if __name__ == "__main__":
    main()
