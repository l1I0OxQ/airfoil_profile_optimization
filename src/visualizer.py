import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import optuna
from utils import generate_airfoil_profile, build_metric_order
import config


def get_optimization_history(study: Optional[optuna.Study]) -> Optional[pd.DataFrame]:
    if study is None:
        return None
    try:
        trials_df = study.trials_dataframe()
        if trials_df.empty:
            return None
        completed_trials = trials_df[trials_df["state"] == "COMPLETE"].copy()
        if completed_trials.empty:
            return None
        if "number" in completed_trials.columns:
            completed_trials["trial_index"] = completed_trials["number"]
        metric_names = build_metric_order()
        for idx, metric_name in enumerate(metric_names):
            value_col = f"values_{idx}"
            if value_col in completed_trials.columns:
                completed_trials[metric_name] = completed_trials[value_col]
        if "trial_index" in completed_trials.columns:
            completed_trials = completed_trials.sort_values("trial_index")
        return completed_trials
    except (KeyError, ValueError, AttributeError, TypeError) as e:
        print(f"Error: Failed to get optimization history: {e}", file=sys.stderr)
        return None


def visualize_results(
    coeffs: Dict,
    trial_index: int = None,
    study: Optional[optuna.Study] = None,
    results: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    figs_dir = config.work_dir / "logs" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    panels = ["current_profile", "ld_history"]
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 7.0))
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    panel_to_ax = {panel: axes_flat[idx] for idx, panel in enumerate(panels)}

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    history_df = get_optimization_history(study)
    valid_history = history_df.copy() if history_df is not None and not history_df.empty else None

    x, y_upper, _, y_lower = generate_airfoil_profile(coeffs)
    ax1 = panel_to_ax["current_profile"]
    ax1.plot(x, y_upper, "b-", linewidth=2, label="Upper")
    ax1.plot(x, y_lower, "r-", linewidth=2, label="Lower")
    ax1.fill_between(x, y_lower, y_upper, alpha=0.2, color="lightblue")

    ax1.set_xlabel("x (m)", fontsize=12)
    ax1.set_ylabel("y (m)", fontsize=12)
    title_text = "Current Airfoil Profile"
    if trial_index is not None:
        title_text += f" - Trial index={trial_index}"
    ax1.set_title(title_text, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, config.chord])
    ax1.set_aspect("equal", adjustable="box")

    ax_ld = panel_to_ax["ld_history"]
    if valid_history is not None and not valid_history.empty:
        trial_indices = (
            valid_history["trial_index"].values
            if "trial_index" in valid_history.columns
            else range(len(valid_history))
        )
        for idx, aoa in enumerate(config.aoas):
            col_name = f"ld_aoa_{aoa}"
            if col_name in valid_history.columns:
                values = valid_history[col_name].values
                color = colors[idx % len(colors)]
                ax_ld.plot(
                    trial_indices,
                    values,
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                    label=f"AoA = {aoa}°",
                    alpha=0.7,
                )
        ax_ld.legend(loc="best", fontsize=9)
    ax_ld.set_xlabel("Trial Index", fontsize=12)
    ax_ld.set_ylabel("L/D", fontsize=12)
    ax_ld.set_title("Optimization History - L/D", fontsize=14, fontweight="bold")
    ax_ld.grid(True, alpha=0.3)
    ax_ld.set_xlim([0, config.n_trials])

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, hspace=0.35)
    filename = f"trial_{trial_index:04d}.png" if trial_index is not None else "trial_latest.png"
    final_path = figs_dir / filename
    tmp_path = figs_dir / (filename + ".tmp")
    plt.savefig(str(tmp_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    os.replace(str(tmp_path), str(final_path))


if __name__ == "__main__":
    from utils import default_coeffs

    visualize_results(coeffs=default_coeffs(), trial_index=0)
