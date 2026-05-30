import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def show_profile(profile_path: str, trial: int = None):
    df = pd.read_csv(profile_path)
    fig, ax = plt.subplots(figsize=(10, 4))

    if "surface" in df.columns:
        for surface in df["surface"].unique():
            sub = df[df["surface"] == surface]
            ax.plot(sub["x"], sub["y"], linewidth=2, label=surface)
    else:
        ax.plot(df["x"], df["y"], "b-", linewidth=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = "Airfoil Profile"
    if trial is not None:
        title += f" (Trial {trial})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_xlim([0, config.chord])

    def _on_click(event):
        if event.inaxes != ax:
            return
        print(f"x={event.xdata:.6f}, y={event.ydata:.6f}")

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive airfoil profile viewer")
    parser.add_argument("--profile", type=str, required=True, help="Path to profile CSV")
    parser.add_argument("--trial", type=int, default=None, help="Trial ID for title")
    parser.add_argument("--work-dir", type=str, default=None, help="工作目录")
    args = parser.parse_args()

    if args.work_dir:
        config.set_work_dir(args.work_dir)

    show_profile(args.profile, trial=args.trial)


if __name__ == "__main__":
    main()
