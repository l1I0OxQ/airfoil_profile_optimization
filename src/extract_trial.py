import argparse
import sys
from pathlib import Path

import config
import optuna
from tool_cader import cader
from utils import generate_airfoil_profile, parameters_to_coeffs


class ExtractError(Exception):
    """Raised when trial extraction fails for a known reason."""
    pass


def extract_trial(trial_id: int, db_path: str, output_dir: str) -> str:
    if not Path(db_path).exists():
        raise ExtractError(f"Database not found: {db_path}. Run optimization first.")

    study = optuna.load_study(
        study_name="airfoil_optimization",
        storage=f"sqlite:///{db_path}",
    )

    if trial_id < 0 or trial_id >= len(study.trials):
        max_id = len(study.trials) - 1
        raise ExtractError(f"Trial {trial_id} not found (valid range: 0..{max_id})")

    trial = study.trials[trial_id]

    if trial.state == optuna.trial.TrialState.PRUNED:
        print(f"Warning: Trial {trial_id} was pruned.")
    elif trial.state == optuna.trial.TrialState.FAIL:
        print(f"Warning: Trial {trial_id} failed.")
    elif trial.state != optuna.trial.TrialState.COMPLETE:
        raise ExtractError(f"Trial {trial_id} is not complete (state: {trial.state})")

    coeffs = parameters_to_coeffs(trial.params)
    x, y_upper, _, y_lower = generate_airfoil_profile(coeffs)

    out_dir = Path(output_dir) / f"trial_{trial_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    params_file = out_dir / f"trial_{trial_id:04d}_params.csv"
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("parameter,value\n")
        for key, value in coeffs.items():
            f.write(f"{key},{value}\n")

    profile_file = out_dir / f"trial_{trial_id:04d}_profile.csv"
    with open(profile_file, "w", encoding="utf-8") as f:
        f.write("surface,x,y\n")
        for xi, yi in zip(x, y_upper):
            f.write(f"upper,{xi:.6f},{yi:.6f}\n")
        for xi, yi in zip(x, y_lower):
            f.write(f"lower,{xi:.6f},{yi:.6f}\n")

    print(f"Parameters saved: {params_file}")
    print(f"Profile saved: {profile_file}")

    stp_path = out_dir / "airfoil.stp"
    if cader(coeffs, output_path=str(stp_path)):
        print(f"STP geometry saved: {stp_path}")
    else:
        raise ExtractError("CAD generation failed.")

    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract trial solution (parameters + profile + STP)")
    parser.add_argument("--work-dir", type=str, default=None, help="工作目录")
    parser.add_argument("--db", type=str, default=None, help="Path to Optuna SQLite database")
    parser.add_argument("--trial", type=int, required=True, help="Trial ID to extract")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory")

    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else config.work_dir

    db = args.db if args.db else str(work_dir / "logs" / "airfoil_optim.db")
    out = args.output_dir if args.output_dir else str(work_dir / "solution")

    if args.work_dir:
        config.set_work_dir(args.work_dir)

    try:
        extract_trial(args.trial, db, out)
    except ExtractError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
