# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

二维翼型多目标优化演示 — CST 参数化翼型 + Optuna (TPE) 贝叶斯优化，目标为各攻角升阻比 L/D；CFD 评估采用绕流 RANS（simpleFoam + forceCoeffs，可选 MPI 域分解）。

All Python source is under `src/`. There are no tests.

## Commands

```bash
# Install / set up
uv sync

# Start the GUI (launches frontend.py which spawns backend.py as QProcess)
./start.sh

# Run backend directly (no GUI)
uv run python src/backend.py

# Resume a previous optimization from its checkpoint
uv run python src/backend.py --work-dir <path> --resume

# Run individual tools standalone (for debugging / manual use)
uv run python src/tool_cader.py
uv run python src/tool_mesher.py
uv run python src/tool_simulator.py
uv run python src/tool_postprocessor.py

# Extract a completed trial's params, profile CSV, and STEP geometry
uv run python src/extract_trial.py --trial <ID> --work-dir <path>

# View an extracted profile CSV interactively (click to read coordinates)
uv run python src/show_profile.py --profile <path> --trial <ID>

# Run any tool with --work-dir to override the working directory
```

## Architecture

### Two-Process Design

- **src/frontend.py** (~1077 lines) — PySide6 GUI. Config editing, process control, log display, Pareto table, trial extraction, per-trial image viewer. Launches `src/backend.py` via `QProcess` and monitors its stdout/stderr + log files.
- **src/backend.py** (~221 lines) — Optuna optimization loop. Lightweight: delegates to tool modules and calls `ProgressTracker` (Optuna callback) for logging/visualization. Accepts `--work-dir` and `--resume`.

**IPC**: Frontend polls `logs/optimization.log` (parsed by regex for progress) and `logs/pareto_front.csv` (parsed as CSV) every 1-2 seconds via QTimers. There is no socket/RPC — it's file-based.

### Config (src/config.py)

Two-layer merge: `src/config_example.json` provides defaults, then `<work_dir>/config.json` overrides. Module exposes fields as `config.<field>` via `__getattr__` delegating to a singleton `Config` dataclass.

Key fields: `chord`, `n`, `n1`, `n2`, `min_cst_coeff`, `max_cst_coeff`, `aoas`, `uinf`, `block` (MPI cores per case), `use_parallel_solver`, `objectives` (`["ld"]` only — single objective type, hardcoded).

Call `config.set_work_dir(path)` to switch working directory and reload config. Auto-creates `logs/` and `sims/` subdirectories.

### Airfoil Parameterization (src/utils.py, ~107 lines)

CST (Class-Shape Transformation) with independent upper/lower Bernstein coefficients (`cst_upper_0..n-1`, `cst_lower_0..n-1`); tail coefficients are hardcoded to 0 for TE closure.

- `generate_airfoil_profile()` → (x_upper, y_upper, x_lower, y_lower) in physical coordinates
- `generate_airfoil_contour()` → closed contour (x, y) for Gmsh / CAD (trailing edge duplicated removed)
- `parameters_to_coeffs()` → converts Optuna trial params → CST coeff dict
- `default_coeffs()` → NACA 0012-approximating default coefficients
- `build_metric_order()` / `build_directions()` → metric names and optimize directions for Optuna

### CAD Export (src/tool_cader.py, ~65 lines)

Optional standalone tool: CST coefficients → CadQuery thin solid → STEP. Not used in the optimization loop; called by `extract_trial.py` for final export.

### Mesh Generation (src/tool_mesher.py, ~229 lines)

Gmsh OCC-based 2D mesh with Z-extrusion (1 layer, recombined for hex/prism). Key features:
- Airfoil contour as OCC spline, farfield rectangular domain at radius `farfield_radius * chord`
- BoundaryLayer field on airfoil surface (Gmsh BL field, fan at trailing edge)
- Surface classification into patches: `airfoil`, `inlet`, `outlet`, `top`, `bottom`, `frontAndBack`
- Output: MSH 2.2 format → `sims/airfoil.msh`

### OpenFOAM Simulator (src/tool_simulator.py, ~112 lines)

Per-AoA workflow:
1. Copy `sims/sim_ref/` → `sims/simulations/AoA=<val>/`
2. Copy mesh into sim dir
3. Edit `keyParameters` dict via PyFoam (Uinf, alpha, chord, block count, nuTilda initial)
4. Run `Allclean` then `Allrun` (serial) or `Allrun-parallel` (MPI decomposePar + runParallel)

### Postprocessor (src/tool_postprocessor.py, ~46 lines)

Reads `postProcessing/forceCoeffs1/0/coefficient.dat`, parses last line for Cl, Cd, and computes L/D = Cl/Cd. Returns None on failure (file missing, Cd near zero).

### Trial Extraction (src/extract_trial.py, ~95 lines)

After optimization, extract a specific trial: loads Optuna study → reconstructs profile → saves:
- `solution/trial_NNNN/trial_NNNN_params.csv` (CST coefficients)
- `solution/trial_NNNN/trial_NNNN_profile.csv` (x,y per surface)
- `solution/trial_NNNN/airfoil.stp` (STEP geometry via cader)

### Progress Tracker (src/progresstracker.py, ~195 lines)

Optuna `callback` that logs each trial's results, best-so-far values, elapsed/ETA time, writes `pareto_front.csv`, and calls `visualize_results()` per trial.

### Visualization (src/visualizer.py, ~109 lines)

Generates per-trial PNG: airfoil profile plot + L/D history from Optuna study dataframe.

### Tool Pipeline (per trial)

```
suggest_coeffs(trial)                  # Optuna TPE suggests CST coeffs
  → prepare_geometry(coeffs)           # mesher(): Gmsh → airfoil.msh
  → calc_parallel(coeffs)              # multiprocessing.Pool, one worker per AoA
      [for each AoA in parallel]:
        run_single_aoa(aoa, uinf):
          simulator()                  # RANS simpleFoam (Allrun-parallel if use_parallel_solver)
          postprocessor()              # forceCoeffs → Cl, Cd, L/D
  → evaluate()                         # collect results, prune if Cd ~ 0
  → visualize_results()                # per-trial PNG (called by ProgressTracker callback)
```

### Objectives

- `"ld"` — one `ld_aoa_{aoa}` metric per AoA (maximize). Multi-objective Optuna study with one direction per AoA.
- If Cd is below `CD_EPS` (1e-8), the trial is pruned via `optuna.TrialPruned`.

### Optimization (src/backend.py)

- Sampler: `TPESampler` with `n_startup_trials=10` and `seed=42`
- DB: SQLite at `logs/airfoil_optim.db`, study name `airfoil_optimization`
- Resume: `--resume` flag loads existing study, validates direction compatibility, skips if target reached
- AoA parallelism: `min(len(aoas), cpu_count // block)` workers via `multiprocessing.Pool`
- The loop runs one trial at a time (`study.optimize(objective, n_trials=1, callbacks=[tracker])`) so the callback fires after each trial

### Key Data Files (WORK_DIR)

| Path | Description |
|------|-------------|
| `logs/airfoil_optim.db` | Optuna SQLite DB (study: `airfoil_optimization`) |
| `logs/pareto_front.csv` | Pareto front (columns: `trial`, `Ld@AoA=...°`) |
| `logs/optimization.log` | Full log with per-trial results, best, timing |
| `logs/figs/trial_NNNN.png` | Per-trial profile + history plots |
| `logs/figs/trial_latest.png` | Overwritten each trial (legacy) |
| `sims/airfoil.msh` | Generated mesh in 2.2 format |
| `sims/airfoil.stp` | CAD geometry (STEP, from `tool_cader.py`) |
| `sims/simulations/AoA=.../` | OpenFOAM cases (one per AoA per trial) |
| `solution/trial_NNNN/` | Extracted params + profile + STP |
| `config.json` | User override config (work-dir root) |

### Source Module Map

| File | Lines | Role |
|------|-------|------|
| `frontend.py` | 1077 | PySide6 GUI (config, process mgmt, Pareto table, extraction UI) |
| `backend.py` | 221 | Optuna optimization loop |
| `tool_mesher.py` | 229 | Gmsh mesh generation |
| `progresstracker.py` | 195 | Optuna callback + logging + timing |
| `tool_simulator.py` | 112 | OpenFOAM case setup & run |
| `utils.py` | 107 | CST parameterization + metric helpers |
| `visualizer.py` | 109 | Trial plots & Pareto visualization |
| `extract_trial.py` | 95 | Post-optimization trial extraction |
| `tool_cader.py` | 65 | CadQuery STEP export |
| `tool_postprocessor.py` | 46 | forceCoeffs → Cl, Cd, L/D |
| `show_profile.py` | 57 | Interactive profile CSV viewer |
| `config.py` | 96 | Config loading/merging |
| `__init__.py` | 1 | Empty (package marker) |

### External Dependencies

- **uv** + Python 3.11 (see `pyproject.toml`, `uv.lock`; run `uv sync` to create `.venv/`)
- **OpenFOAM** (MPI) via `solver_env_path` — `Allrun-parallel` uses `decomposePar` + `runParallel`; set `block` ≤ physical cores
- **gmsh** Python bindings, **cadquery**, **PyFoam**, **PySide6**, **matplotlib**, **optuna**, **numpy**, **pandas**, **scipy**

### Simulation Templates

```
sims/sim_ref/          # OpenFOAM template (simpleFoam, Spalart-Allmaras)
sims/sim_ref/0.orig/   # Initial/boundary condition fields (U, p, nut, omega, etc.)
sims/sim_ref/system/   # controlDict, fvSchemes, fvSolution, decomposeParDict
sims/sim_ref/Allrun         # Serial run script
sims/sim_ref/Allrun-parallel # MPI run script (decomposePar + runParallel)
sims/sim_ref/Allclean        # Cleanup script
```

### GUI Details

- Theme: GitHub-light via `pyqtdarktheme`, plus custom stylesheet
- Working directory: prompted on startup; persist across runs
- Clear History: removes all sims, logs, figs, db, solution — keeps config.json and sim_ref/
- Image viewer: polls `logs/figs/` every 5s for new PNGs; prev/next/open buttons
- Pareto table: sortable by any metric column, ascending/descending toggle
- NoWheelDoubleSpinBox/NoWheelSpinBox: Qt spinboxes that ignore mouse wheel (prevents accidental changes)

### Pruning & Failure Handling

- **Cd too small** (`< 1e-8`): `optuna.TrialPruned` — pruned trials don't count toward the trial target
- **Mesher failure**: `RuntimeError` → backend logs failure
- **Simulator failure**: `RuntimeError` per AoA
- **Postprocessor failure**: `RuntimeError` per AoA
- **SIGSTOP/SIGCONT**: GUI pause uses Unix signals to suspend/resume the backend QProcess
