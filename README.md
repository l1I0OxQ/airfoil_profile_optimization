# ✈️ Airfoil Profile Optimization

**Multi-objective airfoil optimization with Bayesian search and RANS CFD — find the best lift-to-drag ratio across multiple angles of attack.**

## What It Does

Design a better wing. This tool uses **CST parameterization** to describe airfoil shapes, **Optuna's TPE sampler** to intelligently explore the design space, and **OpenFOAM RANS (simpleFoam)** to evaluate each candidate with CFD. The result: a **Pareto front** of optimal airfoils maximizing L/D at each angle of attack you care about.

- 🧬 **CST airfoil parameterization** — smooth, realistic shapes with just a handful of coefficients
- 🧠 **Bayesian optimization (TPE)** — finds good designs in fewer CFD runs than brute-force or genetic algorithms
- 🌪️ **RANS CFD evaluation** — 2D incompressible flow with MPI-parallel OpenFOAM
- 📊 **Live Pareto front** — watch the trade-off surface evolve as trials complete
- 🖥️ **PySide6 GUI** — configure, run, and monitor without touching the command line

## Quick Start

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone & set up
git clone <this-repo>
cd airfoil_profile_optimization
uv sync

# 3. Launch
./start.sh
```

> **Prerequisites:** Python 3.11+ and [OpenFOAM](https://openfoam.org/) with MPI. Point `solver_env_path` to your OpenFOAM installation in the config.

Choose a working directory in the GUI, tweak the design parameters, and hit **Start**.

## How It Works

```
CST coefficients → Gmsh mesh → RANS solve → Cl, Cd → L/D
       ↑                                              ↓
       └────────── Optuna TPE suggests next ←─────────┘
```

1. **Parameterize** — Upper and lower airfoil surfaces defined by CST Bernstein coefficients
2. **Mesh** — Gmsh generates a structured far-field mesh around the shape
3. **Simulate** — OpenFOAM simpleFoam solves RANS at each angle of attack (parallel per-AoA)
4. **Optimize** — Optuna feeds L/D back to the TPE sampler, guiding the next suggestion

## Output

Everything lands in your chosen working directory:

| What | Where |
|------|-------|
| Optimization database | `logs/airfoil_optim.db` |
| Pareto front CSV | `logs/pareto_front.csv` |
| Per-trial plots | `logs/figs/trial_*.png` |
| CAD geometry (STEP) | `sims/airfoil.stp` |
| Extracted trial data | `solution/trial_*/` |

## Project Structure

```
src/
├── frontend.py          # PySide6 GUI
├── backend.py           # Optuna optimization loop
├── config.py            # Configuration management
├── tool_mesher.py       # Gmsh mesh generation
├── tool_simulator.py    # OpenFOAM RANS case setup & run
├── tool_postprocessor.py# forceCoeffs → Cl, Cd, L/D
├── tool_cader.py        # CadQuery STEP export
└── visualizer.py        # Trial plots & Pareto visualization
sims/sim_ref/            # OpenFOAM template case
```

## License

MIT
