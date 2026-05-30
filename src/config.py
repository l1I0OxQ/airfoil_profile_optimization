import json
import os
from dataclasses import dataclass, field
from pathlib import Path

INSTALL_DIR = Path(__file__).resolve().parent.parent
_EXAMPLE_PATH = Path(__file__).resolve().parent / "config_example.json"


@dataclass
class Config:
    """All fields use snake_case, matching the JSON keys in config files."""

    solver_env_path: str = ""

    # CST airfoil parameters
    n: int = 3
    n1: float = 0.5
    n2: float = 1.0
    min_cst_coeff: float = 0.03
    max_cst_coeff: float = 0.15

    # Objectives
    objectives: list = field(default_factory=lambda: ["ld"])

    # Geometry
    chord: float = 1.0
    num_points: int = 80
    farfield_radius: float = 5.0

    # Boundary layer mesh (BoundaryLayer field)
    bl_layers: int = 15
    bl_ratio: float = 1.2
    bl_first_height: float = 3e-4
    bl_recombine: bool = True
    wall_mesh_size: float = 0.002  # tangential spacing along airfoil surface (× chord)

    # Optimization
    n_trials: int = 20
    aoas: list = field(default_factory=lambda: [2.0, 4.0])
    uinf: float = 30.0
    block: int = 4
    use_parallel_solver: bool = True

    # Derived / non-JSON fields
    install_dir: Path = INSTALL_DIR
    work_dir: Path = INSTALL_DIR


_config: Config | None = None
CONFIG_KEYS: frozenset = frozenset()


def _load(work_dir: Path | None = None) -> Config:
    """两层合并：先读 config_example.json 默认值，再用工作目录的 config.json 覆盖。"""
    global CONFIG_KEYS

    defaults = json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    target = work_dir or (INSTALL_DIR if _config is None else _config.work_dir)
    user_path = target / "config.json"
    if user_path.exists():
        defaults.update(json.loads(user_path.read_text(encoding="utf-8")))

    CONFIG_KEYS = frozenset(defaults.keys())

    kwargs = dict(defaults)
    kwargs["install_dir"] = INSTALL_DIR
    kwargs["work_dir"] = target

    return Config(**kwargs)


def set_work_dir(path):
    """设置工作目录并重新加载配置。"""
    global _config
    work_dir = Path(path).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "sims"]:
        (work_dir / sub).mkdir(exist_ok=True)
    _config = _load(work_dir)
    os.chdir(str(work_dir))


def __getattr__(name: str):
    """Delegate module-level access to the singleton Config instance."""
    if name.startswith('_'):
        raise AttributeError(name)
    cfg = _config
    if cfg is None:
        raise AttributeError(name)
    if hasattr(cfg, name):
        return getattr(cfg, name)
    raise AttributeError(name)


_config = _load()
