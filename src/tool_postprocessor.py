import sys
from typing import Dict, Optional

import config
from utils import default_coeffs
from tool_mesher import mesher


def postprocessor(sim_subdir: str) -> Optional[Dict[str, float]]:
    """
    从 forceCoeffs 结果读取 Cl、Cd，计算升阻比 L/D。

    Args:
        sim_subdir: 仿真子文件夹，如 "AoA=2.0"

    Returns:
        {"cl", "cd", "ld"} 或 None
    """
    from utils import CD_EPS

    sim_dir = config.work_dir / "sims" / "simulations" / sim_subdir
    coeff_file = sim_dir / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"

    try:
        with open(coeff_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            return None
        last_line = lines[-1].split()
        # coefficient.dat: Time Cd Cd(f) Cd(r) Cl ...
        cd = float(last_line[1])
        cl = float(last_line[4])
        if abs(cd) < CD_EPS:
            return None
        return {"cl": cl, "cd": cd, "ld": cl / cd}
    except (FileNotFoundError, IndexError, ValueError, OSError) as e:
        print(f"Error in postprocessor: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    result = postprocessor(sim_subdir="AoA=2.0")
    if result:
        print(result)
    else:
        print("postprocess failed", file=sys.stderr)
