import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cadquery as cq

import config
from utils import generate_airfoil_contour


def _default_stp_path() -> Path:
    return config.work_dir / "sims" / "airfoil.stp"


def _airfoil_contour_points(coeffs: Dict) -> List[Tuple[float, float]]:
    """闭合翼型轮廓点，单位 mm（与 mesher 一致去重 TE）。"""
    x_c, y_c = generate_airfoil_contour(coeffs)
    if (
        len(x_c) > 1
        and abs(x_c[0] - x_c[-1]) < 1e-10
        and abs(y_c[0] - y_c[-1]) < 1e-10
    ):
        x_c = x_c[:-1]
        y_c = y_c[:-1]

    scale = 1000.0
    return [(float(x) * scale, float(y) * scale) for x, y in zip(x_c, y_c)]


def _build_airfoil_solid(coeffs: Dict) -> cq.Workplane:
    points = _airfoil_contour_points(coeffs)
    dz_mm = 0.1 * config.chord * 1000.0
    return (
        cq.Workplane("XY")
        .polyline(points)
        .close()
        .extrude(dz_mm)
    )


def cader(coeffs: Dict, output_path: str = None) -> bool:
    """从 CST 系数生成翼型 STP。默认输出 sims/airfoil.stp。"""
    out = Path(output_path) if output_path else _default_stp_path()
    try:
        solid = _build_airfoil_solid(coeffs)
        out.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(solid, str(out), exportType="STEP")
        return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"生成 STP 失败: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    from utils import default_coeffs

    coeffs = default_coeffs()
    if cader(coeffs):
        print(f"✓ STP 已生成: {_default_stp_path()}")
    else:
        print("✗ STP 生成失败", file=sys.stderr)
        sys.exit(1)
