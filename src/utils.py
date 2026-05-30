from scipy.special import comb
import numpy as np
from typing import Dict, Any, List, Tuple

import config

SEM_DEFAULT = 0.01
CD_EPS = 1e-8


def bernstein_poly(i, n, x):
    """Bernstein 基函数"""
    return comb(n, i) * (x**i) * (1 - x)**(n - i)


def _cst_coeff_key(side: str, i: int) -> str:
    return f"cst_{side}_{i}"


def _ensure_cst_tail_coeff(coeffs: Dict, side: str, n: int) -> None:
    """固定尾缘 Bernstein 系数为 0，保证 TE 闭合。"""
    coeffs[_cst_coeff_key(side, n)] = 0.0


def _ensure_all_tail_coeffs(coeffs: Dict) -> None:
    _ensure_cst_tail_coeff(coeffs, "upper", config.n)
    _ensure_cst_tail_coeff(coeffs, "lower", config.n)


def _cst_shape_func(xi: np.ndarray, side: str, coeffs: Dict) -> np.ndarray:
    n = config.n
    coeffs_list = [coeffs[_cst_coeff_key(side, i)] for i in range(n + 1)]
    return sum(w * bernstein_poly(i, n, xi) for i, w in enumerate(coeffs_list))


def _cst_surface_y(xi: np.ndarray, side: str, coeffs: Dict) -> np.ndarray:
    """单面 CST 坐标 y（归一化弦长）。"""
    class_func = xi ** config.n1 * (1 - xi) ** config.n2
    y = class_func * _cst_shape_func(xi, side, coeffs)
    if side == "lower":
        y = -np.abs(y)
    else:
        y = np.abs(y)
    return y


def generate_airfoil_profile(coeffs: Dict, density_power: float = 1.5):
    """
    生成 CST 翼型上下表面。

    Returns:
        (x_upper, y_upper, x_lower, y_lower) 物理坐标 (m)
    """
    _ensure_all_tail_coeffs(coeffs)
    t = np.linspace(0, 1, config.num_points, endpoint=True)
    xi = t ** density_power
    xi = np.clip(xi, 0.0, 1.0)

    x = xi * config.chord
    y_upper = _cst_surface_y(xi, "upper", coeffs) * config.chord
    y_lower = _cst_surface_y(xi, "lower", coeffs) * config.chord
    return x, y_upper, x, y_lower


def generate_airfoil_contour(coeffs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    闭合翼型轮廓：TE(upper) → LE → TE(lower) → TE(upper)。
    """
    x, y_upper, _, y_lower = generate_airfoil_profile(coeffs)

    x_contour = np.concatenate([x[::-1], x[1:]])
    y_contour = np.concatenate([y_upper[::-1], y_lower[1:]])
    return x_contour, y_contour


def build_metric_order() -> List[str]:
    """目标分量顺序：各攻角 L/D。"""
    if "ld" in config.objectives:
        return [f"ld_aoa_{aoa}" for aoa in config.aoas]
    return []


def build_directions() -> List[str]:
    """与 build_metric_order 同序的优化方向。"""
    if "ld" in config.objectives:
        return ["maximize"] * len(config.aoas)
    return []


def parameters_to_coeffs(parameters: Dict[str, Any]) -> Dict:
    """将 Optuna trial.params 转换为 CST 系数字典。"""
    coeffs = {}
    for side in ("upper", "lower"):
        for i in range(config.n):
            coeffs[_cst_coeff_key(side, i)] = parameters[_cst_coeff_key(side, i)]
    _ensure_all_tail_coeffs(coeffs)
    return coeffs


def default_coeffs() -> Dict[str, float]:
    """NACA 0012 近似对称翼型默认 CST 系数（演示用）。"""
    coeffs = {}
    for side in ("upper", "lower"):
        for i in range(config.n):
            coeffs[_cst_coeff_key(side, i)] = 0.1 if side == "upper" else 0.1
    _ensure_all_tail_coeffs(coeffs)
    return coeffs
