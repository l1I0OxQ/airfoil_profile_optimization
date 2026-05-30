import argparse
import logging
import multiprocessing
import os
import sys
from typing import Dict, List, Tuple

import optuna

import config
from utils import (
    build_metric_order,
    build_directions,
    _ensure_cst_tail_coeff,
    _cst_coeff_key,
    CD_EPS,
    SEM_DEFAULT,
)
from tool_mesher import mesher
from tool_simulator import simulator
from tool_postprocessor import postprocessor
from progresstracker import ProgressTracker

logger = logging.getLogger(__name__)

_current_study = None


def prepare_geometry(coeffs: Dict) -> None:
    if not mesher(coeffs):
        raise RuntimeError("网格生成失败")


def run_single_aoa(aoa: float, uinf: float) -> Dict[str, float]:
    sim_subdir = f"AoA={aoa}"
    result: Dict[str, float] = {}

    if "ld" in config.objectives:
        if not simulator(AoA=aoa, uInf=uinf, sim_subdir=sim_subdir):
            raise RuntimeError(f"RANS仿真失败: AoA={aoa}")
        pp = postprocessor(sim_subdir=sim_subdir)
        if pp is None:
            raise RuntimeError(f"后处理失败: AoA={aoa}")
        result["ld"] = pp["ld"]
        result["cl"] = pp["cl"]
        result["cd"] = pp["cd"]

    return result


def calc_parallel(coeffs: Dict, trial_index: int = None) -> List[Dict[str, float]]:
    prepare_geometry(coeffs)
    n_cpu = os.cpu_count() or 1
    block = max(1, config.block)
    n_aoa_workers = min(len(config.aoas), max(1, n_cpu // block))
    logger.info(
        "AoA parallel: %d workers, block=%d (cpus=%d)",
        n_aoa_workers,
        block,
        n_cpu,
    )
    with multiprocessing.Pool(processes=n_aoa_workers) as pool:
        results = pool.starmap(run_single_aoa, [(aoa, config.uinf) for aoa in config.aoas])
    return results


def evaluate(coeffs: Dict[str, float], trial_index: int = None) -> Dict[str, Tuple[float, float]]:
    sim_results: List[Dict[str, float]] = []
    if "ld" in config.objectives:
        sim_results = calc_parallel(coeffs, trial_index=trial_index)

    sem = SEM_DEFAULT
    results = {}
    for i, aoa in enumerate(config.aoas):
        aoa_results = sim_results[i] if sim_results else {}
        if "ld" in config.objectives:
            cd = aoa_results.get("cd", 0.0)
            if abs(cd) < CD_EPS:
                raise optuna.TrialPruned(f"Cd 过小: AoA={aoa}")
            results[f"ld_aoa_{aoa}"] = (aoa_results["ld"], sem)
    return results


def suggest_coeffs(trial: optuna.trial.Trial) -> Dict[str, float]:
    coeffs = {}
    for side in ("upper", "lower"):
        for i in range(config.n):
            coeffs[_cst_coeff_key(side, i)] = trial.suggest_float(
                _cst_coeff_key(side, i),
                config.min_cst_coeff,
                config.max_cst_coeff,
            )
        _ensure_cst_tail_coeff(coeffs, side, config.n)
    return coeffs


def objective(trial: optuna.trial.Trial) -> Tuple[float, ...]:
    coeffs = suggest_coeffs(trial)
    results = evaluate(coeffs, trial_index=trial.number)
    return tuple(results[name][0] for name in build_metric_order())


def _count_existing_trials(study: optuna.Study):
    complete = failed = pruned = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            complete += 1
        elif t.state == optuna.trial.TrialState.FAIL:
            failed += 1
        elif t.state == optuna.trial.TrialState.PRUNED:
            pruned += 1
    return complete, failed, pruned


def _create_study(db_path: str) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        directions=build_directions(),
        sampler=sampler,
        study_name="airfoil_optimization",
    )
    logger.info("使用采样器: TPE (启动试验数: 10, 随机种子: 42)")
    return study


def _resume_study(db_path: str) -> optuna.Study:
    if not os.path.exists(db_path):
        raise RuntimeError(f"无法续算: 数据库文件不存在 ({db_path})")

    study = optuna.load_study(
        study_name="airfoil_optimization",
        storage=f"sqlite:///{db_path}",
    )

    study_directions = [d.name.lower() for d in study.directions]
    current_directions = build_directions()
    if study_directions != current_directions:
        raise RuntimeError(
            f"无法续算: 优化目标方向不匹配。\n"
            f"  数据库: {study_directions}\n"
            f"  当前配置: {current_directions}"
        )

    complete, failed, pruned = _count_existing_trials(study)
    effective = complete + failed
    logger.info(
        f"加载已有 study: {complete} 完成, {failed} 失败, "
        f"{pruned} 修剪, {effective} 有效试验"
    )
    return study


def main():
    global _current_study

    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default=None, help="工作目录")
    parser.add_argument("--resume", action="store_true", help="从已有 checkpoint 续算")
    args = parser.parse_args()

    if args.work_dir:
        config.set_work_dir(args.work_dir)
    else:
        (config.work_dir / "logs" / "figs").mkdir(parents=True, exist_ok=True)
        (config.work_dir / "sims").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                str(config.work_dir / "logs" / "optimization.log"), encoding="utf-8"
            ),
        ],
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = str(config.work_dir / "logs" / "airfoil_optim.db")

    if args.resume:
        try:
            study = _resume_study(db_path)
        except RuntimeError:
            sys.exit(1)
        complete, failed, pruned = _count_existing_trials(study)
        effective_done = complete + failed
        if effective_done >= config.n_trials:
            logger.info(f"已完成目标试验数 ({effective_done} >= {config.n_trials})，无需续算。")
            sys.exit(0)
        tracker = ProgressTracker(
            n_trials=config.n_trials,
            aoas=config.aoas,
            start_trial_count=effective_done,
            start_failed_count=failed,
            start_pruned_count=pruned,
            resume=True,
        )
    else:
        study = _create_study(db_path)
        tracker = ProgressTracker(n_trials=config.n_trials, aoas=config.aoas)

    _current_study = study

    try:
        while not tracker.is_complete():
            study.optimize(objective, n_trials=1, callbacks=[tracker])
    finally:
        tracker.close()

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info(f"总试验次数（包括修剪）: {len(study.trials)}")
    logger.info(f"有效试验次数: {tracker.trial_count}")
    logger.info(f"Pareto 最优解数量：{len(study.best_trials)}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
