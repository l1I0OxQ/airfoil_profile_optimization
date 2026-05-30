import csv
from pathlib import Path
from typing import List, Dict, Tuple
import time
import logging

import config
import optuna
from utils import parameters_to_coeffs, SEM_DEFAULT
from visualizer import visualize_results


def _init_logger():
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger
    if logging.getLogger().handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = config.work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / "optimization.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def _pareto_log_path():
    return config.work_dir / "logs" / "pareto_front.csv"


class ProgressTracker:
    """优化进度追踪器（Optuna 回调）。"""

    def __init__(
        self,
        n_trials: int,
        aoas: List[float],
        start_trial_count: int = 0,
        start_failed_count: int = 0,
        start_pruned_count: int = 0,
        resume: bool = False,
    ):
        self.logger = _init_logger()
        self.n_trials = n_trials
        self.aoas = aoas
        self.resume = resume
        self.start_time = None
        self.best_ld_values = (
            {aoa: float("-inf") for aoa in aoas} if "ld" in config.objectives else {}
        )
        self.trial_count = start_trial_count
        self.failed_count = start_failed_count
        self.pruned_count = start_pruned_count
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            log = self.logger
            self.start_time = time.time()
            if self.resume:
                remaining = max(0, self.n_trials - self.trial_count)
                log.info(
                    f"续算模式: 已完成 {self.trial_count} 次有效试验, "
                    f"剩余 {remaining} 次, 目标总数 {self.n_trials}"
                )
            else:
                log.info(f"开始优化，共 {self.n_trials} 次有效试验")
            log.info(f"攻角列表：{self.aoas}")
            self._initialized = True

    def _build_results_from_values(self, values: Tuple[float, ...]) -> Dict[str, Tuple[float, float]]:
        sem = SEM_DEFAULT
        idx = 0
        results: Dict[str, Tuple[float, float]] = {}
        if "ld" in config.objectives:
            for aoa in self.aoas:
                results[f"ld_aoa_{aoa}"] = (values[idx], sem)
                idx += 1
        return results

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        log = self.logger
        self._ensure_initialized()
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.pruned_count += 1
            log.debug("试验被修剪（不计入总数）")
            return
        if trial.state == optuna.trial.TrialState.FAIL:
            self.trial_count += 1
            self.failed_count += 1
            log.warning(f"试验 #{self.trial_count}/{self.n_trials} 失败")
            return
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.values is None:
            return

        self.trial_count += 1
        values = tuple(float(v) for v in trial.values)
        results = self._build_results_from_values(values)

        ld_values = {}
        if "ld" in config.objectives:
            for aoa in self.aoas:
                key = f"ld_aoa_{aoa}"
                ld_values[aoa] = results[key][0]
                if ld_values[aoa] > self.best_ld_values[aoa]:
                    self.best_ld_values[aoa] = ld_values[aoa]

        elapsed_time = time.time() - self.start_time
        avg_time_per_trial = elapsed_time / self.trial_count if self.trial_count > 0 else 0
        remaining_trials = self.n_trials - self.trial_count
        estimated_remaining = avg_time_per_trial * remaining_trials

        log.info(f"{'='*80}")
        log.info(f"试验 #{self.trial_count}/{self.n_trials} 完成 (Trial {trial.number})")
        formatted_params = {k: f"{v:.4f}" for k, v in trial.params.items()}
        log.info(f"参数: {formatted_params}")
        if "ld" in config.objectives:
            ld_results_str = {
                f"AoA={aoa}°": f"L/D={ld_values[aoa]:.4f}" for aoa in self.aoas
            }
            log.info(f"升阻比: {ld_results_str}")

        log.info("当前最佳值:")
        if "ld" in config.objectives:
            for aoa in self.aoas:
                marker = "★" if ld_values[aoa] == self.best_ld_values[aoa] else " "
                log.info(
                    f"  {marker} AoA={aoa:4.1f}° L/D: "
                    f"{self.best_ld_values[aoa]:.4f} (当前: {ld_values[aoa]:.4f})"
                )
        log.info(
            f"已用时间: {elapsed_time/60:.1f} 分钟 | "
            f"预计剩余: {estimated_remaining/60:.1f} 分钟"
        )
        log.info(f"{'='*80}")

        self._write_pareto_front(study)

        try:
            visualize_results(
                coeffs=parameters_to_coeffs(trial.params),
                trial_index=trial.number,
                study=study,
                results=results,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"可视化失败 (Trial {trial.number}): {e}")

    def _build_pareto_entry(self, trial_number: int, values: Tuple[float, ...]) -> dict:
        entry: dict = {"trial": trial_number}
        idx = 0
        if "ld" in config.objectives:
            for aoa in self.aoas:
                entry[f"Ld@AoA={aoa}°"] = round(values[idx], 6)
                idx += 1
        return entry

    def _write_pareto_front(self, study: optuna.Study):
        pareto_path = _pareto_log_path()
        pareto_entries = []
        for best_trial in study.best_trials:
            entry = self._build_pareto_entry(
                best_trial.number,
                tuple(float(v) for v in best_trial.values),
            )
            pareto_entries.append(entry)
        if not pareto_entries:
            pareto_path.write_text("", encoding="utf-8")
            return
        with open(pareto_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pareto_entries[0].keys()))
            writer.writeheader()
            writer.writerows(pareto_entries)

    def is_complete(self) -> bool:
        return self.trial_count >= self.n_trials

    def close(self):
        log = self.logger
        total_time = time.time() - self.start_time if self.start_time else 0
        log.info("=" * 80)
        log.info("优化完成！")
        log.info(f"有效试验次数: {self.trial_count} (完成+失败)")
        log.info(f"完成次数: {self.trial_count - self.failed_count}")
        log.info(f"失败次数: {self.failed_count}")
        log.info(f"修剪次数: {self.pruned_count} (如 Cd 过小等，不计入总数)")
        log.info(f"总耗时: {total_time/60:.1f} 分钟")
        log.info("最终最佳值:")
        if "ld" in config.objectives:
            for aoa in self.aoas:
                log.info(f"  AoA={aoa:4.1f}° L/D: {self.best_ld_values[aoa]:.4f}")
        log.info("=" * 80)
