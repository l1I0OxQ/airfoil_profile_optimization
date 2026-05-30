import csv
import io
import json
import os
import re
import shutil
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.image as mpimg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel events."""
    def wheelEvent(self, event):
        event.ignore()


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_PATH = BASE_DIR / "src" / "backend.py"

def get_log_path():
    return config.work_dir / "logs" / "optimization.log"

def get_pareto_log_path():
    return config.work_dir / "logs" / "pareto_front.csv"

def get_figs_dir():
    return config.work_dir / "logs" / "figs"


class SortControl(QWidget):
    """Reusable sort key selector: label + combo + direction button."""
    changed = Signal()

    def __init__(self, label: str, show_none: bool = False, parent=None):
        super().__init__(parent)
        self._asc = True
        self._show_none = show_none
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(120)
        layout.addWidget(self.combo)
        self.dir_btn = QPushButton("▲")
        self.dir_btn.setFixedWidth(28)
        self.dir_btn.setToolTip("Toggle sort direction")
        layout.addWidget(self.dir_btn)
        layout.addStretch()

        self.combo.currentIndexChanged.connect(self.changed.emit)
        self.dir_btn.clicked.connect(self._toggle_dir)

    def _toggle_dir(self):
        self._asc = not self._asc
        self.dir_btn.setText("▲" if self._asc else "▼")
        self.changed.emit()

    def key(self):
        return self.combo.currentData()

    def ascending(self):
        return self._asc

    def refresh_keys(self, keys: list, current_key=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        if self._show_none:
            self.combo.addItem("(none)", None)
        for k in keys:
            self.combo.addItem(k, k)
        if current_key is not None:
            idx = self.combo.findData(current_key)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)


class ConfigPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("CONFIGURATION", parent)
        self.values = {k: getattr(config, k) for k in config.CONFIG_KEYS}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred))

        cst_form = QGroupBox("CST SETTINGS")
        f_cst = QFormLayout(cst_form)
        self.n_spin = NoWheelSpinBox()
        self.n_spin.setRange(3, 10)
        self.min_cst = NoWheelDoubleSpinBox()
        self.max_cst = NoWheelDoubleSpinBox()
        for w in [self.min_cst, self.max_cst]:
            w.setDecimals(3)
            w.setRange(0, 10)
            w.setSingleStep(0.01)
        f_cst.addRow("n (Bernstein order)", self.n_spin)
        f_cst.addRow("min_cst_coeff", self.min_cst)
        f_cst.addRow("max_cst_coeff", self.max_cst)
        layout.addWidget(cst_form)

        objective_box = QGroupBox("OPTIMIZATION OBJECTIVES")
        objective_form = QVBoxLayout(objective_box)
        self.obj_ld = QCheckBox("L/D (升阻比) - 各攻角多目标")
        self.obj_ld.setChecked(True)
        self.obj_ld.setEnabled(False)
        objective_form.addWidget(self.obj_ld)
        layout.addWidget(objective_box)

        common = QGroupBox("COMMON PARAMETERS")
        common_form = QFormLayout(common)
        self.n_trials = NoWheelSpinBox()
        self.n_trials.setRange(1, 10000)
        self.aoas = QLineEdit()
        self.uinf = NoWheelDoubleSpinBox()
        self.uinf.setRange(0, 100)
        self.uinf.setDecimals(3)
        self.block = NoWheelSpinBox()
        self.block.setRange(1, 128)
        common_form.addRow("n_trials", self.n_trials)
        common_form.addRow("aoas (deg) (e.g. 2.0,6.0)", self.aoas)
        common_form.addRow("uinf (m/s)", self.uinf)
        common_form.addRow("block (MPI cores)", self.block)
        layout.addWidget(common)

        self._load_ui_values()

    def _load_ui_values(self):
        self.n_spin.setValue(int(self.values["n"]))
        self.min_cst.setValue(float(self.values["min_cst_coeff"]))
        self.max_cst.setValue(float(self.values["max_cst_coeff"]))

        self.n_trials.setValue(int(self.values["n_trials"]))
        aoas = self.values["aoas"]
        self.aoas.setText(",".join(str(x) for x in aoas))
        self.obj_ld.setChecked(True)
        self.uinf.setValue(float(self.values["uinf"]))
        self.block.setValue(int(self.values.get("block", 4)))

    def _selected_objectives(self) -> List[str]:
        return ["ld"]

    def save_config(self):
        """Collect UI values, validate, and write config.json. Returns (success, message)."""
        try:
            aoa_vals = [float(x.strip()) for x in self.aoas.text().split(",") if x.strip()]
            if not aoa_vals:
                return False, "请至少配置一个攻角"
            payload = {
                "objectives": self._selected_objectives(),
                "chord": float(self.values["chord"]),
                "farfield_radius": float(self.values.get("farfield_radius", 20.0)),
                "n": int(self.n_spin.value()),
                "n1": float(self.values.get("n1", 0.5)),
                "n2": float(self.values.get("n2", 1.0)),
                "min_cst_coeff": float(self.min_cst.value()),
                "max_cst_coeff": float(self.max_cst.value()),
                "n_trials": int(self.n_trials.value()),
                "aoas": aoa_vals,
                "uinf": float(self.uinf.value()),
                "block": int(self.block.value()),
            }
            config_path = config.work_dir / "config.json"
            filtered = {k: v for k, v in payload.items() if k in config.CONFIG_KEYS}
            config_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
            return True, "Config saved"
        except (TypeError, OSError, ValueError) as exc:
            return False, f"Failed to save config: {exc}"


class VisualizationPanel(QGroupBox):
    info = Signal(str)

    def __init__(self, parent=None):
        super().__init__("VISUALIZATION", parent)
        self.current_idx = -1
        self.image_files: List[Path] = []
        self._manual_images: set[Path] = set()
        self._build_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(5000)
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.pick_btn = QPushButton("Open PNG")
        self.status = QLabel("No figure yet")
        top_row.addWidget(self.prev_btn)
        top_row.addWidget(self.next_btn)
        top_row.addWidget(self.pick_btn)
        top_row.addStretch()
        top_row.addWidget(self.status)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")

        layout.addLayout(top_row)
        layout.addWidget(self.canvas)

        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)
        self.pick_btn.clicked.connect(self.pick_file)

    def refresh(self):
        files = sorted(get_figs_dir().glob("*.png"), key=lambda p: p.stat().st_mtime)
        merged = sorted(set(files) | self._manual_images, key=lambda p: p.stat().st_mtime)
        if merged != self.image_files:
            self.image_files = merged
            if merged:
                self.current_idx = len(merged) - 1
                self.show_current()

    def show_current(self):
        if not self.image_files:
            self.status.setText("No figure found in figs/")
            return
        image_path = self.image_files[self.current_idx]
        try:
            img = mpimg.imread(str(image_path))
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.axis("off")
            self.figure.subplots_adjust(left=0, right=1, top=0.97, bottom=0)
            self.canvas.draw_idle()
            self.status.setText(f"{self.current_idx + 1}/{len(self.image_files)} {image_path.name}")
        except (OSError, ValueError) as exc:
            self.info.emit(f"Figure load failed: {exc}")

    def show_prev(self):
        if not self.image_files:
            return
        self.current_idx = max(0, self.current_idx - 1)
        self.show_current()

    def show_next(self):
        if not self.image_files:
            return
        self.current_idx = min(len(self.image_files) - 1, self.current_idx + 1)
        self.show_current()

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Figure", str(get_figs_dir()), "PNG (*.png)")
        if not path:
            return
        p = Path(path)
        self._manual_images.add(p)
        if p not in self.image_files:
            self.image_files.append(p)
            self.image_files = sorted(self.image_files, key=lambda x: x.stat().st_mtime)
        self.current_idx = self.image_files.index(p)
        self.show_current()

    def clear_manual_images(self):
        self._manual_images.clear()


class LogConsole(QGroupBox):
    progress = Signal(int, int)
    pareto_front = Signal(list)
    time_info = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__("LOG CONSOLE", parent)
        self.offset = 0
        self._last_pareto_content = ""
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_log_file)
        self.timer.start(1000)
        self.pareto_timer = QTimer(self)
        self.pareto_timer.timeout.connect(self._poll_pareto_file)
        self.pareto_timer.start(2000)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(5000)
        layout.addWidget(self.text)

    def append_line(self, line: str):
        line = line.rstrip("\n")
        if not line:
            return
        fmt = QTextCharFormat()
        if "ERROR" in line:
            fmt.setForeground(QColor("#cf222e"))
        elif "WARNING" in line:
            fmt.setForeground(QColor("#9a6700"))
        elif "★" in line:
            fmt.setForeground(QColor("#1a7f37"))
        elif "INFO" in line:
            fmt.setForeground(QColor("#0550ae"))
        else:
            fmt.setForeground(QColor("#24292f"))

        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(line + "\n", fmt)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()
        self._parse_status(line)

    def poll_log_file(self):
        if not get_log_path().exists():
            return
        with get_log_path().open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        if not chunk:
            return
        for line in chunk.splitlines():
            self.append_line(line)

    def _parse_status(self, line: str):
        complete_match = re.search(r"试验\s+#(\d+)/(\d+)\s+完成\s+\(Trial\s+(\d+)\)", line)
        if complete_match:
            trial_idx = int(complete_match.group(1))
            trial_total = int(complete_match.group(2))
            self.progress.emit(trial_idx, trial_total)

        time_match = re.search(r"已用时间:\s*([\d.]+)\s*分钟\s*\|\s*预计剩余:\s*([\d.]+)\s*分钟", line)
        if time_match:
            self.time_info.emit(time_match.group(1), time_match.group(2))

    def _poll_pareto_file(self):
        if not get_pareto_log_path().exists():
            return
        content = get_pareto_log_path().read_text(encoding="utf-8", errors="ignore")
        if content == self._last_pareto_content:
            return
        self._last_pareto_content = content

        front_data = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            entry = {}
            for k, v in row.items():
                if k == "trial":
                    entry[k] = int(v)
                else:
                    entry[k] = float(v)
            front_data.append(entry)

        if front_data:
            self.pareto_front.emit(front_data)


class ProcessPanel(QWidget):
    log_message = Signal(str)
    process_state = Signal(str)
    run_clicked = Signal()
    continue_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: Optional[QProcess] = None
        self._paused: bool = False
        self._build_ui()
        self._set_state("IDLE")

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        state_row = QHBoxLayout()
        self.led = QLabel("●")
        self.state_label = QLabel("IDLE")
        state_row.addWidget(self.led)
        state_row.addWidget(self.state_label)
        state_row.addStretch()
        layout.addLayout(state_row)

        self.run_btn = QPushButton("▶ RUN")
        self.run_btn.setToolTip("Start a new optimization from scratch")
        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setToolTip("Suspend the running optimization")
        self.unpause_btn = QPushButton("↻ UNPAUSE")
        self.unpause_btn.setToolTip("Resume the paused optimization process")
        self.continue_btn = QPushButton("▶▶ CONTINUE")
        self.continue_btn.setToolTip("Continue optimization from previous checkpoint")
        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setToolTip("Terminate the current optimization process")
        layout.addWidget(self.run_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.unpause_btn)
        layout.addWidget(self.continue_btn)
        layout.addWidget(self.stop_btn)

        self.run_btn.clicked.connect(self._on_run_clicked)
        self.pause_btn.clicked.connect(self.pause_backend)
        self.unpause_btn.clicked.connect(self.unpause_backend)
        self.continue_btn.clicked.connect(self._on_continue_clicked)
        self.stop_btn.clicked.connect(self.stop_backend)

    def _set_state(self, state: str):
        colors = {
            "IDLE": "#9aa5b1",
            "RUNNING": "#0969da",
            "PAUSED": "#ffbd2e",
            "DONE": "#39ff14",
            "ERROR": "#ff5f56",
        }
        self.state_label.setText(state)
        self.led.setStyleSheet(f"color:{colors.get(state, '#9aa5b1')}; font-size:20px;")
        self.process_state.emit(state)

    def _on_run_clicked(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.log_message.emit("Process already running.")
            return
        self.run_clicked.emit()

    def _on_continue_clicked(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.log_message.emit("Process already running.")
            return
        self.continue_clicked.emit()

    def start_backend(self, resume: bool = False):
        if not resume:
            clear_working_dir()
        self._paused = False
        self.process = QProcess(self)
        args = [str(BACKEND_PATH), "--work-dir", str(config.work_dir)]
        if resume:
            args.insert(1, "--resume")
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.setWorkingDirectory(str(BASE_DIR))
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.start()
        ok = self.process.waitForStarted(2000)
        if ok:
            self._set_state("RUNNING")
            mode = "( resume mode)" if resume else ""
            self.log_message.emit(f"backend.py started{mode}.")
        else:
            self._set_state("ERROR")
            self.log_message.emit("Failed to start backend.py")
            self.process.kill()
            if not self.process.waitForFinished(3000):
                self.log_message.emit("WARNING: failed to kill orphaned backend process")
            self.process = None

    def pause_backend(self):
        if self._paused:
            return
        if not self.process or self.process.state() == QProcess.NotRunning:
            return
        pid = self.process.processId()
        if pid and os.name != "nt":
            os.kill(pid, signal.SIGSTOP)
            self._paused = True
            self._set_state("PAUSED")
            self.log_message.emit("Process paused.")

    def unpause_backend(self):
        if not self._paused:
            return
        if not self.process or self.process.state() == QProcess.NotRunning:
            return
        pid = self.process.processId()
        if pid and os.name != "nt":
            os.kill(pid, signal.SIGCONT)
            self._paused = False
            self._set_state("RUNNING")
            self.log_message.emit("Process unpaused.")

    def stop_backend(self):
        if not self.process or self.process.state() == QProcess.NotRunning:
            return
        ret = QMessageBox.question(
            self, "Stop Run", "Stop current optimization process?"
        )
        if ret != QMessageBox.Yes:
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self._paused = False
        self._set_state("IDLE")
        self.log_message.emit("Process stopped.")

    def _on_stdout(self):
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for line in data.splitlines():
            self.log_message.emit(line)

    def _on_stderr(self):
        if not self.process:
            return
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="ignore")
        for line in data.splitlines():
            if line.strip():
                self.log_message.emit(f"ERROR: {line}")

    def _on_finished(self, exit_code, _status):
        self._paused = False
        if exit_code == 0:
            self._set_state("DONE")
            self.log_message.emit("Process completed successfully.")
        else:
            self._set_state("ERROR")
            self.log_message.emit(f"Process exited with code {exit_code}.")


class ParetoView(QGroupBox):
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__("PARETO FRONT", parent)
        self._front_data_cache: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.sort_primary = SortControl("Sort by:")
        self.sort_secondary = SortControl("  then:", show_none=True)
        layout.addWidget(self.sort_primary)
        layout.addWidget(self.sort_secondary)

        self.pareto_table = QTableWidget(0, 0)
        self.pareto_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pareto_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pareto_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.pareto_table)
        self.pareto_count = QLabel("Pareto solutions: 0")
        layout.addWidget(self.pareto_count)

        self.sort_primary.changed.connect(self._on_sort_changed)
        self.sort_secondary.changed.connect(self._on_sort_changed)

    def update_pareto_front(self, front_data: list):
        if not front_data:
            return

        self._front_data_cache = front_data

        keys = list(front_data[0].keys())
        if "trial" in keys:
            keys.remove("trial")
            keys.insert(0, "trial")

        self._refresh_sort_combos(keys)
        sorted_data = self._sort_pareto_data(front_data)

        self.pareto_table.setColumnCount(len(keys))
        self.pareto_table.setHorizontalHeaderLabels(keys)
        self.pareto_table.setRowCount(len(sorted_data))

        for i, data in enumerate(sorted_data):
            for col, key in enumerate(keys):
                val = data.get(key)
                if val is None:
                    display = "-"
                elif key == "trial":
                    display = str(int(val))
                elif isinstance(val, float):
                    display = f"{val:.4f}"
                else:
                    display = str(val)
                self.pareto_table.setItem(i, col, QTableWidgetItem(display))

        self.pareto_count.setText(f"Pareto solutions: {len(front_data)}")

    def _refresh_sort_combos(self, keys: list):
        primary_key = self.sort_primary.key()
        secondary_key = self.sort_secondary.key()

        self.sort_primary.refresh_keys(keys, primary_key if primary_key in keys else keys[0] if keys else None)
        self.sort_secondary.refresh_keys(keys, secondary_key)

    def _sort_pareto_data(self, front_data: list) -> list:
        sort_keys = []
        pk = self.sort_primary.key()
        if pk:
            sort_keys.append((pk, self.sort_primary.ascending()))
        sk = self.sort_secondary.key()
        if sk:
            sort_keys.append((sk, self.sort_secondary.ascending()))

        if not sort_keys:
            return list(front_data)

        def sort_key(row):
            result = []
            for k, asc in sort_keys:
                val = row.get(k)
                if val is None:
                    val = float("inf") if asc else float("-inf")
                result.append(val if asc else -val)
            return tuple(result)

        return sorted(front_data, key=sort_key)

    def _on_sort_changed(self):
        if self._front_data_cache:
            self.update_pareto_front(self._front_data_cache)

    def reset(self):
        self._front_data_cache.clear()
        self.pareto_table.setRowCount(0)
        self.pareto_table.setColumnCount(0)
        self.pareto_count.setText("Pareto solutions: 0")


class ExtractPanel(QGroupBox):
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__("EXTRACT TRIAL SOLUTION", parent)
        self.extract_process: Optional[QProcess] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        extract_input_row = QHBoxLayout()
        extract_input_row.addWidget(QLabel("Trial ID:"))
        self.trial_id_input = NoWheelSpinBox()
        self.trial_id_input.setRange(0, 999999)
        self.trial_id_input.setToolTip("Enter a trial number to extract its geometry")
        extract_input_row.addWidget(self.trial_id_input)
        layout.addLayout(extract_input_row)
        self.extract_btn = QPushButton("EXTRACT")
        self.extract_btn.setToolTip("Generate profile CSV + parameter files into solution/")
        layout.addWidget(self.extract_btn)
        self.extract_status = QLabel("")
        layout.addWidget(self.extract_status)

        self.extract_btn.clicked.connect(self.extract_trial)

    def extract_trial(self):
        trial_id = self.trial_id_input.value()
        self.extract_btn.setEnabled(False)
        self.extract_status.setText("Extracting...")

        extract_script = str(BASE_DIR / "src" / "extract_trial.py")

        args = [extract_script, "--trial", str(trial_id), "--work-dir", str(config.work_dir)]
        self.extract_process = QProcess(self)
        self.extract_process.setProgram(sys.executable)
        self.extract_process.setArguments(args)
        self.extract_process.setWorkingDirectory(str(BASE_DIR))
        self.extract_process.finished.connect(self._on_extract_finished)
        self.extract_process.start()

    def _on_extract_finished(self, exit_code, _status):
        self.extract_btn.setEnabled(True)
        self.extract_status.setText("Completed" if exit_code == 0 else "Failed")
        if exit_code == 0:
            output = bytes(self.extract_process.readAllStandardOutput()).decode(
                "utf-8", errors="ignore"
            )
            self.log_message.emit(output.strip())

            profile_path = None
            for line in output.splitlines():
                if line.startswith("Profile saved: "):
                    profile_path = line[len("Profile saved: "):].strip()
                    break
            if profile_path:
                viewer_script = str(BASE_DIR / "src" / "show_profile.py")
                trial_id = self.trial_id_input.value()
                proc = QProcess(self)
                proc.setProgram(sys.executable)
                proc.setArguments([viewer_script, "--profile", profile_path,
                                   "--trial", str(trial_id)])
                proc.setWorkingDirectory(str(BASE_DIR))
                proc.start()
        else:
            stderr = bytes(self.extract_process.readAllStandardError()).decode(
                "utf-8", errors="ignore"
            )
            msg = stderr.strip() or "Unknown error"
            self.log_message.emit(f"ERROR: Trial extraction failed: {msg}")


class ControlPanel(QGroupBox):
    log_message = Signal(str)
    process_state = Signal(str)
    run_requested = Signal()
    continue_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("CONTROL", parent)
        self.trial_done = 0
        self.trial_total = 0

        self._build_ui()
        self._wire_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.process_panel = ProcessPanel()
        layout.addWidget(self.process_panel)

        self.progress_label = QLabel("Progress 0/0")
        self.progress = QProgressBar()
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)

        self.time_info = QLabel("Elapsed -- | ETA --")
        layout.addWidget(self.time_info)

        self.pareto_view = ParetoView()
        layout.addWidget(self.pareto_view)

        self.extract_panel = ExtractPanel()
        layout.addWidget(self.extract_panel)

        layout.addStretch()

    def _wire_signals(self):
        self.process_panel.log_message.connect(self.log_message)
        self.process_panel.process_state.connect(self.process_state)
        self.process_panel.run_clicked.connect(self._on_run)
        self.process_panel.continue_clicked.connect(self._on_continue)
        self.pareto_view.log_message.connect(self.log_message)
        self.extract_panel.log_message.connect(self.log_message)

    def _on_run(self):
        self.run_requested.emit()

    def _on_continue(self):
        if not _db_path().exists():
            QMessageBox.warning(
                self, "Cannot Continue",
                "No previous optimization database found.\n"
                "Please run a new optimization first (RUN)."
            )
            return
        self.continue_requested.emit()

    def bind_log(self, log_console: LogConsole):
        log_console.progress.connect(self.update_progress)
        log_console.pareto_front.connect(self.pareto_view.update_pareto_front)
        log_console.time_info.connect(self.update_time_info)

    def start_backend(self, resume: bool = False):
        self.process_panel.start_backend(resume)

    def pause_backend(self):
        self.process_panel.pause_backend()

    def unpause_backend(self):
        self.process_panel.unpause_backend()

    def stop_backend(self):
        self.process_panel.stop_backend()

    def update_progress(self, done: int, total: int):
        self.trial_done, self.trial_total = done, total
        self.progress_label.setText(f"Progress {done}/{total}")
        if total > 0:
            self.progress.setValue(int(done * 100 / total))

    def update_time_info(self, elapsed_min: str, eta_min: str):
        self.time_info.setText(f"Elapsed {elapsed_min} min | ETA {eta_min} min")

    def set_state(self, state: str):
        self.process_panel._set_state(state)

    def reset(self):
        self.process_panel._set_state("IDLE")
        self.progress_label.setText("Progress 0/0")
        self.progress.setValue(0)
        self.time_info.setText("Elapsed -- | ETA --")
        self.pareto_view.reset()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airfoil Multi-Objective Optimizer")
        self.resize(1680, 980)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.config_panel = ConfigPanel()
        self.visual_panel = VisualizationPanel()
        self.log_console = LogConsole()
        self.control_panel = ControlPanel()
        self.control_panel.bind_log(self.log_console)

        self.visual_panel.info.connect(self.log_console.append_line)
        self.control_panel.log_message.connect(self.log_console.append_line)
        self.control_panel.run_requested.connect(self._handle_run_requested)
        self.control_panel.continue_requested.connect(self._handle_continue_requested)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        work_dir_bar = QWidget()
        wdb_layout = QHBoxLayout(work_dir_bar)
        wdb_layout.setContentsMargins(0, 0, 0, 0)
        wdb_layout.addWidget(QLabel("Working Directory:"))
        self.wd_display = QLineEdit(str(config.work_dir))
        self.wd_display.setReadOnly(True)
        wd_change_btn = QPushButton("Change")
        wd_change_btn.clicked.connect(self._change_work_dir)
        wd_clear_btn = QPushButton("Clear History")
        wd_clear_btn.setToolTip("Clear all optimization history in the current working directory and reset the frontend to initial state")
        wd_clear_btn.clicked.connect(self._clear_and_reset)
        wdb_layout.addWidget(self.wd_display)
        wdb_layout.addWidget(wd_change_btn)
        wdb_layout.addWidget(wd_clear_btn)
        center_layout.addWidget(work_dir_bar)

        center_layout.addWidget(self.visual_panel, 6)
        center_layout.addWidget(self.log_console, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap_scroll(self.config_panel, resizable=False))
        splitter.addWidget(center_widget)
        splitter.addWidget(self._wrap_scroll(self.control_panel))
        splitter.setSizes([340, 1040, 300])
        root_layout.addWidget(splitter)

        self.setCentralWidget(root)

    def _wrap_scroll(self, widget: QWidget, *, resizable: bool = True) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(resizable)
        area.setWidget(widget)
        area.setFrameStyle(QFrame.NoFrame)
        if not resizable:
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return area

    def _change_work_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if path:
            config.set_work_dir(path)
            self.wd_display.setText(str(config.work_dir))
            self.log_console.append_line(f"工作目录已设为: {path}")

    def _handle_run_requested(self):
        success, msg = self.config_panel.save_config()
        self.log_console.append_line(msg)
        if success:
            self.control_panel.start_backend(resume=False)

    def _handle_continue_requested(self):
        success, msg = self.config_panel.save_config()
        self.log_console.append_line(msg)
        if success:
            self.control_panel.start_backend(resume=True)

    def _clear_and_reset(self):
        ret = QMessageBox.warning(
            self, "Clear Working Directory",
            "Are you sure you want to clear all optimization history in the current working directory and reset the frontend to initial state?\n\n"
            "This operation cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        clear_working_dir()

        self.visual_panel.image_files.clear()
        self.visual_panel.current_idx = -1
        self.visual_panel.clear_manual_images()
        self.visual_panel.ax.clear()
        self.visual_panel.ax.axis("off")
        self.visual_panel.canvas.draw_idle()
        self.visual_panel.status.setText("No figure yet")

        self.log_console.text.clear()
        self.log_console.offset = 0
        self.log_console._last_pareto_content = ""

        self.control_panel.reset()

        self.log_console.append_line("工作目录已清空，可以开始新的优化。")


def setup_theme(app: QApplication):
    try:
        import qdarktheme
        qdarktheme.setup_theme("light")
    except (ImportError, OSError, RuntimeError):
        raise RuntimeError("setup_theme failed.")
    app.setStyleSheet(
        """
        QWidget { background-color: #f6f8fa; color: #24292f; font-size: 13px; }
        QGroupBox {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: bold;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            color: #0969da;
            padding: 0 4px;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #0969da;
            border-radius: 6px;
            padding: 6px 10px;
            color: #24292f;
            font-weight: 600;
        }
        QPushButton:hover { background-color: #f3f4f6; }
        QPushButton:pressed { background-color: #e8ecf0; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget, QPlainTextEdit {
            background-color: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 4px;
        }
        QHeaderView::section {
            background-color: #f0f3f6;
            color: #24292f;
            border: 0px;
            padding: 4px;
        }
        QProgressBar {
            border: 1px solid #d0d7de;
            border-radius: 5px;
            text-align: center;
            background-color: #eaeef2;
        }
        QProgressBar::chunk { background-color: #0969da; }
        """
    )


def clear_working_dir() -> None:
    wd = config.work_dir
    figs = wd / "logs" / "figs"
    if figs.exists():
        shutil.rmtree(figs, ignore_errors=True)
    figs.mkdir(parents=True, exist_ok=True)

    for p in [get_log_path(), get_pareto_log_path(), wd / "logs" / "airfoil_optim.db"]:
        if p.exists():
            p.unlink()

    sims = wd / "sims"
    if sims.is_dir():
        for item in sims.iterdir():
            if item.name in ("sim_ref",):
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    sol = wd / "solution"
    if sol.exists():
        shutil.rmtree(sol, ignore_errors=True)


def _db_path() -> Path:
    return config.work_dir / "logs" / "airfoil_optim.db"


def _prompt_work_dir_at_startup():
    while True:
        dialog = QDialog()
        dialog.setWindowTitle("选择工作目录")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("请选择工作目录（所有输出将写入此目录）："))

        h_layout = QHBoxLayout()
        path_input = QLineEdit()
        path_input.setPlaceholderText("点击浏览选择目录...")
        browse_btn = QPushButton("浏览")
        h_layout.addWidget(path_input)
        h_layout.addWidget(browse_btn)
        layout.addLayout(h_layout)

        confirm_btn = QPushButton("确认")
        layout.addWidget(confirm_btn)

        def browse():
            path = QFileDialog.getExistingDirectory(dialog, "选择工作目录")
            if path:
                path_input.setText(path)

        def confirm():
            path = path_input.text().strip()
            if not path:
                QMessageBox.warning(dialog, "提示", "请选择工作目录")
                return
            config.set_work_dir(path)
            dialog.accept()

        browse_btn.clicked.connect(browse)
        confirm_btn.clicked.connect(confirm)

        if dialog.exec() == QDialog.Accepted:
            break


def main():
    app = QApplication(sys.argv)
    setup_theme(app)
    _prompt_work_dir_at_startup()
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
