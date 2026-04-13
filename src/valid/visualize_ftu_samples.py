"""FTU export sample Qt visualizer (time-domain + range profile + 3D surface)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QSplitter,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "缺少 Qt 依赖，请安装 PyQt5 或 PySide6 后重试。"
        ) from e

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES_DIR = ROOT / "exports" / "FTU" / "samples"
DEFAULT_META_DIR = ROOT / "exports" / "FTU" / "meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTU Qt Visualizer")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="样本目录（默认 exports/FTU/samples）",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=DEFAULT_META_DIR,
        help="meta目录（默认 exports/FTU/meta）",
    )
    return parser.parse_args()


def list_sample_files(samples_dir: Path) -> List[Path]:
    return sorted(samples_dir.glob("*.npz"))


def load_meta(meta_dir: Path, sample_tag: str) -> Dict[str, object]:
    meta_path = meta_dir / f"{sample_tag}.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class MplCanvas(FigureCanvas):
    def __init__(self, is_3d: bool = False) -> None:
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        if is_3d:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class FTUVisualizerMainWindow(QMainWindow):
    def __init__(self, samples_dir: Path, meta_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("FTU Sample Visualizer (Qt)")
        self.resize(1400, 900)

        self.samples_dir = samples_dir
        self.meta_dir = meta_dir
        self.current_data: Optional[Dict[str, np.ndarray]] = None
        self.current_meta: Dict[str, object] = {}
        self.current_tag: Optional[str] = None
        self._colorbar_3d = None

        self._build_ui()
        self._refresh_samples()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # 左侧控制栏
        left = QWidget()
        left_layout = QVBoxLayout(left)

        dir_form = QFormLayout()
        self.samples_dir_edit = QLineEdit(str(self.samples_dir))
        self.meta_dir_edit = QLineEdit(str(self.meta_dir))
        dir_form.addRow("Samples Dir", self.samples_dir_edit)
        dir_form.addRow("Meta Dir", self.meta_dir_edit)
        left_layout.addLayout(dir_form)

        btn_row = QHBoxLayout()
        self.btn_browse_samples = QPushButton("Browse Samples")
        self.btn_browse_meta = QPushButton("Browse Meta")
        self.btn_refresh = QPushButton("Refresh")
        btn_row.addWidget(self.btn_browse_samples)
        btn_row.addWidget(self.btn_browse_meta)
        btn_row.addWidget(self.btn_refresh)
        left_layout.addLayout(btn_row)

        left_layout.addWidget(QLabel("Samples"))
        self.sample_list = QListWidget()
        left_layout.addWidget(self.sample_list, stretch=1)

        param_form = QFormLayout()
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.percentile_spin = QSpinBox()
        self.percentile_spin.setRange(80, 100)
        self.percentile_spin.setValue(95)
        self.range_max_m_spin = QSpinBox()
        self.range_max_m_spin.setRange(1, 20)
        self.range_max_m_spin.setValue(5)
        param_form.addRow("Frame Idx", self.frame_spin)
        param_form.addRow("3D Floor Percentile", self.percentile_spin)
        param_form.addRow("Range Max (m)", self.range_max_m_spin)
        left_layout.addLayout(param_form)

        self.btn_render = QPushButton("Render Views")
        left_layout.addWidget(self.btn_render)

        left_layout.addWidget(QLabel("Info"))
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        left_layout.addWidget(self.info_box, stretch=1)

        # 右侧图形区域
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()

        self.canvas_time = MplCanvas(is_3d=False)
        self.canvas_fft2d = MplCanvas(is_3d=False)
        self.canvas_fft3d = MplCanvas(is_3d=True)

        tab_time = QWidget()
        tab_time_layout = QVBoxLayout(tab_time)
        tab_time_layout.addWidget(self.canvas_time)
        self.tabs.addTab(tab_time, "Time Domain")

        tab_2d = QWidget()
        tab_2d_layout = QVBoxLayout(tab_2d)
        tab_2d_layout.addWidget(self.canvas_fft2d)
        self.tabs.addTab(tab_2d, "Range Profile")

        tab_3d = QWidget()
        tab_3d_layout = QVBoxLayout(tab_3d)
        tab_3d_layout.addWidget(self.canvas_fft3d)
        self.tabs.addTab(tab_3d, "3D FFT")

        right_layout.addWidget(self.tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 980])

        self.btn_browse_samples.clicked.connect(self._browse_samples_dir)
        self.btn_browse_meta.clicked.connect(self._browse_meta_dir)
        self.btn_refresh.clicked.connect(self._refresh_samples)
        self.sample_list.currentItemChanged.connect(self._on_sample_selected)
        self.btn_render.clicked.connect(self._render_all_views)

    def _browse_samples_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Samples Dir", self.samples_dir_edit.text())
        if selected:
            self.samples_dir_edit.setText(selected)

    def _browse_meta_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Meta Dir", self.meta_dir_edit.text())
        if selected:
            self.meta_dir_edit.setText(selected)

    def _refresh_samples(self) -> None:
        self.samples_dir = Path(self.samples_dir_edit.text()).expanduser().resolve()
        self.meta_dir = Path(self.meta_dir_edit.text()).expanduser().resolve()

        self.sample_list.clear()
        if not self.samples_dir.exists():
            QMessageBox.warning(self, "Path Error", f"samples目录不存在:\n{self.samples_dir}")
            return

        files = list_sample_files(self.samples_dir)
        for p in files:
            item = QListWidgetItem(p.stem)
            item.setData(Qt.UserRole, str(p))
            self.sample_list.addItem(item)

        self.info_box.setPlainText(f"Loaded {len(files)} samples from:\n{self.samples_dir}")
        if files:
            self.sample_list.setCurrentRow(0)

    def _load_sample(self, npz_path: Path) -> Dict[str, np.ndarray]:
        with np.load(npz_path) as data:
            required = {"radar", "time", "heart_rate", "respiration_rate"}
            if set(data.files) != required:
                raise ValueError(f"npz keys异常: {data.files}")
            return {
                "radar": data["radar"],
                "time": data["time"],
                "heart_rate": data["heart_rate"],
                "respiration_rate": data["respiration_rate"],
            }

    def _on_sample_selected(self, current: Optional[QListWidgetItem], _: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        npz_path = Path(current.data(Qt.UserRole))
        try:
            loaded = self._load_sample(npz_path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", f"加载失败:\n{npz_path}\n\n{e}")
            return

        self.current_data = loaded
        self.current_tag = npz_path.stem
        self.current_meta = load_meta(self.meta_dir, self.current_tag)

        radar = loaded["radar"]
        frames, _, _, _ = radar.shape
        self.frame_spin.setMaximum(max(0, frames - 1))
        self.frame_spin.setValue(0)

        self._update_info()
        self._render_all_views()

    def _update_info(self) -> None:
        if self.current_data is None or self.current_tag is None:
            self.info_box.setPlainText("No sample selected.")
            return

        radar = self.current_data["radar"]
        time = self.current_data["time"]
        hr = self.current_data["heart_rate"]
        rr = self.current_data["respiration_rate"]
        lines = [
            f"Sample: {self.current_tag}",
            f"Radar shape: {tuple(radar.shape)}",
            f"Radar dtype: {radar.dtype}",
            f"Time len: {len(time)}",
            f"HR finite ratio: {np.isfinite(hr).mean():.2%}",
            f"RR finite ratio: {np.isfinite(rr).mean():.2%}",
        ]
        if self.current_meta:
            lines.append(f"Meta preprocess: {self.current_meta.get('radar_preprocess', 'N/A')}")
            lines.append(f"Meta axes: {self.current_meta.get('radar_axes', 'N/A')}")
        self.info_box.setPlainText("\n".join(lines))

    def _render_all_views(self) -> None:
        if self.current_data is None:
            return
        try:
            self._render_time_domain()
            self._render_fft2d()
            self._render_fft3d()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Render Error", str(e))

    def _render_time_domain(self) -> None:
        data = self.current_data
        assert data is not None
        radar = data["radar"]
        time = data["time"]
        hr = data["heart_rate"]
        rr = data["respiration_rate"]

        ax = self.canvas_time.ax
        ax.clear()

        ax.plot(time, hr, label="heart_rate", lw=1.6)
        if np.isfinite(rr).any():
            ax.plot(time, rr, label="respiration_rate", lw=1.2, alpha=0.8)

        # 叠加雷达最强bin的时域包络（归一化到右轴）
        mag = np.abs(radar)
        energy_bin = mag.mean(axis=0)  # (d,a,r)
        d0, a0, r0 = np.unravel_index(np.argmax(energy_bin), energy_bin.shape)
        envelope = mag[:, d0, a0, r0]
        env_norm = (envelope - envelope.min()) / (np.ptp(envelope) + 1e-8)
        env_plot = env_norm * 100.0  # 缩放到 bpm 附近便于共图观察
        ax.plot(np.linspace(time.min(), time.max(), len(env_plot)), env_plot, label="radar_envelope(norm*100)", lw=1.0)

        ax.set_title(f"Time-domain Waveforms ({self.current_tag})")
        ax.set_xlabel("Time")
        ax.set_ylabel("BPM / Scaled")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        self.canvas_time.draw()

    def _render_fft2d(self) -> None:
        data = self.current_data
        assert data is not None
        radar = data["radar"]
        frame_idx = int(self.frame_spin.value())
        # User-required 2D view:
        # x-axis: range bin, y-axis: amplitude (linear magnitude)
        frame_cube = radar[frame_idx]  # (d, a, r)
        range_profile = np.abs(frame_cube).mean(axis=(0, 1))  # (r,)

        x_bins = np.arange(range_profile.size)
        y_amp = range_profile

        ax = self.canvas_fft2d.ax
        ax.clear()
        ax.plot(x_bins, y_amp, lw=1.8, color="tab:blue")
        ax.set_title(f"Range FFT Profile @ frame={frame_idx}")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        self.canvas_fft2d.draw()

    def _render_fft3d(self) -> None:
        data = self.current_data
        assert data is not None
        radar = data["radar"]
        percentile = int(self.percentile_spin.value())
        range_max_m = float(self.range_max_m_spin.value())

        # 生成类似示例图的 Time-Range-Amplitude 3D 曲面：
        # 先对 doppler/angle 求均值得到 (frames, range)，再转 dB。
        tr_amp = np.abs(radar).mean(axis=(1, 2))  # (frames, range)
        tr_db = 20.0 * np.log10(tr_amp + 1e-8)

        # 使用分位值抑制低能噪声，保留主体结构
        floor = np.percentile(tr_db, percentile)
        tr_db_masked = np.maximum(tr_db, floor)

        frames, range_bins = tr_db_masked.shape
        if len(data["time"]) >= 2:
            t_axis = np.linspace(float(data["time"].min()), float(data["time"].max()), frames)
            t_label = "Time (s)"
        else:
            t_axis = np.arange(frames, dtype=float)
            t_label = "Frame"
        r_axis = np.linspace(0.0, range_max_m, range_bins)
        tt, rr = np.meshgrid(t_axis, r_axis, indexing="xy")  # (range, frames)
        zz = tr_db_masked.T  # (range, frames)

        ax = self.canvas_fft3d.ax
        ax.clear()
        # Prevent colorbar accumulation on repeated re-render
        if self._colorbar_3d is not None:
            try:
                self._colorbar_3d.remove()
            except Exception:
                pass
            self._colorbar_3d = None

        surf = ax.plot_surface(
            tt,
            rr,
            zz,
            cmap="jet",
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        self._colorbar_3d = self.canvas_fft3d.fig.colorbar(
            surf,
            ax=ax,
            fraction=0.046,
            pad=0.08,
        )
        ax.view_init(elev=28, azim=-25)
        ax.set_title(f"3D Surface (Time-Range-Amplitude), floor={percentile}th")
        ax.set_xlabel(t_label)
        ax.set_ylabel("Range (m)")
        ax.set_zlabel("Amplitude (dB)")
        self.canvas_fft3d.draw()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    win = FTUVisualizerMainWindow(
        samples_dir=args.samples_dir.resolve(),
        meta_dir=args.meta_dir.resolve(),
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
