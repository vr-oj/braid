import os
import logging
from PySide6.QtCore import Slot, Signal, QEvent, QSettings
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QTabWidget, QMessageBox, QApplication, QLineEdit, \
    QHBoxLayout, QLabel, QSizePolicy, QFileDialog
from data_pipeline import DataPipeline
from processing.task_manager import TaskManager
from widgets.file_picker import FilePickerWidget
from processing.data_loader import (
    extract_buti_settings,
    get_system_username,
    parse_and_validate_csv,
)
from tabs.plot_tab import PlotTab
from tabs.scale_tab import ScaleTab
from tabs.roi_tab import ROITab
from tabs.seed_tab import SeedTab
from tabs.thresh_tab import ThreshTab
from tabs.geometry_tab import GeometryTab
from tabs.mechanics_tab import MechanicsTab
from tabs.last_pull_tab import LastPullTab
from tabs.relaxation_tab import RelaxationTab
from tabs.export_tab import ExportTab
from tabs.voxel_tab import VoxelTab
from pathlib import Path


log = logging.getLogger(__name__)

class AnalysisWidget(QWidget):
    """
    A widget that encapsulates a single, self-contained analysis session.
    It manages its own DataPipeline and the UI components related to it.
    """
    tab_name_requested = Signal(str)

    def __init__(self, task_manager: TaskManager, settings: QSettings, parent=None):
        """
        Initializes the session widget.

        Args:
            task_manager (TaskManager): The shared TaskManager instance from MainWindow.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.task_manager = task_manager
        self.settings = settings
        self.pipeline = DataPipeline(self)
        self.pipeline.task_manager = self.task_manager

        self.init_ui()
        self.connect_signals()
        QApplication.instance().installEventFilter(self)
        self._load_settings_into_pipeline()
        log.info("AnalysisWidget created.")

    def init_ui(self):
        """Initializes the user interface for this session."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Use the full space of the tab

        # File selection widgets for this session
        self.file_pickers = FilePickerWidget()
        self.file_pickers.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_pickers, 1)

        # Text field for author name
        author_widget = QWidget()
        author_widget.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Preferred
        )
        author_layout = QHBoxLayout(author_widget)
        author_label = QLabel("Author:")
        author_layout.addWidget(author_label)
        self.tab_author = QLineEdit()
        self.tab_author.setPlaceholderText("Enter name...")
        self.username = get_system_username()
        authorname = self.settings.value("author", defaultValue=self.username, type=str)
        self.pipeline.on_author_changed(authorname)
        self.tab_author.setText(authorname)
        self.tab_author.setClearButtonEnabled(True)
        self.tab_author.textEdited.connect(self._save_author)
        author_layout.addWidget(self.tab_author)

        top_layout.addWidget(author_widget, 0)

        main_layout.addLayout(top_layout)

        self.buti_settings_label = QLabel()
        self.buti_settings_label.setObjectName("butiSettingsLabel")
        self.buti_settings_label.setWordWrap(True)
        self.buti_settings_label.hide()
        main_layout.addWidget(self.buti_settings_label)

        # A visual separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep)

        # The tab widget for all the analysis steps (Plot, Scale, etc.)
        self.analysis_tabs = QTabWidget()
        main_layout.addWidget(self.analysis_tabs)

        self.init_analysis_tabs()


    def eventFilter(self, watched, event):
        """
        Filters events for the entire application. Used here to clear focus
        from the new_tab_name_input when the user clicks elsewhere.
        """
        if event.type() == QEvent.MouseButtonPress:
            # Check if the input field currently has focus
            if self.tab_author.hasFocus():
                # Find out which widget was actually clicked on
                clicked_widget = QApplication.widgetAt(event.globalPos())

                # If the click was outside the line edit and its children (like the clear button), then unfocus it.
                if clicked_widget and clicked_widget != self.tab_author and not self.tab_author.isAncestorOf(
                        clicked_widget):
                    self.tab_author.clearFocus()

        # Pass the event on to the parent class for default processing
        return super().eventFilter(watched, event)

    def init_analysis_tabs(self):
        """Creates and adds all the analysis-specific tabs."""
        self.plot_tab = PlotTab(self.pipeline)
        self.analysis_tabs.addTab(self.plot_tab, "📈 Plot")
        self.scale_tab = ScaleTab(self.pipeline)
        self.analysis_tabs.addTab(self.scale_tab, "📏 Scale")
        self.roi_tab = ROITab(self.pipeline)
        self.analysis_tabs.addTab(self.roi_tab, "🎯 ROI")
        self.seed_tab = SeedTab(self.pipeline)
        self.analysis_tabs.addTab(self.seed_tab, "🌱 Seed")
        self.thresh_tab = ThreshTab(self.pipeline)
        self.analysis_tabs.addTab(self.thresh_tab, "🏁 Threshold")
        self.geometry_tab = GeometryTab(self.pipeline)
        self.analysis_tabs.addTab(self.geometry_tab, "📊 Geometry")
        self.voxel_tab = VoxelTab(self.pipeline)
        self.analysis_tabs.addTab(self.voxel_tab, "🧊 3D View")
        self.mechanics_tab = MechanicsTab(self.pipeline)
        self.analysis_tabs.addTab(self.mechanics_tab, "🧮 Mechanics")
        self.lastpull_tab = LastPullTab(self.pipeline)
        self.analysis_tabs.addTab(self.lastpull_tab, "⛓️ Last Pull")
        self.relaxation_tab = RelaxationTab(self.pipeline)
        self.analysis_tabs.addTab(self.relaxation_tab, "⏳ Relaxation")
        self.export_tab = ExportTab(self.pipeline)
        self.analysis_tabs.addTab(self.export_tab, "📦 Export")


    def connect_signals(self):
        """Connects signals from widgets to the appropriate slots in this session."""
        self.file_pickers.video_selected.connect(self.on_file_selected)
        self.pipeline.plot_selection_changed.connect(self._save_plot_selection)
        self.pipeline.data_available.connect(self._show_buti_settings)
        self.pipeline.cycle_selection_changed.connect(self._save_cycle_selection)
        self.pipeline.known_length_changed.connect(self._save_known_length)
        self.pipeline.scale_is_manual_changed.connect(self._save_scale_is_manual)
        self.pipeline.manual_conversion_factor_changed.connect(self._save_manual_conversion_factor)
        self.pipeline.state_loaded.connect(self._restore_loaded_session_header)
        #self.pipeline.scale_changed.connect(self._save_scale)

    @Slot(dict)
    def _show_buti_settings(self, data: dict) -> None:
        """Show acquisition context without changing analysis calculations."""

        settings = extract_buti_settings(data)
        if not settings:
            self.buti_settings_label.clear()
            self.buti_settings_label.hide()
            return

        parts = []
        experiment_type = settings.get("experiment_type")
        if experiment_type:
            parts.append(str(experiment_type))

        def add_value(label, key, unit="", precision=3):
            value = settings.get(key)
            if value is None or value == "":
                return
            try:
                rendered = f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                rendered = str(value)
            parts.append(f"{label}: {rendered}{unit}")

        add_value("Preload", "preload_mm", " mm")
        add_value("Deformation", "deformation_mm", " mm")
        add_value("Deformation", "deformation_percent", "%", precision=1)
        add_value("Steps", "steps", precision=0)
        add_value("Forward rate", "rate_forward_mm_s", " mm/s")
        add_value("Reverse rate", "rate_reverse_mm_s", " mm/s")
        add_value("Cycles", "cycles", precision=0)
        add_value("Wire diameter", "wire_diameter_mm", " mm")
        add_value("Constant tension", "constant_tension_mn", " mN", precision=1)

        self.buti_settings_label.setText("BUTI setup — " + "  |  ".join(parts))
        self.buti_settings_label.show()

    @Slot()
    def _restore_loaded_session_header(self):
        """Refresh header fields after the pipeline applies a saved session."""
        author = getattr(self.pipeline, "author", None)
        if author:
            self.tab_author.setText(str(author))
        if self.pipeline.video:
            self.file_pickers.set_video_label(str(self.pipeline.video))

    def _load_settings_into_pipeline(self):
        """Reads values from QSettings and populates the pipeline."""
        log.info("Loading persistent settings into data pipeline...")
        plot_selection = self.settings.value("plot/plot_selection", defaultValue="Time vs. Force", type=str)
        self.pipeline.set_plot_selection(plot_selection)
        cycle_selection = self.settings.value("plot/cycle_selection", defaultValue="All Cycles", type=str)
        self.pipeline.set_cycle_selection(cycle_selection)
        # Use the pipeline's setter method. This will also emit the signal
        # that the ScaleTab is listening to, automatically updating the UI.
        known_length = self.settings.value("scale/known_length", defaultValue=0.0, type=float)
        self.pipeline.set_known_length(known_length)
        manual_factor = self.settings.value("scale/manual_factor", defaultValue=0.0, type=float)
        self.pipeline.set_manual_conversion_factor(manual_factor)
        is_manual = self.settings.value("scale/is_manual", defaultValue=False, type=bool)
        self.pipeline.set_scale_is_manual(is_manual)

    @Slot(str)
    def on_file_selected(self, path: str):
        """Handles the selection of a video file for this session."""
        log.info(f"Session received video file path: {path}")
        file_name = os.path.basename(path)
        base_name, _ = os.path.splitext(file_name)
        new_name = base_name.removeprefix("recording_")
        new_name = new_name.replace("_video", "")
        new_name = new_name.replace("_c", "")
        self.tab_name_requested.emit(new_name)
        self.file_pickers.set_video_label(path)

        # Only try to resolve CSVs for MKVs. TIFFs handle themselves.
        if path.lower().endswith(".mkv"):
            sidecar_data = self.resolve_sidecar_data(path)
            if not sidecar_data:
                log.warning("User cancelled data loading or provided invalid data. Aborting video load.")
                return  # Prevent the crash by stopping here

            # Inject the parsed data directly into the pipeline
            self.pipeline.data = sidecar_data

        self._handle_video_load(path)

    def _handle_video_load(self, path: str):
        """Single source of truth for loading a video file within this session."""
        log.info(f"Loading Video {path} for session.")
        self.pipeline.load_video_file(path)


    def resolve_sidecar_data(self, video_path: str) -> dict | None:
        """Attempts to automatically load sidecar data, falling back to a user prompt."""
        p = Path(video_path)

        # 1. Automatic Discovery
        stems = [
            p.stem.replace("_video_c", "_force"),
            p.stem.replace("_video", "_force"),
        ]

        # New Case: Strip everything from "_video" onward and append "_force"
        # Example: "test_video_custom_suffix" -> "test_force"
        if "_video" in p.stem:
            stems.append(f"{p.stem.split('_video')[0]}_force")

        stems.append(p.stem)  # Fallback to exact same name

        stems = list(dict.fromkeys(stems))

        csv_path = None
        for stem in stems:
            candidate = p.with_stem(stem).with_suffix(".csv")
            if candidate.exists():
                csv_path = candidate
                break

        # 2. Validation
        data = None
        if csv_path:
            log.info(f"Found matching CSV: {csv_path}")
            data = parse_and_validate_csv(csv_path)

        # 3. Prompt User if Missing or Invalid
        if not data:
            log.warning("Valid CSV not found automatically. Prompting user.")
            reply = QMessageBox.warning(
                self,
                "Data File Missing",
                "Could not automatically locate a valid sidecar csv file for this video.\nWould you like to locate it manually?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                manual_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select CSV Data File",
                    str(p.parent),
                    "CSV Files (*.csv)"
                )

                if manual_path:
                    data = parse_and_validate_csv(Path(manual_path))
                    if not data:
                        QMessageBox.critical(self, "Invalid Data",
                                             "The selected CSV does not contain the required columns.")

        return data



    # --- SLOTS FOR SAVING ---

    @Slot(str)
    def _save_plot_selection(self, selection_text: str):
        """Saves the selected plot to settings."""
        self.settings.setValue("plot/plot_selection", selection_text)

    @Slot(str)
    def _save_cycle_selection(self, selection_text: str):
        """Saves the selected cycle filter to settings."""
        self.settings.setValue("plot/cycle_selection", selection_text)

    @Slot(float)
    def _save_known_length(self, length: float):
        """Saves the known_length to settings."""
        self.settings.setValue("scale/known_length", length)

    @Slot(bool)
    def _save_scale_is_manual(self, is_manual: bool):
        """Saves the manual mode state to settings."""
        self.settings.setValue("scale/is_manual", is_manual)

    @Slot(float)
    def _save_manual_conversion_factor(self, factor: float):
        """Saves the manual conversion factor to settings."""
        self.settings.setValue("scale/manual_factor", factor)

    @Slot(str)
    def _save_author(self, author: str):
        if author.strip() == "":
            author = self.username
        author = author.strip()
        self.pipeline.on_author_changed(author)
        self.settings.setValue("author", author)
