"""Main GUI Window with Categorized API Widget.

Updated version that uses the categorized API parameter layout.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit, QSplitter,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QCheckBox, QListWidget,
    QScrollArea, QHeaderView, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QProcess, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QAction

from workspace_manager import WorkspaceManager
from categorized_api_widget import CategorizedAPIWidget
from product_grid_widget import ProductGridWidget
from population_tab_widget import PopulationTabWidget

# Import app_api for catalog integration
sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
try:
    from lmp_pkg import app_api
    # Test if catalog is actually working
    test_entries = app_api.list_catalog_entries("subject")
    CATALOG_AVAILABLE = True
except Exception:
    CATALOG_AVAILABLE = False
    app_api = None

# Also try direct catalog access
CATALOG_ROOT = Path(__file__).parent.parent / "lmp_pkg" / "src" / "lmp_pkg" / "catalog" / "builtin"
CATALOG_DIRECT = CATALOG_ROOT.exists()


class WorkspaceTab(QWidget):
    """Home/Workspace management tab."""

    workspace_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("LMP Workspace")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Workspace selection
        workspace_group = QGroupBox("Workspace")
        workspace_layout = QFormLayout()

        self.workspace_path_edit = QLineEdit()
        self.workspace_path_edit.setPlaceholderText("Select workspace directory...")

        workspace_buttons = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_workspace)
        create_btn = QPushButton("Create New")
        create_btn.clicked.connect(self.create_workspace)

        workspace_buttons.addWidget(browse_btn)
        workspace_buttons.addWidget(create_btn)
        workspace_buttons.addStretch()

        workspace_layout.addRow("Directory:", self.workspace_path_edit)
        workspace_layout.addRow("", workspace_buttons)

        workspace_group.setLayout(workspace_layout)
        layout.addWidget(workspace_group)

        # Recent projects
        recent_group = QGroupBox("Recent Workspaces")
        recent_layout = QVBoxLayout()
        self.recent_list = QListWidget()
        self.recent_list.addItem("No recent workspaces")
        recent_layout.addWidget(self.recent_list)
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)

        # Package info
        info_group = QGroupBox("Package Information")
        info_layout = QFormLayout()
        info_layout.addRow("LMP Version:", QLabel("0.1.0"))
        info_layout.addRow("PySide6:", QLabel("6.9.2"))
        catalog_status = "Available" if CATALOG_AVAILABLE else "Direct" if CATALOG_DIRECT else "Unavailable"
        info_layout.addRow("Catalog:", QLabel(catalog_status))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.setLayout(layout)

    def browse_workspace(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Workspace Directory")
        if directory:
            self.set_workspace(directory)

    def create_workspace(self):
        directory = QFileDialog.getExistingDirectory(self, "Create Workspace In Directory")
        if directory:
            workspace_path = Path(directory) / "lmp_workspace"
            workspace_path.mkdir(exist_ok=True)
            self.set_workspace(str(workspace_path))

    def set_workspace(self, path: str):
        try:
            self.workspace_manager = WorkspaceManager(path)
            self.workspace_path_edit.setText(path)
            self.workspace_changed.emit(path)
            self.recent_list.clear()
            self.recent_list.addItem(f"Current: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not set workspace: {str(e)}")


class ProductTab(QWidget):
    """Product parameters tab with dropdown selection."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("Product Parameters")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Product selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Product:"))

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.load_product_parameters)
        selection_layout.addWidget(self.product_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_products)
        selection_layout.addWidget(refresh_btn)

        save_btn = QPushButton("Save to Config")
        save_btn.clicked.connect(self.save_to_config)
        selection_layout.addWidget(save_btn)

        selection_layout.addStretch()
        layout.addLayout(selection_layout)

        # Product parameter table
        self.product_table = QTableWidget()
        self.product_table.setColumnCount(2)
        self.product_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.product_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.product_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.product_table)

        self.setLayout(layout)

    def load_data(self):
        self.refresh_products()

    def refresh_products(self):
        try:
            if CATALOG_AVAILABLE:
                entries = app_api.list_catalog_entries("product")
            elif CATALOG_DIRECT:
                entries = self.get_direct_catalog_entries()
            else:
                entries = ["HFA_MDI_PT210", "DPI_RS01", "SMI_K1"]

            self.product_combo.clear()
            self.product_combo.addItems(entries)

            if entries:
                self.load_product_parameters()

        except Exception as e:
            print(f"Error loading products: {e}")
            self.product_combo.clear()
            self.product_combo.addItem("Error loading products")

    def get_direct_catalog_entries(self):
        catalog_dir = CATALOG_ROOT / "product"
        if not catalog_dir.exists():
            return []

        entries = []
        for file_path in catalog_dir.glob("*.toml"):
            if not file_path.name.startswith("Variability_"):
                entries.append(file_path.stem)

        return sorted(entries)

    def load_product_parameters(self):
        product_name = self.product_combo.currentText()
        if not product_name or product_name.startswith("Error"):
            self.product_table.setRowCount(0)
            return

        try:
            if CATALOG_AVAILABLE:
                data = app_api.get_catalog_entry("product", product_name)
            elif CATALOG_DIRECT:
                data = self.load_direct_catalog_entry(product_name)
            else:
                data = {"name": product_name, "device_type": "pMDI", "label_claim_mg": 0.1}

            self.populate_parameter_table(data)

        except Exception as e:
            print(f"Error loading product parameters: {e}")
            self.product_table.setRowCount(1)
            self.product_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.product_table.setItem(0, 1, QTableWidgetItem(str(e)))

    def load_direct_catalog_entry(self, product_name: str) -> dict:
        file_path = CATALOG_ROOT / "product" / f"{product_name}.toml"
        if not file_path.exists():
            raise FileNotFoundError(f"Product file not found: {file_path}")

        try:
            try:
                import tomllib
                with open(file_path, 'rb') as f:
                    return tomllib.load(f)
            except ImportError:
                import toml
                with open(file_path, 'r', encoding='utf-8') as f:
                    return toml.load(f)
        except Exception as e:
            raise Exception(f"Error parsing TOML file {file_path}: {e}")

    def populate_parameter_table(self, data: dict):
        flattened = self.flatten_dict(data)
        self.product_table.setRowCount(len(flattened))

        for row, (key, value) in enumerate(flattened.items()):
            param_item = QTableWidgetItem(key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.product_table.setItem(row, 0, param_item)

            if isinstance(value, list):
                value_str = ", ".join(map(str, value))
            else:
                value_str = str(value)

            value_item = QTableWidgetItem(value_str)
            self.product_table.setItem(row, 1, value_item)

    def flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def save_to_config(self):
        product_name = self.product_combo.currentText()
        if not product_name or product_name.startswith("Error"):
            QMessageBox.warning(self, "No Selection", "Please select a product item.")
            return

        data = {"name": product_name}
        for row in range(self.product_table.rowCount()):
            param_item = self.product_table.item(row, 0)
            value_item = self.product_table.item(row, 1)

            if param_item and value_item:
                param_name = param_item.text()
                value_text = value_item.text()

                try:
                    if "," in value_text:
                        value = [self.convert_value(v.strip()) for v in value_text.split(",")]
                    else:
                        value = self.convert_value(value_text)
                except:
                    value = value_text

                data[param_name] = value

        self.config_updated.emit("product", data)
        QMessageBox.information(self, "Saved", f"Product '{product_name}' saved to config.")

    def convert_value(self, value_str: str):
        value_str = value_str.strip()

        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            return int(value_str)

        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
        except ValueError:
            pass

        return value_str


class PopulationTab(QWidget):
    """Population tab - simplified for this demo."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Population (Subject & Maneuver)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        placeholder = QLabel("Population parameters (Subject & Maneuver selection)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        self.setLayout(layout)


class ManualCFDDialog(QDialog):
    """Dialog for manual CFD configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual CFD Configuration")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Set ET row (fixed name, editable value)
        param_item = QTableWidgetItem("ET")
        param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(0, 0, param_item)
        self.table.setItem(0, 1, QTableWidgetItem("1.0"))

        # Resize columns
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.resize(400, 200)

    def get_values(self):
        """Get the configured values."""
        values = {}
        for row in range(self.table.rowCount()):
            param = self.table.item(row, 0).text()
            value = self.table.item(row, 1).text()
            values[param] = value
        return values


class ManualLungDepositionDialog(QDialog):
    """Dialog for manual lung deposition configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Lung Deposition Configuration")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create table with 24 rows (0-23)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(24)
        self.table.setHorizontalHeaderLabels(["Generation", "Value"])

        # Add rows 0-23
        for row in range(24):
            param_item = QTableWidgetItem(str(row))
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, param_item)
            self.table.setItem(row, 1, QTableWidgetItem("0.0"))

        # Resize columns
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        # Add scroll area for the table
        scroll = QScrollArea()
        scroll.setWidget(self.table)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.resize(400, 500)

    def get_values(self):
        """Get the configured values."""
        values = {}
        for row in range(self.table.rowCount()):
            generation = self.table.item(row, 0).text()
            value = self.table.item(row, 1).text()
            values[generation] = value
        return values


class SimulationWorker(QThread):
    """Worker thread for running simulations."""

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict

    def run(self):
        """Run the simulation in a separate thread."""
        try:
            # Import app_api
            from lmp_pkg import app_api
            from lmp_pkg.config.model import AppConfig
            from lmp_pkg.engine.workflow import Workflow

            self.progress.emit("Creating configuration...")

            # Create AppConfig from dict
            config = AppConfig.model_validate(self.config_dict)

            self.progress.emit("Preparing workflow...")

            # Create custom workflow based on selected stages
            # This avoids issues with CFD when not needed
            stages = self.config_dict.get('run', {}).get('stages', [])

            # Build stage configs based on what's in the config
            stage_configs = {}
            if 'cfd' in stages:
                stage_configs['cfd'] = {'model': 'ml'}
            if 'deposition' in stages:
                deposition_config = self.config_dict.get('deposition', {})
                stage_configs['deposition'] = {
                    'model': deposition_config.get('model', 'null'),
                    'particle_grid': deposition_config.get('particle_grid', 'medium')
                }
            if 'pbbm' in stages:
                pbbm_config = self.config_dict.get('pbbm', {})
                stage_configs['pbbm'] = {
                    'model': pbbm_config.get('model', 'null'),
                    'epi_layers': pbbm_config.get('epi_layers', [2, 2, 1, 1])
                }
            if 'pk' in stages:
                pk_config = self.config_dict.get('pk', {})
                stage_configs['pk'] = {
                    'model': pk_config.get('model', 'null')
                }

            # Create custom workflow
            custom_workflow = Workflow(
                name="custom_gui_workflow",
                stages=stages,
                stage_configs=stage_configs
            )

            self.progress.emit("Running simulation...")

            # Run the simulation with custom workflow
            result = app_api.run_single_simulation(config, workflow=custom_workflow)

            self.progress.emit("Simulation complete!")

            # Extract key results
            result_summary = {
                "run_id": result.run_id,
                "status": "success",
                "stages_completed": list(result.stages.keys()) if hasattr(result, 'stages') else [],
                "has_deposition": result.deposition is not None if hasattr(result, 'deposition') else False,
                "has_pbbk": result.pbbk is not None if hasattr(result, 'pbbk') else False,
                "has_pk": result.pk is not None if hasattr(result, 'pk') else False,
            }

            # Add PK results if available
            if hasattr(result, 'pk') and result.pk:
                pk_data = result.pk
                if hasattr(pk_data, 'cmax_plasma'):
                    result_summary['cmax'] = pk_data.cmax_plasma
                if hasattr(pk_data, 'tmax_plasma'):
                    result_summary['tmax'] = pk_data.tmax_plasma
                if hasattr(pk_data, 'auc_plasma'):
                    result_summary['auc'] = pk_data.auc_plasma

            self.finished.emit(result_summary)

        except Exception as e:
            self.error.emit(f"Simulation failed: {str(e)}")


class StudyDesignerTab(QWidget):
    """Study Designer tab with two-column layout."""

    config_ready = Signal(dict)
    simulation_started = Signal()
    simulation_finished = Signal(dict)

    def __init__(self):
        super().__init__()
        self.current_config = {}
        self.available_study_configs = []
        self.manual_cfd_values = {}
        self.manual_lung_dep_values = {}
        self.simulation_worker = None

        # Track configured entities from other tabs
        self.configured_apis = []  # List of configured API names
        self.configured_products = []  # List of configured product names
        self.configured_subjects = []  # List of configured subject names
        self.configured_maneuvers = []  # List of configured maneuver names

        self.init_ui()
        self.load_available_configs()
        self.load_initial_entities()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Header
        header = QLabel("Study Design")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(header)

        # Create two-column layout
        columns_widget = QWidget()
        columns_layout = QHBoxLayout()
        columns_widget.setLayout(columns_layout)

        # LEFT COLUMN
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_column.setLayout(left_layout)

        # (i) Model Configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()

        # CFD Model (only ml is registered in registry)
        self.cfd_combo = QComboBox()
        self.cfd_combo.addItems(["ml"])  # Only registered model from registry
        model_layout.addRow("CFD:", self.cfd_combo)

        # Lung Deposition Model (from registry)
        self.lung_dep_combo = QComboBox()
        self.lung_dep_combo.addItems(["clean_lung", "null"])
        self.lung_dep_combo.setCurrentText("clean_lung")
        model_layout.addRow("Lung Deposition:", self.lung_dep_combo)

        # PBBM Model (from registry)
        self.pbbm_combo = QComboBox()
        self.pbbm_combo.addItems(["numba"])
        model_layout.addRow("PBBM:", self.pbbm_combo)

        # PK Model
        self.pk_combo = QComboBox()
        self.pk_combo.addItems(["pk_1c", "pk_2c", "pk_3c", "null"])
        model_layout.addRow("PK:", self.pk_combo)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # (ii) Workflow
        workflow_group = QGroupBox("Workflow")
        workflow_layout = QFormLayout()

        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems([
            "basic",
            "deposition_only",
            "modular_lung_pk",
            "modular_gi_pk",
            "modular_pk_only",
            "modular_full_pbbm",
            "modular_comparison",
            "pk_sweep",
            "custom"
        ])
        workflow_layout.addRow("Select Workflow:", self.workflow_combo)

        workflow_group.setLayout(workflow_layout)
        left_layout.addWidget(workflow_group)

        # (iii) Run Configuration
        run_group = QGroupBox("Run Configuration")
        run_layout = QVBoxLayout()

        # Stages with checkboxes
        stages_label = QLabel("Pipeline Stages:")
        run_layout.addWidget(stages_label)

        self.stage_checkboxes = {}
        stages = ["cfd", "deposition", "pbbm", "pk"]
        for stage in stages:
            checkbox = QCheckBox(stage)
            checkbox.setChecked(True)
            self.stage_checkboxes[stage] = checkbox
            run_layout.addWidget(checkbox)

        # Seed
        seed_layout = QFormLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(123)
        seed_layout.addRow("Seed:", self.seed_spin)
        run_layout.addLayout(seed_layout)

        run_group.setLayout(run_layout)
        left_layout.addWidget(run_group)

        left_layout.addStretch()

        # RIGHT COLUMN
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_column.setLayout(right_layout)

        # (i) Study Config
        study_config_group = QGroupBox("Study Config")
        study_config_layout = QFormLayout()

        # Study configuration fields
        self.study_type_combo = QComboBox()
        self.study_type_combo.addItems(["bioequivalence", "dose_response", "single_dose", "steady_state"])
        study_config_layout.addRow("Study Type:", self.study_type_combo)

        self.study_design_combo = QComboBox()
        self.study_design_combo.addItems(["parallel", "crossover", "sequential"])
        study_config_layout.addRow("Design:", self.study_design_combo)

        self.n_subjects_spin = QSpinBox()
        self.n_subjects_spin.setRange(1, 1000)
        self.n_subjects_spin.setValue(6)
        study_config_layout.addRow("N Subjects:", self.n_subjects_spin)

        # Population dropdown - will be populated from lung geometry configuration
        self.population_combo = QComboBox()
        self.population_combo.setEditable(False)
        self.population_combo.addItem("(No lung geometry configured)")
        study_config_layout.addRow("Population:", self.population_combo)

        self.charcoal_block_checkbox = QCheckBox("Enable GI charcoal block")
        self.charcoal_block_checkbox.setToolTip("Zero GI absorption")
        study_config_layout.addRow("", self.charcoal_block_checkbox)

        # No examples loader - removed per user request

        study_config_group.setLayout(study_config_layout)
        right_layout.addWidget(study_config_group)

        # (ii) Data
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()

        data_label = QLabel("Load data file for analysis:")
        data_layout.addWidget(data_label)

        self.data_file_edit = QLineEdit()
        self.data_file_edit.setPlaceholderText("No file loaded...")
        self.data_file_edit.setReadOnly(True)
        data_layout.addWidget(self.data_file_edit)

        load_data_btn = QPushButton("Load Data File")
        load_data_btn.clicked.connect(self.load_data_file)
        data_layout.addWidget(load_data_btn)

        data_group.setLayout(data_layout)
        right_layout.addWidget(data_group)

        # Entity references (moved from original)
        entities_group = QGroupBox("Entity References")
        entities_layout = QFormLayout()

        # Subject dropdown - will be populated from configured subjects
        self.subject_ref_combo = QComboBox()
        self.subject_ref_combo.setEditable(True)
        self.subject_ref_combo.addItem("(No subjects configured)")
        self.subject_ref_edit = self.subject_ref_combo.lineEdit()  # Keep for compatibility

        # API dropdown - will be populated from configured APIs
        self.api_ref_combo = QComboBox()
        self.api_ref_combo.setEditable(True)
        self.api_ref_combo.addItem("(No APIs configured)")
        self.api_ref_edit = self.api_ref_combo.lineEdit()  # Keep for compatibility

        # Product dropdown - will be populated from configured products
        self.product_ref_combo = QComboBox()
        self.product_ref_combo.setEditable(True)
        self.product_ref_combo.addItem("(No products configured)")
        self.product_ref_edit = self.product_ref_combo.lineEdit()  # Keep for compatibility

        # Maneuver dropdown - will be populated from configured maneuvers
        self.maneuver_ref_combo = QComboBox()
        self.maneuver_ref_combo.setEditable(True)
        self.maneuver_ref_combo.addItem("(No maneuvers configured)")
        self.maneuver_ref_edit = self.maneuver_ref_combo.lineEdit()  # Keep for compatibility

        entities_layout.addRow("Subject:", self.subject_ref_combo)
        entities_layout.addRow("API:", self.api_ref_combo)
        entities_layout.addRow("Product:", self.product_ref_combo)
        entities_layout.addRow("Maneuver:", self.maneuver_ref_combo)

        # Add info label instead of refresh button
        info_label = QLabel("Configure entities in other tabs first")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        entities_layout.addRow("", info_label)

        entities_group.setLayout(entities_layout)
        right_layout.addWidget(entities_group)

        right_layout.addStretch()

        # Add columns to main layout
        columns_layout.addWidget(left_column)
        columns_layout.addWidget(right_column)

        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidget(columns_widget)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        # Results display area
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setPlaceholderText("Simulation results will appear here...")
        results_layout.addWidget(self.results_text)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        results_layout.addWidget(self.progress_bar)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # Bottom buttons
        button_layout = QHBoxLayout()
        validate_btn = QPushButton("Validate Config")
        validate_btn.clicked.connect(self.validate_config)
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.save_config)

        # Add Run Simulation button
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        button_layout.addWidget(validate_btn)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(self.run_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_available_configs(self):
        """Load available study configs from lmp_pkg examples."""
        examples_dir = Path(__file__).parent.parent / "lmp_pkg" / "examples"
        if examples_dir.exists():
            config_files = list(examples_dir.glob("*.toml"))
            self.available_study_configs = [f.stem for f in config_files]
            self.example_config_combo.clear()
            self.example_config_combo.addItem("Select a config...")
            self.example_config_combo.addItems(self.available_study_configs)

    def load_initial_entities(self):
        """Load some initial default entities from catalog to populate dropdowns."""
        try:
            # Only load defaults if we have access to catalog
            global app_api, CATALOG_AVAILABLE
            if CATALOG_AVAILABLE and app_api:
                # Load a few default entities
                try:
                    # Add default subject
                    subjects = app_api.list_catalog_entries('subject')
                    if subjects and not self.configured_subjects:
                        # Add first subject as default
                        self.configured_subjects = [subjects[0]]
                        self._refresh_subject_dropdown()
                except:
                    # Add fallback default
                    if not self.configured_subjects:
                        self.configured_subjects = ['healthy_reference']
                        self._refresh_subject_dropdown()

                try:
                    # Add default API
                    apis = app_api.list_catalog_entries('api')
                    if apis and not self.configured_apis:
                        # Add first API as default
                        self.configured_apis = [apis[0]]
                        self._refresh_api_dropdown()
                except:
                    # Add fallback default
                    if not self.configured_apis:
                        self.configured_apis = ['BD']
                        self._refresh_api_dropdown()

                try:
                    # Add default product
                    products = app_api.list_catalog_entries('product')
                    if products and not self.configured_products:
                        # Add first product as default
                        self.configured_products = [products[0]]
                        self._refresh_product_dropdown()
                except:
                    # Add fallback default
                    if not self.configured_products:
                        self.configured_products = ['HFA_MDI_PT210']
                        self._refresh_product_dropdown()

                try:
                    # Add default maneuver
                    maneuvers = []
                    try:
                        maneuvers = app_api.list_catalog_entries('maneuver')
                    except:
                        maneuvers = app_api.list_catalog_entries('inhalation')

                    if maneuvers and not self.configured_maneuvers:
                        # Add first maneuver as default
                        self.configured_maneuvers = [maneuvers[0]]
                        self._refresh_maneuver_dropdown()
                except:
                    # Add fallback default
                    if not self.configured_maneuvers:
                        self.configured_maneuvers = ['pMDI_variable_trapezoid']
                        self._refresh_maneuver_dropdown()
            else:
                # No catalog available, use hardcoded defaults
                if not self.configured_subjects:
                    self.configured_subjects = ['healthy_reference']
                    self._refresh_subject_dropdown()
                if not self.configured_apis:
                    self.configured_apis = ['BD']
                    self._refresh_api_dropdown()
                if not self.configured_products:
                    self.configured_products = ['HFA_MDI_PT210']
                    self._refresh_product_dropdown()
                if not self.configured_maneuvers:
                    self.configured_maneuvers = ['pMDI_variable_trapezoid']
                    self._refresh_maneuver_dropdown()
        except:
            # Silent fail, dropdowns will remain with defaults
            pass

    def update_configured_api(self, api_name: str):
        """Add a configured API to the dropdown."""
        if api_name and api_name not in self.configured_apis:
            self.configured_apis.append(api_name)
        self._refresh_api_dropdown()

    def update_configured_product(self, product_data: dict):
        """Add a configured product to the dropdown."""
        # Extract product name from the data
        if isinstance(product_data, dict):
            # Check if it's from the grid widget or simple widget
            if 'products' in product_data:
                # From ProductGridWidget - multiple products
                for product in product_data.get('products', []):
                    name = product.get('name', '')
                    if name and name not in self.configured_products:
                        self.configured_products.append(name)
            elif 'name' in product_data:
                # Single product
                name = product_data.get('name', '')
                if name and name not in self.configured_products:
                    self.configured_products.append(name)
        self._refresh_product_dropdown()

    def update_configured_subject(self, subject_name: str):
        """Add a configured subject to the dropdown."""
        if subject_name and subject_name not in self.configured_subjects:
            self.configured_subjects.append(subject_name)
        self._refresh_subject_dropdown()

    def update_configured_maneuver(self, maneuver_name: str):
        """Add a configured maneuver to the dropdown."""
        if maneuver_name and maneuver_name not in self.configured_maneuvers:
            self.configured_maneuvers.append(maneuver_name)
        self._refresh_maneuver_dropdown()

    def update_population_from_lung_geometry(self, lung_geometry_name: str):
        """Update population dropdown from lung geometry configuration."""
        if lung_geometry_name:
            self.population_combo.clear()
            self.population_combo.addItem(lung_geometry_name)
            self.population_combo.setCurrentText(lung_geometry_name)
        else:
            self.population_combo.clear()
            self.population_combo.addItem("(No lung geometry configured)")

    def _refresh_subject_dropdown(self):
        """Refresh subject dropdown with configured subjects."""
        current = self.subject_ref_combo.currentText()
        self.subject_ref_combo.clear()

        if self.configured_subjects:
            self.subject_ref_combo.addItems(self.configured_subjects)
            if current in self.configured_subjects:
                self.subject_ref_combo.setCurrentText(current)
            else:
                self.subject_ref_combo.setCurrentText(self.configured_subjects[0])
        else:
            self.subject_ref_combo.addItem("healthy_reference")  # Default fallback
            self.subject_ref_combo.setCurrentText("healthy_reference")

    def _refresh_api_dropdown(self):
        """Refresh API dropdown with configured APIs."""
        current = self.api_ref_combo.currentText()
        self.api_ref_combo.clear()

        if self.configured_apis:
            self.api_ref_combo.addItems(self.configured_apis)
            if current in self.configured_apis:
                self.api_ref_combo.setCurrentText(current)
            else:
                self.api_ref_combo.setCurrentText(self.configured_apis[0])
        else:
            self.api_ref_combo.addItem("BD")  # Default fallback
            self.api_ref_combo.setCurrentText("BD")

    def _refresh_product_dropdown(self):
        """Refresh product dropdown with configured products."""
        current = self.product_ref_combo.currentText()
        self.product_ref_combo.clear()

        if self.configured_products:
            self.product_ref_combo.addItems(self.configured_products)
            if current in self.configured_products:
                self.product_ref_combo.setCurrentText(current)
            else:
                self.product_ref_combo.setCurrentText(self.configured_products[0])
        else:
            self.product_ref_combo.addItem("HFA_MDI_PT210")  # Default fallback
            self.product_ref_combo.setCurrentText("HFA_MDI_PT210")

    def _refresh_maneuver_dropdown(self):
        """Refresh maneuver dropdown with configured maneuvers."""
        current = self.maneuver_ref_combo.currentText()
        self.maneuver_ref_combo.clear()

        if self.configured_maneuvers:
            self.maneuver_ref_combo.addItems(self.configured_maneuvers)
            if current in self.configured_maneuvers:
                self.maneuver_ref_combo.setCurrentText(current)
            else:
                self.maneuver_ref_combo.setCurrentText(self.configured_maneuvers[0])
        else:
            self.maneuver_ref_combo.addItem("pMDI_variable_trapezoid")  # Default fallback
            self.maneuver_ref_combo.setCurrentText("pMDI_variable_trapezoid")

    # Removed model change handlers - no longer needed since manual option removed

    def open_cfd_manual_dialog(self):
        """Open manual CFD configuration dialog."""
        dialog = ManualCFDDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.manual_cfd_values = dialog.get_values()
            QMessageBox.information(self, "CFD Configuration", "Manual CFD values saved.")

    def open_lung_dep_manual_dialog(self):
        """Open manual Lung Deposition configuration dialog."""
        dialog = ManualLungDepositionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.manual_lung_dep_values = dialog.get_values()
            QMessageBox.information(self, "Lung Deposition Configuration", "Manual values saved.")

    # Removed example config methods - no longer needed per user request

    def load_data_file(self):
        """Load a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data File",
            "",
            "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;TOML Files (*.toml)"
        )
        if file_path:
            self.data_file_edit.setText(file_path)
            QMessageBox.information(self, "Data Loaded", f"Data file loaded: {Path(file_path).name}")

    def run_simulation(self):
        """Run a simulation with the current configuration."""
        try:
            # Build configuration
            config = self.build_appconfig()

            # Clear previous results
            self.results_text.clear()
            self.results_text.append("Starting simulation...\n")

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            # Disable run button
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Running...")

            # Create and start worker thread
            self.simulation_worker = SimulationWorker(config)
            self.simulation_worker.progress.connect(self.on_simulation_progress)
            self.simulation_worker.finished.connect(self.on_simulation_finished)
            self.simulation_worker.error.connect(self.on_simulation_error)
            self.simulation_worker.start()

            self.simulation_started.emit()

        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to prepare configuration: {str(e)}")
            self.results_text.append(f"\nError: {str(e)}")
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Simulation")

    def on_simulation_progress(self, message: str):
        """Handle simulation progress updates."""
        self.results_text.append(message)

    def on_simulation_finished(self, results: dict):
        """Handle successful simulation completion."""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Simulation")

        # Display results
        self.results_text.append("\n" + "="*50)
        self.results_text.append("SIMULATION COMPLETED SUCCESSFULLY")
        self.results_text.append("="*50 + "\n")
        self.results_text.append(f"Run ID: {results.get('run_id', 'N/A')}")
        self.results_text.append(f"Stages completed: {', '.join(results.get('stages_completed', []))}")

        if results.get('has_deposition'):
            self.results_text.append("✓ Deposition results available")
        if results.get('has_pbbk'):
            self.results_text.append("✓ PBBK results available")
        if results.get('has_pk'):
            self.results_text.append("✓ PK results available")

        # Display PK metrics if available
        if 'cmax' in results:
            self.results_text.append(f"\nPK Metrics:")
            self.results_text.append(f"  Cmax: {results['cmax']:.4f}")
        if 'tmax' in results:
            self.results_text.append(f"  Tmax: {results['tmax']:.4f}")
        if 'auc' in results:
            self.results_text.append(f"  AUC: {results['auc']:.4f}")

        self.simulation_finished.emit(results)

        QMessageBox.information(self, "Simulation Complete", "Simulation completed successfully!")

    def on_simulation_error(self, error_msg: str):
        """Handle simulation error."""
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Simulation")

        self.results_text.append("\n" + "="*50)
        self.results_text.append("SIMULATION FAILED")
        self.results_text.append("="*50 + "\n")
        self.results_text.append(f"Error: {error_msg}")

        QMessageBox.critical(self, "Simulation Failed", f"Simulation failed: {error_msg}")

    def build_appconfig(self) -> dict:
        """Build AppConfig dict from GUI values."""
        # Get selected stages
        stages = [stage for stage, checkbox in self.stage_checkboxes.items() if checkbox.isChecked()]

        # Build the configuration dict for AppConfig
        config = {
            "run": {
                "stages": stages,
                "seed": self.seed_spin.value(),
                "threads": 1,
                "enable_numba": False,
                "artifact_dir": "results",
            },
            "study": {
                "study_type": self.study_type_combo.currentText(),
                "design": self.study_design_combo.currentText(),
                "n_subjects": self.n_subjects_spin.value(),
                "charcoal_block": self.charcoal_block_checkbox.isChecked()
            },
            "deposition": {
                "model": self.lung_dep_combo.currentText(),
                "particle_grid": "medium"
            },
            "pbbm": {
                "model": self.pbbm_combo.currentText(),
                "epi_layers": [2, 2, 1, 1]
            },
            "pk": {
                "model": self.pk_combo.currentText()
            },
            "subject": {"ref": self.subject_ref_edit.text() or "healthy_reference"},
            "api": {"ref": self.api_ref_edit.text() or "BD"},
            "product": {"ref": self.product_ref_edit.text() or "HFA_MDI_PT210"},
            "maneuver": {"ref": self.maneuver_ref_edit.text() or "pMDI_variable_trapezoid"}
        }

        # Add workflow if not custom
        workflow = self.workflow_combo.currentText()
        if workflow != "custom":
            config["run"]["workflow_name"] = workflow

        # Add population if specified (from lung geometry configuration)
        pop_text = self.population_combo.currentText()
        if pop_text and pop_text != "(No lung geometry configured)":
            config["study"]["population"] = pop_text

        # Add CFD config if needed
        if "cfd" in stages:
            if self.cfd_combo.currentText() == "Manual" and self.manual_cfd_values:
                # For now, CFD is not a standard config section, we'll skip manual CFD
                pass

        # Add manual deposition params if configured
        if self.lung_dep_combo.currentText() == "Manual" and self.manual_lung_dep_values:
            config["deposition"]["params"] = self.manual_lung_dep_values

        return config

    def build_config(self) -> Dict[str, Any]:
        """Build legacy config dict (for backward compatibility)."""
        # Get selected stages
        stages = [stage for stage, checkbox in self.stage_checkboxes.items() if checkbox.isChecked()]

        config = {
            "run": {
                "stages": stages,
                "seed": self.seed_spin.value(),
                "threads": 1,
                "enable_numba": False,
                "artifact_dir": "results",
                "workflow_name": self.workflow_combo.currentText() if self.workflow_combo.currentText() != "custom" else None
            },
            "study": {
                "study_type": self.study_type_combo.currentText(),
                "design": self.study_design_combo.currentText(),
                "n_subjects": self.n_subjects_spin.value(),
                "population": self.population_combo.currentText() if self.population_combo.currentText() != "(No lung geometry configured)" else None,
                "charcoal_block": self.charcoal_block_checkbox.isChecked()
            },
            "deposition": {
                "model": self.lung_dep_combo.currentText(),  # Now using actual model names
                "particle_grid": "medium"
            },
            "pbbm": {
                "model": self.pbbm_combo.currentText(),
                "epi_layers": [2, 2, 1, 1]
            },
            "pk": {
                "model": self.pk_combo.currentText()
            },
            "subject": {"ref": self.subject_ref_edit.text() or "healthy_reference"},
            "api": {"ref": self.api_ref_edit.text() or "BD"},
            "product": {"ref": self.product_ref_edit.text() or "HFA_MDI_PT210"},
            "maneuver": {"ref": self.maneuver_ref_edit.text() or "pMDI_variable_trapezoid"}
        }

        # Add CFD configuration
        if self.cfd_combo.currentText() == "Manual" and self.manual_cfd_values:
            config["cfd"] = {
                "model": "manual",
                "params": self.manual_cfd_values
            }
        else:
            config["cfd"] = {"model": "ml"}

        # Add manual lung deposition values if configured
        if self.lung_dep_combo.currentText() == "Manual" and self.manual_lung_dep_values:
            config["deposition"]["params"] = self.manual_lung_dep_values

        # Add data file if loaded
        if self.data_file_edit.text():
            config["data_file"] = self.data_file_edit.text()

        return config

    def validate_config(self):
        try:
            current_config = self.build_config()
            QMessageBox.information(self, "Validation", "Configuration is valid!")
            self.current_config = current_config
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Configuration error: {str(e)}")

    def save_config(self):
        try:
            current_config = self.build_config()
            QMessageBox.information(self, "Save Config", "Configuration saved!")
            self.current_config = current_config
            self.config_ready.emit(current_config)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save: {str(e)}")

    def update_from_catalog(self, category: str, data: Dict[str, Any]):
        """Update entity references when items are configured in other tabs."""
        if category == "subject":
            name = data.get('name', '')
            if name:
                self.update_configured_subject(name)
                self.subject_ref_combo.setCurrentText(name)
        elif category == "api":
            name = data.get('name', '')
            if name:
                self.update_configured_api(name)
                self.api_ref_combo.setCurrentText(name)
        elif category == "product":
            # Could be single product or multiple from grid
            self.update_configured_product(data)
            # Set to first product if available
            if self.configured_products:
                self.product_ref_combo.setCurrentText(self.configured_products[-1])
        elif category == "products":
            # From ProductGridWidget
            self.update_configured_product(data)
            if self.configured_products:
                self.product_ref_combo.setCurrentText(self.configured_products[-1])
        elif category == "maneuver":
            name = data.get('name', '')
            if name:
                self.update_configured_maneuver(name)
                self.maneuver_ref_combo.setCurrentText(name)


class SimpleTab(QWidget):
    """Simple placeholder tab."""

    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout()
        header = QLabel(title)
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        placeholder = QLabel(f"{title} functionality")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
        self.setLayout(layout)


class LMPMainWindow(QMainWindow):
    """Main application window with categorized API widget."""

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.init_ui()
        self.setup_menu()

    def init_ui(self):
        self.setWindowTitle("LMP - Lung Modeling Platform (Categorized)")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.tab_widget = QTabWidget()

        # Create tabs
        self.workspace_tab = WorkspaceTab()
        self.api_tab = CategorizedAPIWidget()  # Use categorized API widget
        self.product_tab = ProductGridWidget()  # Use new product grid widget
        self.population_tab = PopulationTabWidget()  # Use new population tab widget
        self.study_designer_tab = StudyDesignerTab()
        self.run_queue_tab = SimpleTab("Run Queue")
        self.results_tab = SimpleTab("Results Viewer")
        self.logs_tab = SimpleTab("Logs & Diagnostics")

        # Add tabs
        self.tab_widget.addTab(self.workspace_tab, "Home / Workspace")
        self.tab_widget.addTab(self.api_tab, "API Parameters")
        self.tab_widget.addTab(self.product_tab, "Product Parameters")
        self.tab_widget.addTab(self.population_tab, "Population")
        self.tab_widget.addTab(self.study_designer_tab, "Study Designer")
        self.tab_widget.addTab(self.run_queue_tab, "Run Queue")
        self.tab_widget.addTab(self.results_tab, "Results Viewer")
        self.tab_widget.addTab(self.logs_tab, "Logs & Diagnostics")

        layout.addWidget(self.tab_widget)

        # Connect signals
        self.workspace_tab.workspace_changed.connect(self.on_workspace_changed)
        self.api_tab.config_updated.connect(self.on_catalog_config_updated)
        self.api_tab.api_configured.connect(self.on_api_configured)  # API-Product integration
        self.product_tab.config_updated.connect(self.on_catalog_config_updated)
        self.population_tab.config_updated.connect(self.on_catalog_config_updated)
        self.population_tab.subject_configured.connect(self.on_subject_configured)
        self.population_tab.maneuver_configured.connect(self.on_maneuver_configured)
        self.study_designer_tab.config_ready.connect(self.on_config_ready)

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        new_workspace_action = QAction("New Workspace...", self)
        new_workspace_action.triggered.connect(self.workspace_tab.create_workspace)
        file_menu.addAction(new_workspace_action)

        open_workspace_action = QAction("Open Workspace...", self)
        open_workspace_action.triggered.connect(self.workspace_tab.browse_workspace)
        file_menu.addAction(open_workspace_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About LMP", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def on_workspace_changed(self, workspace_path: str):
        try:
            self.workspace_manager = WorkspaceManager(workspace_path)
            self._sync_saved_api_entries()
        except Exception as e:
            QMessageBox.critical(self, "Workspace Error", f"Could not set workspace: {str(e)}")

    def on_catalog_config_updated(self, category: str, data: Dict[str, Any]):
        self.study_designer_tab.update_from_catalog(category, data)

    def on_api_configured(self, api_name: str):
        """Handle when an API is configured - update product tab, population tab, and study designer."""
        # Update product tab
        if hasattr(self.product_tab, 'configured_apis'):
            if api_name not in self.product_tab.configured_apis:
                self.product_tab.configured_apis.append(api_name)
        else:
            self.product_tab.configured_apis = [api_name]

        # Update population tab's GI tract dropdown if applicable
        if hasattr(self.population_tab, 'update_saved_apis'):
            configured_apis = []
            for name in getattr(self.product_tab, 'configured_apis', []) or []:
                configured_apis.append({"name": name, "ref": name})
            try:
                self.population_tab.update_saved_apis(configured_apis)
            except Exception:
                pass

        # Update API dropdowns in the product grid
        self.product_tab.set_configured_apis(self.product_tab.configured_apis)

        # Update Study Designer tab
        self.study_designer_tab.update_configured_api(api_name)

    def on_subject_configured(self, subject_name: str):
        """Handle when a subject is configured - update study designer tab."""
        self.study_designer_tab.update_configured_subject(subject_name)

    def on_maneuver_configured(self, maneuver_name: str):
        """Handle when a maneuver is configured - update study designer tab."""
        self.study_designer_tab.update_configured_maneuver(maneuver_name)

    def on_config_ready(self, _: Dict[str, Any]):
        self.tab_widget.setCurrentWidget(self.run_queue_tab)

    def show_about(self):
        catalog_status = "Available" if CATALOG_AVAILABLE else "Direct" if CATALOG_DIRECT else "Unavailable"
        QMessageBox.about(
            self, "About LMP",
            "LMP - Lung Modeling Platform\\n\\n"
            "Version: 0.1.0\\n"
            "GUI Framework: PySide6\\n"
            f"Catalog Integration: {catalog_status}\\n\\n"
            "Features categorized API parameters like the reference interface."
        )

    def _sync_saved_api_entries(self) -> None:
        if not hasattr(self.population_tab, "update_saved_apis"):
            return

        api_info: List[Dict[str, str]] = []
        if self.workspace_manager is not None:
            try:
                entries = self.workspace_manager.list_catalog_entries("api")
            except Exception:
                entries = []

            seen: Set[Tuple[str, str]] = set()

            def _record(name_value: Optional[Any], ref_value: Optional[Any]) -> None:
                name_str = str(name_value) if name_value is not None else None
                ref_str = str(ref_value) if ref_value is not None else None
                if name_str is None and ref_str is None:
                    return
                display = name_str or ref_str
                ref = ref_str or display
                key = (display, ref)
                if key in seen:
                    return
                seen.add(key)
                api_info.append({"name": display, "ref": ref})

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                ref = entry.get("ref") or entry.get("name") or entry.get("id")
                display = entry.get("name") or entry.get("display_name") or ref
                _record(display, ref)
                overrides = entry.get("overrides")
                if isinstance(overrides, dict):
                    _record(overrides.get("name"), ref)

        self.population_tab.update_saved_apis(api_info, replace_existing=True)

        configured_names = [entry["name"] for entry in api_info if entry.get("name")]
        if hasattr(self.product_tab, "configured_apis"):
            self.product_tab.configured_apis = configured_names
            self.product_tab.set_configured_apis(configured_names)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LMP Categorized")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("LMP Team")

    window = LMPMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
