"""Population Tab Widget with Subject and Maneuver Configuration.

Provides a 2-column layout for configuring population parameters:
- Left column: Subject parameters and Inhalation maneuver selection
- Right column: Lung geometry selection and configuration
Uses Pydantic EntityRef models for configuration management.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable, Tuple
import toml
import copy

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QGroupBox, QFormLayout, QScrollArea, QSplitter, QMessageBox,
    QCheckBox, QDialog, QTabWidget, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

# Import app_api for catalog integration
sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
try:
    from lmp_pkg import app_api
    CATALOG_AVAILABLE = True
except Exception:
    CATALOG_AVAILABLE = False
    app_api = None

try:
    from lmp_pkg.catalog.builtin_loader import BuiltinDataLoader
except Exception:
    BuiltinDataLoader = None

CATALOG_ROOT = Path(__file__).parent.parent / "lmp_pkg" / "src" / "lmp_pkg" / "catalog" / "builtin"


class ParameterTableMixin:
    """Mixin class with common parameter table functionality."""

    def flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary structure."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle list of dicts (like generations in lung_geometry)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self.flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def convert_value(self, value_str: str):
        """Convert string value to appropriate type."""
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


class ParameterTable(QTableWidget, ParameterTableMixin):
    """Table widget for displaying and editing TOML parameters."""

    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)

    def populate_from_toml(self, data: Dict[str, Any]):
        """Populate table from TOML data."""
        # Check if this is Tabulated Flow with flow profile
        has_flow_profile = 'tabulated_flow_profile' in data and isinstance(data['tabulated_flow_profile'], list)

        # Separate regular parameters from flow profile
        regular_params = {k: v for k, v in data.items() if k != 'tabulated_flow_profile'}
        flattened = self.flatten_dict(regular_params)

        # Add one extra row for flow profile if present
        total_rows = len(flattened) + (1 if has_flow_profile else 0)
        self.setRowCount(total_rows)

        # Populate regular parameters
        for row, (key, value) in enumerate(flattened.items()):
            # Parameter name (read-only)
            param_item = QTableWidgetItem(key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, param_item)

            # Parameter value (editable)
            if isinstance(value, list):
                value_str = ", ".join(map(str, value))
            else:
                value_str = str(value)

            value_item = QTableWidgetItem(value_str)
            self.setItem(row, 1, value_item)

        # Add flow profile as a special widget if present
        if has_flow_profile:
            row = len(flattened)
            # Parameter name
            param_item = QTableWidgetItem("tabulated_flow_profile")
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, param_item)

            # Create a button to show flow profile
            show_btn = QPushButton(f"View/Edit Flow Profile ({len(data['tabulated_flow_profile'])} points)")
            show_btn.clicked.connect(lambda: self.show_flow_profile(data['tabulated_flow_profile']))
            self.setCellWidget(row, 1, show_btn)
            self.flow_profile_data = data['tabulated_flow_profile']

    def show_flow_profile(self, flow_profile):
        """Show flow profile in a separate dialog."""
        dialog = FlowProfileDialog(self, flow_profile)
        if dialog.exec() == QDialog.Accepted:
            self.flow_profile_data = dialog.get_flow_profile()
            # Update button text
            btn = self.cellWidget(self.rowCount() - 1, 1)
            if btn:
                btn.setText(f"View/Edit Flow Profile ({len(self.flow_profile_data)} points)")

    def get_values(self) -> Dict[str, Any]:
        """Get all values from table."""
        values = {}
        for row in range(self.rowCount()):
            param_item = self.item(row, 0)
            if param_item:
                param_name = param_item.text()

                # Check if this is the flow profile row
                if param_name == "tabulated_flow_profile" and hasattr(self, 'flow_profile_data'):
                    values[param_name] = self.flow_profile_data
                else:
                    value_item = self.item(row, 1)
                    if value_item:
                        value_text = value_item.text()
                        # Try to convert to appropriate type
                        try:
                            if '.' in value_text:
                                values[param_name] = float(value_text)
                            else:
                                values[param_name] = int(value_text)
                        except ValueError:
                            # Keep as string if conversion fails
                            values[param_name] = value_text
        return values


class GenerationsTable(QTableWidget):
    """Specialized table widget for lung geometry generations data."""

    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

    def populate_from_generations(self, generations_data: List[Dict[str, Any]]):
        """Populate table from generations array data."""
        if not generations_data:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Preserve the original order of parameters from the TOML file
        # Use the first generation to establish column order
        if generations_data:
            columns = list(generations_data[0].keys())

            # Add any additional columns from other generations (maintain order)
            all_columns = set(columns)
            for gen in generations_data:
                for key in gen.keys():
                    if key not in all_columns:
                        columns.append(key)
                        all_columns.add(key)
        else:
            columns = []

        # Set up table structure
        self.setRowCount(len(generations_data))
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)

        # Populate data
        for row, generation in enumerate(generations_data):
            for col, column_name in enumerate(columns):
                value = generation.get(column_name, "")

                # Format value appropriately
                if isinstance(value, float):
                    if abs(value) >= 1e10:  # Handle very large numbers like 1e+50
                        value_str = f"{value:.2e}"
                    else:
                        value_str = f"{value:.6g}"  # Use general format to avoid unnecessary decimals
                elif isinstance(value, (int, bool)):
                    value_str = str(value)
                else:
                    value_str = str(value)

                item = QTableWidgetItem(value_str)

                # Make generation column read-only to preserve ordering
                if column_name == "generation":
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.setItem(row, col, item)

        # Resize columns to content
        self.resizeColumnsToContents()

        # Set minimum column widths for readability
        for col in range(self.columnCount()):
            if self.columnWidth(col) < 80:
                self.setColumnWidth(col, 80)

    def get_generations_values(self) -> List[Dict[str, Any]]:
        """Get all generation values from table."""
        generations = []

        if self.rowCount() == 0 or self.columnCount() == 0:
            return generations

        # Get column headers
        headers = []
        for col in range(self.columnCount()):
            header_item = self.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"col_{col}")

        # Extract data for each generation
        for row in range(self.rowCount()):
            generation = {}
            for col, header in enumerate(headers):
                item = self.item(row, col)
                if item:
                    value_text = item.text()
                    # Convert back to appropriate type
                    generation[header] = self.convert_value(value_text)
                else:
                    generation[header] = ""
            generations.append(generation)

        return generations

    def convert_value(self, value_str: str):
        """Convert string value to appropriate type."""
        value_str = value_str.strip()

        if not value_str:
            return ""

        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"

        # Handle scientific notation
        if "e" in value_str.lower():
            try:
                return float(value_str)
            except ValueError:
                pass

        if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            return int(value_str)

        try:
            if "." in value_str:
                return float(value_str)
        except ValueError:
            pass

        return value_str

    def get_values(self) -> Dict[str, Any]:
        """Get all parameter values from table."""
        values = {}
        for row in range(self.rowCount()):
            param_item = self.item(row, 0)
            value_item = self.item(row, 1)

            if param_item and value_item:
                param_name = param_item.text()
                value_text = value_item.text()

                try:
                    # Try to convert value back to appropriate type
                    if "," in value_text:
                        value = [self.convert_value(v.strip()) for v in value_text.split(",")]
                    else:
                        value = self.convert_value(value_text)
                except:
                    value = value_text

                values[param_name] = value

        return values


class PopulationTabWidget(QWidget):
    """Population tab with subject and maneuver configuration."""

    config_updated = Signal(str, dict)  # category, config_data
    subject_configured = Signal(str)    # subject_name
    maneuver_configured = Signal(str)   # maneuver_name

    def __init__(self):
        super().__init__()
        self.current_subject = None
        self.current_inhalation = None
        self.current_lung_geometry = None
        self.current_gi_tract = None
        self.current_gi_data_key = None

        # Variability configurations
        self.subject_variability_config = None
        self.inhalation_variability_config = None
        self.lung_variability_config = None
        self.gi_variability_config = None
        self.pk_variability_config = None

        # Track saved APIs for GI dropdown
        self.saved_apis: List[str] = []
        self.saved_api_refs: List[str] = []
        self.saved_api_mapping: Dict[str, str] = {}
        self._builtin_api_names: List[str] = []
        if BuiltinDataLoader is not None:
            try:
                self._builtin_api_names = BuiltinDataLoader().get_available_api_names()
            except Exception:
                self._builtin_api_names = []

        self.init_ui()
        self.load_default_data()
        self._update_pk_variability_controls()

    # ------------------------------------------------------------------
    # Helpers for loading/merging data

    @staticmethod
    def _merge_dicts(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        if overrides:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = PopulationTabWidget._merge_dicts(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _category_dir(category: str) -> str:
        mapping = {
            "subject": "subject",
            "maneuver": "inhalation",
            "lung_geometry": "lung_geometry",
            "gi_tract": "gi_tract",
        }
        return mapping.get(category, category)

    def _load_builtin_data(self, category: str, ref: str) -> Dict[str, Any]:
        directory = self._category_dir(category)
        file_path = CATALOG_ROOT / directory / f"{ref}.toml"
        if not file_path.exists():
            return {}
        try:
            with open(file_path, 'rb') as handle:
                try:
                    import tomllib
                except ImportError:  # Python <3.11
                    import tomli as tomllib
                return tomllib.load(handle)
        except Exception:
            try:
                with open(file_path, 'r', encoding='utf-8') as handle:
                    return toml.load(handle)
            except Exception:
                return {}


    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Header
        header = QLabel("Population (Subject & Maneuver)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Main content in splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left column
        left_widget = self.create_left_column()
        splitter.addWidget(left_widget)

        # Right column
        right_widget = self.create_right_column()
        splitter.addWidget(right_widget)

        # Set equal sizes
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_left_column(self) -> QWidget:
        """Create left column with subject and inhalation configuration."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Subject section
        subject_group = QGroupBox("Subject Parameters")
        subject_layout = QVBoxLayout()

        # Subject table (loads healthy_reference.toml by default)
        self.subject_table = ParameterTable()
        subject_layout.addWidget(self.subject_table)

        # Variability controls for subject
        variability_layout = QHBoxLayout()
        self.subject_variability_checkbox = QCheckBox("Enable Variability")
        self.subject_variability_checkbox.toggled.connect(self.on_subject_variability_toggled)
        variability_layout.addWidget(self.subject_variability_checkbox)

        self.subject_variability_btn = QPushButton("Configure Variability...")
        self.subject_variability_btn.clicked.connect(self.open_subject_variability_window)
        self.subject_variability_btn.setEnabled(False)  # Initially disabled
        variability_layout.addWidget(self.subject_variability_btn)

        variability_layout.addStretch()
        subject_layout.addLayout(variability_layout)

        # Save subject button
        save_subject_btn = QPushButton("Save Subject Config")
        save_subject_btn.clicked.connect(self.save_subject_config)
        subject_layout.addWidget(save_subject_btn)

        subject_group.setLayout(subject_layout)
        layout.addWidget(subject_group)

        # Inhalation maneuver section
        inhalation_group = QGroupBox("Inhalation Maneuver")
        inhalation_layout = QVBoxLayout()

        # Dropdown selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Maneuver:"))

        self.inhalation_combo = QComboBox()
        self.inhalation_combo.currentTextChanged.connect(self.load_inhalation_parameters)
        selection_layout.addWidget(self.inhalation_combo)

        refresh_inhalation_btn = QPushButton("Refresh")
        refresh_inhalation_btn.clicked.connect(self.refresh_inhalation_list)
        selection_layout.addWidget(refresh_inhalation_btn)

        inhalation_layout.addLayout(selection_layout)

        # Inhalation parameter table
        self.inhalation_table = ParameterTable()
        inhalation_layout.addWidget(self.inhalation_table)

        # Variability controls for inhalation
        inhalation_variability_layout = QHBoxLayout()
        self.inhalation_variability_checkbox = QCheckBox("Enable Variability")
        self.inhalation_variability_checkbox.toggled.connect(self.on_inhalation_variability_toggled)
        inhalation_variability_layout.addWidget(self.inhalation_variability_checkbox)

        self.inhalation_variability_btn = QPushButton("Configure Variability...")
        self.inhalation_variability_btn.clicked.connect(self.open_inhalation_variability_window)
        self.inhalation_variability_btn.setEnabled(False)  # Initially disabled
        inhalation_variability_layout.addWidget(self.inhalation_variability_btn)

        inhalation_variability_layout.addStretch()
        inhalation_layout.addLayout(inhalation_variability_layout)

        # Save inhalation button
        save_inhalation_btn = QPushButton("Save Maneuver Config")
        save_inhalation_btn.clicked.connect(self.save_inhalation_config)
        inhalation_layout.addWidget(save_inhalation_btn)

        inhalation_group.setLayout(inhalation_layout)
        layout.addWidget(inhalation_group)

        widget.setLayout(layout)
        return widget

    def create_right_column(self) -> QWidget:
        """Create right column with lung geometry configuration."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Lung geometry section
        lung_group = QGroupBox("Lung Geometry")
        lung_layout = QVBoxLayout()

        # Dropdown selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Geometry:"))

        self.lung_geometry_combo = QComboBox()
        self.lung_geometry_combo.currentTextChanged.connect(self.load_lung_geometry_parameters)
        selection_layout.addWidget(self.lung_geometry_combo)

        refresh_lung_btn = QPushButton("Refresh")
        refresh_lung_btn.clicked.connect(self.refresh_lung_geometry_list)
        selection_layout.addWidget(refresh_lung_btn)

        lung_layout.addLayout(selection_layout)

        # Lung geometry parameter table (specialized for generations data)
        scroll_area = QScrollArea()
        self.lung_geometry_table = GenerationsTable()
        scroll_area.setWidget(self.lung_geometry_table)
        scroll_area.setWidgetResizable(True)
        lung_layout.addWidget(scroll_area)

        # Variability controls for lung geometry
        lung_variability_layout = QHBoxLayout()
        self.lung_variability_checkbox = QCheckBox("Enable Variability")
        self.lung_variability_checkbox.toggled.connect(self.on_lung_variability_toggled)
        lung_variability_layout.addWidget(self.lung_variability_checkbox)

        self.lung_variability_btn = QPushButton("Configure Variability...")
        self.lung_variability_btn.clicked.connect(self.open_lung_variability_window)
        self.lung_variability_btn.setEnabled(False)  # Initially disabled
        lung_variability_layout.addWidget(self.lung_variability_btn)

        lung_variability_layout.addStretch()
        lung_layout.addLayout(lung_variability_layout)

        # Save lung geometry button
        save_lung_btn = QPushButton("Save Lung Geometry Config")
        save_lung_btn.clicked.connect(self.save_lung_geometry_config)
        lung_layout.addWidget(save_lung_btn)

        lung_group.setLayout(lung_layout)
        layout.addWidget(lung_group)

        # GI Tract section
        gi_group = QGroupBox("GI Tract")
        gi_layout = QVBoxLayout()

        # Dropdown selection
        gi_selection_layout = QHBoxLayout()
        gi_selection_layout.addWidget(QLabel("Select GI Type:"))

        self.gi_combo = QComboBox()
        self.gi_combo.currentTextChanged.connect(self.load_gi_tract_data)
        gi_selection_layout.addWidget(self.gi_combo)

        refresh_gi_btn = QPushButton("Refresh")
        refresh_gi_btn.clicked.connect(self.refresh_gi_tract_list)
        gi_selection_layout.addWidget(refresh_gi_btn)

        gi_layout.addLayout(gi_selection_layout)

        # GI Tract table
        self.gi_table = GITractTable()
        gi_layout.addWidget(self.gi_table)

        # Variability controls for GI Tract
        gi_variability_layout = QHBoxLayout()

        self.gi_variability_checkbox = QCheckBox("Enable GI Variability")
        self.gi_variability_checkbox.toggled.connect(self.on_gi_variability_toggled)
        gi_variability_layout.addWidget(self.gi_variability_checkbox)

        self.gi_variability_btn = QPushButton("Configure Variability...")
        self.gi_variability_btn.clicked.connect(self.open_gi_variability_window)
        self.gi_variability_btn.setEnabled(False)  # Initially disabled
        gi_variability_layout.addWidget(self.gi_variability_btn)

        gi_variability_layout.addStretch()
        gi_layout.addLayout(gi_variability_layout)

        # Save GI Tract button
        save_gi_btn = QPushButton("Save GI Tract Config")
        save_gi_btn.clicked.connect(self.save_gi_tract_config)
        gi_layout.addWidget(save_gi_btn)

        gi_group.setLayout(gi_layout)
        layout.addWidget(gi_group)

        # PK Variability section
        pk_group = QGroupBox("PK Variability")
        pk_layout = QVBoxLayout()

        pk_info = QLabel("Configure pharmacokinetic variability factors per API.")
        pk_info.setWordWrap(True)
        pk_layout.addWidget(pk_info)

        pk_controls = QHBoxLayout()
        self.pk_variability_checkbox = QCheckBox("Enable PK Variability")
        self.pk_variability_checkbox.toggled.connect(self.on_pk_variability_toggled)
        pk_controls.addWidget(self.pk_variability_checkbox)

        self.pk_variability_btn = QPushButton("Configure Variability...")
        self.pk_variability_btn.clicked.connect(self.open_pk_variability_window)
        self.pk_variability_btn.setEnabled(False)
        pk_controls.addWidget(self.pk_variability_btn)
        pk_controls.addStretch()
        pk_layout.addLayout(pk_controls)

        pk_group.setLayout(pk_layout)
        layout.addWidget(pk_group)

        widget.setLayout(layout)
        return widget

    def load_default_data(self):
        """Load default data on initialization."""
        # Load subject data (healthy_reference.toml)
        self.load_subject_data()

        # Refresh dropdown lists
        self.refresh_inhalation_list()
        self.refresh_lung_geometry_list()
        self.refresh_gi_tract_list()

    # ------------------------------------------------------------------
    # External loading helpers

    def load_subject_entry(self, ref: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = self._load_builtin_data("subject", ref)
        merged = self._merge_dicts(base, overrides)
        variability = merged.pop("variability", None)
        if "variability_overrides" in merged:
            variability = merged.pop("variability_overrides")
        self.subject_table.populate_from_toml(merged)
        self.current_subject = ref
        self.subject_variability_checkbox.blockSignals(True)
        self.subject_variability_checkbox.setChecked(bool(variability))
        self.subject_variability_checkbox.blockSignals(False)
        self.subject_variability_btn.setEnabled(bool(variability))
        self.subject_variability_config = variability

        pk_variability = merged.pop("pk_variability", None)
        self.pk_variability_config = pk_variability
        self.pk_variability_checkbox.blockSignals(True)
        self.pk_variability_checkbox.setChecked(bool(pk_variability))
        self.pk_variability_checkbox.blockSignals(False)
        self._update_pk_variability_controls()

    def load_maneuver_entry(self, ref: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = self._load_builtin_data("maneuver", ref)
        merged = self._merge_dicts(base, overrides)
        variability = merged.pop("variability", None)
        if "variability_overrides" in merged:
            variability = merged.pop("variability_overrides")
        self.inhalation_table.populate_from_toml(merged)
        self.current_inhalation = ref
        self.inhalation_combo.blockSignals(True)
        index = self.inhalation_combo.findText(ref)
        if index >= 0:
            self.inhalation_combo.setCurrentIndex(index)
        self.inhalation_combo.blockSignals(False)
        self.inhalation_variability_checkbox.blockSignals(True)
        self.inhalation_variability_checkbox.setChecked(bool(variability))
        self.inhalation_variability_checkbox.blockSignals(False)
        self.inhalation_variability_btn.setEnabled(bool(variability))
        self.inhalation_variability_config = variability

    def load_lung_geometry_entry(self, ref: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = self._load_builtin_data("lung_geometry", ref)
        merged = self._merge_dicts(base, overrides)
        variability = merged.pop("variability", None)
        if "variability_overrides" in merged:
            variability = merged.pop("variability_overrides")
        generations = merged.get("generations") if isinstance(merged.get("generations"), list) else []
        self.lung_geometry_table.populate_from_generations(generations)
        self.current_lung_geometry = ref
        self.lung_geometry_combo.blockSignals(True)
        index = self.lung_geometry_combo.findText(ref)
        if index >= 0:
            self.lung_geometry_combo.setCurrentIndex(index)
        self.lung_geometry_combo.blockSignals(False)
        self.lung_variability_checkbox.blockSignals(True)
        self.lung_variability_checkbox.setChecked(bool(variability))
        self.lung_variability_checkbox.blockSignals(False)
        self.lung_variability_btn.setEnabled(bool(variability))
        self.lung_variability_config = variability

    def load_gi_entry(self, ref: str, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = self._load_builtin_data("gi_tract", "default")
        merged = self._merge_dicts(base, overrides)
        variability = merged.pop("variability", None)
        if "variability_overrides" in merged:
            variability = merged.pop("variability_overrides")
        gi_type = ref
        target_api = gi_type
        if target_api in self.saved_api_mapping:
            target_api = self.saved_api_mapping[target_api]
        if target_api and target_api.lower() == "default":
            if self.saved_api_refs:
                target_api = self.saved_api_refs[0]
            elif self.saved_apis and self.saved_apis[0] in self.saved_api_mapping:
                target_api = self.saved_api_mapping[self.saved_apis[0]]
            elif self._builtin_api_names:
                target_api = self._builtin_api_names[0]
        self.gi_table.populate_from_gi_toml(merged, target_api or gi_type)
        self.current_gi_tract = gi_type
        self.current_gi_data_key = target_api or gi_type
        self.gi_combo.blockSignals(True)
        index = self.gi_combo.findText(ref)
        if index >= 0:
            self.gi_combo.setCurrentIndex(index)
        self.gi_combo.blockSignals(False)
        self.gi_variability_checkbox.blockSignals(True)
        self.gi_variability_checkbox.setChecked(bool(variability))
        self.gi_variability_checkbox.blockSignals(False)
        self.gi_variability_btn.setEnabled(bool(variability))
        self.gi_variability_config = variability

    def load_subject_data(self):
        """Load subject data from healthy_reference.toml."""
        try:
            subject_file = CATALOG_ROOT / "subject" / "healthy_reference.toml"
            if subject_file.exists():
                with open(subject_file, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                self.subject_table.populate_from_toml(data)
                self.current_subject = "healthy_reference"
            else:
                # Fallback data
                fallback_data = {
                    "name": "healthy_reference",
                    "age_years": 35.6,
                    "weight_kg": 97.8,
                    "height_cm": 183.8,
                    "sex": "M"
                }
                self.subject_table.populate_from_toml(fallback_data)
                self.current_subject = "healthy_reference"
        except Exception as e:
            print(f"Error loading subject data: {e}")
        finally:
            self._update_pk_variability_controls()

    def refresh_inhalation_list(self):
        """Refresh inhalation maneuver dropdown list."""
        try:
            inhalation_dir = CATALOG_ROOT / "inhalation"
            if inhalation_dir.exists():
                entries = []
                for file_path in inhalation_dir.glob("*.toml"):
                    if not file_path.name.startswith("Variability_"):
                        entries.append(file_path.stem)

                self.inhalation_combo.clear()
                self.inhalation_combo.addItems(sorted(entries))

                if entries:
                    self.load_inhalation_parameters()
            else:
                self.inhalation_combo.clear()
                self.inhalation_combo.addItem("No inhalation files found")

        except Exception as e:
            print(f"Error loading inhalation list: {e}")
            self.inhalation_combo.clear()
            self.inhalation_combo.addItem("Error loading inhalation files")

    def refresh_lung_geometry_list(self):
        """Refresh lung geometry dropdown list."""
        try:
            lung_dir = CATALOG_ROOT / "lung_geometry"
            if lung_dir.exists():
                entries = []
                for file_path in lung_dir.glob("*.toml"):
                    if not file_path.name.startswith("Variability_"):
                        entries.append(file_path.stem)

                self.lung_geometry_combo.clear()
                self.lung_geometry_combo.addItems(sorted(entries))

                if entries:
                    self.load_lung_geometry_parameters()
            else:
                self.lung_geometry_combo.clear()
                self.lung_geometry_combo.addItem("No lung geometry files found")

        except Exception as e:
            print(f"Error loading lung geometry list: {e}")
            self.lung_geometry_combo.clear()
            self.lung_geometry_combo.addItem("Error loading lung geometry files")

    def load_inhalation_parameters(self):
        """Load parameters for selected inhalation maneuver."""
        maneuver_name = self.inhalation_combo.currentText()
        if not maneuver_name or maneuver_name.startswith("Error") or maneuver_name.startswith("No "):
            self.inhalation_table.setRowCount(0)
            return

        try:
            file_path = CATALOG_ROOT / "inhalation" / f"{maneuver_name}.toml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                self.inhalation_table.populate_from_toml(data)
                self.current_inhalation = maneuver_name
            else:
                self.inhalation_table.setRowCount(1)
                self.inhalation_table.setItem(0, 0, QTableWidgetItem("Error"))
                self.inhalation_table.setItem(0, 1, QTableWidgetItem(f"File not found: {file_path}"))

        except Exception as e:
            print(f"Error loading inhalation parameters: {e}")
            self.inhalation_table.setRowCount(1)
            self.inhalation_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.inhalation_table.setItem(0, 1, QTableWidgetItem(str(e)))

    def load_lung_geometry_parameters(self):
        """Load parameters for selected lung geometry."""
        geometry_name = self.lung_geometry_combo.currentText()
        if not geometry_name or geometry_name.startswith("Error") or geometry_name.startswith("No "):
            self.lung_geometry_table.setRowCount(0)
            self.lung_geometry_table.setColumnCount(0)
            return

        try:
            file_path = CATALOG_ROOT / "lung_geometry" / f"{geometry_name}.toml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = toml.load(f)

                # Extract generations array and populate specialized table
                if 'generations' in data and isinstance(data['generations'], list):
                    self.lung_geometry_table.populate_from_generations(data['generations'])
                else:
                    # Fallback to regular table for non-generations data
                    self.lung_geometry_table.setRowCount(1)
                    self.lung_geometry_table.setColumnCount(2)
                    self.lung_geometry_table.setHorizontalHeaderLabels(["Parameter", "Value"])
                    self.lung_geometry_table.setItem(0, 0, QTableWidgetItem("Error"))
                    self.lung_geometry_table.setItem(0, 1, QTableWidgetItem("No generations data found"))

                self.current_lung_geometry = geometry_name
            else:
                self.lung_geometry_table.setRowCount(1)
                self.lung_geometry_table.setColumnCount(2)
                self.lung_geometry_table.setHorizontalHeaderLabels(["Parameter", "Value"])
                self.lung_geometry_table.setItem(0, 0, QTableWidgetItem("Error"))
                self.lung_geometry_table.setItem(0, 1, QTableWidgetItem(f"File not found: {file_path}"))

        except Exception as e:
            print(f"Error loading lung geometry parameters: {e}")
            self.lung_geometry_table.setRowCount(1)
            self.lung_geometry_table.setColumnCount(2)
            self.lung_geometry_table.setHorizontalHeaderLabels(["Parameter", "Value"])
            self.lung_geometry_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.lung_geometry_table.setItem(0, 1, QTableWidgetItem(str(e)))

    def save_subject_config(self):
        """Save subject configuration using Pydantic EntityRef."""
        if not self.current_subject:
            QMessageBox.warning(self, "No Subject", "No subject data to save.")
            return

        # Import Pydantic model
        from lmp_pkg.config.model import EntityRef

        # Get parameter overrides from table
        overrides = self.subject_table.get_values()

        # Add variability configuration if enabled
        if self.subject_variability_checkbox.isChecked() and self.subject_variability_config:
            overrides["variability"] = self.subject_variability_config

        if self.pk_variability_checkbox.isChecked() and self.pk_variability_config:
            overrides["pk_variability"] = copy.deepcopy(self.pk_variability_config)
        elif "pk_variability" in overrides:
            overrides.pop("pk_variability")

        # Create EntityRef with reference and parameter overrides
        entity_ref = EntityRef(
            ref=self.current_subject,
            overrides=overrides
        )

        # Convert to dict using model_dump()
        config_data = entity_ref.model_dump()

        # Add name for backward compatibility
        config_data["name"] = self.current_subject

        self.config_updated.emit("subject", config_data)
        self.subject_configured.emit(self.current_subject)

        QMessageBox.information(self, "Saved", f"Subject '{self.current_subject}' saved using Pydantic model.")

    def save_inhalation_config(self):
        """Save inhalation maneuver configuration using Pydantic EntityRef."""
        if not self.current_inhalation:
            QMessageBox.warning(self, "No Maneuver", "Please select an inhalation maneuver.")
            return

        # Import Pydantic model
        from lmp_pkg.config.model import EntityRef

        # Get parameter overrides from table
        overrides = self.inhalation_table.get_values()

        # Add variability configuration if enabled
        if self.inhalation_variability_checkbox.isChecked() and self.inhalation_variability_config:
            overrides["variability"] = self.inhalation_variability_config

        # Create EntityRef with reference and parameter overrides
        entity_ref = EntityRef(
            ref=self.current_inhalation,
            overrides=overrides
        )

        # Convert to dict using model_dump()
        config_data = entity_ref.model_dump()

        # Add name for backward compatibility
        config_data["name"] = self.current_inhalation

        self.config_updated.emit("maneuver", config_data)
        self.maneuver_configured.emit(self.current_inhalation)

        QMessageBox.information(self, "Saved", f"Maneuver '{self.current_inhalation}' saved using Pydantic model.")

    def save_lung_geometry_config(self):
        """Save lung geometry configuration using Pydantic EntityRef."""
        if not self.current_lung_geometry:
            QMessageBox.warning(self, "No Geometry", "Please select a lung geometry.")
            return

        # Import Pydantic model
        from lmp_pkg.config.model import EntityRef

        # Get parameter overrides from generations table
        generations_data = self.lung_geometry_table.get_generations_values()

        # Structure overrides appropriately for lung geometry
        overrides = {}
        if generations_data:
            overrides["generations"] = generations_data

        # Add variability configuration if enabled
        if self.lung_variability_checkbox.isChecked() and self.lung_variability_config:
            overrides["variability"] = self.lung_variability_config

        # Create EntityRef with reference and parameter overrides
        entity_ref = EntityRef(
            ref=self.current_lung_geometry,
            overrides=overrides
        )

        # Convert to dict using model_dump()
        config_data = entity_ref.model_dump()

        # Add name for backward compatibility
        config_data["name"] = self.current_lung_geometry

        self.config_updated.emit("lung_geometry", config_data)

        QMessageBox.information(self, "Saved", f"Lung geometry '{self.current_lung_geometry}' with {len(generations_data)} generations saved using Pydantic model.")

    # Variability control methods
    def on_subject_variability_toggled(self, checked: bool):
        """Handle subject variability checkbox toggle."""
        self.subject_variability_btn.setEnabled(checked)

    def on_inhalation_variability_toggled(self, checked: bool):
        """Handle inhalation variability checkbox toggle."""
        self.inhalation_variability_btn.setEnabled(checked)

    def on_lung_variability_toggled(self, checked: bool):
        """Handle lung geometry variability checkbox toggle."""
        self.lung_variability_btn.setEnabled(checked)

    def open_subject_variability_window(self):
        """Open subject variability configuration window."""
        if not self.current_subject:
            QMessageBox.warning(self, "No Subject", "Please select a subject first.")
            return

        variability_file = f"Variability_{self.current_subject}.toml"
        dialog = VariabilityConfigDialog(self, "subject", self.current_subject, variability_file)
        if dialog.exec() == QDialog.Accepted:
            self.subject_variability_config = dialog.get_variability_config()

    def open_inhalation_variability_window(self):
        """Open inhalation variability configuration window."""
        if not self.current_inhalation:
            QMessageBox.warning(self, "No Maneuver", "Please select an inhalation maneuver first.")
            return

        variability_file = f"Variability_{self.current_inhalation}.toml"
        dialog = VariabilityConfigDialog(self, "inhalation", self.current_inhalation, variability_file)
        if dialog.exec() == QDialog.Accepted:
            self.inhalation_variability_config = dialog.get_variability_config()

    def open_lung_variability_window(self):
        """Open lung geometry variability configuration window."""
        if not self.current_lung_geometry:
            QMessageBox.warning(self, "No Geometry", "Please select a lung geometry first.")
            return

        variability_file = f"Variability_{self.current_lung_geometry}.toml"
        dialog = VariabilityConfigDialog(self, "lung_geometry", self.current_lung_geometry, variability_file)
        if dialog.exec() == QDialog.Accepted:
            self.lung_variability_config = dialog.get_variability_config()

    def on_gi_variability_toggled(self, checked: bool):
        """Handle GI variability checkbox toggle."""
        self.gi_variability_btn.setEnabled(checked)

    def open_gi_variability_window(self):
        """Open GI Tract variability configuration window."""
        if not self.current_gi_tract:
            QMessageBox.warning(self, "No GI Type", "Please select a GI tract type first.")
            return

        variability_file = f"Variability_default.toml"  # GI uses default variability
        dialog = GIVariabilityConfigDialog(self, "gi_tract", self.current_gi_tract, variability_file)
        if dialog.exec() == QDialog.Accepted:
            self.gi_variability_config = dialog.get_variability_config()

    def on_pk_variability_toggled(self, checked: bool) -> None:
        """Handle PK variability checkbox toggle."""
        self.pk_variability_btn.setEnabled(checked)
        if checked and self.pk_variability_config is None:
            if not self.open_pk_variability_window():
                self.pk_variability_checkbox.blockSignals(True)
                self.pk_variability_checkbox.setChecked(False)
                self.pk_variability_checkbox.blockSignals(False)
                self.pk_variability_btn.setEnabled(False)
        self._update_pk_variability_controls()

    def open_pk_variability_window(self) -> bool:
        """Open PK variability configuration dialog."""
        available_apis = self._available_pk_api_names()
        if not available_apis:
            QMessageBox.information(
                self,
                "PK Variability",
                "Configure and save at least one API before editing PK variability."
            )
            return False
        variability_file = "Variability_default.toml"
        dialog = PKVariabilityDialog(
            self,
            available_apis=available_apis,
            variability_file=variability_file,
            initial_config=self.pk_variability_config,
        )
        result = dialog.exec()
        if result == QDialog.Accepted:
            self.pk_variability_config = dialog.get_variability_config()
            self._update_pk_variability_controls()
            return True
        return False

    def refresh_gi_tract_list(self):
        """Refresh GI tract dropdown list."""
        try:
            previous_selection = self.current_gi_tract or self.gi_combo.currentText()

            options: List[str] = ["Default"]
            for api_name in self.saved_apis:
                if api_name and api_name not in options:
                    options.append(api_name)

            self.gi_combo.blockSignals(True)
            self.gi_combo.clear()
            self.gi_combo.addItems(options)

            if previous_selection in options:
                target_selection = previous_selection
            else:
                target_selection = options[0] if options else ""

            if target_selection:
                index = self.gi_combo.findText(target_selection)
                if index >= 0:
                    self.gi_combo.setCurrentIndex(index)
            self.gi_combo.blockSignals(False)

            if target_selection:
                self.load_gi_tract_data(target_selection)

        except Exception as e:
            print(f"Error loading GI tract list: {e}")
            self.gi_combo.clear()
            self.gi_combo.addItem("Default")

    def load_gi_tract_data(self, gi_type=None):
        """Load GI tract data for selected type."""
        try:
            if gi_type is None:
                gi_type = self.gi_combo.currentText()

            if not gi_type:
                self.gi_table.setRowCount(0)
                self.current_gi_tract = None
                self.current_gi_data_key = None
                return

            desired_key = self.saved_api_mapping.get(gi_type, gi_type)

            gi_file = CATALOG_ROOT / "gi_tract" / "default.toml"
            if gi_file.exists():
                try:
                    import tomllib
                    with open(gi_file, 'rb') as f:
                        data = tomllib.load(f)
                except ImportError:
                    import tomli as tomllib
                    with open(gi_file, 'rb') as f:
                        data = tomllib.load(f)

                resolved_key = self._resolve_gi_data_key(data, desired_key, gi_type)
                if resolved_key:
                    self.gi_table.populate_from_gi_toml(data, resolved_key)
                    self.current_gi_data_key = resolved_key
                else:
                    self.gi_table.setRowCount(0)
                    self.current_gi_data_key = None
                self.current_gi_tract = gi_type

        except Exception as e:
            print(f"Error loading GI tract data: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load GI tract data: {str(e)}")

    def save_gi_tract_config(self):
        """Save GI tract configuration using Pydantic EntityRef."""
        if not self.current_gi_tract:
            QMessageBox.warning(self, "No GI Type", "Please select a GI tract type.")
            return

        # Import Pydantic model
        from lmp_pkg.config.model import EntityRef

        # Get parameter overrides from table
        overrides = self.gi_table.get_values()

        # Add variability configuration if enabled
        if self.gi_variability_checkbox.isChecked() and self.gi_variability_config:
            overrides["variability"] = self.gi_variability_config

        # Create EntityRef with reference and parameter overrides
        target_ref = self.saved_api_mapping.get(
            self.current_gi_tract, self.current_gi_data_key or self.current_gi_tract
        )
        entity_ref = EntityRef(
            ref=f"gi_tract_{target_ref}",
            overrides=overrides
        )

        # Convert to dict using model_dump()
        config_data = entity_ref.model_dump()

        # Add name for backward compatibility
        config_data["name"] = target_ref

        self.config_updated.emit("gi_tract", config_data)

        QMessageBox.information(self, "Saved", f"GI Tract '{self.current_gi_tract}' saved using Pydantic model.")

    def update_saved_apis(
        self,
        saved_api_info: Iterable[Dict[str, Any]],
        *,
        replace_existing: bool = False,
    ) -> None:
        """Update the list of saved APIs (display and refs) for GI/PK editors."""
        prev_mapping = {} if replace_existing else (getattr(self, "saved_api_mapping", {}) or {})

        display_names: List[str] = []
        refs: List[str] = []
        mapping: Dict[str, str] = {}
        for info in saved_api_info or []:
            if not isinstance(info, dict):
                continue
            name = info.get("name")
            ref = info.get("ref") or prev_mapping.get(str(name), name)
            if name and str(name) not in display_names:
                display_names.append(str(name))
            if ref and str(ref) not in refs:
                refs.append(str(ref))
            if name and ref:
                mapping[str(name)] = str(ref)

        if not mapping and prev_mapping and not replace_existing:
            mapping = prev_mapping
        if not display_names and prev_mapping and not replace_existing:
            display_names = list(prev_mapping.keys())
        if not refs and prev_mapping and not replace_existing:
            refs = list(dict.fromkeys(prev_mapping.values()))

        self.saved_apis = display_names
        self.saved_api_refs = refs
        self.saved_api_mapping = mapping
        self.refresh_gi_tract_list()
        self._update_pk_variability_controls()

    def _available_pk_api_names(self) -> List[str]:
        names: List[str] = []
        preferred_sources = self.saved_api_refs or self.saved_apis
        for api_name in preferred_sources:
            if api_name and api_name not in names:
                names.append(api_name)

        if not names and self.pk_variability_config:
            for payload in self.pk_variability_config.values():
                if isinstance(payload, dict):
                    for api_name in payload.keys():
                        if api_name and api_name not in names:
                            names.append(api_name)

        return names

    def _update_pk_variability_controls(self) -> None:
        """Enable or disable PK variability controls based on available APIs/config."""
        has_api_targets = bool(self.saved_api_refs or self.saved_apis)
        has_existing_config = bool(self.pk_variability_config)
        can_configure = has_api_targets or has_existing_config

        self.pk_variability_checkbox.setEnabled(True)
        if not can_configure:
            self.pk_variability_checkbox.blockSignals(True)
            self.pk_variability_checkbox.setChecked(False)
            self.pk_variability_checkbox.blockSignals(False)
            self.pk_variability_btn.setEnabled(False)
            self.pk_variability_btn.setToolTip(
                "Save at least one API in the API/Product tabs to configure PK variability."
            )
        else:
            self.pk_variability_btn.setEnabled(self.pk_variability_checkbox.isChecked())
            self.pk_variability_btn.setToolTip("")

    def _resolve_gi_data_key(
        self,
        data: Dict[str, Any],
        desired_key: Optional[str],
        display_key: str,
    ) -> Optional[str]:
        available_keys: List[str] = []
        for section in ("gi_area", "gi_tg", "gi_vol"):
            section_data = data.get(section)
            if isinstance(section_data, dict):
                for key in section_data.keys():
                    if key and key not in available_keys:
                        available_keys.append(key)

        if desired_key and desired_key in available_keys:
            return desired_key

        if display_key and display_key in available_keys:
            return display_key

        if display_key.lower() == "default":
            for api_name in self.saved_api_refs:
                if api_name in available_keys:
                    return api_name
            for api_name in self.saved_apis:
                mapped = self.saved_api_mapping.get(api_name, api_name)
                if mapped in available_keys:
                    return mapped

        return available_keys[0] if available_keys else None


class VariabilityConfigDialog(QDialog):
    """Dialog for configuring variability parameters."""

    def __init__(self, parent, category: str, item_name: str, variability_file: str):
        super().__init__(parent)
        self.parent_widget = parent
        self.category = category
        self.item_name = item_name
        self.variability_file = variability_file
        self.variability_config = None
        self.setWindowTitle(f"Variability Configuration - {item_name}")

        # Adjust window size based on category
        if category == "lung_geometry":
            self.setGeometry(200, 200, 1200, 700)
        else:
            self.setGeometry(200, 200, 900, 600)

        self.setModal(True)  # Make it modal
        self.init_ui()
        self.load_variability_data()

    def init_ui(self):
        """Initialize the variability dialog UI."""
        layout = QVBoxLayout()

        # Header
        header = QLabel(f"Variability Configuration for {self.item_name}")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Info label
        if self.category == "lung_geometry":
            info = QLabel("Configure lung geometry variability.\nShowing inter_sigma and inter_dist for each generation parameter.")
        else:
            info = QLabel("Configure inter/intra-subject variability parameters.\nEach parameter has: [inter_sigma, inter_dist, intra_sigma, intra_dist, dependent_field]")
        layout.addWidget(info)

        # Create appropriate table based on category
        if self.category == "lung_geometry":
            self.variability_table = LungGeometryVariabilityTable()
        else:
            self.variability_table = StandardVariabilityTable()
        layout.addWidget(self.variability_table)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)  # Use accept() for QDialog
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)  # Use reject() for QDialog
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_variability_data(self):
        """Load variability data from TOML file."""
        try:
            file_path = CATALOG_ROOT / self.category / self.variability_file
            if file_path.exists():
                try:
                    import tomllib
                    with open(file_path, 'rb') as f:
                        data = tomllib.load(f)
                except ImportError:
                    import tomli as tomllib
                    with open(file_path, 'rb') as f:
                        data = tomllib.load(f)

                self.variability_table.populate_from_variability_toml(data)
            else:
                self._load_fallback_variability()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading variability data: {str(e)}")

    def _load_fallback_variability(self) -> None:
        """Populate variability table when no builtin variability file exists."""
        if self._apply_existing_variability_config():
            return

        base_ref = self._resolve_base_reference()
        fallback_path = None
        if base_ref:
            candidate = CATALOG_ROOT / self.category / f"Variability_{base_ref}.toml"
            if candidate.exists():
                fallback_path = candidate

        if fallback_path is None:
            default_refs = {
                "subject": "healthy_reference",
                "inhalation": "pMDI_variable_trapezoid",
                "gi_tract": "default",
            }
            default_ref = default_refs.get(self.category)
            if default_ref and default_ref != base_ref:
                candidate = CATALOG_ROOT / self.category / f"Variability_{default_ref}.toml"
                if candidate.exists():
                    fallback_path = candidate
                    base_ref = default_ref

        if fallback_path is not None:
            try:
                try:
                    import tomllib  # type: ignore[attr-defined]
                except ImportError:  # pragma: no cover - Python < 3.11
                    import tomli as tomllib  # type: ignore
                with open(fallback_path, 'rb') as handle:
                    data = tomllib.load(handle)
                self.variability_table.populate_from_variability_toml(data)
                self._ensure_variability_checkbox_enabled()
                QMessageBox.information(
                    self,
                    "Variability",
                    f"Loaded fallback variability from {base_ref}."
                )
                return
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Variability",
                    f"Could not load fallback variability: {exc}"
                )

        parameter_names = self._collect_parameter_names()
        if isinstance(self.variability_table, StandardVariabilityTable) and parameter_names:
            self.variability_table.populate_from_parameter_names(parameter_names)
            self._ensure_variability_checkbox_enabled()
            QMessageBox.information(
                self,
                "Variability",
                "No variability template found; starting with default entries."
            )
        else:
            QMessageBox.warning(
                self,
                "File Not Found",
                f"Variability file not found: {self.variability_file}"
            )
            self._disable_variability_toggle()

    def _apply_existing_variability_config(self) -> bool:
        config_map = {
            "subject": "subject_variability_config",
            "inhalation": "inhalation_variability_config",
            "gi_tract": "gi_variability_config",
            "pk": "pk_variability_config",
        }
        attr = config_map.get(self.category)
        if attr:
            config = getattr(self.parent_widget, attr, None)
            if isinstance(config, dict) and config:
                self.variability_table.populate_from_variability_toml(config)
                self._ensure_variability_checkbox_enabled()
                return True
        return False

    def _resolve_base_reference(self) -> Optional[str]:
        mapping = {
            "subject": getattr(self.parent_widget, "current_subject", None),
            "inhalation": getattr(self.parent_widget, "current_inhalation", None),
            "gi_tract": getattr(self.parent_widget, "current_gi_tract", None),
        }
        return mapping.get(self.category)

    def _disable_variability_toggle(self) -> None:
        checkbox_map = {
            "subject": "subject_variability_checkbox",
            "inhalation": "inhalation_variability_checkbox",
            "gi_tract": "gi_variability_checkbox",
            "pk": "pk_variability_checkbox",
        }
        attr = checkbox_map.get(self.category)
        if not attr:
            return
        checkbox = getattr(self.parent_widget, attr, None)
        if checkbox is None:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)
        checkbox.setEnabled(False)

    def _ensure_variability_checkbox_enabled(self) -> None:
        checkbox_map = {
            "subject": "subject_variability_checkbox",
            "inhalation": "inhalation_variability_checkbox",
            "gi_tract": "gi_variability_checkbox",
            "pk": "pk_variability_checkbox",
        }
        attr = checkbox_map.get(self.category)
        if not attr:
            return
        checkbox = getattr(self.parent_widget, attr, None)
        if checkbox is not None:
            checkbox.setEnabled(True)

    def _collect_parameter_names(self) -> List[str]:
        """Gather parameter names for the active category to seed variability rows."""
        try:
            if self.category == "subject":
                table = getattr(self.parent_widget, "subject_table", None)
                values = table.get_values() if table else {}
                return [name for name in values.keys() if isinstance(name, str)]

            if self.category == "inhalation":
                table = getattr(self.parent_widget, "inhalation_table", None)
                values = table.get_values() if table else {}
                return [
                    name for name in values.keys()
                    if isinstance(name, str) and name != "tabulated_flow_profile"
                ]

            if self.category == "gi_tract":
                table = getattr(self.parent_widget, "gi_table", None)
                if table and hasattr(table, "get_parameter_keys"):
                    return table.get_parameter_keys()

            if self.category == "pk":
                return [
                    "clearance_L_h",
                    "hepatic_extraction",
                    "volume_central_L",
                    "q_inter_L_h",
                    "ka_h",
                    "f_bioavail",
                ]

        except Exception:
            pass
        return []


    def accept(self):
        """Save configuration and accept dialog."""
        self.variability_config = self.variability_table.get_variability_values()
        super().accept()

    def get_variability_config(self):
        """Get the saved variability configuration."""
        return self.variability_config


class StandardVariabilityTable(QTableWidget):
    """Table for standard variability parameters (subject/inhalation) - 2 column layout."""

    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Variability Values"])
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)

    def populate_from_variability_toml(self, data: Dict[str, Any]):
        """Populate table from variability TOML data."""
        variability_params = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 5:
                variability_params[key] = value

        self.setRowCount(len(variability_params))

        for row, (param_name, param_values) in enumerate(variability_params.items()):
            self._populate_row(row, param_name, param_values)

    def populate_from_parameter_names(self, parameter_names: Iterable[str]) -> None:
        """Populate table with default variability rows for provided parameters."""
        names = [str(name) for name in parameter_names if name]
        if not names:
            self.setRowCount(0)
            return

        unique_names = sorted(dict.fromkeys(names))
        self.setRowCount(len(unique_names))
        for row, name in enumerate(unique_names):
            self._populate_row(row, name, None)

    def _populate_row(self, row: int, param_name: str, param_values: Optional[Iterable[Any]]) -> None:
        """Populate a single table row with controls for variability editing."""
        defaults: List[Any] = [0.0, "lognormal", 0.0, "lognormal", ""]
        if param_values is not None:
            values = list(defaults)
            for idx, value in enumerate(param_values):
                if idx < len(values) and value not in (None, ""):
                    values[idx] = value
        else:
            values = defaults

        param_item = QTableWidgetItem(param_name)
        param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 0, param_item)

        value_widget = QWidget()
        value_layout = QHBoxLayout()
        value_layout.setContentsMargins(5, 0, 5, 0)

        inter_sigma_edit = QLineEdit(str(values[0]))
        inter_sigma_edit.setMaximumWidth(60)
        value_layout.addWidget(QLabel("Inter σ:"))
        value_layout.addWidget(inter_sigma_edit)

        inter_dist_combo = QComboBox()
        inter_dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
        inter_dist_combo.setCurrentText(str(values[1]))
        inter_dist_combo.setMaximumWidth(100)
        value_layout.addWidget(inter_dist_combo)

        intra_sigma_edit = QLineEdit(str(values[2]))
        intra_sigma_edit.setMaximumWidth(60)
        value_layout.addWidget(QLabel("Intra σ:"))
        value_layout.addWidget(intra_sigma_edit)

        intra_dist_combo = QComboBox()
        intra_dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
        intra_dist_combo.setCurrentText(str(values[3]))
        intra_dist_combo.setMaximumWidth(100)
        value_layout.addWidget(intra_dist_combo)

        dep_field_edit = QLineEdit(str(values[4]))
        dep_field_edit.setMaximumWidth(120)
        value_layout.addWidget(QLabel("Dependent Field:"))
        value_layout.addWidget(dep_field_edit)

        value_layout.addStretch()

        value_widget.inter_sigma = inter_sigma_edit
        value_widget.inter_dist = inter_dist_combo
        value_widget.intra_sigma = intra_sigma_edit
        value_widget.intra_dist = intra_dist_combo
        value_widget.dep_field = dep_field_edit

        value_widget.setLayout(value_layout)
        self.setCellWidget(row, 1, value_widget)

    def get_variability_values(self) -> Dict[str, Any]:
        """Get all variability values from table."""
        values = {}

        for row in range(self.rowCount()):
            param_item = self.item(row, 0)
            if param_item:
                param_name = param_item.text()

                # Get widget containing all variability controls
                value_widget = self.cellWidget(row, 1)
                if value_widget and hasattr(value_widget, 'inter_sigma'):
                    # Extract values from the widget controls
                    try:
                        inter_sigma = float(value_widget.inter_sigma.text())
                    except ValueError:
                        inter_sigma = 0.0

                    inter_dist = value_widget.inter_dist.currentText()

                    try:
                        intra_sigma = float(value_widget.intra_sigma.text())
                    except ValueError:
                        intra_sigma = 0.0

                    intra_dist = value_widget.intra_dist.currentText()
                    dep_field = value_widget.dep_field.text()

                    values[param_name] = [inter_sigma, inter_dist, intra_sigma, intra_dist, dep_field]

        return values


class LungGeometryVariabilityTable(QTableWidget):
    """Specialized table for lung geometry variability - matches GenerationsTable layout."""

    def __init__(self):
        super().__init__()
        # Set up table with generations as rows and parameters as columns
        self.setRowCount(25)  # 25 generations

        # Column headers: Generation + 7 parameters (each with sigma and distribution)
        headers = ["Generation"]
        param_names = ["N_airways", "Length", "Diameter", "Area", "Volume", "FlowRate", "Other"]
        for param in param_names:
            headers.extend([f"{param}\nσ", f"{param}\nDist"])

        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        # Set generation column width
        self.setColumnWidth(0, 80)

        # Set parameter column widths
        for i in range(1, self.columnCount()):
            if i % 2 == 1:  # Sigma columns
                self.setColumnWidth(i, 60)
            else:  # Distribution columns
                self.setColumnWidth(i, 90)

        self.setAlternatingRowColors(True)

        # Initialize generation labels
        for i in range(25):
            gen_item = QTableWidgetItem(f"Gen {i-1}")
            gen_item.setFlags(gen_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(i, 0, gen_item)

    def populate_from_variability_toml(self, data: Dict[str, Any]):
        """Populate table from lung geometry variability TOML data."""
        if 'lung_geometry' not in data or not isinstance(data['lung_geometry'], list):
            return

        lung_matrix = data['lung_geometry']

        for gen_idx, generation_data in enumerate(lung_matrix[:25]):
            if isinstance(generation_data, list) and len(generation_data) >= 7:
                # For each parameter in this generation
                for param_idx, param_variability in enumerate(generation_data[:7]):
                    if isinstance(param_variability, list) and len(param_variability) >= 2:
                        # Column index: 1 + param_idx * 2 for sigma, +1 for distribution
                        sigma_col = 1 + param_idx * 2
                        dist_col = sigma_col + 1

                        # Set sigma value (inter_sigma)
                        sigma_item = QTableWidgetItem(str(param_variability[0]))
                        self.setItem(gen_idx, sigma_col, sigma_item)

                        # Set distribution dropdown (inter_dist)
                        dist_combo = QComboBox()
                        dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
                        dist_combo.setCurrentText(str(param_variability[1]))
                        self.setCellWidget(gen_idx, dist_col, dist_combo)

    def get_variability_values(self) -> Dict[str, Any]:
        """Get all variability values from table and reconstruct lung_geometry matrix."""
        lung_geometry_matrix = []

        for gen_idx in range(25):
            generation_data = []

            # For each parameter (7 total)
            for param_idx in range(7):
                sigma_col = 1 + param_idx * 2
                dist_col = sigma_col + 1

                # Get sigma value
                sigma_item = self.item(gen_idx, sigma_col)
                if sigma_item:
                    try:
                        inter_sigma = float(sigma_item.text())
                    except ValueError:
                        inter_sigma = 0.0
                else:
                    inter_sigma = 0.0

                # Get distribution from combo
                dist_combo = self.cellWidget(gen_idx, dist_col)
                if isinstance(dist_combo, QComboBox):
                    inter_dist = dist_combo.currentText()
                else:
                    inter_dist = "lognormal"

                # Build full variability structure (using defaults for intra values)
                param_variability = [inter_sigma, inter_dist, 0.0, "lognormal", ""]
                generation_data.append(param_variability)

            lung_geometry_matrix.append(generation_data)

        return {'lung_geometry': lung_geometry_matrix}


class GITractTable(QTableWidget):
    """Table widget for displaying GI Tract parameters."""

    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)

    def populate_from_gi_toml(self, data: Dict[str, Any], gi_type: str):
        """Populate table from GI Tract TOML data."""
        # Set up table structure with compartment headers
        compartment_names = [
            "Comp 1", "Comp 2", "Comp 3", "Comp 4", "Comp 5",
            "Comp 6", "Comp 7", "Comp 8", "Comp 9"
        ]

        # Parameters to display
        param_configs = [
            ("gi_area", "GI Area (cm²)"),
            ("gi_tg", "Transit Time (s)"),
            ("gi_vol", "Volume (mL)")
        ]

        self.setRowCount(len(param_configs))
        self.setColumnCount(len(compartment_names) + 1)  # +1 for parameter name column

        # Set headers
        headers = ["Parameter"] + compartment_names
        self.setHorizontalHeaderLabels(headers)

        # Set first column width
        self.setColumnWidth(0, 120)

        # Populate data
        for row, (param_key, param_label) in enumerate(param_configs):
            # Parameter name (read-only)
            param_item = QTableWidgetItem(param_label)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, param_item)

            # Get values for this parameter and GI type
            if param_key in data:
                param_data = data[param_key]
                if gi_type in param_data:
                    values = param_data[gi_type]
                    for col, value in enumerate(values[:9]):  # Max 9 compartments
                        value_item = QTableWidgetItem(str(value))
                        self.setItem(row, col + 1, value_item)

    def get_values(self) -> Dict[str, Any]:
        """Get all values from table."""
        values = {}

        param_keys = ["gi_area", "gi_tg", "gi_vol"]

        for row, param_key in enumerate(param_keys):
            param_values = []
            for col in range(1, self.columnCount()):
                item = self.item(row, col)
                if item:
                    try:
                        param_values.append(float(item.text()))
                    except ValueError:
                        param_values.append(0.0)
                else:
                    param_values.append(0.0)

            values[param_key] = param_values

        return values


    def get_parameter_keys(self) -> List[str]:
        """Return the parameter labels shown in the table."""
        keys: List[str] = []
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item:
                keys.append(item.text())
        return keys


class GIVariabilityConfigDialog(QDialog):
    """Dialog for configuring GI tract variability parameters."""

    def __init__(self, parent, category: str, item_name: str, variability_file: str):
        super().__init__(parent)
        self.parent_widget = parent
        self.category = category
        self.item_name = item_name
        self.variability_file = variability_file
        self.variability_config = None
        self.setWindowTitle(f"GI Tract Variability - {item_name}")
        self.setGeometry(200, 200, 1200, 600)
        self.setModal(True)
        self.init_ui()
        self.load_variability_data()

    def init_ui(self):
        """Initialize the variability dialog UI."""
        layout = QVBoxLayout()

        # Header
        header = QLabel(f"GI Tract Variability Configuration for {self.item_name}")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Info label
        info = QLabel("Configure GI tract variability.\nShowing inter_sigma and inter_dist for each compartment.")
        layout.addWidget(info)

        # Create GI variability table
        self.variability_table = GIVariabilityTable(self.item_name)
        layout.addWidget(self.variability_table)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_variability_data(self):
        """Load variability data from TOML file."""
        try:
            file_path = CATALOG_ROOT / self.category / self.variability_file
            if file_path.exists():
                try:
                    import tomllib
                    with open(file_path, 'rb') as f:
                        data = tomllib.load(f)
                except ImportError:
                    import tomli as tomllib
                    with open(file_path, 'rb') as f:
                        data = tomllib.load(f)

                self.variability_table.populate_from_variability_toml(data, self.item_name)
            else:
                QMessageBox.warning(self, "File Not Found", f"Variability file not found: {self.variability_file}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading variability data: {str(e)}")

    def accept(self):
        """Save configuration and accept dialog."""
        self.variability_config = self.variability_table.get_variability_values()
        super().accept()

    def get_variability_config(self):
        """Get the saved variability configuration."""
        return self.variability_config


class PKVariabilityDialog(QDialog):
    """Dialog for configuring per-API PK variability parameters."""

    def __init__(
        self,
        parent,
        available_apis: Iterable[str],
        variability_file: str = "Variability_default.toml",
        initial_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.parent_widget = parent
        self.available_apis = [name for name in available_apis if name]
        self.variability_file = variability_file
        self.initial_config = copy.deepcopy(initial_config) if initial_config else None
        self.variability_config: Optional[Dict[str, Any]] = None
        self.setWindowTitle("PK Variability Configuration")
        self.setGeometry(200, 200, 900, 600)
        self.setModal(True)

        self.init_ui()
        self.load_variability_data()

    def init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Configure inter/intra-subject PK variability by API")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        info = QLabel(
            "Each row captures [inter_sigma, inter_dist, intra_sigma, intra_dist] for an API-specific parameter."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.variability_table = PKVariabilityTable(self.available_apis)
        layout.addWidget(self.variability_table)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        buttons.addWidget(self.save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.setLayout(layout)

    def load_variability_data(self) -> None:
        data: Dict[str, Any] = {}
        file_path = CATALOG_ROOT / "pk" / self.variability_file
        if file_path.exists():
            try:
                try:
                    import tomllib  # type: ignore[attr-defined]
                except ImportError:  # pragma: no cover - Python < 3.11
                    import tomli as tomllib  # type: ignore
                with open(file_path, 'rb') as handle:
                    raw = tomllib.load(handle)
                if self.available_apis:
                    data = {
                        param: {
                            api: spec
                            for api, spec in payload.items()
                            if api in self.available_apis
                        }
                        for param, payload in raw.items()
                        if isinstance(payload, dict)
                    }
                else:
                    data = {}
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Variability",
                    f"Could not load builtin PK variability: {exc}"
                )

        self.variability_table.populate_from_data(data, self.initial_config)

        has_rows = self.variability_table.rowCount() > 0
        self.save_btn.setEnabled(has_rows)
        if not has_rows:
            QMessageBox.information(
                self,
                "PK Variability",
                "No PK variability entries available. Save APIs in the API & Products tab first to configure PK variability."
            )

    def accept(self) -> None:
        self.variability_config = self.variability_table.get_variability_values()
        super().accept()

    def get_variability_config(self) -> Dict[str, Any]:
        return self.variability_config or {}


class PKVariabilityTable(QTableWidget):
    """Table widget for editing PK variability per API."""

    PARAMETERS = [
        "clearance_L_h",
        "hepatic_extraction",
        "volume_central_L",
        "q_inter_L_h",
        "ka_h",
        "f_bioavail",
    ]

    def __init__(self, api_names: Iterable[str]):
        super().__init__()
        self._api_names = [name for name in api_names if name]
        self._row_mapping: List[Tuple[str, str]] = []
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels([
            "Parameter",
            "API",
            "Inter σ",
            "Inter Dist",
            "Intra σ",
            "Intra Dist",
            "Dependent"
        ])
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def populate_from_data(
        self,
        base_data: Optional[Dict[str, Any]],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        base_data = base_data or {}
        overrides = overrides or {}

        all_params: List[str] = list(dict.fromkeys(self.PARAMETERS + list(base_data.keys()) + list(overrides.keys())))
        all_params = [param for param in all_params if param]

        all_apis: List[str] = []
        for name in self._api_names:
            if name and name not in all_apis:
                all_apis.append(name)

        for source in (base_data, overrides):
            for payload in source.values():
                if isinstance(payload, dict):
                    for api_name in payload.keys():
                        if api_name and api_name not in all_apis:
                            all_apis.append(api_name)

        if not all_params or not all_apis:
            self.setRowCount(0)
            self._row_mapping = []
            return

        rows = []
        for param in all_params:
            payload_base = base_data.get(param, {}) if isinstance(base_data.get(param), dict) else {}
            payload_override = overrides.get(param, {}) if isinstance(overrides.get(param), dict) else {}
            for api_name in all_apis:
                if self._api_names and api_name not in self._api_names:
                    continue
                spec = payload_override.get(api_name)
                if spec is None:
                    spec = payload_base.get(api_name)
                rows.append((param, api_name, spec))

        self.setRowCount(len(rows))
        self._row_mapping = []

        for row_index, (param, api_name, spec) in enumerate(rows):
            self._row_mapping.append((param, api_name))

            param_item = QTableWidgetItem(param)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row_index, 0, param_item)

            api_item = QTableWidgetItem(api_name)
            api_item.setFlags(api_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row_index, 1, api_item)

            values = self._normalize_spec(spec)

            inter_sigma_edit = QLineEdit(str(values[0]))
            inter_sigma_edit.setMaximumWidth(80)
            self.setCellWidget(row_index, 2, inter_sigma_edit)

            inter_dist_combo = QComboBox()
            inter_dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
            inter_dist_combo.setCurrentText(str(values[1]))
            inter_dist_combo.setMaximumWidth(110)
            self.setCellWidget(row_index, 3, inter_dist_combo)

            intra_sigma_edit = QLineEdit(str(values[2]))
            intra_sigma_edit.setMaximumWidth(80)
            self.setCellWidget(row_index, 4, intra_sigma_edit)

            intra_dist_combo = QComboBox()
            intra_dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
            intra_dist_combo.setCurrentText(str(values[3]))
            intra_dist_combo.setMaximumWidth(110)
            self.setCellWidget(row_index, 5, intra_dist_combo)

            dep_field_edit = QLineEdit(str(values[4]))
            dep_field_edit.setMaximumWidth(140)
            self.setCellWidget(row_index, 6, dep_field_edit)

    @staticmethod
    def _normalize_spec(spec: Optional[Iterable[Any]]) -> List[Any]:
        defaults = [0.0, "lognormal", 0.0, "lognormal", ""]
        if spec is None:
            return defaults
        values = list(defaults)
        for idx, value in enumerate(list(spec)[:5]):
            values[idx] = value
        if not values[1]:
            values[1] = "lognormal"
        if not values[3]:
            values[3] = "lognormal"
        return values

    def get_variability_values(self) -> Dict[str, Dict[str, List[Any]]]:
        results: Dict[str, Dict[str, List[Any]]] = {}
        for row_index, (param, api_name) in enumerate(self._row_mapping):
            inter_sigma = self._to_float(self.cellWidget(row_index, 2))
            inter_dist = self._combo_value(self.cellWidget(row_index, 3))
            intra_sigma = self._to_float(self.cellWidget(row_index, 4))
            intra_dist = self._combo_value(self.cellWidget(row_index, 5))
            dep_field = self._line_value(self.cellWidget(row_index, 6))

            spec = [inter_sigma, inter_dist or "lognormal", intra_sigma, intra_dist or "lognormal", dep_field or api_name]
            results.setdefault(param, {})[api_name] = spec
        return results

    @staticmethod
    def _to_float(widget: Optional[QLineEdit]) -> float:
        if isinstance(widget, QLineEdit):
            try:
                return float(widget.text())
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _combo_value(widget: Optional[QComboBox]) -> str:
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return "lognormal"

    @staticmethod
    def _line_value(widget: Optional[QLineEdit]) -> str:
        if isinstance(widget, QLineEdit):
            return widget.text()
        return ""

class GIVariabilityTable(QTableWidget):
    """Specialized table for GI tract variability - matches GITractTable layout."""

    def __init__(self, gi_type: str):
        super().__init__()
        self.gi_type = gi_type
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)

        # Set up table structure
        compartment_names = [
            "Comp 1", "Comp 2", "Comp 3", "Comp 4", "Comp 5",
            "Comp 6", "Comp 7", "Comp 8", "Comp 9"
        ]

        # Parameters with variability (3 params × 2 values each)
        self.param_configs = [
            ("gi_area", "GI Area"),
            ("gi_tg", "Transit Time"),
            ("gi_vol", "Volume")
        ]

        self.setRowCount(len(self.param_configs))

        # Columns: Parameter + (9 compartments × 2 values)
        headers = ["Parameter"]
        for comp in compartment_names:
            headers.extend([f"{comp}\nσ", f"{comp}\nDist"])

        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        # Set column widths
        self.setColumnWidth(0, 100)
        for i in range(1, self.columnCount()):
            if i % 2 == 1:  # Sigma columns
                self.setColumnWidth(i, 50)
            else:  # Distribution columns
                self.setColumnWidth(i, 80)

        # Initialize parameter labels
        for row, (param_key, param_label) in enumerate(self.param_configs):
            param_item = QTableWidgetItem(param_label)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, param_item)

    def populate_from_variability_toml(self, data: Dict[str, Any], gi_type: str):
        """Populate table from GI variability TOML data."""
        for row, (param_key, param_label) in enumerate(self.param_configs):
            if param_key not in data:
                continue
            per_api = data[param_key]
            compartment_variabilities = None
            if isinstance(per_api, dict):
                if gi_type in per_api:
                    compartment_variabilities = per_api[gi_type]
                elif per_api:
                    # fallback to first available API entry
                    first_key = next(iter(per_api))
                    compartment_variabilities = per_api[first_key]
            elif isinstance(per_api, list):
                compartment_variabilities = per_api

            if not compartment_variabilities:
                continue

            for comp_idx, comp_variability in enumerate(compartment_variabilities[:9]):
                if isinstance(comp_variability, list) and len(comp_variability) >= 2:
                    sigma_col = 1 + comp_idx * 2
                    dist_col = sigma_col + 1

                    sigma_item = QTableWidgetItem(str(comp_variability[0]))
                    self.setItem(row, sigma_col, sigma_item)

                    dist_combo = QComboBox()
                    dist_combo.addItems(["lognormal", "normal", "uniform", "constant"])
                    dist_combo.setCurrentText(str(comp_variability[1]))
                    self.setCellWidget(row, dist_col, dist_combo)

    def get_variability_values(self) -> Dict[str, Any]:
        """Get all variability values from table."""
        values = {}

        for row, (param_key, param_label) in enumerate(self.param_configs):
            compartment_data = []

            for comp_idx in range(9):  # 9 compartments
                sigma_col = 1 + comp_idx * 2
                dist_col = sigma_col + 1

                # Get sigma value
                sigma_item = self.item(row, sigma_col)
                if sigma_item:
                    try:
                        inter_sigma = float(sigma_item.text())
                    except ValueError:
                        inter_sigma = 0.0
                else:
                    inter_sigma = 0.0

                # Get distribution from combo
                dist_combo = self.cellWidget(row, dist_col)
                if isinstance(dist_combo, QComboBox):
                    inter_dist = dist_combo.currentText()
                else:
                    inter_dist = "lognormal"

                # Build full variability structure
                comp_variability = [inter_sigma, inter_dist, 0.0, "lognormal", ""]
                compartment_data.append(comp_variability)

            values[param_key] = {self.gi_type: compartment_data}

        return values


class FlowProfileDialog(QDialog):
    """Dialog for viewing and editing tabulated flow profile."""

    def __init__(self, parent, flow_profile):
        super().__init__(parent)
        self.flow_profile = flow_profile
        self.setWindowTitle("Tabulated Flow Profile")
        self.setGeometry(300, 300, 600, 500)
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        """Initialize the flow profile dialog UI."""
        layout = QVBoxLayout()

        # Header
        header = QLabel("Tabulated Flow Profile")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Info
        info = QLabel(f"Total points: {len(self.flow_profile)}")
        layout.addWidget(info)

        # Create table
        self.table = QTableWidget()
        self.table.setRowCount(len(self.flow_profile))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Flow (L/min)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Populate table
        for row, (time, flow) in enumerate(self.flow_profile):
            time_item = QTableWidgetItem(str(time))
            flow_item = QTableWidgetItem(str(flow))
            self.table.setItem(row, 0, time_item)
            self.table.setItem(row, 1, flow_item)

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()

        add_btn = QPushButton("Add Point")
        add_btn.clicked.connect(self.add_point)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_point)
        button_layout.addWidget(remove_btn)

        button_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def add_point(self):
        """Add a new point to the flow profile."""
        current_row = self.table.currentRow()
        insert_row = current_row + 1 if current_row >= 0 else self.table.rowCount()

        self.table.insertRow(insert_row)
        self.table.setItem(insert_row, 0, QTableWidgetItem("0.0"))
        self.table.setItem(insert_row, 1, QTableWidgetItem("0.0"))
        self.table.selectRow(insert_row)

    def remove_point(self):
        """Remove the selected point from the flow profile."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def get_flow_profile(self):
        """Get the edited flow profile."""
        flow_profile = []
        for row in range(self.table.rowCount()):
            time_item = self.table.item(row, 0)
            flow_item = self.table.item(row, 1)
            if time_item and flow_item:
                try:
                    time = float(time_item.text())
                    flow = float(flow_item.text())
                    flow_profile.append([time, flow])
                except ValueError:
                    pass  # Skip invalid entries
        return flow_profile
