"""Categorized API Parameter Widget.

Organizes API parameters into logical categories with checkboxes and grouped sections,
similar to the reference image structure.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import toml

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QLineEdit, QComboBox, QGroupBox, QFormLayout, QScrollArea,
    QFrame, QGridLayout, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QRadioButton, QButtonGroup
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

CATALOG_ROOT = Path(__file__).parent.parent / "lmp_pkg" / "src" / "lmp_pkg" / "catalog" / "builtin"


class ParameterGroup(QWidget):
    """A group of related parameters with checkbox and proper table layout."""

    def __init__(self, title: str, parameters: Dict[str, Any], unit_options: List[str] = None):
        super().__init__()
        self.title = title
        self.parameters = parameters
        self.unit_options = unit_options or ["-"]
        self.enabled = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with checkbox
        header_layout = QHBoxLayout()
        self.checkbox = QCheckBox(self.title)
        self.checkbox.setChecked(self.enabled)
        self.checkbox.toggled.connect(self.toggle_enabled)

        font = QFont()
        font.setBold(True)
        self.checkbox.setFont(font)

        header_layout.addWidget(self.checkbox)
        header_layout.addStretch()

        # Unit dropdown
        if len(self.unit_options) > 1:
            self.unit_combo = QComboBox()
            self.unit_combo.addItems(self.unit_options)
            self.unit_combo.setFixedWidth(80)
            header_layout.addWidget(self.unit_combo)

        layout.addLayout(header_layout)

        # Parameters container
        self.params_widget = QWidget()
        self.setup_parameters_layout()
        layout.addWidget(self.params_widget)

        self.setLayout(layout)

    def calculate_table_height(self, row_count: int) -> int:
        """Calculate dynamic table height based on row count."""
        row_height = 30  # Approximate row height
        header_height = 25  # Header height
        margin = 10  # Top/bottom margin
        return header_height + (row_count * row_height) + margin

    def setup_parameters_layout(self):
        """Setup the parameters layout based on data structure."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 0, 0, 0)

        # Analyze parameter structure and create appropriate layout
        structure_type = self.analyze_structure()

        if structure_type == "regional_compartment":
            self.setup_regional_compartment_table(layout)
        elif structure_type == "regional_in_out":
            self.setup_regional_in_out_table(layout)
        elif structure_type == "regional_simple":
            self.setup_regional_simple_table(layout)
        elif structure_type == "compartment_simple":
            self.setup_compartment_table(layout)
        elif structure_type == "simple_in_out":
            self.setup_simple_in_out_table(layout)
        elif structure_type == "fraction_unbound_special":
            self.setup_fraction_unbound_table(layout)
        else:
            self.setup_simple_table(layout)

        self.params_widget.setLayout(layout)

    def analyze_structure(self):
        """Analyze the structure of parameters to determine layout type."""
        # Special case for fraction_unbound which has ELF, Epithelium, Tissue, Plasma
        if "fraction_unbound" in self.parameters:
            return "fraction_unbound_special"

        # Check for different parameter structures
        has_regions = self.has_regions()
        has_compartments = self.has_compartments()
        has_in_out = self.has_in_out_structure()

        if has_regions and has_compartments:
            return "regional_compartment"  # pscale_Kin.ET.Epithelium
        elif has_regions and has_in_out:
            return "regional_in_out"       # pscale.ET.In/Out
        elif has_regions:
            return "regional_simple"       # pscale_para.ET
        elif has_compartments:
            return "compartment_simple"    # k_in.Epithelium
        elif has_in_out:
            return "simple_in_out"         # peff.In/Out
        else:
            return "simple"                # molecular_weight

    def has_regions(self) -> bool:
        """Check if any parameter has regional structure."""
        for key, value in self.parameters.items():
            if isinstance(value, dict):
                if any(region in value for region in ["ET", "BB", "bb", "Al"]):
                    return True
                # Check nested
                for subkey, subvalue in value.items():
                    if subkey in ["ET", "BB", "bb", "Al"]:
                        return True
                    if isinstance(subvalue, dict) and any(region in subvalue for region in ["ET", "BB", "bb", "Al"]):
                        return True
        return False

    def has_compartments(self) -> bool:
        """Check if any parameter has compartment structure."""
        for key, value in self.parameters.items():
            if isinstance(value, dict):
                if any(comp in value for comp in ["Epithelium", "Tissue"]):
                    return True
                # Check nested
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and any(comp in subvalue for comp in ["Epithelium", "Tissue"]):
                        return True
        return False

    def has_in_out_structure(self) -> bool:
        """Check if any parameter has In/Out structure."""
        for key, value in self.parameters.items():
            if isinstance(value, dict):
                if "In" in value and "Out" in value:
                    return True
                # Check nested
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "In" in subvalue and "Out" in subvalue:
                        return True
        return False

    def flatten_parameters(self) -> Dict[str, Any]:
        """Flatten nested parameter dictionaries."""
        flat = {}
        for key, value in self.parameters.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}.{subkey}"] = subvalue
                    flat[subkey] = subvalue  # Also add direct key for easier lookup
                    if isinstance(subvalue, dict):
                        for subsubkey, subsubvalue in subvalue.items():
                            flat[f"{key}.{subkey}.{subsubkey}"] = subsubvalue
                            flat[f"{subkey}.{subsubkey}"] = subsubvalue
            else:
                flat[key] = value
        return flat

    def setup_regional_compartment_table(self, layout):
        """Setup table for regional compartment parameters (pscale_Kin.ET.Epithelium)."""
        self.add_radio_buttons(layout)

        # Add compartment label
        label = QLabel("compartment")
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)

        # Create regional compartment table
        self.table = QTableWidget()
        self.table.setColumnCount(3)  # Region, Epithelium, Tissue
        self.table.setHorizontalHeaderLabels(["Region", "Epithelium", "Tissue"])

        regions = ["ET", "BB", "bb", "Al"]
        self.table.setRowCount(len(regions))

        for row, region in enumerate(regions):
            # Region name (read-only)
            region_item = QTableWidgetItem(region)
            region_item.setFlags(region_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, region_item)

            # Extract compartment values for this region
            epi_value = self.extract_value([region, "Epithelium"])
            tissue_value = self.extract_value([region, "Tissue"])

            self.table.setItem(row, 1, QTableWidgetItem(str(epi_value)))
            self.table.setItem(row, 2, QTableWidgetItem(str(tissue_value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setAlternatingRowColors(True)  # Better visual separation
        layout.addWidget(self.table)

    def setup_regional_in_out_table(self, layout):
        """Setup table for regional In/Out parameters (pscale.ET.In/Out)."""
        self.add_radio_buttons(layout)

        # Add transport label
        label = QLabel("transport")
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)

        # Create regional In/Out table
        self.table = QTableWidget()
        self.table.setColumnCount(3)  # Region, In, Out
        self.table.setHorizontalHeaderLabels(["Region", "In", "Out"])

        regions = ["ET", "BB", "bb", "Al"]
        self.table.setRowCount(len(regions))

        for row, region in enumerate(regions):
            # Region name (read-only)
            region_item = QTableWidgetItem(region)
            region_item.setFlags(region_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, region_item)

            # Extract In/Out values for this region
            in_value = self.extract_value([region, "In"])
            out_value = self.extract_value([region, "Out"])

            self.table.setItem(row, 1, QTableWidgetItem(str(in_value)))
            self.table.setItem(row, 2, QTableWidgetItem(str(out_value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setAlternatingRowColors(True)  # Better visual separation
        layout.addWidget(self.table)

    def setup_regional_simple_table(self, layout):
        """Setup table for simple regional parameters (pscale_para.ET)."""
        self.add_radio_buttons(layout)

        # Create simple regional table
        self.table = QTableWidget()
        self.table.setColumnCount(2)  # Region, Value
        self.table.setHorizontalHeaderLabels(["Region", "Value"])

        regions = ["ET", "BB", "bb", "Al"]
        self.table.setRowCount(len(regions))

        for row, region in enumerate(regions):
            # Region name (read-only)
            region_item = QTableWidgetItem(region)
            region_item.setFlags(region_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, region_item)

            # Extract value for this region
            value = self.extract_value([region])
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setAlternatingRowColors(True)  # Better visual separation
        layout.addWidget(self.table)

    def setup_compartment_table(self, layout):
        """Setup table for compartment parameters (k_in/k_out) - just Epithelium/Tissue columns, single row."""
        self.add_radio_buttons(layout)

        # Add compartment label
        label = QLabel("compartment")
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)

        # Create simple compartment table (just 2 columns, 1 row)
        self.table = QTableWidget()
        self.table.setColumnCount(2)  # Epithelium, Tissue
        self.table.setHorizontalHeaderLabels(["Epithelium", "Tissue"])
        self.table.setRowCount(1)  # Just one row

        # Get the compartment values
        base_values = {}
        for param_key, param_value in self.parameters.items():
            if isinstance(param_value, dict):
                base_values = param_value
                break

        # Single row with Epithelium/Tissue values
        epi_value = base_values.get("Epithelium", "0.25")
        tissue_value = base_values.get("Tissue", "0.25")

        self.table.setItem(0, 0, QTableWidgetItem(str(epi_value)))
        self.table.setItem(0, 1, QTableWidgetItem(str(tissue_value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def setup_simple_in_out_table(self, layout):
        """Setup table for simple In/Out parameters (peff) - just In/Out columns, single row."""
        self.add_radio_buttons(layout)

        # Add transport label
        label = QLabel("transport")
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)

        # Create simple In/Out table (just 2 columns, 1 row)
        self.table = QTableWidget()
        self.table.setColumnCount(2)  # In, Out
        self.table.setHorizontalHeaderLabels(["In", "Out"])
        self.table.setRowCount(1)  # Just one row

        # Get the In/Out values
        base_values = {}
        for param_key, param_value in self.parameters.items():
            if isinstance(param_value, dict):
                base_values = param_value
                break

        # Single row with In/Out values
        in_value = base_values.get("In", "2.3e-8")
        out_value = base_values.get("Out", "2.3e-8")

        self.table.setItem(0, 0, QTableWidgetItem(str(in_value)))
        self.table.setItem(0, 1, QTableWidgetItem(str(out_value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def add_radio_buttons(self, layout):
        """Add uniform vs per element radio buttons."""
        radio_layout = QHBoxLayout()
        self.uniform_radio = QRadioButton("Uniform for entire array")
        self.per_element_radio = QRadioButton("Per element")
        self.per_element_radio.setChecked(True)

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.uniform_radio)
        self.radio_group.addButton(self.per_element_radio)

        radio_layout.addWidget(self.uniform_radio)
        radio_layout.addWidget(self.per_element_radio)
        radio_layout.addStretch()
        layout.addLayout(radio_layout)


    def setup_fraction_unbound_table(self, layout):
        """Setup table layout for fraction_unbound with ELF, Epithelium, Tissue, Plasma."""
        self.add_radio_buttons(layout)

        # Add compartment label
        label = QLabel("compartment")
        label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(label)

        # Create table for fraction_unbound
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Compartment", "Value"])

        compartments = ["ELF", "Epithelium", "Tissue", "Plasma"]
        self.table.setRowCount(len(compartments))

        if "fraction_unbound" in self.parameters:
            fraction_data = self.parameters["fraction_unbound"]
            if isinstance(fraction_data, dict):
                for row, compartment in enumerate(compartments):
                    # Compartment name (read-only)
                    comp_item = QTableWidgetItem(compartment)
                    comp_item.setFlags(comp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, 0, comp_item)

                    # Compartment value
                    value = fraction_data.get(compartment, "")
                    self.table.setItem(row, 1, QTableWidgetItem(str(value)))

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def setup_simple_table(self, layout):
        """Setup table layout for simple parameters."""
        # Add radio buttons for consistency
        self.add_radio_buttons(layout)

        # Create table with Parameter and Value columns
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Get all parameters and create rows
        param_list = list(self.parameters.items())
        self.table.setRowCount(len(param_list))

        for row, (param_key, param_value) in enumerate(param_list):
            # Parameter name (read-only)
            param_item = QTableWidgetItem(param_key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, param_item)

            # Parameter value (editable)
            value_item = QTableWidgetItem(str(param_value))
            self.table.setItem(row, 1, value_item)

        self.table.setFixedWidth(400)
        self.table.setFixedHeight(self.calculate_table_height(self.table.rowCount()))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def setup_simple_parameters(self, layout):
        """Setup simple parameter list."""
        form_layout = QFormLayout()
        self.widgets = {}

        for key, value in self.parameters.items():
            widget = self.create_simple_widget(value)
            self.widgets[key] = widget
            form_layout.addRow(f"{key}:", widget)

        layout.addLayout(form_layout)

    def create_simple_widget(self, value):
        """Create appropriate widget for simple value."""
        if isinstance(value, (int, float)):
            widget = QDoubleSpinBox()
            widget.setRange(-999999, 999999)
            widget.setDecimals(6)
            widget.setValue(float(value))
        elif isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
        elif isinstance(value, list):
            widget = QLineEdit()
            widget.setText(", ".join(map(str, value)))
        else:
            widget = QLineEdit()
            widget.setText(str(value))
        return widget

    def extract_value(self, path: List[str]):
        """Extract value following a path through nested parameters."""
        # Try to find the value by following the path
        for param_key, param_value in self.parameters.items():
            result = self._follow_path(param_value, path)
            if result is not None:
                return result

            # Also try including the param_key as part of the path
            full_path = [param_key] + path
            result = self._follow_path(self.parameters, full_path)
            if result is not None:
                return result

        # If not found, return default based on path
        return self._get_default_value(path)

    def _follow_path(self, data, path):
        """Follow a path through nested dictionaries."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _get_default_value(self, path):
        """Get default value based on path components."""
        path_str = ".".join(path)

        # Regional defaults
        if "ET" in path:
            if "In" in path or "Out" in path:
                return "100"
            else:
                return "1.0"
        elif "BB" in path:
            if "In" in path or "Out" in path:
                return "15"
            else:
                return "1.0"
        elif "bb" in path:
            if "In" in path or "Out" in path:
                return "15"
            else:
                return "1.0"
        elif "Al" in path:
            return "1"

        # Compartment defaults
        elif "Epithelium" in path:
            return "0.8"
        elif "Tissue" in path:
            return "0.8"

        # Direction defaults
        elif "In" in path:
            return "2.3e-8"
        elif "Out" in path:
            return "2.3e-8"

        return "0"

    def toggle_enabled(self, enabled: bool):
        """Toggle the enabled state of the parameter group."""
        self.enabled = enabled
        self.params_widget.setEnabled(enabled)

    def get_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        if not self.enabled:
            return {}

        values = {}
        structure_type = self.analyze_structure()

        if hasattr(self, 'table') and self.table.rowCount() > 0:
            headers = [self.table.horizontalHeaderItem(i).text()
                      for i in range(self.table.columnCount())]

            if structure_type == "simple":
                # Extract simple parameter-value pairs from table
                for row in range(self.table.rowCount()):
                    param_name = self.table.item(row, 0).text()
                    param_value = self.convert_value(self.table.item(row, 1).text())
                    values[param_name] = param_value

            elif structure_type == "regional_compartment":
                # Rebuild nested regional compartment structure
                for param_key in self.parameters:
                    if isinstance(self.parameters[param_key], dict):
                        values[param_key] = {}
                        for row in range(self.table.rowCount()):
                            region = self.table.item(row, 0).text()
                            epi_value = self.convert_value(self.table.item(row, 1).text())
                            tissue_value = self.convert_value(self.table.item(row, 2).text())
                            values[param_key][region] = {"Epithelium": epi_value, "Tissue": tissue_value}

            elif structure_type == "regional_in_out":
                # Rebuild nested regional In/Out structure
                for param_key in self.parameters:
                    if isinstance(self.parameters[param_key], dict):
                        values[param_key] = {}
                        for row in range(self.table.rowCount()):
                            region = self.table.item(row, 0).text()
                            in_value = self.convert_value(self.table.item(row, 1).text())
                            out_value = self.convert_value(self.table.item(row, 2).text())
                            values[param_key][region] = {"In": in_value, "Out": out_value}

            elif structure_type == "regional_simple":
                # Rebuild simple regional structure
                for param_key in self.parameters:
                    if isinstance(self.parameters[param_key], dict):
                        values[param_key] = {}
                        for row in range(self.table.rowCount()):
                            region = self.table.item(row, 0).text()
                            value = self.convert_value(self.table.item(row, 1).text())
                            values[param_key][region] = value

            elif structure_type == "compartment_simple":
                # Extract compartment table values - simple 2-column format
                for param_key in self.parameters:
                    # For compartment_simple, we store the base values from single row
                    values[param_key] = {}
                    if self.table.rowCount() > 0:
                        # Single row with Epithelium (col 0) and Tissue (col 1)
                        epi_value = self.convert_value(self.table.item(0, 0).text())
                        tissue_value = self.convert_value(self.table.item(0, 1).text())
                        values[param_key]["Epithelium"] = epi_value
                        values[param_key]["Tissue"] = tissue_value

            elif structure_type == "simple_in_out":
                # Extract In/Out table values - simple 2-column format
                for param_key in self.parameters:
                    if isinstance(self.parameters[param_key], dict):
                        values[param_key] = {}
                        if self.table.rowCount() > 0:
                            # Single row with In (col 0) and Out (col 1)
                            in_value = self.convert_value(self.table.item(0, 0).text())
                            out_value = self.convert_value(self.table.item(0, 1).text())
                            values[param_key]["In"] = in_value
                            values[param_key]["Out"] = out_value

            elif structure_type == "fraction_unbound_special":
                # Extract fraction_unbound table values
                values["fraction_unbound"] = {}
                for row in range(self.table.rowCount()):
                    comp_name = self.table.item(row, 0).text()
                    comp_value = self.convert_value(self.table.item(row, 1).text())
                    values["fraction_unbound"][comp_name] = comp_value

        elif hasattr(self, 'widgets'):
            # Extract values from form widgets (should not be used anymore)

            # Simple parameters (fallback for any remaining widget-based forms)
            for key, widget in self.widgets.items():
                if isinstance(widget, QDoubleSpinBox):
                    values[key] = widget.value()
                elif isinstance(widget, QCheckBox):
                    values[key] = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if "," in text:
                        values[key] = [self.convert_value(v.strip()) for v in text.split(",")]
                    else:
                        values[key] = self.convert_value(text)

        return values

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


class CategorizedAPIWidget(QWidget):
    """Categorized API parameter widget matching the reference interface."""

    config_updated = Signal(str, dict)
    api_configured = Signal(str)  # Signal when an API is configured

    def __init__(self):
        super().__init__()
        self.current_api = None
        self.parameter_groups = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("API Parameters")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # API selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select API:"))

        self.api_combo = QComboBox()
        self.api_combo.currentTextChanged.connect(self.load_api_parameters)
        selection_layout.addWidget(self.api_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_apis)
        selection_layout.addWidget(refresh_btn)

        save_btn = QPushButton("Save to Config")
        save_btn.clicked.connect(self.save_to_config)
        selection_layout.addWidget(save_btn)

        selection_layout.addStretch()
        layout.addLayout(selection_layout)

        # Scrollable parameters area with two columns
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.params_widget = QWidget()

        # Create two-column layout
        self.params_layout = QHBoxLayout()

        # Left column
        self.left_column = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Top align content
        self.left_column.setLayout(self.left_layout)

        # Right column
        self.right_column = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Top align content
        self.right_column.setLayout(self.right_layout)

        # Add columns to main layout with top alignment
        self.params_layout.addWidget(self.left_column, alignment=Qt.AlignmentFlag.AlignTop)
        self.params_layout.addWidget(self.right_column, alignment=Qt.AlignmentFlag.AlignTop)

        self.params_widget.setLayout(self.params_layout)
        scroll.setWidget(self.params_widget)
        layout.addWidget(scroll)

        self.setLayout(layout)

        # Load initial data
        self.refresh_apis()

    def refresh_apis(self):
        """Refresh the API dropdown."""
        try:
            # Try direct file access first, then app_api
            entries = self.get_direct_catalog_entries()
            if not entries or entries == ["BD", "FF", "GP"]:
                if CATALOG_AVAILABLE:
                    try:
                        entries = app_api.list_catalog_entries("api")
                    except:
                        pass  # Keep direct entries

            self.api_combo.clear()
            self.api_combo.addItems(entries)

            if entries and not entries[0].startswith("Error"):
                self.load_api_parameters()

        except Exception as e:
            print(f"Error loading APIs: {e}")
            self.api_combo.clear()
            self.api_combo.addItem("Error loading APIs")

    def get_direct_catalog_entries(self) -> List[str]:
        """Get API entries directly from filesystem."""
        api_dir = CATALOG_ROOT / "api"
        if not api_dir.exists():
            return ["BD", "FF", "GP"]  # Fallback

        entries = []
        for file_path in api_dir.glob("*.toml"):
            if not file_path.name.startswith("Variability_"):
                entries.append(file_path.stem)

        if not entries:
            return ["BD", "FF", "GP"]  # Fallback if no files found

        return sorted(entries)

    def load_api_parameters(self):
        """Load and categorize parameters for the selected API."""
        api_name = self.api_combo.currentText()
        if not api_name or api_name.startswith("Error"):
            return

        try:
            # Load API data - try direct first, then app_api
            data = None
            try:
                data = self.load_direct_api_data(api_name)
            except:
                if CATALOG_AVAILABLE:
                    try:
                        data = app_api.get_catalog_entry("api", api_name)
                    except:
                        pass

            if data:
                self.current_api = api_name
                self.create_parameter_groups(data)
            else:
                print(f"Could not load data for API: {api_name}")

        except Exception as e:
            print(f"Error loading API parameters: {e}")

    def load_direct_api_data(self, api_name: str) -> Dict[str, Any]:
        """Load API data directly from TOML file."""
        file_path = CATALOG_ROOT / "api" / f"{api_name}.toml"
        if not file_path.exists():
            raise FileNotFoundError(f"API file not found: {file_path}")

        try:
            try:
                import tomllib
                with open(file_path, 'rb') as f:
                    return tomllib.load(f)
            except ImportError:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return toml.load(f)
        except Exception as e:
            raise Exception(f"Error parsing TOML file {file_path}: {e}")

    def create_parameter_groups(self, data: Dict[str, Any]):
        """Create categorized parameter groups from API data."""
        # Clear existing groups
        for group in self.parameter_groups:
            group.deleteLater()
        self.parameter_groups.clear()

        # Clear both column layouts
        for i in reversed(range(self.left_layout.count())):
            item = self.left_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        for i in reversed(range(self.right_layout.count())):
            item = self.right_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        # Define column assignments
        left_column_categories = {
            "Basic Properties",
            "Physical Properties",
            "Fraction Unbound",
            "Blood Plasma ratio",
            "Pharmacokinetics"
        }

        right_column_categories = {
            "Trans-cellular Transport Peff",
            "Permeability Scaling Factor",
            "Paracellular Transport Peff",
            "Para-cellular Transport scaling factor",
            "K_in",
            "pscale_Kin",
            "K_out",
            "pscale_Kout",
            "Cell Binding (0 = No, 1 = Yes)",
            "GI Absorption"
        }

        # Define parameter categories with units based on the reference image
        categories = [
            ("Basic Properties", ["name", "description", "molecular_weight"], ["-"]),
            ("Trans-cellular Transport Peff", ["peff"], ["cm/s", "m/s", "mm/s"]),
            ("Passive Trans-Cell Transport (0 = No, 1 = Yes)", ["passive_transport"], ["-"]),
            ("Permeability Scaling Factor", ["pscale"], ["-"]),
            ("Paracellular Transport Peff", ["peff_para"], ["cm/s", "m/s", "mm/s"]),
            ("Para-cellular Transport scaling factor", ["pscale_para"], ["-"]),
            ("Fraction surface availability for para-cellular transport", ["surface_availability"], ["-"]),
            ("Cell Binding (0 = No, 1 = Yes)", ["cell_binding"], ["-"]),
            ("K_in", ["k_in"], ["h⁻¹", "1/h", "min⁻¹"]),
            ("pscale_Kin", ["pscale_Kin"], ["-"]),
            ("K_out", ["k_out"], ["h⁻¹", "1/h", "min⁻¹"]),
            ("pscale_Kout", ["pscale_Kout"], ["-"]),
            ("Fraction Unbound", ["fraction_unbound"], ["-"]),
            ("Blood Plasma ratio", ["blood_plasma_ratio"], ["-"]),
            ("Pharmacokinetics", ["n_pk_compartments", "volume_central_L", "clearance_L_h", "k12_h", "k21_h", "k13_h", "k31_h"], ["L/h", "L", "1/h"]),
            ("GI Absorption", ["hepatic_extraction_pct", "peff_GI"], ["cm/s", "%"]),
            ("Physical Properties", ["diffusion_coeff", "density_g_m3", "solubility_pg_ml"], ["cm²/s", "g/m³", "pg/ml"])
        ]

        # Special handling for peff parameter which contains both peff and peff_para
        if "peff" in data:
            peff_data = data["peff"]

            # Create Trans-cellular Transport Peff with In/Out only
            if isinstance(peff_data, dict) and "In" in peff_data and "Out" in peff_data:
                trans_peff = {"In": peff_data["In"], "Out": peff_data["Out"]}
                group = ParameterGroup("Trans-cellular Transport Peff", {"peff": trans_peff}, ["cm/s", "m/s", "mm/s"])
                self.parameter_groups.append(group)

                # Create Paracellular Transport Peff if peff_para exists
                if "peff_para" in peff_data:
                    para_peff = {"peff_para": peff_data["peff_para"]}
                    group = ParameterGroup("Paracellular Transport Peff", para_peff, ["cm/s", "m/s", "mm/s"])
                    self.parameter_groups.append(group)

        # Group parameters by category - each parameter gets its own category for proper display
        for category_name, param_keys, unit_options in categories:
            # Skip peff since we handled it specially above
            if "peff" in param_keys:
                param_keys = [k for k in param_keys if k != "peff"]
                if not param_keys:
                    continue

            for param_key in param_keys:
                if param_key in data:
                    # Create individual category for each parameter
                    single_param = {param_key: data[param_key]}

                    # Use specific category name if only one parameter
                    if len(param_keys) == 1:
                        display_name = category_name
                    else:
                        # For multi-parameter categories like Pharmacokinetics, keep them together
                        display_name = category_name
                        # But for the first occurrence, add all parameters at once
                        if param_key == param_keys[0]:
                            multi_params = {k: data[k] for k in param_keys if k in data}
                            if multi_params:
                                group = ParameterGroup(display_name, multi_params, unit_options)
                                self.parameter_groups.append(group)
                        continue  # Skip individual processing for multi-param categories

                    group = ParameterGroup(display_name, single_param, unit_options)
                    self.parameter_groups.append(group)

        # Add remaining uncategorized parameters
        categorized_keys = set()
        for _, param_keys, _ in categories:
            categorized_keys.update(param_keys)

        remaining_params = {k: v for k, v in data.items() if k not in categorized_keys}
        if remaining_params:
            group = ParameterGroup("Other Parameters", remaining_params, ["-"])
            self.parameter_groups.append(group)

        # Assign parameter groups to appropriate columns
        for group in self.parameter_groups:
            if group.title in left_column_categories:
                self.left_layout.addWidget(group)
            else:
                self.right_layout.addWidget(group)

        # Add stretch to both columns to push content to top
        self.left_layout.addStretch()
        self.right_layout.addStretch()

    def save_to_config(self):
        """Save current API configuration using Pydantic EntityRef."""
        if not self.current_api:
            return

        # Import Pydantic model
        sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
        from lmp_pkg.config.model import EntityRef

        # Collect all parameter values from groups as overrides
        overrides = {}
        for group in self.parameter_groups:
            group_values = group.get_values()
            overrides.update(group_values)

        # Create EntityRef with reference and parameter overrides
        entity_ref = EntityRef(
            ref=self.current_api,
            overrides=overrides
        )

        # Convert to dict using model_dump() for JSON serialization
        config_data = entity_ref.model_dump()

        # Add name for backward compatibility with GUI
        config_data["name"] = self.current_api

        self.config_updated.emit("api", config_data)
        self.api_configured.emit(self.current_api)  # Emit API name for product tab

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Saved", f"API '{self.current_api}' saved to config using Pydantic model.")


# Fix the missing import
from PySide6.QtWidgets import QPushButton