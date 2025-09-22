"""Product Grid Widget for LMP GUI.

Creates a configurable grid of N products × M APIs with dropdown selections
and automatic population of default values from product templates.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import toml

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QComboBox,
    QSpinBox, QHeaderView, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

# Path setup for catalog access
CATALOG_ROOT = Path(__file__).parent.parent / "lmp_pkg" / "src" / "lmp_pkg" / "catalog" / "builtin"

class ProductGridWidget(QWidget):
    """Product grid widget for configuring N products with M APIs each."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.num_products = 1
        self.num_apis = 1
        self.configured_apis = []  # List of API names from API tab
        self.product_templates = {}  # Cache for product templates
        self.grid_table = None
        self.all_non_api_fields = set()  # All non-API fields across templates
        self.all_api_fields = set()      # All API fields across templates
        self.row_field_mapping = {}      # Maps row index to field info
        self.init_ui()
        self.load_product_templates()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("Products Configuration")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(header)

        # Configuration controls
        config_layout = QHBoxLayout()

        # Number of products input
        config_layout.addWidget(QLabel("Number of Products:"))
        self.products_spin = QSpinBox()
        self.products_spin.setMinimum(1)
        self.products_spin.setMaximum(10)
        self.products_spin.setValue(self.num_products)
        config_layout.addWidget(self.products_spin)

        # Number of APIs input
        config_layout.addWidget(QLabel("Number of APIs:"))
        self.apis_spin = QSpinBox()
        self.apis_spin.setMinimum(1)
        self.apis_spin.setMaximum(10)
        self.apis_spin.setValue(self.num_apis)
        config_layout.addWidget(self.apis_spin)

        # Generate grid button
        generate_btn = QPushButton("Add Product Data")
        generate_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        generate_btn.clicked.connect(self.generate_grid)
        config_layout.addWidget(generate_btn)

        config_layout.addStretch()
        layout.addLayout(config_layout)

        # Scrollable grid area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.grid_widget = QWidget()
        self.grid_layout = QVBoxLayout()
        self.grid_widget.setLayout(self.grid_layout)
        scroll.setWidget(self.grid_widget)

        layout.addWidget(scroll)

        # Save button
        save_layout = QHBoxLayout()
        save_btn = QPushButton("Save Product Configuration")
        save_btn.clicked.connect(self.save_configuration)
        save_layout.addWidget(save_btn)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        self.setLayout(layout)

    def update_template_combos_visibility(self):
        """Update visibility of template combo boxes based on number of products."""
        for i, combo in enumerate(self.template_combos):
            combo.setVisible(i < self.num_products)
            # Also update the associated label
            label_idx = i * 2 + 1  # Labels are at odd indices in the template layout
            if hasattr(self, 'template_layout'):
                label_widget = self.template_layout.itemAt(label_idx)
                if label_widget:
                    label_widget.widget().setVisible(i < self.num_products)

    def load_product_templates(self):
        """Load product templates from catalog."""
        product_dir = CATALOG_ROOT / "product"
        if not product_dir.exists():
            print(f"Product directory not found: {product_dir}")
            return

        self.product_templates = {}
        for file_path in product_dir.glob("*.toml"):
            try:
                try:
                    import tomllib
                    with open(file_path, 'rb') as f:
                        template_data = tomllib.load(f)
                except ImportError:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = toml.load(f)

                # Only include templates that have APIs
                if 'apis' in template_data and template_data['apis']:
                    template_name = template_data.get('name', file_path.stem)
                    self.product_templates[template_name] = template_data
                    print(f"Loaded product template: {template_name}")
                else:
                    print(f"Skipped template {file_path.stem}: no APIs found")

            except Exception as e:
                print(f"Error loading product template {file_path}: {e}")


        # Generate initial basic grid
        self.generate_grid()


    def set_configured_apis(self, api_list: List[str]):
        """Set the list of configured APIs from the API tab."""
        self.configured_apis = api_list
        if self.grid_table:
            self.update_api_dropdowns()

    def update_api_dropdowns(self):
        """Update API dropdowns in existing grid."""
        if not self.grid_table:
            return

        # Find API rows and update dropdowns
        for row in range(self.grid_table.rowCount()):
            item = self.grid_table.item(row, 0)
            if item and item.text().startswith("API"):
                for col in range(1, self.grid_table.columnCount()):
                    widget = self.grid_table.cellWidget(row, col)
                    if isinstance(widget, QComboBox):
                        current_value = widget.currentText()
                        widget.clear()
                        widget.addItems(["Select API..."] + self.configured_apis)
                        # Restore previous selection if still available
                        if current_value in self.configured_apis:
                            widget.setCurrentText(current_value)

    def generate_grid(self):
        """Generate basic product grid with template selection."""
        if hasattr(self, 'products_spin'):
            self.num_products = self.products_spin.value()
            self.num_apis = self.apis_spin.value()

        # Clear existing grid
        if self.grid_table:
            self.grid_table.deleteLater()

        # Reset row mapping
        self.row_field_mapping = {}

        # Create new grid table
        self.grid_table = QTableWidget()

        # Start with minimal rows: Product name + Template selection + placeholder for dynamic fields
        initial_rows = 3  # Name, Template, (dynamic fields will be added when template selected)
        # Calculate columns: Parameter names + (N products × M APIs)
        total_cols = 1 + (self.num_products * self.num_apis)

        self.grid_table.setRowCount(initial_rows)
        self.grid_table.setColumnCount(total_cols)

        # Set up headers - each product has M API columns
        headers = ["Product Data"]
        for product_idx in range(self.num_products):
            for api_idx in range(self.num_apis):
                headers.append(f"Product {product_idx+1} | API {api_idx+1}")
        self.grid_table.setHorizontalHeaderLabels(headers)

        row = 0

        # Product Name row
        self.grid_table.setItem(row, 0, QTableWidgetItem("Product Name"))
        self.row_field_mapping[row] = {'type': 'product', 'field': 'name'}
        col = 1
        for product_idx in range(self.num_products):
            # First API column gets the product name input
            name_input = QLineEdit(f"Product_{product_idx + 1}")
            self.grid_table.setCellWidget(row, col, name_input)
            col += 1
            # Remaining API columns for this product get empty cells
            for api_idx in range(1, self.num_apis):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.grid_table.setItem(row, col, item)
                col += 1
        row += 1

        # Template selection row
        self.grid_table.setItem(row, 0, QTableWidgetItem("Template"))
        self.row_field_mapping[row] = {'type': 'template', 'field': 'template'}
        col = 1
        for product_idx in range(self.num_products):
            # First API column gets the template selection
            template_combo = QComboBox()
            template_combo.addItems(["Select Template..."] + list(self.product_templates.keys()))
            template_combo.setProperty("product_idx", product_idx)
            template_combo.currentTextChanged.connect(self.on_template_selection_changed)
            self.grid_table.setCellWidget(row, col, template_combo)
            col += 1
            # Remaining API columns for this product get empty cells
            for api_idx in range(1, self.num_apis):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.grid_table.setItem(row, col, item)
                col += 1
        row += 1

        # Placeholder row for dynamic fields (will be populated when template selected)
        self.grid_table.setItem(row, 0, QTableWidgetItem("Select a template to see fields..."))
        col = 1
        for product_idx in range(self.num_products):
            for api_idx in range(self.num_apis):
                self.grid_table.setItem(row, col, QTableWidgetItem(""))
                col += 1

        # Configure table appearance
        self.grid_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, total_cols):
            self.grid_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        self.grid_table.verticalHeader().setVisible(False)
        self.grid_table.setAlternatingRowColors(True)

        # Make parameter names column read-only
        for row in range(self.grid_table.rowCount()):
            item = self.grid_table.item(row, 0)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self.grid_layout.addWidget(self.grid_table)

    def on_template_selection_changed(self, template_name: str):
        """Handle template selection change - rebuild grid for that template's fields."""
        if template_name == "Select Template...":
            return

        sender = self.sender()
        if not sender:
            return

        product_idx = sender.property("product_idx")
        if product_idx is None:
            return

        # Get template data
        template_data = self.product_templates.get(template_name)
        if not template_data:
            return

        # Rebuild the grid to accommodate this template's fields
        self.rebuild_grid_for_templates()

    def rebuild_grid_for_templates(self):
        """Rebuild grid based on all selected templates."""
        if not self.grid_table:
            return

        # Collect all field names from all selected templates
        all_non_api_fields = set()
        all_api_fields = set()
        selected_templates = {}

        # Check what templates are currently selected
        # Templates are in the first column of each product group
        col = 1
        for product_idx in range(self.num_products):
            template_widget = self.grid_table.cellWidget(1, col)  # Template row is row 1
            if isinstance(template_widget, QComboBox):
                template_name = template_widget.currentText()
                if template_name != "Select Template..." and template_name in self.product_templates:
                    selected_templates[product_idx] = template_name
                    template_data = self.product_templates[template_name]

                    # Collect fields from this template
                    for field_name, field_value in template_data.items():
                        if field_name not in ['name', 'apis'] and field_value is not None:
                            all_non_api_fields.add(field_name)

                    if 'apis' in template_data:
                        for api_data in template_data['apis']:
                            for field_name, field_value in api_data.items():
                                if field_name != 'name' and field_value is not None:
                                    all_api_fields.add(field_name)

            # Move to the first column of the next product
            col += self.num_apis

        # Calculate new row count
        new_row_count = 2  # Name + Template
        if all_non_api_fields:
            new_row_count += len(all_non_api_fields)
        if all_api_fields:
            new_row_count += 1  # API selection row
            new_row_count += len(all_api_fields)

        # Store current values before rebuilding
        current_values = {}
        for row in range(min(2, self.grid_table.rowCount())):
            for col in range(1, self.grid_table.columnCount()):
                widget = self.grid_table.cellWidget(row, col)
                if widget:
                    if isinstance(widget, QLineEdit):
                        current_values[(row, col)] = widget.text()
                    elif isinstance(widget, QComboBox):
                        current_values[(row, col)] = widget.currentText()

        # Resize grid
        self.grid_table.setRowCount(new_row_count)

        # Rebuild rows starting from row 2 (after Name and Template)
        row = 2

        # Add non-API fields
        for field_name in sorted(all_non_api_fields):
            self.grid_table.setItem(row, 0, QTableWidgetItem(field_name))
            self.row_field_mapping[row] = {'type': 'non_api', 'field': field_name}

            col = 1
            for product_idx in range(self.num_products):
                template_name = selected_templates.get(product_idx)

                # First API column gets the field input
                field_input = QLineEdit("")

                # Auto-populate if this template has this field
                if template_name and template_name in self.product_templates:
                    template_data = self.product_templates[template_name]
                    if field_name in template_data:
                        field_input.setText(str(template_data[field_name]))

                self.grid_table.setCellWidget(row, col, field_input)
                col += 1

                # Remaining API columns for this product get empty cells
                for api_idx in range(1, self.num_apis):
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.grid_table.setItem(row, col, item)
                    col += 1
            row += 1

        # Add API selection row if we have API fields
        if all_api_fields:
            self.grid_table.setItem(row, 0, QTableWidgetItem("API"))
            self.row_field_mapping[row] = {'type': 'api', 'field': 'name'}

            col = 1
            for product_idx in range(self.num_products):
                for api_idx in range(self.num_apis):
                    api_combo = QComboBox()
                    api_combo.addItems(["Select API..."] + self.configured_apis)
                    self.grid_table.setCellWidget(row, col, api_combo)
                    col += 1
            row += 1

            # Add API fields
            for field_name in sorted(all_api_fields):
                self.grid_table.setItem(row, 0, QTableWidgetItem(field_name))
                self.row_field_mapping[row] = {'type': 'api', 'field': field_name}

                col = 1
                for product_idx in range(self.num_products):
                    template_name = selected_templates.get(product_idx)

                    for api_idx in range(self.num_apis):
                        field_input = QLineEdit("0.0")

                        # Auto-populate if this template has this API field
                        if template_name and template_name in self.product_templates:
                            template_data = self.product_templates[template_name]
                            if 'apis' in template_data and template_data['apis']:
                                # Get default value from first API in template
                                first_api = template_data['apis'][0]
                                if field_name in first_api:
                                    field_input.setText(str(first_api[field_name]))

                        self.grid_table.setCellWidget(row, col, field_input)
                        col += 1
                row += 1

        # Restore values for name and template rows
        for (old_row, old_col), value in current_values.items():
            if old_row < 2:  # Only restore Name and Template rows
                widget = self.grid_table.cellWidget(old_row, old_col)
                if widget:
                    if isinstance(widget, QLineEdit):
                        widget.setText(value)
                    elif isinstance(widget, QComboBox) and value in [widget.itemText(i) for i in range(widget.count())]:
                        widget.setCurrentText(value)

        # Make parameter names column read-only
        for row in range(self.grid_table.rowCount()):
            item = self.grid_table.item(row, 0)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def get_template_field(self, template_data: Dict[str, Any], field_names: List[str], default_value: Any = None) -> Any:
        """Get a field value from template data trying multiple possible field names."""
        for field_name in field_names:
            if field_name in template_data and template_data[field_name] is not None:
                return template_data[field_name]
        return default_value

    def get_api_field_with_conversion(self, api_data: Dict[str, Any], field_type: str) -> Optional[float]:
        """Get API field value with automatic unit conversion based on field type."""
        if field_type == 'dose':
            # Handle dose fields with different units
            field_mappings = {
                'dose_pg': lambda x: x / 1e6,  # Convert pg to μg
                'dose_ug': lambda x: x,        # Already in μg
                'dose_mg': lambda x: x * 1000, # Convert mg to μg
                'dose_ng': lambda x: x / 1000, # Convert ng to μg
                'dose': lambda x: x,           # Assume μg if no unit specified
                'label_claim_mg': lambda x: x * 1000,  # Convert mg to μg
                'label_claim_ug': lambda x: x,         # Already in μg
            }
        elif field_type == 'usp_deposition':
            # Handle USP deposition fraction fields
            field_mappings = {
                'usp_depo_fraction': lambda x: x,      # Assume percentage
                'usp_deposition_fraction': lambda x: x,
                'deposition_fraction': lambda x: x,
                'fine_particle_fraction': lambda x: x * 100,  # Convert fraction to percentage
                'fpf': lambda x: x * 100,                      # Convert fraction to percentage
                'emitted_dose_percent': lambda x: x,
            }
        elif field_type == 'mmad':
            # Handle MMAD fields with different units
            field_mappings = {
                'mmad': lambda x: x,                           # Assume μm
                'mmad_um': lambda x: x,                        # Already in μm
                'mmad_nm': lambda x: x / 1000,                 # Convert nm to μm
                'mass_median_diameter_um': lambda x: x,        # Already in μm
                'mass_median_diameter': lambda x: x,           # Assume μm
                'particle_size_um': lambda x: x,
            }
        elif field_type == 'gsd':
            # Handle GSD fields
            field_mappings = {
                'gsd': lambda x: x,
                'geometric_standard_deviation': lambda x: x,
                'particle_gsd': lambda x: x,
                'size_distribution_gsd': lambda x: x,
            }
        else:
            return None

        # Try each field mapping
        for field_name, converter in field_mappings.items():
            if field_name in api_data and api_data[field_name] is not None:
                try:
                    return float(converter(api_data[field_name]))
                except (ValueError, TypeError):
                    continue

        return None

    def on_template_combo_changed(self, template_name: str):
        """Handle template combo box selection change."""
        sender = self.sender()
        if sender:
            product_idx = sender.property("product_idx")
            self.on_template_selected(template_name, product_idx)

    def on_template_selected(self, template_name: str, product_idx: int):
        """Handle product template selection and populate default values."""
        if template_name == "Select Template..." or template_name not in self.product_templates:
            return

        if not self.grid_table or product_idx >= self.num_products:
            return

        template_data = self.product_templates[template_name]

        # Calculate the starting column for this product (each product has M API columns)
        start_col = 1 + (product_idx * self.num_apis)

        # Set product name (in first API column of this product)
        name_widget = self.grid_table.cellWidget(0, start_col)
        if isinstance(name_widget, QLineEdit):
            name_widget.setText(template_data.get('name', f"Product_{product_idx + 1}"))

        # Set propellant using flexible field mapping
        propellant_widget = self.grid_table.cellWidget(1, start_col)
        if isinstance(propellant_widget, QComboBox):
            propellant = self.get_template_field(template_data, ['propellant', 'propellant_type', 'carrier_gas'], 'PT210')
            if propellant in ["PT210", "PT010", "HFA"]:
                propellant_widget.setCurrentText(propellant)

        # Set device using flexible field mapping
        device_widget = self.grid_table.cellWidget(2, start_col)
        if isinstance(device_widget, QComboBox):
            device = self.get_template_field(template_data, ['device', 'device_type', 'inhaler_type'], 'DFP')
            # Map various device values to UI values
            device_mapping = {
                'SMI': 'DPI',
                'soft_mist_inhaler': 'DPI',
                'MDI': 'MDI',
                'metered_dose_inhaler': 'MDI',
                'DPI': 'DPI',
                'dry_powder_inhaler': 'DPI',
                'DFP': 'DFP'
            }
            device = device_mapping.get(device.lower() if device else '', device)
            if device in ["DFP", "MDI", "DPI"]:
                device_widget.setCurrentText(device)

        # Clear all API data for this product first
        for api_idx in range(self.num_apis):
            api_col = start_col + api_idx

            # Clear API dropdown (row 3)
            api_widget = self.grid_table.cellWidget(3, api_col)
            if isinstance(api_widget, QComboBox):
                api_widget.setCurrentText("Select API...")

            # Clear dose (row 4)
            dose_widget = self.grid_table.cellWidget(4, api_col)
            if isinstance(dose_widget, QLineEdit):
                dose_widget.setText("0.0")

            # Clear USP deposition fraction (row 5)
            usp_widget = self.grid_table.cellWidget(5, api_col)
            if isinstance(usp_widget, QLineEdit):
                usp_widget.setText("0.0")

            # Clear MMAD (row 6)
            mmad_widget = self.grid_table.cellWidget(6, api_col)
            if isinstance(mmad_widget, QLineEdit):
                mmad_widget.setText("0.0")

            # Clear GSD (row 7)
            gsd_widget = self.grid_table.cellWidget(7, api_col)
            if isinstance(gsd_widget, QLineEdit):
                gsd_widget.setText("0.0")

        # Populate API data if available
        if 'apis' in template_data:
            template_apis = template_data['apis']

            for api_idx, api_data in enumerate(template_apis):
                if api_idx >= self.num_apis:
                    break

                # Calculate column for this specific API within this product
                api_col = start_col + api_idx

                # Set API name (row 3)
                api_widget = self.grid_table.cellWidget(3, api_col)
                if isinstance(api_widget, QComboBox) and api_data.get('name') in self.configured_apis:
                    api_widget.setCurrentText(api_data.get('name', ''))

                # Set dose (row 4) with flexible field mapping and unit conversion
                dose_widget = self.grid_table.cellWidget(4, api_col)
                if isinstance(dose_widget, QLineEdit):
                    dose = self.get_api_field_with_conversion(api_data, 'dose')
                    if dose is not None:
                        dose_widget.setText(f"{dose:.2f}")

                # Set USP deposition fraction (row 5) with flexible field mapping
                usp_widget = self.grid_table.cellWidget(5, api_col)
                if isinstance(usp_widget, QLineEdit):
                    usp = self.get_api_field_with_conversion(api_data, 'usp_deposition')
                    if usp is not None:
                        usp_widget.setText(f"{usp:.2f}")

                # Set MMAD (row 6) with flexible field mapping
                mmad_widget = self.grid_table.cellWidget(6, api_col)
                if isinstance(mmad_widget, QLineEdit):
                    mmad = self.get_api_field_with_conversion(api_data, 'mmad')
                    if mmad is not None:
                        mmad_widget.setText(f"{mmad:.2f}")

                # Set GSD (row 7) with flexible field mapping
                gsd_widget = self.grid_table.cellWidget(7, api_col)
                if isinstance(gsd_widget, QLineEdit):
                    gsd = self.get_api_field_with_conversion(api_data, 'gsd')
                    if gsd is not None:
                        gsd_widget.setText(f"{gsd:.2f}")


    def save_configuration(self):
        """Save the current product configuration using Pydantic EntityRef."""
        if not self.grid_table:
            return

        # Import Pydantic model
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
        from lmp_pkg.config.model import EntityRef

        products = []

        for col in range(1, self.num_products + 1):
            # Get product name
            name_widget = self.grid_table.cellWidget(0, col)
            product_name = name_widget.text() if isinstance(name_widget, QLineEdit) else f"Product_{col}"

            # Get template
            template_widget = self.grid_table.cellWidget(1, col)
            template_name = template_widget.currentText() if isinstance(template_widget, QComboBox) else ""

            # Collect product-level overrides
            product_overrides = {}
            if template_name and template_name != "Select Template...":
                product_overrides["template"] = template_name

            # Get APIs for this product
            apis = []
            for api_idx in range(self.num_apis):
                api_row = 2 + (api_idx * 5)

                # Get API data
                api_widget = self.grid_table.cellWidget(api_row, col)
                api_name = api_widget.currentText() if isinstance(api_widget, QComboBox) else ""

                if api_name and api_name != "Select API...":
                    dose_widget = self.grid_table.cellWidget(api_row + 1, col)
                    usp_widget = self.grid_table.cellWidget(api_row + 2, col)
                    mmad_widget = self.grid_table.cellWidget(api_row + 3, col)
                    gsd_widget = self.grid_table.cellWidget(api_row + 4, col)

                    # Create API overrides for this product
                    api_overrides = {}
                    if dose_widget.text():
                        api_overrides['dose_ug'] = float(dose_widget.text())
                    if usp_widget.text():
                        api_overrides['usp_depo_fraction'] = float(usp_widget.text())
                    if mmad_widget.text():
                        api_overrides['mmad'] = float(mmad_widget.text())
                    if gsd_widget.text():
                        api_overrides['gsd'] = float(gsd_widget.text())

                    # Create EntityRef for API
                    api_entity = EntityRef(
                        ref=api_name,
                        overrides=api_overrides
                    )
                    apis.append(api_entity.model_dump())

            # Add APIs to product overrides
            if apis:
                product_overrides["apis"] = apis

            # Create EntityRef for product
            product_entity = EntityRef(
                ref=product_name,
                overrides=product_overrides
            )

            products.append(product_entity.model_dump())

        # Create main configuration using Pydantic structure
        config_data = {
            'num_products': self.num_products,
            'num_apis_per_product': self.num_apis,
            'products': products
        }

        self.config_updated.emit("products", config_data)

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Saved", f"Product configuration with {len(products)} products saved using Pydantic models.")

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration for external use."""
        if not self.grid_table:
            return {'num_products': self.num_products, 'num_apis_per_product': self.num_apis, 'products': []}

        products = []

        # New layout: APIs are in columns within each product
        for product_idx in range(self.num_products):
            # Calculate column range for this product
            start_col = 1 + (product_idx * self.num_apis)

            # Get product name (from first API column of this product, row 0)
            name_widget = self.grid_table.cellWidget(0, start_col)
            product_name = name_widget.text() if isinstance(name_widget, QLineEdit) else f"Product_{product_idx + 1}"

            # Get propellant (from first API column of this product, row 1)
            propellant_widget = self.grid_table.cellWidget(1, start_col)
            propellant = propellant_widget.currentText() if isinstance(propellant_widget, QComboBox) else ""

            # Get device (from first API column of this product, row 2)
            device_widget = self.grid_table.cellWidget(2, start_col)
            device = device_widget.currentText() if isinstance(device_widget, QComboBox) else ""

            # Get APIs for this product (each API in its own column)
            apis = []
            for api_idx in range(self.num_apis):
                api_col = start_col + api_idx

                # Get API name (row 3)
                api_widget = self.grid_table.cellWidget(3, api_col)
                api_name = api_widget.currentText() if isinstance(api_widget, QComboBox) else ""

                if api_name and api_name != "Select API...":
                    try:
                        # Get API-specific data (rows 4-7)
                        dose_widget = self.grid_table.cellWidget(4, api_col)
                        usp_widget = self.grid_table.cellWidget(5, api_col)
                        mmad_widget = self.grid_table.cellWidget(6, api_col)
                        gsd_widget = self.grid_table.cellWidget(7, api_col)

                        api_data = {
                            'name': api_name,
                            'dose_ug': float(dose_widget.text()) if isinstance(dose_widget, QLineEdit) and dose_widget.text() else 0.0,
                            'usp_depo_fraction': float(usp_widget.text()) if isinstance(usp_widget, QLineEdit) and usp_widget.text() else 0.0,
                            'mmad': float(mmad_widget.text()) if isinstance(mmad_widget, QLineEdit) and mmad_widget.text() else 0.0,
                            'gsd': float(gsd_widget.text()) if isinstance(gsd_widget, QLineEdit) and gsd_widget.text() else 0.0
                        }
                        apis.append(api_data)
                    except (ValueError, AttributeError):
                        continue  # Skip invalid entries

            if apis:  # Only add products that have at least one API configured
                product_data = {
                    'name': product_name,
                    'propellant': propellant if propellant else "",
                    'device': device if device else "",
                    'apis': apis
                }
                products.append(product_data)

        return {
            'num_products': self.num_products,
            'num_apis_per_product': self.num_apis,
            'products': products
        }