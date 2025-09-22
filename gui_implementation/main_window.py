"""Main GUI Window for LMP Application.

Implements the tabbed interface design from the GUI plan with:
1. Home/Workspace
2. Catalog & Libraries
3. Study Designer
4. Run Queue
5. Results Viewer
6. Logs & Diagnostics
"""

import sys
import json
import shutil
import logging
import copy
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterable, Mapping, Tuple
import pandas as pd
import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FigureCanvas = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit, QSplitter,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QScrollArea, QHeaderView, QAbstractItemView, QGridLayout,
    QDoubleSpinBox
)
from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from PySide6.QtGui import QFont, QAction, QTextCursor

from workspace_manager import WorkspaceManager
from process_manager import ProcessManager

# Import app_api for catalog integration
sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
try:
    from lmp_pkg import app_api
    from lmp_pkg.config import AppConfig, check_catalog_coverage
    test_entries = app_api.list_catalog_entries("subject")
    CATALOG_AVAILABLE = True
    CONFIG_MODEL_AVAILABLE = True
except Exception:
    CATALOG_AVAILABLE = False
    CONFIG_MODEL_AVAILABLE = False
    app_api = None
    AppConfig = None
    check_catalog_coverage = None

try:
    from lmp_pkg.catalog.builtin_loader import BuiltinDataLoader
except Exception:
    BuiltinDataLoader = None


STAGE_DISPLAY_NAMES = {
    "cfd": "CFD",
    "deposition": "Deposition",
    "pbbm": "PBPK",
    "pbpk": "PBPK",
    "pk": "PK",
    "vbe": "VBE",
    "analysis": "Analysis",
    "analysis_bioequivalence": "Bioequivalence",
    "overall": "Overall",
}


logger = logging.getLogger(__name__)


def _sanitise_product_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name.strip())
    safe = safe.strip("-")
    return safe or name.replace(" ", "_")


def _flatten_entries(data: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], Any]]:
    """Recursively flatten a nested mapping into path tuples."""
    items: List[Tuple[Tuple[str, ...], Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            items.extend(_flatten_entries(value, prefix + (str(key),)))
    else:
        items.append((prefix, data))
    return items


def _format_path(path: Tuple[str, ...]) -> str:
    return ".".join(path)


def _parse_path(path: str) -> Tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


def _format_cell_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_cell_value(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _unflatten_entries(flat: Dict[str, Any]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for path, value in flat.items():
        if not path:
            continue
        parts = _parse_path(path)
        if not parts:
            continue
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def _apply_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return copy.deepcopy(base)

    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            base_child = result.get(key)
            if isinstance(base_child, dict):
                result[key] = _apply_overrides(base_child, value)
            else:
                result[key] = copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _compute_overrides(base: Dict[str, Any], edited: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, edited_value in edited.items():
        base_value = base.get(key, None)
        if isinstance(edited_value, dict) and isinstance(base_value, dict):
            nested = _compute_overrides(base_value, edited_value)
            if nested:
                overrides[key] = nested
        else:
            if key not in base or base_value != edited_value:
                overrides[key] = edited_value

    # Handle keys removed in edited data by setting to None
    for key in base.keys() - edited.keys():
        overrides[key] = None

    return overrides


class KeyValueTable(QTableWidget):
    """Editable table for flattened configuration key/value pairs."""

    def __init__(self, key_header: str = "Parameter", value_header: str = "Value"):
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels([key_header, value_header])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def set_data(self, data: Dict[str, Any]) -> None:
        items = _flatten_entries(data)
        self.setRowCount(len(items))

        for row, (path, value) in enumerate(items):
            key_item = QTableWidgetItem(_format_path(path))
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, key_item)

            value_item = QTableWidgetItem(_format_cell_value(value))
            self.setItem(row, 1, value_item)

    def get_data(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for row in range(self.rowCount()):
            key_item = self.item(row, 0)
            value_item = self.item(row, 1)
            if key_item is None:
                continue
            key = key_item.text().strip()
            if not key:
                continue
            value_text = value_item.text() if value_item is not None else ""
            flat[key] = _parse_cell_value(value_text)
        return _unflatten_entries(flat)



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

        # Recent projects (placeholder)
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
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.setLayout(layout)

    def browse_workspace(self):
        """Browse for existing workspace directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Workspace Directory"
        )
        if directory:
            self.set_workspace(directory)

    def create_workspace(self):
        """Create new workspace directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Create Workspace In Directory"
        )
        if directory:
            workspace_path = Path(directory) / "lmp_workspace"
            workspace_path.mkdir(exist_ok=True)
            self.set_workspace(str(workspace_path))

    def set_workspace(self, path: str):
        """Set the current workspace."""
        try:
            self.workspace_manager = WorkspaceManager(path)
            self.workspace_path_edit.setText(path)
            self.workspace_changed.emit(path)

            # Update recent list (placeholder)
            self.recent_list.clear()
            self.recent_list.addItem(f"Current: {path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not set workspace: {str(e)}")


class APIProductsTab(QWidget):
    """Workspace-aware catalog editor for APIs and products."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self._builtin_loader = BuiltinDataLoader() if BuiltinDataLoader is not None else None
        self._editors: Dict[str, Dict[str, Any]] = {}
        self._saved_entries: Dict[str, List[Dict[str, Any]]] = {"api": [], "product": []}
        self._catalog_names: Dict[str, List[str]] = {"api": [], "product": []}
        self._state: Dict[str, Dict[str, Any]] = {
            "api": {
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "loading": False,
                "overrides": {},
                "variability_overrides": {},
            },
            "product": {
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "loading": False,
                "overrides": {},
                "variability_overrides": {},
            },
        }
        self.init_ui()
        self.refresh_catalog_options("api")
        self.refresh_catalog_options("product")

    # ------------------------------------------------------------------
    # UI setup helpers

    def init_ui(self) -> None:
        layout = QVBoxLayout()

        header = QLabel("API & Products")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        columns = QHBoxLayout()
        self._editors["api"] = self._build_editor("api", "APIs")
        self._editors["product"] = self._build_editor("product", "Products")
        columns.addWidget(self._editors["api"]["group"])
        columns.addWidget(self._editors["product"]["group"])
        layout.addLayout(columns)
        layout.addStretch()

        self.setLayout(layout)

    def _build_editor(self, category: str, title: str) -> Dict[str, Any]:
        group = QGroupBox(title)
        group_layout = QVBoxLayout()

        # Saved configurations
        saved_layout = QHBoxLayout()
        saved_combo = QComboBox()
        saved_combo.addItem("Saved configurations…", None)
        saved_combo.currentIndexChanged.connect(
            lambda _idx, cat=category: self.on_saved_entry_changed(cat)
        )
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(lambda _=False, cat=category: self.delete_entry(cat))
        saved_layout.addWidget(saved_combo)
        saved_layout.addWidget(delete_btn)
        group_layout.addLayout(saved_layout)

        # Editable metadata
        form_layout = QFormLayout()
        name_edit = QLineEdit()
        form_layout.addRow("Name:", name_edit)

        base_row = QHBoxLayout()
        base_combo = QComboBox()
        base_combo.currentIndexChanged.connect(
            lambda _=False, cat=category: self.on_base_changed(cat)
        )
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda _=False, cat=category: self.refresh_catalog_options(cat, preserve_selection=True))
        base_row.addWidget(base_combo)
        base_row.addWidget(refresh_btn)
        form_layout.addRow("Catalog ref:", base_row)

        help_label = QLabel("Edit parameter values to capture overrides. Use JSON for lists or nested values.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        form_layout.addRow("", help_label)

        group_layout.addLayout(form_layout)

        tab_widget = QTabWidget()
        parameters_table = KeyValueTable("Parameter", "Value")
        tab_widget.addTab(parameters_table, "Parameters")
        variability_table = KeyValueTable("Parameter", "Value")
        tab_widget.addTab(variability_table, "Variability")
        group_layout.addWidget(tab_widget)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda _=False, cat=category: self.save_entry(cat, update_existing=True))
        save_as_btn = QPushButton("Save As")
        save_as_btn.clicked.connect(lambda _=False, cat=category: self.save_entry(cat, update_existing=False))
        revert_btn = QPushButton("Revert to catalog")
        revert_btn.clicked.connect(lambda _=False, cat=category: self.revert_to_catalog(cat))
        button_row.addWidget(save_btn)
        button_row.addWidget(save_as_btn)
        button_row.addWidget(revert_btn)
        button_row.addStretch()
        group_layout.addLayout(button_row)

        group.setLayout(group_layout)

        return {
            "group": group,
            "saved_combo": saved_combo,
            "delete_btn": delete_btn,
            "name_edit": name_edit,
            "base_combo": base_combo,
            "parameters_table": parameters_table,
            "variability_table": variability_table,
            "tab_widget": tab_widget,
            "save_btn": save_btn,
            "save_as_btn": save_as_btn,
            "revert_btn": revert_btn,
        }

    # ------------------------------------------------------------------
    # Workspace integration

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]) -> None:
        self.workspace_manager = workspace_manager
        for category in ("api", "product"):
            self.refresh_saved_entries(category)
            self._update_button_states(category)

    # ------------------------------------------------------------------
    # Catalog loading helpers

    def refresh_catalog_options(self, category: str, preserve_selection: bool = False) -> None:
        combo = self._editors[category]["base_combo"]
        previous = combo.currentData()

        names = self._fetch_catalog_references(category)
        self._catalog_names[category] = names

        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name, name)
        combo.blockSignals(False)

        if preserve_selection and previous in names:
            combo.setCurrentIndex(combo.findData(previous))
        elif names:
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(-1)

        self.on_base_changed(category)

    def _fetch_catalog_references(self, category: str) -> List[str]:
        names: List[str] = []
        if CATALOG_AVAILABLE and app_api is not None:
            try:
                names = app_api.list_catalog_entries(category)
            except Exception:
                names = []

        if not names:
            names = self._fallback_catalog_names(category)

        return sorted(dict.fromkeys(names))

    def _fallback_catalog_names(self, category: str) -> List[str]:
        if not self._builtin_loader:
            return []

        try:
            catalog_root = self._builtin_loader.catalog_root
            category_dir = catalog_root / category
            if not category_dir.exists():
                return []
            names = []
            for path in category_dir.glob("*.toml"):
                if path.stem.startswith("Variability_"):
                    continue
                names.append(path.stem)
            return names
        except Exception:
            return []

    def _fetch_catalog_entity(self, category: str, ref: str) -> Dict[str, Any]:
        if not ref:
            return {}
        if CATALOG_AVAILABLE and app_api is not None:
            try:
                return app_api.get_catalog_entry(category, ref)
            except Exception:
                pass
        if self._builtin_loader:
            try:
                if category == "api":
                    return self._builtin_loader.load_api_parameters(ref)
                if category == "product":
                    return self._builtin_loader.load_product_parameters(ref)
            except Exception:
                pass
        raise ValueError(f"Could not load {category} '{ref}' from catalog")

    def _fetch_variability_data(self, category: str, ref: str) -> Optional[Dict[str, Any]]:
        if not self._builtin_loader:
            return None
        filename = f"Variability_{ref}"
        try:
            return self._builtin_loader.load_variability_file(category, filename)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # State transitions

    def on_base_changed(self, category: str) -> None:
        state = self._state[category]
        editor = self._editors[category]

        if state["loading"]:
            return

        base_combo = editor["base_combo"]
        ref = base_combo.currentData()
        if not ref:
            state.update({
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "overrides": {},
                "variability_overrides": {},
            })
            editor["parameters_table"].setRowCount(0)
            editor["variability_table"].setRowCount(0)
            self._set_variability_tab_enabled(category, False)
            self._update_button_states(category)
            return

        try:
            base_data = self._fetch_catalog_entity(category, ref)
        except Exception as exc:
            QMessageBox.warning(self, "Catalog", f"Could not load {category} '{ref}': {exc}")
            return

        variability_data = self._fetch_variability_data(category, ref) or {}

        state.update({
            "base_ref": ref,
            "base_data": base_data,
            "variability_base": variability_data,
            "current_id": None,
            "overrides": {},
            "variability_overrides": {},
        })

        self.populate_tables(category, overrides=None, variability_overrides=None)
        editor["name_edit"].setPlaceholderText(ref)

        saved_combo = editor["saved_combo"]
        saved_combo.blockSignals(True)
        saved_combo.setCurrentIndex(0)
        saved_combo.blockSignals(False)
        self._update_button_states(category)

    def populate_tables(
        self,
        category: str,
        overrides: Optional[Dict[str, Any]],
        variability_overrides: Optional[Dict[str, Any]],
    ) -> None:
        state = self._state[category]
        editor = self._editors[category]

        resolved = _apply_overrides(state["base_data"], overrides)
        editor["parameters_table"].set_data(resolved)
        state["overrides"] = copy.deepcopy(overrides) if overrides else {}

        variability_base = state.get("variability_base") or {}
        if variability_base:
            resolved_var = _apply_overrides(variability_base, variability_overrides)
            editor["variability_table"].set_data(resolved_var)
            state["variability_overrides"] = copy.deepcopy(variability_overrides) if variability_overrides else {}
            self._set_variability_tab_enabled(category, True)
        else:
            editor["variability_table"].setRowCount(0)
            state["variability_overrides"] = {}
            self._set_variability_tab_enabled(category, False)

    def refresh_saved_entries(self, category: str, select_id: Optional[str] = None) -> None:
        editor = self._editors[category]
        combo = editor["saved_combo"]

        entries: List[Dict[str, Any]] = []
        if self.workspace_manager is not None:
            try:
                entries = self.workspace_manager.list_catalog_entries(category)
            except Exception as exc:
                logger.warning("list catalog entries failed", category=category, error=str(exc))
        self._saved_entries[category] = entries

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Saved configurations…", None)

        selected_index = 0
        for idx, entry in enumerate(entries, start=1):
            entry_id = entry.get("id")
            display_name = entry.get("name") or entry.get("ref") or entry_id
            combo.addItem(display_name, entry_id)
            if select_id and entry_id == select_id:
                selected_index = idx

        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

        if select_id:
            self.on_saved_entry_changed(category)
        else:
            self._update_button_states(category)

    def on_saved_entry_changed(self, category: str) -> None:
        editor = self._editors[category]
        combo = editor["saved_combo"]
        entry_id = combo.currentData()
        if not entry_id:
            self._state[category]["current_id"] = None
            self._update_button_states(category)
            return

        entry = self._get_saved_entry(category, entry_id)
        if not entry:
            QMessageBox.warning(self, "Catalog", f"Saved {category} '{entry_id}' not found")
            self.refresh_saved_entries(category)
            return

        state = self._state[category]
        state["loading"] = True
        try:
            base_ref = entry.get("ref")
            if base_ref:
                base_combo = editor["base_combo"]
                if base_combo.findData(base_ref) == -1:
                    base_combo.addItem(base_ref, base_ref)
                base_combo.setCurrentIndex(base_combo.findData(base_ref))
                base_data = self._fetch_catalog_entity(category, base_ref)
                variability_data = self._fetch_variability_data(category, base_ref) or {}
            else:
                base_data = {}
                variability_data = {}

            state.update({
                "base_ref": base_ref,
                "base_data": base_data,
                "variability_base": variability_data,
                "current_id": entry.get("id"),
            })

            overrides = entry.get("overrides") or {}
            variability_overrides = entry.get("variability_overrides") or {}
            self.populate_tables(category, overrides, variability_overrides if variability_data else None)
            editor["name_edit"].setText(entry.get("name", ""))
            editor["name_edit"].setPlaceholderText(base_ref or "")
            self._update_button_states(category)
        finally:
            state["loading"] = False

    def _get_saved_entry(self, category: str, entry_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._saved_entries.get(category, []):
            if entry.get("id") == entry_id:
                return entry
        return None

    def revert_to_catalog(self, category: str) -> None:
        state = self._state[category]
        editor = self._editors[category]
        base_ref = state.get("base_ref")
        if not base_ref:
            QMessageBox.information(self, "Catalog", "Select a catalog entry first.")
            return

        self.populate_tables(category, overrides=None, variability_overrides=None)
        editor["name_edit"].clear()
        editor["saved_combo"].blockSignals(True)
        editor["saved_combo"].setCurrentIndex(0)
        editor["saved_combo"].blockSignals(False)
        state["current_id"] = None
        self._update_button_states(category)

    # ------------------------------------------------------------------
    # Save / delete operations

    def save_entry(self, category: str, update_existing: bool) -> None:
        if not self._ensure_workspace():
            return

        state = self._state[category]
        editor = self._editors[category]
        base_ref = state.get("base_ref")
        if not base_ref:
            QMessageBox.warning(self, "Catalog", f"Select a base {category} before saving.")
            return

        base_data = state.get("base_data") or {}
        parameters = editor["parameters_table"].get_data()
        overrides = _compute_overrides(base_data, parameters)
        # Remove None overrides to avoid wiping defaults unless explicit
        overrides = {k: v for k, v in overrides.items() if v is not None}

        variability_base = state.get("variability_base") or {}
        variability_overrides: Dict[str, Any] = {}
        if variability_base:
            variability_values = editor["variability_table"].get_data()
            variability_overrides = _compute_overrides(variability_base, variability_values)
            variability_overrides = {k: v for k, v in variability_overrides.items() if v is not None}

        name = editor["name_edit"].text().strip() or base_ref

        payload: Dict[str, Any] = {
            "name": name,
            "ref": base_ref,
            "overrides": overrides,
        }
        if variability_overrides:
            payload["variability_overrides"] = variability_overrides

        entry_id = state.get("current_id") if update_existing else None
        stored = self.workspace_manager.save_catalog_entry(category, payload, entry_id=entry_id)
        state["current_id"] = stored.get("id")
        state["overrides"] = overrides
        state["variability_overrides"] = variability_overrides

        self.refresh_saved_entries(category, select_id=state["current_id"])
        QMessageBox.information(self, "Saved", f"Saved {category} configuration '{stored.get('name', base_ref)}'.")

        self._emit_config_update(category, stored)

    def delete_entry(self, category: str) -> None:
        if not self.workspace_manager:
            return

        state = self._state[category]
        editor = self._editors[category]
        entry_id = state.get("current_id") or editor["saved_combo"].currentData()
        if not entry_id:
            QMessageBox.information(self, "Catalog", "Select a saved configuration to delete.")
            return

        display_name = editor["saved_combo"].currentText()

        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete saved {category} '{entry_id}'?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.workspace_manager.delete_catalog_entry(category, entry_id)
        except Exception as exc:
            QMessageBox.warning(self, "Delete", f"Could not delete entry: {exc}")
            return

        state["current_id"] = None
        self.refresh_saved_entries(category)
        QMessageBox.information(self, "Deleted", f"Removed saved {category} '{entry_id}'.")
        self.config_updated.emit(
            category,
            {
                "id": entry_id,
                "ref": entry_id,
                "name": display_name,
                "deleted": True,
            },
        )

    def _ensure_workspace(self) -> bool:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "Workspace", "Select a workspace before saving configurations.")
            return False
        return True

    def _emit_config_update(self, category: str, stored_payload: Dict[str, Any]) -> None:
        data = {
            "id": stored_payload.get("id"),
            "name": stored_payload.get("name"),
            "ref": stored_payload.get("ref"),
            "overrides": stored_payload.get("overrides", {}),
            "variability_overrides": stored_payload.get("variability_overrides", {}),
        }
        self.config_updated.emit(category, data)

    # ------------------------------------------------------------------
    # UI state helpers

    def _update_button_states(self, category: str) -> None:
        editor = self._editors[category]
        state = self._state[category]
        has_workspace = self.workspace_manager is not None
        has_saved = bool(state.get("current_id"))

        editor["save_btn"].setEnabled(has_workspace and has_saved)
        editor["save_as_btn"].setEnabled(has_workspace)
        editor["delete_btn"].setEnabled(has_workspace and has_saved)
        editor["revert_btn"].setEnabled(bool(state.get("base_ref")))

    def _set_variability_tab_enabled(self, category: str, enabled: bool) -> None:
        editor = self._editors[category]
        tab_widget: QTabWidget = editor["tab_widget"]
        variability_table = editor["variability_table"]
        index = tab_widget.indexOf(variability_table)
        if index != -1:
            tab_widget.setTabEnabled(index, enabled)
            if not enabled:
                variability_table.setRowCount(0)


class PopulationTab(QWidget):
    """Population editor for subjects and inhalation maneuvers."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self._builtin_loader = BuiltinDataLoader() if BuiltinDataLoader is not None else None
        self._category_titles = {"subject": "Subjects", "maneuver": "Inhalation Maneuvers"}
        self._display_names = {"subject": "subject", "maneuver": "maneuver"}
        self._editors: Dict[str, Dict[str, Any]] = {}
        self._saved_entries: Dict[str, List[Dict[str, Any]]] = {"subject": [], "maneuver": []}
        self._catalog_names: Dict[str, List[str]] = {"subject": [], "maneuver": []}
        self._state: Dict[str, Dict[str, Any]] = {
            "subject": {
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "loading": False,
                "overrides": {},
                "variability_overrides": {},
            },
            "maneuver": {
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "loading": False,
                "overrides": {},
                "variability_overrides": {},
            },
        }
        self.init_ui()
        self.refresh_catalog_options("subject")
        self.refresh_catalog_options("maneuver")

    def init_ui(self) -> None:
        layout = QVBoxLayout()

        header = QLabel("Population")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        columns = QHBoxLayout()
        self._editors["subject"] = self._build_editor("subject", self._category_titles["subject"])
        self._editors["maneuver"] = self._build_editor("maneuver", self._category_titles["maneuver"])
        columns.addWidget(self._editors["subject"]["group"])
        columns.addWidget(self._editors["maneuver"]["group"])
        layout.addLayout(columns)
        layout.addStretch()

        self.setLayout(layout)

    def _build_editor(self, category: str, title: str) -> Dict[str, Any]:
        group = QGroupBox(title)
        group_layout = QVBoxLayout()

        saved_layout = QHBoxLayout()
        saved_combo = QComboBox()
        saved_combo.addItem("Saved configurations…", None)
        saved_combo.currentIndexChanged.connect(
            lambda _idx, cat=category: self.on_saved_entry_changed(cat)
        )
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(lambda _=False, cat=category: self.delete_entry(cat))
        saved_layout.addWidget(saved_combo)
        saved_layout.addWidget(delete_btn)
        group_layout.addLayout(saved_layout)

        form_layout = QFormLayout()
        name_edit = QLineEdit()
        form_layout.addRow("Name:", name_edit)

        base_row = QHBoxLayout()
        base_combo = QComboBox()
        base_combo.currentIndexChanged.connect(
            lambda _=False, cat=category: self.on_base_changed(cat)
        )
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda _=False, cat=category: self.refresh_catalog_options(cat, preserve_selection=True))
        base_row.addWidget(base_combo)
        base_row.addWidget(refresh_btn)
        form_layout.addRow("Catalog ref:", base_row)

        help_label = QLabel("Adjust subject and maneuver parameters, capturing overrides in your workspace.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        form_layout.addRow("", help_label)

        group_layout.addLayout(form_layout)

        tab_widget = QTabWidget()
        parameters_table = KeyValueTable("Parameter", "Value")
        tab_widget.addTab(parameters_table, "Parameters")
        variability_table = KeyValueTable("Parameter", "Value")
        tab_widget.addTab(variability_table, "Variability")
        group_layout.addWidget(tab_widget)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda _=False, cat=category: self.save_entry(cat, update_existing=True))
        save_as_btn = QPushButton("Save As")
        save_as_btn.clicked.connect(lambda _=False, cat=category: self.save_entry(cat, update_existing=False))
        revert_btn = QPushButton("Revert to catalog")
        revert_btn.clicked.connect(lambda _=False, cat=category: self.revert_to_catalog(cat))
        button_row.addWidget(save_btn)
        button_row.addWidget(save_as_btn)
        button_row.addWidget(revert_btn)
        button_row.addStretch()
        group_layout.addLayout(button_row)

        group.setLayout(group_layout)

        return {
            "group": group,
            "saved_combo": saved_combo,
            "delete_btn": delete_btn,
            "name_edit": name_edit,
            "base_combo": base_combo,
            "parameters_table": parameters_table,
            "variability_table": variability_table,
            "tab_widget": tab_widget,
            "save_btn": save_btn,
            "save_as_btn": save_as_btn,
            "revert_btn": revert_btn,
        }

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]) -> None:
        self.workspace_manager = workspace_manager
        for category in ("subject", "maneuver"):
            self.refresh_saved_entries(category)
            self._update_button_states(category)

    def refresh_catalog_options(self, category: str, preserve_selection: bool = False) -> None:
        combo = self._editors[category]["base_combo"]
        previous = combo.currentData()

        names = self._fetch_catalog_references(category)
        self._catalog_names[category] = names

        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name, name)
        combo.blockSignals(False)

        if preserve_selection and previous in names:
            combo.setCurrentIndex(combo.findData(previous))
        elif names:
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(-1)

        self.on_base_changed(category)

    def _fetch_catalog_references(self, category: str) -> List[str]:
        names: List[str] = []
        if CATALOG_AVAILABLE and app_api is not None:
            try:
                names = app_api.list_catalog_entries(category)
            except Exception:
                names = []

        if not names:
            names = self._fallback_catalog_names(category)

        return sorted(dict.fromkeys(names))

    def _fallback_catalog_names(self, category: str) -> List[str]:
        if not self._builtin_loader:
            return []

        try:
            catalog_root = self._builtin_loader.catalog_root
            directory = "inhalation" if category == "maneuver" else category
            category_dir = catalog_root / directory
            if not category_dir.exists():
                return []
            names: List[str] = []
            for path in category_dir.glob("*.toml"):
                if path.stem.startswith("Variability_"):
                    continue
                names.append(path.stem)
            return names
        except Exception:
            return []

    def _fetch_catalog_entity(self, category: str, ref: str) -> Dict[str, Any]:
        if not ref:
            return {}
        if CATALOG_AVAILABLE and app_api is not None:
            try:
                return app_api.get_catalog_entry(category, ref)
            except Exception:
                pass
        if self._builtin_loader:
            try:
                if category == "subject":
                    return self._builtin_loader.load_subject_physiology(ref)
                if category == "maneuver":
                    return self._builtin_loader.load_inhalation_profile(ref)
            except Exception:
                pass
        raise ValueError(f"Could not load {category} '{ref}' from catalog")

    def _fetch_variability_data(self, category: str, ref: str) -> Optional[Dict[str, Any]]:
        if not self._builtin_loader:
            return None
        directory = "inhalation" if category == "maneuver" else category
        filename = f"Variability_{ref}"
        try:
            return self._builtin_loader.load_variability_file(directory, filename)
        except Exception:
            return None

    def on_base_changed(self, category: str) -> None:
        state = self._state[category]
        editor = self._editors[category]

        if state["loading"]:
            return

        base_combo = editor["base_combo"]
        ref = base_combo.currentData()
        if not ref:
            state.update({
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "overrides": {},
                "variability_overrides": {},
            })
            editor["parameters_table"].setRowCount(0)
            editor["variability_table"].setRowCount(0)
            self._set_variability_tab_enabled(category, False)
            self._update_button_states(category)
            return

        try:
            base_data = self._fetch_catalog_entity(category, ref)
        except Exception as exc:
            QMessageBox.warning(self, "Catalog", f"Could not load {category} '{ref}': {exc}")
            return

        variability_data = self._fetch_variability_data(category, ref) or {}

        state.update({
            "base_ref": ref,
            "base_data": base_data,
            "variability_base": variability_data,
            "current_id": None,
            "overrides": {},
            "variability_overrides": {},
        })

        self.populate_tables(category, overrides=None, variability_overrides=None)
        editor["name_edit"].setPlaceholderText(ref)

        saved_combo = editor["saved_combo"]
        saved_combo.blockSignals(True)
        saved_combo.setCurrentIndex(0)
        saved_combo.blockSignals(False)
        self._update_button_states(category)

    def populate_tables(
        self,
        category: str,
        overrides: Optional[Dict[str, Any]],
        variability_overrides: Optional[Dict[str, Any]],
    ) -> None:
        state = self._state[category]
        editor = self._editors[category]

        resolved = _apply_overrides(state["base_data"], overrides)
        editor["parameters_table"].set_data(resolved)
        state["overrides"] = copy.deepcopy(overrides) if overrides else {}

        variability_base = state.get("variability_base") or {}
        if variability_base:
            resolved_var = _apply_overrides(variability_base, variability_overrides)
            editor["variability_table"].set_data(resolved_var)
            state["variability_overrides"] = copy.deepcopy(variability_overrides) if variability_overrides else {}
            self._set_variability_tab_enabled(category, True)
        else:
            editor["variability_table"].setRowCount(0)
            state["variability_overrides"] = {}
            self._set_variability_tab_enabled(category, False)

    def refresh_saved_entries(self, category: str, select_id: Optional[str] = None) -> None:
        editor = self._editors[category]
        combo = editor["saved_combo"]

        entries: List[Dict[str, Any]] = []
        if self.workspace_manager is not None:
            try:
                entries = self.workspace_manager.list_catalog_entries(category)
            except Exception as exc:
                logger.warning("list catalog entries failed", category=category, error=str(exc))
        self._saved_entries[category] = entries

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Saved configurations…", None)

        selected_index = 0
        for idx, entry in enumerate(entries, start=1):
            entry_id = entry.get("id")
            display_name = entry.get("name") or entry.get("ref") or entry_id
            combo.addItem(display_name, entry_id)
            if select_id and entry_id == select_id:
                selected_index = idx

        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

        if select_id:
            self.on_saved_entry_changed(category)
        else:
            self._update_button_states(category)

    def on_saved_entry_changed(self, category: str) -> None:
        editor = self._editors[category]
        combo = editor["saved_combo"]
        entry_id = combo.currentData()
        if not entry_id:
            self._state[category]["current_id"] = None
            self._update_button_states(category)
            return

        entry = self._get_saved_entry(category, entry_id)
        if not entry:
            QMessageBox.warning(self, "Catalog", f"Saved {category} '{entry_id}' not found")
            self.refresh_saved_entries(category)
            return

        state = self._state[category]
        state["loading"] = True
        try:
            base_ref = entry.get("ref")
            if base_ref:
                base_combo = editor["base_combo"]
                if base_combo.findData(base_ref) == -1:
                    base_combo.addItem(base_ref, base_ref)
                base_combo.setCurrentIndex(base_combo.findData(base_ref))
                base_data = self._fetch_catalog_entity(category, base_ref)
                variability_data = self._fetch_variability_data(category, base_ref) or {}
            else:
                base_data = {}
                variability_data = {}

            state.update({
                "base_ref": base_ref,
                "base_data": base_data,
                "variability_base": variability_data,
                "current_id": entry.get("id"),
            })

            overrides = entry.get("overrides") or {}
            variability_overrides = entry.get("variability_overrides") or {}
            self.populate_tables(category, overrides, variability_overrides if variability_data else None)
            editor["name_edit"].setText(entry.get("name", ""))
            editor["name_edit"].setPlaceholderText(base_ref or "")
            self._update_button_states(category)
        finally:
            state["loading"] = False

    def _get_saved_entry(self, category: str, entry_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._saved_entries.get(category, []):
            if entry.get("id") == entry_id:
                return entry
        return None

    def revert_to_catalog(self, category: str) -> None:
        state = self._state[category]
        editor = self._editors[category]
        base_ref = state.get("base_ref")
        if not base_ref:
            QMessageBox.information(self, "Catalog", f"Select a base {self._display_names[category]} first.")
            return

        self.populate_tables(category, overrides=None, variability_overrides=None)
        editor["name_edit"].clear()
        editor["saved_combo"].blockSignals(True)
        editor["saved_combo"].setCurrentIndex(0)
        editor["saved_combo"].blockSignals(False)
        state["current_id"] = None
        self._update_button_states(category)

    def save_entry(self, category: str, update_existing: bool) -> None:
        if not self._ensure_workspace():
            return

        state = self._state[category]
        editor = self._editors[category]
        base_ref = state.get("base_ref")
        if not base_ref:
            QMessageBox.warning(self, "Catalog", f"Select a base {self._display_names[category]} before saving.")
            return

        base_data = state.get("base_data") or {}
        parameters = editor["parameters_table"].get_data()
        overrides = _compute_overrides(base_data, parameters)
        overrides = {k: v for k, v in overrides.items() if v is not None}

        variability_base = state.get("variability_base") or {}
        variability_overrides: Dict[str, Any] = {}
        if variability_base:
            variability_values = editor["variability_table"].get_data()
            variability_overrides = _compute_overrides(variability_base, variability_values)
            variability_overrides = {k: v for k, v in variability_overrides.items() if v is not None}

        name = editor["name_edit"].text().strip() or base_ref

        payload: Dict[str, Any] = {
            "name": name,
            "ref": base_ref,
            "overrides": overrides,
        }
        if variability_overrides:
            payload["variability_overrides"] = variability_overrides

        entry_id = state.get("current_id") if update_existing else None
        stored = self.workspace_manager.save_catalog_entry(category, payload, entry_id=entry_id)
        state["current_id"] = stored.get("id")
        state["overrides"] = overrides
        state["variability_overrides"] = variability_overrides

        self.refresh_saved_entries(category, select_id=state["current_id"])
        display = stored.get("name", base_ref)
        QMessageBox.information(self, "Saved", f"Saved {self._display_names[category]} configuration '{display}'.")

        self._emit_config_update(category, stored)

    def delete_entry(self, category: str) -> None:
        if not self.workspace_manager:
            return

        state = self._state[category]
        editor = self._editors[category]
        entry_id = state.get("current_id") or editor["saved_combo"].currentData()
        if not entry_id:
            QMessageBox.information(self, "Catalog", f"Select a saved {self._display_names[category]} to delete.")
            return

        display_name = editor["saved_combo"].currentText()

        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete saved {self._display_names[category]} '{entry_id}'?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.workspace_manager.delete_catalog_entry(category, entry_id)
        except Exception as exc:
            QMessageBox.warning(self, "Delete", f"Could not delete entry: {exc}")
            return

        state["current_id"] = None
        self.refresh_saved_entries(category)
        QMessageBox.information(self, "Deleted", f"Removed saved {self._display_names[category]} '{entry_id}'.")
        self.config_updated.emit(
            category,
            {
                "id": entry_id,
                "ref": entry_id,
                "name": display_name,
                "deleted": True,
            },
        )

    def _ensure_workspace(self) -> bool:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "Workspace", "Select a workspace before saving configurations.")
            return False
        return True

    def _emit_config_update(self, category: str, stored_payload: Dict[str, Any]) -> None:
        data = {
            "id": stored_payload.get("id"),
            "name": stored_payload.get("name"),
            "ref": stored_payload.get("ref"),
            "overrides": stored_payload.get("overrides", {}),
            "variability_overrides": stored_payload.get("variability_overrides", {}),
        }
        self.config_updated.emit(category, data)

    def _update_button_states(self, category: str) -> None:
        editor = self._editors[category]
        state = self._state[category]
        has_workspace = self.workspace_manager is not None
        has_saved = bool(state.get("current_id"))

        editor["save_btn"].setEnabled(has_workspace and has_saved)
        editor["save_as_btn"].setEnabled(has_workspace)
        editor["delete_btn"].setEnabled(has_workspace and has_saved)
        editor["revert_btn"].setEnabled(bool(state.get("base_ref")))

    def _set_variability_tab_enabled(self, category: str, enabled: bool) -> None:
        editor = self._editors[category]
        tab_widget: QTabWidget = editor["tab_widget"]
        variability_table = editor["variability_table"]
        index = tab_widget.indexOf(variability_table)
        if index != -1:
            tab_widget.setTabEnabled(index, enabled)
            if not enabled:
                variability_table.setRowCount(0)


class StudyDesignerTab(QWidget):
    """Study Designer tab for creating simulation configurations."""

    config_ready = Signal(dict)

    def __init__(self):
        super().__init__()
        self.current_config: Dict[str, Any] = {}
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.last_saved_config_path: Optional[str] = None
        self.run_type_definitions = [
            (
                "single",
                "Single Simulation",
                "Execute one simulation run using the configuration below.",
            ),
            (
                "sweep",
                "Parameter Sweep",
                "Enumerate a grid of parameter overrides and run each scenario.",
            ),
            (
                "sensitivity",
                "Sensitivity Analysis",
                "Perturb inputs to understand local sensitivities.",
            ),
            (
                "parameter_estimation",
                "Parameter Estimation",
                "Fit model parameters against observed data using optimisation.",
            ),
            (
                "virtual_trial",
                "Virtual Trial",
                "Simulate a virtual population for a single product.",
            ),
            (
                "virtual_bioequivalence",
                "Virtual Bioequivalence",
                "Compare multiple products in a virtual population.",
            ),
        ]
        self.saved_catalog_entries: Dict[str, List[Dict[str, Any]]] = {
            "subject": [],
            "api": [],
            "product": [],
            "maneuver": [],
        }
        self.selected_entities: Dict[str, Optional[Dict[str, Any]]] = {
            "subject": None,
            "api": None,
            "product": None,
            "maneuver": None,
        }
        self.entity_combos: Dict[str, QComboBox] = {}
        self.entity_placeholder: Dict[str, str] = {
            "subject": "Configure subjects in Population tab",
            "maneuver": "Configure maneuvers in Population tab",
            "api": "Configure APIs in API & Products tab",
            "product": "Configure products in API & Products tab",
        }
        self.entity_display: Dict[str, str] = {
            "subject": "subject",
            "api": "API",
            "product": "product",
            "maneuver": "maneuver",
        }
        self.init_ui()
        self.populate_catalog_defaults()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("Study Designer")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Scroll area for form
        scroll = QScrollArea()
        form_widget = QWidget()
        form_layout = QVBoxLayout()

        # Study metadata
        study_group = QGroupBox("Study Details")
        study_layout = QFormLayout()

        self.study_name_edit = QLineEdit()
        self.study_name_edit.setPlaceholderText("Study or configuration name")
        study_layout.addRow("Config Name:", self.study_name_edit)

        study_group.setLayout(study_layout)
        form_layout.addWidget(study_group)

        # Subject configuration
        subject_group = QGroupBox("Subject Configuration")
        subject_layout = QFormLayout()

        self.subject_ref_combo = QComboBox()
        self.subject_ref_combo.setEditable(False)
        self.subject_ref_combo.currentIndexChanged.connect(
            lambda _=0, cat="subject": self.on_entity_selection_changed(cat)
        )

        subject_layout.addRow("Subject Reference:", self.subject_ref_combo)
        subject_group.setLayout(subject_layout)
        form_layout.addWidget(subject_group)

        # API configuration
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout()

        self.api_ref_combo = QComboBox()
        self.api_ref_combo.setEditable(False)
        self.api_ref_combo.currentIndexChanged.connect(
            lambda _=0, cat="api": self.on_entity_selection_changed(cat)
        )

        api_layout.addRow("API Reference:", self.api_ref_combo)
        api_group.setLayout(api_layout)
        form_layout.addWidget(api_group)

        # Product configuration
        product_group = QGroupBox("Product Configuration")
        product_layout = QFormLayout()

        self.product_ref_combo = QComboBox()
        self.product_ref_combo.setEditable(False)
        self.product_ref_combo.currentIndexChanged.connect(
            lambda _=0, cat="product": self.on_entity_selection_changed(cat)
        )

        product_layout.addRow("Product Reference:", self.product_ref_combo)
        product_group.setLayout(product_layout)
        form_layout.addWidget(product_group)

        # Maneuver configuration
        maneuver_group = QGroupBox("Maneuver Configuration")
        maneuver_layout = QFormLayout()

        self.maneuver_ref_combo = QComboBox()
        self.maneuver_ref_combo.setEditable(False)
        self.maneuver_ref_combo.currentIndexChanged.connect(
            lambda _=0, cat="maneuver": self.on_entity_selection_changed(cat)
        )

        maneuver_layout.addRow("Maneuver Reference:", self.maneuver_ref_combo)
        maneuver_group.setLayout(maneuver_layout)
        form_layout.addWidget(maneuver_group)

        self.entity_combos = {
            "subject": self.subject_ref_combo,
            "api": self.api_ref_combo,
            "product": self.product_ref_combo,
            "maneuver": self.maneuver_ref_combo,
        }
        for category in self.entity_combos:
            self._set_combo_placeholder(category)

        # Model configuration
        models_group = QGroupBox("Model Configuration")
        models_layout = QFormLayout()

        self.deposition_model_combo = QComboBox()
        self.deposition_model_combo.addItems(["clean_lung", "null"])

        self.pbbm_model_combo = QComboBox()
        self.pbbm_model_combo.addItems(["numba"])

        self.pk_model_combo = QComboBox()
        self.pk_model_combo.addItems(["pk_1c", "pk_2c", "pk_3c", "null"])

        models_layout.addRow("Deposition Model:", self.deposition_model_combo)
        models_layout.addRow("PBBM Model:", self.pbbm_model_combo)
        models_layout.addRow("PK Model:", self.pk_model_combo)

        models_group.setLayout(models_layout)
        form_layout.addWidget(models_group)

        # Run configuration
        run_group = QGroupBox("Run Configuration")
        run_layout = QFormLayout()

        self.run_type_combo = QComboBox()
        for value, label, _ in self.run_type_definitions:
            self.run_type_combo.addItem(label, userData=value)
        self.run_type_combo.currentIndexChanged.connect(self.on_run_type_changed)
        run_layout.addRow("Run Type:", self.run_type_combo)

        self.run_label_edit = QLineEdit()
        self.run_label_edit.setPlaceholderText("Friendly run label (optional)")
        run_layout.addRow("Run Label:", self.run_label_edit)

        self.run_type_help_label = QLabel()
        self.run_type_help_label.setWordWrap(True)
        self.run_type_help_label.setStyleSheet("color: #555; font-size: 11px;")
        run_layout.addRow(self.run_type_help_label)

        self.run_type_notice_label = QLabel()
        self.run_type_notice_label.setWordWrap(True)
        self.run_type_notice_label.setStyleSheet("color: #888; font-size: 11px;")
        run_layout.addRow(self.run_type_notice_label)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(123)

        self.stages_edit = QLineEdit("deposition,pbbm,pk")

        run_layout.addRow("Seed:", self.seed_spin)
        run_layout.addRow("Stages:", self.stages_edit)

        run_group.setLayout(run_layout)
        form_layout.addWidget(run_group)

        # Sweep configuration (basic)
        self.sweep_group = QGroupBox("Parameter Sweep")
        sweep_layout = QVBoxLayout()

        self.sweep_enabled = QCheckBox("Enable Parameter Sweep")
        sweep_layout.addWidget(self.sweep_enabled)

        self.sweep_params_edit = QTextEdit()
        self.sweep_params_edit.setPlaceholderText(
            'Enter sweep parameters as JSON:\n'
            '{\n'
            '  "pk.model": ["pk_1c", "pk_2c"],\n'
            '  "run.seed": [123, 456, 789]\n'
            '}'
        )
        self.sweep_params_edit.setMaximumHeight(100)
        self.sweep_params_edit.setEnabled(False)
        self.sweep_enabled.toggled.connect(self.sweep_params_edit.setEnabled)

        sweep_layout.addWidget(self.sweep_params_edit)
        self.sweep_group.setLayout(sweep_layout)
        form_layout.addWidget(self.sweep_group)

        # Parameter estimation configuration
        self.parameter_estimation_group = QGroupBox("Parameter Estimation")
        pe_layout = QVBoxLayout()

        # Observed dataset controls
        observed_form = QFormLayout()

        self.observed_label_edit = QLineEdit()
        self.observed_label_edit.setPlaceholderText("Dataset label (optional)")
        observed_form.addRow("Label:", self.observed_label_edit)

        obs_path_row = QHBoxLayout()
        self.observed_path_edit = QLineEdit()
        self.observed_path_edit.setPlaceholderText("Path to observed PK CSV (optional)")
        browse_observed_btn = QPushButton("Browse…")
        browse_observed_btn.clicked.connect(self.browse_observed_dataset)
        obs_path_row.addWidget(self.observed_path_edit)
        obs_path_row.addWidget(browse_observed_btn)
        obs_path_container = QWidget()
        obs_path_container.setLayout(obs_path_row)
        observed_form.addRow("Observed CSV:", obs_path_container)

        self.observed_time_col_edit = QLineEdit("time_h")
        observed_form.addRow("Time column:", self.observed_time_col_edit)

        self.observed_value_col_edit = QLineEdit("conc_ng_ml")
        observed_form.addRow("Concentration column:", self.observed_value_col_edit)

        self.observed_time_unit_combo = QComboBox()
        self.observed_time_unit_combo.addItem("Hours", userData="h")
        self.observed_time_unit_combo.addItem("Minutes", userData="min")
        self.observed_time_unit_combo.addItem("Seconds", userData="s")
        observed_form.addRow("Time unit:", self.observed_time_unit_combo)

        self.observed_manual_data_edit = QTextEdit()
        self.observed_manual_data_edit.setPlaceholderText(
            "Optional JSON series overrides (e.g. {\"time_s\": [...], \"values\": [...]} or "
            "[{\"time\": 0, \"value\": 0}, …])"
        )
        self.observed_manual_data_edit.setMaximumHeight(100)
        observed_form.addRow("Observed series JSON:", self.observed_manual_data_edit)

        self.timeseries_weight_spin = QDoubleSpinBox()
        self.timeseries_weight_spin.setDecimals(4)
        self.timeseries_weight_spin.setRange(0.0, 1_000_000.0)
        self.timeseries_weight_spin.setValue(1.0)
        observed_form.addRow("Timeseries weight:", self.timeseries_weight_spin)

        pe_layout.addLayout(observed_form)

        # Parameter table controls
        self.parameter_table = QTableWidget(0, 6)
        self.parameter_table.setHorizontalHeaderLabels([
            "Name", "Config Path", "Mode", "Step", "Lower Bound", "Upper Bound"
        ])
        self.parameter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.parameter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.parameter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.parameter_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        table_buttons = QHBoxLayout()
        add_param_btn = QPushButton("Add Parameter")
        add_param_btn.clicked.connect(self.add_parameter_row)
        remove_param_btn = QPushButton("Remove Selected")
        remove_param_btn.clicked.connect(self.remove_selected_parameter_row)
        table_buttons.addWidget(add_param_btn)
        table_buttons.addWidget(remove_param_btn)
        table_buttons.addStretch()

        pe_layout.addWidget(self.parameter_table)
        pe_layout.addLayout(table_buttons)

        settings_form = QFormLayout()

        self.include_baseline_checkbox = QCheckBox("Include baseline run")
        self.include_baseline_checkbox.setChecked(True)
        settings_form.addRow("Baseline:", self.include_baseline_checkbox)

        self.default_step_spin = QDoubleSpinBox()
        self.default_step_spin.setDecimals(6)
        self.default_step_spin.setRange(1e-6, 10.0)
        self.default_step_spin.setValue(0.1)
        settings_form.addRow("Default relative step:", self.default_step_spin)

        self.target_metric_combo = QComboBox()
        self.target_metric_combo.addItem("Sum of squared errors (SSE)", userData="sse")
        self.target_metric_combo.addItem("Mean absolute error (MAE)", userData="mae")
        settings_form.addRow("Objective:", self.target_metric_combo)

        pe_layout.addLayout(settings_form)

        scalar_group = QGroupBox("Additional Scalar Targets")
        scalar_layout = QGridLayout()
        scalar_layout.setColumnStretch(3, 1)

        def _make_scalar_row(row: int, label: str):
            checkbox = QCheckBox(label)
            value_spin = QDoubleSpinBox()
            value_spin.setDecimals(6)
            value_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            weight_spin = QDoubleSpinBox()
            weight_spin.setDecimals(4)
            weight_spin.setRange(0.0, 1_000_000.0)
            weight_spin.setValue(1.0)
            scalar_layout.addWidget(checkbox, row, 0)
            scalar_layout.addWidget(QLabel("Observed"), row, 1)
            scalar_layout.addWidget(value_spin, row, 2)
            scalar_layout.addWidget(QLabel("Weight"), row, 3)
            scalar_layout.addWidget(weight_spin, row, 4)
            return checkbox, value_spin, weight_spin

        self.pk_auc_checkbox, self.pk_auc_value_spin, self.pk_auc_weight_spin = _make_scalar_row(0, "PK AUC0_t")
        self.cfd_mmad_checkbox, self.cfd_mmad_value_spin, self.cfd_mmad_weight_spin = _make_scalar_row(1, "CFD MMAD (um)")
        self.cfd_gsd_checkbox, self.cfd_gsd_value_spin, self.cfd_gsd_weight_spin = _make_scalar_row(2, "CFD GSD")
        self.cfd_mt_checkbox, self.cfd_mt_value_spin, self.cfd_mt_weight_spin = _make_scalar_row(3, "CFD MT fraction")

        scalar_group.setLayout(scalar_layout)
        pe_layout.addWidget(scalar_group)

        deposition_group = QGroupBox("Deposition Fraction Targets")
        deposition_layout = QVBoxLayout()
        self.deposition_fraction_enable = QCheckBox("Enable deposition fraction targets")
        deposition_layout.addWidget(self.deposition_fraction_enable)
        self.deposition_fraction_text = QTextEdit()
        self.deposition_fraction_text.setPlaceholderText(
            "Enter region fractions (JSON mapping or CSV rows: region,value)\n"
            "Example:\n"
            "central,0.25\nperipheral,0.60"
        )
        self.deposition_fraction_text.setMaximumHeight(100)
        self.deposition_fraction_text.setEnabled(False)
        self.deposition_fraction_enable.toggled.connect(self.deposition_fraction_text.setEnabled)
        deposition_layout.addWidget(self.deposition_fraction_text)

        fraction_weight_row = QHBoxLayout()
        fraction_weight_row.addWidget(QLabel("Weight:"))
        self.deposition_fraction_weight_spin = QDoubleSpinBox()
        self.deposition_fraction_weight_spin.setDecimals(4)
        self.deposition_fraction_weight_spin.setRange(0.0, 1_000_000.0)
        self.deposition_fraction_weight_spin.setValue(1.0)
        self.deposition_fraction_weight_spin.setEnabled(False)
        self.deposition_fraction_enable.toggled.connect(self.deposition_fraction_weight_spin.setEnabled)
        fraction_weight_row.addWidget(self.deposition_fraction_weight_spin)
        fraction_weight_row.addStretch()
        deposition_layout.addLayout(fraction_weight_row)

        deposition_group.setLayout(deposition_layout)
        pe_layout.addWidget(deposition_group)

        self.parameter_estimation_group.setLayout(pe_layout)
        form_layout.addWidget(self.parameter_estimation_group)
        self.parameter_estimation_group.setVisible(False)

        # Virtual trial configuration
        self.virtual_trial_group = QGroupBox("Virtual Trial")
        vt_layout = QFormLayout()
        self.virtual_trial_subjects_spin = QSpinBox()
        self.virtual_trial_subjects_spin.setRange(1, 5000)
        self.virtual_trial_subjects_spin.setValue(10)
        vt_layout.addRow("Subjects:", self.virtual_trial_subjects_spin)

        self.virtual_trial_seed_spin = QSpinBox()
        self.virtual_trial_seed_spin.setRange(0, 1_000_000)
        self.virtual_trial_seed_spin.setValue(1234)
        vt_layout.addRow("Base seed:", self.virtual_trial_seed_spin)

        self.virtual_trial_variability_check = QCheckBox("Enable variability sampling")
        self.virtual_trial_variability_check.setChecked(True)
        vt_layout.addRow("Variability:", self.virtual_trial_variability_check)

        self.virtual_trial_products_edit = QLineEdit()
        self.virtual_trial_products_edit.setPlaceholderText("Comma-separated additional products (optional)")
        vt_layout.addRow("Additional products:", self.virtual_trial_products_edit)

        self.virtual_trial_group.setLayout(vt_layout)
        form_layout.addWidget(self.virtual_trial_group)
        self.virtual_trial_group.setVisible(False)

        # Virtual bioequivalence configuration
        self.vbe_group = QGroupBox("Virtual Bioequivalence")
        vbe_layout = QFormLayout()
        self.vbe_subjects_spin = QSpinBox()
        self.vbe_subjects_spin.setRange(2, 5000)
        self.vbe_subjects_spin.setValue(24)
        vbe_layout.addRow("Subjects:", self.vbe_subjects_spin)

        self.vbe_seed_spin = QSpinBox()
        self.vbe_seed_spin.setRange(0, 1_000_000)
        self.vbe_seed_spin.setValue(4321)
        vbe_layout.addRow("Base seed:", self.vbe_seed_spin)

        self.vbe_variability_check = QCheckBox("Enable variability sampling")
        self.vbe_variability_check.setChecked(True)
        vbe_layout.addRow("Variability:", self.vbe_variability_check)

        self.vbe_reference_product_edit = QLineEdit()
        self.vbe_reference_product_edit.setPlaceholderText("Reference product (defaults to config value)")
        vbe_layout.addRow("Reference product:", self.vbe_reference_product_edit)

        self.vbe_test_products_edit = QLineEdit()
        self.vbe_test_products_edit.setPlaceholderText("Comma-separated test products (e.g. test_product_a,test_product_b)")
        vbe_layout.addRow("Test products:", self.vbe_test_products_edit)

        self.vbe_group.setLayout(vbe_layout)
        form_layout.addWidget(self.vbe_group)
        self.vbe_group.setVisible(False)

        # Buttons
        button_layout = QHBoxLayout()

        validate_btn = QPushButton("Validate Config")
        validate_btn.clicked.connect(self.validate_config)

        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.save_config)

        preview_btn = QPushButton("Preview Manifest")
        preview_btn.clicked.connect(self.preview_manifest)

        button_layout.addWidget(validate_btn)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(preview_btn)
        button_layout.addStretch()

        form_layout.addLayout(button_layout)

        form_widget.setLayout(form_layout)
        scroll.setWidget(form_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)
        self.setLayout(layout)

        # Initialise run-type dependent UI state
        self.on_run_type_changed()

    def _entry_identifier(self, entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        return entry.get("id") or entry.get("ref")

    def _set_combo_placeholder(self, category: str) -> None:
        combo = self.entity_combos.get(category)
        if not combo:
            return
        combo.blockSignals(True)
        combo.clear()
        placeholder = self.entity_placeholder.get(category, "No entries available")
        combo.addItem(placeholder, None)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self.selected_entities[category] = None

    def _populate_combo_from_saved(
        self,
        category: str,
        entries: List[Dict[str, Any]],
        select_id: Optional[str] = None,
    ) -> None:
        combo = self.entity_combos.get(category)
        if not combo:
            return

        combo.blockSignals(True)
        combo.clear()

        if not entries:
            combo.blockSignals(False)
            self._set_combo_placeholder(category)
            return

        target_index = 0
        for idx, entry in enumerate(entries):
            identifier = self._entry_identifier(entry)
            display = entry.get("name") or entry.get("ref") or identifier or "(unnamed)"
            combo.addItem(display, copy.deepcopy(entry))
            if select_id and identifier == select_id:
                target_index = idx

        combo.setCurrentIndex(target_index)
        combo.blockSignals(False)
        self.on_entity_selection_changed(category)

    def on_entity_selection_changed(self, category: str) -> None:
        combo = self.entity_combos.get(category)
        if not combo:
            return
        data = combo.currentData()
        if isinstance(data, dict):
            self.selected_entities[category] = copy.deepcopy(data)
        else:
            self.selected_entities[category] = None

    def reload_workspace_catalog_entries(self) -> None:
        if self.workspace_manager is None:
            self.populate_catalog_defaults()
            return

        for category in ("subject", "api", "product", "maneuver"):
            try:
                entries = self.workspace_manager.list_catalog_entries(category)
            except Exception as exc:
                logger.warning(
                    "list catalog entries failed",
                    category=category,
                    error=str(exc),
                )
                entries = []
            self.saved_catalog_entries[category] = entries
            select_id = entries[0].get("id") if entries else None
            self._populate_combo_from_saved(category, entries, select_id=select_id)

    def _entity_config_payload(self, category: str) -> Dict[str, Any]:
        entry = self.selected_entities.get(category)
        label = self.entity_display.get(category, category)
        if not entry:
            raise ValueError(
                f"Select a {label} from its configuration tab before building the configuration."
            )

        ref = entry.get("ref")
        if not ref:
            raise ValueError(f"Saved {label} entry is missing a catalog reference.")

        payload: Dict[str, Any] = {"ref": ref}
        overrides = entry.get("overrides") or {}
        if overrides:
            payload["overrides"] = copy.deepcopy(overrides)
        return payload

    def populate_catalog_defaults(self):
        """Populate reference selectors with catalog data when available."""
        if self.workspace_manager is not None:
            return

        if not CATALOG_AVAILABLE or app_api is None:
            for category in self.entity_combos.keys():
                self._set_combo_placeholder(category)
            return

        try:
            for category in ("subject", "api", "product", "maneuver"):
                refs = app_api.list_catalog_entries(category) or []
                entries = [
                    {
                        "id": ref,
                        "name": ref,
                        "ref": ref,
                        "overrides": {},
                    }
                    for ref in refs
                ]
                self.saved_catalog_entries[category] = entries
                self._populate_combo_from_saved(category, entries)
        except Exception as exc:
            print(f"Catalog population failed: {exc}")

    # --- Run type helpers ---------------------------------------------------

    def current_config_name(self) -> Optional[str]:
        name = self.study_name_edit.text().strip()
        return name or None

    def current_run_type(self) -> str:
        if not hasattr(self, "run_type_combo"):
            return "single"
        value = self.run_type_combo.currentData()
        return value or "single"

    def get_run_label(self) -> Optional[str]:
        text = self.run_label_edit.text().strip() if hasattr(self, "run_label_edit") else ""
        if text:
            return text
        return self.current_config_name()

    def on_run_type_changed(self):
        run_type = self.current_run_type()
        description = next((desc for value, _, desc in self.run_type_definitions if value == run_type), "")
        self.run_type_help_label.setText(description)

        if run_type == "single":
            notice = ""
        elif run_type == "sweep":
            notice = "Provide sweep parameter overrides in the section below."
        elif run_type == "parameter_estimation":
            notice = "Configure observed data and parameter step sizes for calibration runs below."
        elif run_type == "virtual_trial":
            notice = "Configure cohort size, seeds, and optional products for virtual trials below."
        elif run_type == "virtual_bioequivalence":
            notice = "Configure cohort size and reference/test products for bioequivalence runs below."
        else:
            notice = "Detailed configuration for this run type will arrive in upcoming builds."
        self.run_type_notice_label.setText(notice)

        is_sweep = run_type == "sweep"
        is_parameter_estimation = run_type == "parameter_estimation"
        is_virtual_trial = run_type == "virtual_trial"
        is_virtual_bioequivalence = run_type == "virtual_bioequivalence"
        if hasattr(self, "sweep_group"):
            self.sweep_group.setVisible(is_sweep)
            self.sweep_enabled.blockSignals(True)
            self.sweep_enabled.setChecked(is_sweep)
            self.sweep_enabled.setEnabled(is_sweep)
            self.sweep_enabled.blockSignals(False)
            self.sweep_params_edit.setEnabled(is_sweep)

        if hasattr(self, "parameter_estimation_group"):
            self.parameter_estimation_group.setVisible(is_parameter_estimation)
            if is_parameter_estimation and self.parameter_table.rowCount() == 0:
                self.add_parameter_row()

        if hasattr(self, "virtual_trial_group"):
            self.virtual_trial_group.setVisible(is_virtual_trial)

        if hasattr(self, "vbe_group"):
            self.vbe_group.setVisible(is_virtual_bioequivalence)

    def get_sweep_parameters(self) -> Dict[str, Any]:
        if self.current_run_type() != "sweep":
            return {}
        sweep_text = self.sweep_params_edit.toPlainText().strip()
        if not sweep_text:
            return {}
        return json.loads(sweep_text)

    def browse_observed_dataset(self):
        if self.workspace_manager is not None:
            start_dir = str(self.workspace_manager.workspace_path)
        else:
            start_dir = str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Observed PK CSV",
            start_dir,
            "CSV files (*.csv *.tsv);;All files (*)"
        )
        if file_path:
            self.observed_path_edit.setText(file_path)

    def add_parameter_row(
        self,
        name: str = "",
        path: str = "",
        mode: str = "relative",
        delta: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        row = self.parameter_table.rowCount()
        self.parameter_table.insertRow(row)

        name_edit = QLineEdit(name)
        self.parameter_table.setCellWidget(row, 0, name_edit)

        path_edit = QLineEdit(path)
        path_edit.setPlaceholderText("e.g. pk.params.clearance_L_h")
        self.parameter_table.setCellWidget(row, 1, path_edit)

        mode_combo = QComboBox()
        mode_combo.addItem("Relative", userData="relative")
        mode_combo.addItem("Absolute", userData="absolute")
        current_index = mode_combo.findData(mode)
        if current_index >= 0:
            mode_combo.setCurrentIndex(current_index)
        self.parameter_table.setCellWidget(row, 2, mode_combo)

        step_spin = QDoubleSpinBox()
        step_spin.setDecimals(6)
        step_spin.setRange(0.0, 1e6)
        step_spin.setSingleStep(0.01)
        default_delta = delta if delta is not None else self.default_step_spin.value()
        step_spin.setValue(default_delta)
        self.parameter_table.setCellWidget(row, 3, step_spin)

        lower_edit = QLineEdit("" if lower is None else str(lower))
        lower_edit.setPlaceholderText("Optional")
        self.parameter_table.setCellWidget(row, 4, lower_edit)

        upper_edit = QLineEdit("" if upper is None else str(upper))
        upper_edit.setPlaceholderText("Optional")
        self.parameter_table.setCellWidget(row, 5, upper_edit)

    def remove_selected_parameter_row(self) -> None:
        selection_model = self.parameter_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            return
        row_index = selected_rows[0].row()
        self.parameter_table.removeRow(row_index)

    def _apply_run_plan_to_ui(self, run_plan: Mapping[str, Any]) -> None:
        run_type = run_plan.get("run_type", "single")
        index = self.run_type_combo.findData(run_type)
        if index >= 0:
            self.run_type_combo.blockSignals(True)
            self.run_type_combo.setCurrentIndex(index)
            self.run_type_combo.blockSignals(False)
            self.on_run_type_changed()

        label_text = run_plan.get("run_label")
        if label_text:
            self.run_label_edit.setText(str(label_text))

        if run_type == "sweep":
            params = run_plan.get("sweep_parameters") or {}
            try:
                self.sweep_params_edit.setPlainText(json.dumps(params, indent=2))
                self.sweep_enabled.setChecked(bool(params))
            except Exception:
                pass
        elif run_type == "parameter_estimation":
            estimation_plan = run_plan.get("estimation") or {}
            self._load_parameter_estimation_plan(estimation_plan)
        elif run_type == "virtual_trial":
            vt_plan = run_plan.get("virtual_trial") or {}
            try:
                self.virtual_trial_subjects_spin.setValue(int(vt_plan.get("n_subjects", self.virtual_trial_subjects_spin.value())))
            except Exception:
                pass
            try:
                self.virtual_trial_seed_spin.setValue(int(vt_plan.get("base_seed", self.virtual_trial_seed_spin.value())))
            except Exception:
                pass
            self.virtual_trial_variability_check.setChecked(bool(vt_plan.get("apply_variability", True)))
            additional = vt_plan.get("additional_products") or []
            if isinstance(additional, list):
                self.virtual_trial_products_edit.setText(", ".join(additional))
            else:
                self.virtual_trial_products_edit.setText(str(additional))
        elif run_type == "virtual_bioequivalence":
            vbe_plan = run_plan.get("virtual_bioequivalence") or {}
            try:
                self.vbe_subjects_spin.setValue(int(vbe_plan.get("n_subjects", self.vbe_subjects_spin.value())))
            except Exception:
                pass
            try:
                self.vbe_seed_spin.setValue(int(vbe_plan.get("base_seed", self.vbe_seed_spin.value())))
            except Exception:
                pass
            self.vbe_variability_check.setChecked(bool(vbe_plan.get("apply_variability", True)))
            reference = vbe_plan.get("reference_product")
            self.vbe_reference_product_edit.setText(reference or "")
            tests = vbe_plan.get("test_products") or []
            if isinstance(tests, list):
                self.vbe_test_products_edit.setText(", ".join(tests))
            else:
                self.vbe_test_products_edit.setText(str(tests) if tests else "")

    def _load_parameter_estimation_plan(self, estimation_plan: Mapping[str, Any]) -> None:
        self.include_baseline_checkbox.setChecked(bool(estimation_plan.get("include_baseline", True)))
        default_step = estimation_plan.get("default_relative_step")
        if default_step is not None:
            try:
                self.default_step_spin.setValue(float(default_step))
            except Exception:
                pass

        self.parameter_table.setRowCount(0)
        params = estimation_plan.get("parameters") or []
        if isinstance(params, list):
            for entry in params:
                if not isinstance(entry, Mapping):
                    continue
                try:
                    delta_value = float(entry.get("delta", self.default_step_spin.value()))
                except Exception:
                    delta_value = self.default_step_spin.value()
                self.add_parameter_row(
                    name=str(entry.get("name", entry.get("path", ""))),
                    path=str(entry.get("path", "")),
                    mode=str(entry.get("mode", "relative")),
                    delta=delta_value,
                    lower=entry.get("lower"),
                    upper=entry.get("upper"),
                )
        if self.parameter_table.rowCount() == 0:
            self.add_parameter_row()

        # Reset target controls
        self.observed_path_edit.clear()
        self.observed_label_edit.clear()
        self.observed_time_col_edit.setText("time_h")
        self.observed_value_col_edit.setText("conc_ng_ml")
        self.observed_time_unit_combo.setCurrentIndex(self.observed_time_unit_combo.findData("h"))
        self.observed_manual_data_edit.clear()
        self.timeseries_weight_spin.setValue(1.0)
        self.target_metric_combo.setCurrentIndex(0)

        self.pk_auc_checkbox.setChecked(False)
        self.pk_auc_value_spin.setValue(0.0)
        self.pk_auc_weight_spin.setValue(1.0)

        self.cfd_mmad_checkbox.setChecked(False)
        self.cfd_mmad_value_spin.setValue(0.0)
        self.cfd_mmad_weight_spin.setValue(1.0)

        self.cfd_gsd_checkbox.setChecked(False)
        self.cfd_gsd_value_spin.setValue(0.0)
        self.cfd_gsd_weight_spin.setValue(1.0)

        self.cfd_mt_checkbox.setChecked(False)
        self.cfd_mt_value_spin.setValue(0.0)
        self.cfd_mt_weight_spin.setValue(1.0)

        self.deposition_fraction_enable.setChecked(False)
        self.deposition_fraction_text.clear()
        self.deposition_fraction_weight_spin.setValue(1.0)

        targets = estimation_plan.get("targets") or []
        for entry in targets:
            if not isinstance(entry, Mapping):
                continue
            metric = entry.get("metric")
            observed = entry.get("observed") or {}
            weight = float(entry.get("weight", 1.0))
            if metric == "pk_concentration":
                path_value = observed.get("path") or ""
                self.observed_path_edit.setText(str(path_value or ""))
                self.observed_label_edit.setText(str(observed.get("label") or ""))
                self.observed_time_col_edit.setText(str(observed.get("time_column") or "time_h"))
                self.observed_value_col_edit.setText(str(observed.get("value_column") or "conc_ng_ml"))
                unit_idx = self.observed_time_unit_combo.findData(observed.get("time_unit") or "h")
                if unit_idx >= 0:
                    self.observed_time_unit_combo.setCurrentIndex(unit_idx)
                series = observed.get("series")
                if series:
                    try:
                        self.observed_manual_data_edit.setPlainText(json.dumps(series, indent=2))
                    except Exception:
                        self.observed_manual_data_edit.setPlainText("")
                self.timeseries_weight_spin.setValue(weight)
                loss = entry.get("loss")
                if loss is not None:
                    idx = self.target_metric_combo.findData(loss)
                    if idx >= 0:
                        self.target_metric_combo.setCurrentIndex(idx)
            elif metric == "pk_auc_0_t":
                self.pk_auc_checkbox.setChecked(True)
                value = observed.get("value")
                if value is not None:
                    try:
                        self.pk_auc_value_spin.setValue(float(value))
                    except Exception:
                        pass
                self.pk_auc_weight_spin.setValue(weight)
            elif metric == "cfd_mmad":
                self.cfd_mmad_checkbox.setChecked(True)
                value = observed.get("value")
                if value is not None:
                    try:
                        self.cfd_mmad_value_spin.setValue(float(value))
                    except Exception:
                        pass
                self.cfd_mmad_weight_spin.setValue(weight)
            elif metric == "cfd_gsd":
                self.cfd_gsd_checkbox.setChecked(True)
                value = observed.get("value")
                if value is not None:
                    try:
                        self.cfd_gsd_value_spin.setValue(float(value))
                    except Exception:
                        pass
                self.cfd_gsd_weight_spin.setValue(weight)
            elif metric == "cfd_mt_fraction":
                self.cfd_mt_checkbox.setChecked(True)
                value = observed.get("value")
                if value is not None:
                    try:
                        self.cfd_mt_value_spin.setValue(float(value))
                    except Exception:
                        pass
                self.cfd_mt_weight_spin.setValue(weight)
            elif metric == "deposition_fraction":
                regions = observed.get("regions") or {}
                if regions:
                    try:
                        self.deposition_fraction_text.setPlainText(json.dumps(regions, indent=2))
                    except Exception:
                        formatted = "\n".join(f"{key},{value}" for key, value in regions.items())
                        self.deposition_fraction_text.setPlainText(formatted)
                self.deposition_fraction_enable.setChecked(True)
                self.deposition_fraction_weight_spin.setValue(weight)


    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Could not parse numeric value '{text}'") from exc

    @staticmethod
    def _normalise_observed_payload(payload: Any) -> Dict[str, List[float]]:
        def _to_float_list(values: Iterable[Any], name: str) -> List[float]:
            try:
                return [float(v) for v in values]
            except Exception as exc:
                raise ValueError(f"Observed series '{name}' contains non-numeric values") from exc

        if isinstance(payload, dict):
            time_series = payload.get("time_s") or payload.get("time") or payload.get("t")
            value_series = payload.get("values") or payload.get("value") or payload.get("concentration")
            if time_series is None or value_series is None:
                raise ValueError("Observed series JSON dict must include time_s/time and values/value")
            times = _to_float_list(time_series, "time_s")
            values = _to_float_list(value_series, "values")
        elif isinstance(payload, list):
            times = []
            values = []
            for entry in payload:
                if not isinstance(entry, Mapping):
                    raise ValueError("Observed series list entries must be JSON objects")
                t_value = entry.get("time_s")
                if t_value is None:
                    t_value = entry.get("time")
                if t_value is None:
                    raise ValueError("Observed series entry missing time/time_s")
                conc_value = entry.get("value")
                if conc_value is None:
                    conc_value = entry.get("values")
                if conc_value is None:
                    conc_value = entry.get("concentration")
                if conc_value is None:
                    raise ValueError("Observed series entry missing value/concentration")
                times.append(float(t_value))
                values.append(float(conc_value))
        else:
            raise ValueError("Observed series JSON must be an object or list")

        if len(times) != len(values):
            raise ValueError("Observed series time and value lengths do not match")

        return {"time_s": times, "values": values}

    @staticmethod
    def _parse_deposition_fraction_text(text: str) -> Dict[str, float]:
        text = (text or "").strip()
        if not text:
            return {}

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        regions: Dict[str, float] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                try:
                    regions[str(key)] = float(value)
                except Exception as exc:
                    raise ValueError(f"Deposition fraction for region '{key}' is not numeric") from exc
            return regions

        lines = text.splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if "," in line:
                region, value = [part.strip() for part in line.split(",", 1)]
            elif "\t" in line:
                region, value = [part.strip() for part in line.split("\t", 1)]
            else:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError("Deposition fraction lines must be 'region,value'")
                region, value = parts
            if not region:
                raise ValueError("Deposition fraction region name is required")
            try:
                regions[region] = float(value)
            except Exception as exc:
                raise ValueError(f"Deposition fraction for region '{region}' is not numeric") from exc

        return regions

    def build_parameter_estimation_plan(self) -> Dict[str, Any]:
        parameters: List[Dict[str, Any]] = []
        for row in range(self.parameter_table.rowCount()):
            name_widget = self.parameter_table.cellWidget(row, 0)
            path_widget = self.parameter_table.cellWidget(row, 1)
            mode_widget = self.parameter_table.cellWidget(row, 2)
            step_widget = self.parameter_table.cellWidget(row, 3)
            lower_widget = self.parameter_table.cellWidget(row, 4)
            upper_widget = self.parameter_table.cellWidget(row, 5)

            if not isinstance(path_widget, QLineEdit) or not isinstance(step_widget, QDoubleSpinBox):
                continue

            path_value = path_widget.text().strip()
            if not path_value:
                raise ValueError("Parameter rows must include a config path")

            name_value = path_value
            if isinstance(name_widget, QLineEdit):
                candidate = name_widget.text().strip()
                if candidate:
                    name_value = candidate

            mode_value = "relative"
            if isinstance(mode_widget, QComboBox):
                data = mode_widget.currentData()
                if data in {"relative", "absolute"}:
                    mode_value = data

            delta_value = float(step_widget.value())
            if delta_value <= 0:
                raise ValueError(f"Parameter '{name_value}' must have a positive step size")

            lower_value = None
            if isinstance(lower_widget, QLineEdit):
                lower_value = self._parse_float(lower_widget.text())

            upper_value = None
            if isinstance(upper_widget, QLineEdit):
                upper_value = self._parse_float(upper_widget.text())

            param_entry: Dict[str, Any] = {
                "name": name_value,
                "path": path_value,
                "mode": mode_value,
                "delta": delta_value,
            }
            if lower_value is not None:
                param_entry["lower"] = lower_value
            if upper_value is not None:
                param_entry["upper"] = upper_value
            if lower_value is not None and upper_value is not None and lower_value > upper_value:
                raise ValueError(f"Lower bound exceeds upper bound for '{name_value}'")

            parameters.append(param_entry)

        if not parameters:
            raise ValueError("Add at least one parameter to run parameter estimation")

        targets: List[Dict[str, Any]] = []

        observed_csv = self.observed_path_edit.text().strip()
        manual_text = self.observed_manual_data_edit.toPlainText().strip()
        manual_series = None
        if manual_text:
            try:
                manual_payload = json.loads(manual_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Observed series JSON invalid: {exc}") from exc
            manual_series = self._normalise_observed_payload(manual_payload)

        if observed_csv or manual_series is not None:
            observed_spec: Dict[str, Any] = {
                "path": observed_csv or None,
                "time_column": self.observed_time_col_edit.text().strip() or "time_h",
                "value_column": self.observed_value_col_edit.text().strip() or "conc",
                "time_unit": self.observed_time_unit_combo.currentData() or "h",
            }
            label_text = self.observed_label_edit.text().strip()
            if label_text:
                observed_spec["label"] = label_text
            if manual_series is not None:
                observed_spec["series"] = manual_series

            targets.append({
                "metric": "pk_concentration",
                "weight": float(self.timeseries_weight_spin.value()),
                "loss": self.target_metric_combo.currentData() or "sse",
                "observed": observed_spec,
            })

        if self.pk_auc_checkbox.isChecked():
            targets.append({
                "metric": "pk_auc_0_t",
                "weight": float(self.pk_auc_weight_spin.value()),
                "observed": {"value": float(self.pk_auc_value_spin.value())},
            })

        if self.cfd_mmad_checkbox.isChecked():
            targets.append({
                "metric": "cfd_mmad",
                "weight": float(self.cfd_mmad_weight_spin.value()),
                "observed": {"value": float(self.cfd_mmad_value_spin.value())},
            })

        if self.cfd_gsd_checkbox.isChecked():
            targets.append({
                "metric": "cfd_gsd",
                "weight": float(self.cfd_gsd_weight_spin.value()),
                "observed": {"value": float(self.cfd_gsd_value_spin.value())},
            })

        if self.cfd_mt_checkbox.isChecked():
            targets.append({
                "metric": "cfd_mt_fraction",
                "weight": float(self.cfd_mt_weight_spin.value()),
                "observed": {"value": float(self.cfd_mt_value_spin.value())},
            })

        if self.deposition_fraction_enable.isChecked():
            regions = self._parse_deposition_fraction_text(self.deposition_fraction_text.toPlainText())
            if not regions:
                raise ValueError("Provide at least one deposition fraction when enabled")
            targets.append({
                "metric": "deposition_fraction",
                "weight": float(self.deposition_fraction_weight_spin.value()),
                "observed": {"regions": regions},
            })

        if not targets:
            raise ValueError("Add at least one estimation target (timeseries, scalar, or deposition fraction)")

        plan = {
            "parameters": parameters,
            "include_baseline": self.include_baseline_checkbox.isChecked(),
            "default_relative_step": float(self.default_step_spin.value()),
            "targets": targets,
        }

        return plan

    @staticmethod
    def _parse_product_list(text: str) -> List[str]:
        if not text:
            return []
        items = [item.strip() for item in text.replace("\n", ",").split(",")]
        products = [item for item in items if item]
        seen = set()
        ordered: List[str] = []
        for product in products:
            key = product.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(product)
        return ordered

    def build_virtual_trial_plan(self) -> Dict[str, Any]:
        return {
            "n_subjects": self.virtual_trial_subjects_spin.value(),
            "base_seed": self.virtual_trial_seed_spin.value(),
            "apply_variability": self.virtual_trial_variability_check.isChecked(),
            "additional_products": self._parse_product_list(self.virtual_trial_products_edit.text()),
        }

    def build_virtual_bioequivalence_plan(self) -> Dict[str, Any]:
        return {
            "n_subjects": self.vbe_subjects_spin.value(),
            "base_seed": self.vbe_seed_spin.value(),
            "apply_variability": self.vbe_variability_check.isChecked(),
            "reference_product": self.vbe_reference_product_edit.text().strip() or None,
            "test_products": self._parse_product_list(self.vbe_test_products_edit.text()),
        }

    @staticmethod
    def _build_manifest_parameter_map(parameters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        manifest: Dict[str, Dict[str, Any]] = {}
        for entry in parameters:
            path = entry.get("path")
            if not path:
                continue
            manifest[path] = {
                "delta": float(entry.get("delta", 0.1)),
                "mode": entry.get("mode", "relative"),
            }
        return manifest

    def build_run_plan(self) -> Dict[str, Any]:
        plan: Dict[str, Any] = {
            "run_type": self.current_run_type(),
            "run_label": self.get_run_label(),
            "config_name": self.current_config_name(),
        }

        if plan["run_type"] == "sweep":
            plan["sweep_parameters"] = self.get_sweep_parameters() if self.sweep_enabled.isChecked() else {}
        elif plan["run_type"] == "parameter_estimation":
            plan["estimation"] = self.build_parameter_estimation_plan()
        elif plan["run_type"] == "virtual_trial":
            plan["virtual_trial"] = self.build_virtual_trial_plan()
        elif plan["run_type"] == "virtual_bioequivalence":
            plan["virtual_bioequivalence"] = self.build_virtual_bioequivalence_plan()

        return plan

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]):
        """Assign workspace manager for saving configurations."""
        self.workspace_manager = workspace_manager
        self.reload_workspace_catalog_entries()

    def build_config(self) -> Dict[str, Any]:
        """Build configuration dictionary from form inputs."""
        config = {
            "run": {
                "stages": [s.strip() for s in self.stages_edit.text().split(",")],
                "seed": self.seed_spin.value(),
                "threads": 1,
                "enable_numba": False,
                "artifact_dir": "results"
            },
            "deposition": {
                "model": self.deposition_model_combo.currentText(),
                "particle_grid": "medium"
            },
            "pbbm": {
                "model": self.pbbm_model_combo.currentText(),
                "epi_layers": [2, 2, 1, 1]
            },
            "pk": {
                "model": self.pk_model_combo.currentText()
            }
        }

        config["subject"] = self._entity_config_payload("subject")
        config["api"] = self._entity_config_payload("api")
        config["product"] = self._entity_config_payload("product")
        config["maneuver"] = self._entity_config_payload("maneuver")

        return config

    def build_app_config(self):
        """Convert current form into AppConfig model."""
        if not CONFIG_MODEL_AVAILABLE or AppConfig is None:
            raise RuntimeError("LMP configuration model is unavailable")
        return AppConfig.model_validate(self.build_config())

    def validate_config(self):
        """Validate the current configuration."""
        try:
            app_config = self.build_app_config()
            if app_api is not None:
                app_api.validate_configuration(app_config)

            QMessageBox.information(self, "Validation", "Configuration validated successfully.")
            self.current_config = app_config.model_dump()
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Configuration error: {str(e)}")

    def save_config(self):
        """Save configuration to workspace."""
        try:
            if self.workspace_manager is None:
                QMessageBox.warning(self, "No Workspace", "Please select a workspace before saving configurations.")
                return

            app_config = self.build_app_config()
            if app_api is not None:
                app_api.validate_configuration(app_config)

            if check_catalog_coverage is not None:
                coverage = check_catalog_coverage(app_config)
                missing = [category for category, available in coverage.items() if not available]
                if missing:
                    categories = ", ".join(missing)
                    QMessageBox.critical(
                        self,
                        "Missing Catalog Entries",
                        f"The following selections are not available in the catalog: {categories}"
                    )
                    return

            try:
                run_plan = self.build_run_plan()
            except json.JSONDecodeError:
                QMessageBox.critical(self, "Sweep Parameters", "Invalid JSON in sweep parameters")
                return
            except ValueError as exc:
                QMessageBox.critical(self, "Run Plan", str(exc))
                return

            if run_plan["run_type"] == "sweep" and not run_plan.get("sweep_parameters"):
                QMessageBox.warning(
                    self,
                    "Sweep Parameters",
                    "Provide at least one sweep axis to execute a parameter sweep.",
                )
                return

            config_name = self.study_name_edit.text().strip()
            safe_name = None
            if config_name:
                sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in config_name)
                safe_name = sanitized or None
            config_path = self.workspace_manager.save_config(
                app_config.model_dump(),
                name=safe_name,
                run_plan=run_plan,
            )

            self.current_config = app_config.model_dump()
            self.last_saved_config_path = str(config_path)

            if not run_plan.get("run_label") and config_name:
                run_plan["run_label"] = config_name

            self.config_ready.emit({
                "config": self.current_config,
                "config_path": str(config_path),
                "run_plan": run_plan,
            })

            QMessageBox.information(self, "Save Config", f"Configuration saved to {config_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save: {str(e)}")

    def preview_manifest(self):
        """Preview simulation manifest."""
        try:
            if app_api is None:
                QMessageBox.warning(self, "Catalog Unavailable", "app_api is not available to plan manifests.")
                return

            app_config = self.build_app_config()

            sweep_params = {}
            if self.current_run_type() == "sweep" and self.sweep_enabled.isChecked():
                sweep_params = self.get_sweep_parameters()

            manifest = None
            run_type = self.current_run_type()
            if run_type == "sweep":
                manifest = app_api.plan_simulation_manifest(app_config, sweep_params)
                run_count = len(manifest) if hasattr(manifest, "__len__") else 1
                message = f"Sweep will generate {run_count} run(s)"
            elif run_type == "parameter_estimation":
                estimation_plan = self.build_parameter_estimation_plan()
                param_map = self._build_manifest_parameter_map(estimation_plan.get("parameters", []))
                manifest = app_api.plan_parameter_estimation_runs(
                    app_config,
                    param_map,
                    include_baseline=estimation_plan.get("include_baseline", True),
                    default_relative_step=estimation_plan.get("default_relative_step", 0.1),
                )
                run_count = len(manifest) if hasattr(manifest, "__len__") else 1
                message = f"Parameter estimation will queue {run_count} run(s)"
            else:
                manifest = app_api.plan_simulation_manifest(app_config, {})
                message = "Single run configuration"

            QMessageBox.information(self, "Manifest Preview", message)

        except json.JSONDecodeError:
            QMessageBox.critical(self, "JSON Error", "Invalid JSON in sweep parameters")
        except ValueError as exc:
            QMessageBox.critical(self, "Run Plan", str(exc))
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Could not preview: {str(e)}")

    def update_from_catalog(self, category: str, data: Dict[str, Any]):
        """Update local catalog selection based on saved entry."""
        if category not in self.saved_catalog_entries:
            return

        entry_id = data.get("id") or data.get("ref")
        if not entry_id:
            QMessageBox.warning(
                self,
                "Catalog",
                f"Could not determine identifier for {category} selection.",
            )
            return

        entries = self.saved_catalog_entries.get(category, [])
        if data.get("deleted"):
            entries = [
                entry for entry in entries
                if self._entry_identifier(entry) != entry_id
            ]
            self.saved_catalog_entries[category] = entries
            self._populate_combo_from_saved(category, entries)
            friendly = self.entity_display.get(category, category)
            display_name = data.get("name") or data.get("ref") or entry_id
            QMessageBox.information(
                self,
                "Removed",
                f"Removed saved {friendly} '{display_name}'.",
            )
            return

        entries = self.saved_catalog_entries.get(category, [])
        replaced = False
        for idx, entry in enumerate(entries):
            if self._entry_identifier(entry) == entry_id:
                entries[idx] = data
                replaced = True
                break
        if not replaced:
            entries.append(data)
        self.saved_catalog_entries[category] = entries

        self._populate_combo_from_saved(category, entries, select_id=entry_id)

        display_name = data.get("name") or data.get("ref") or entry_id
        friendly = self.entity_display.get(category, category)
        QMessageBox.information(self, "Updated", f"{friendly.capitalize()} set to: {display_name}")


class RunQueueTab(QWidget):
    """Run Queue management tab."""

    run_completed = Signal(str, str)  # run_id, status

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.process_manager: Optional[ProcessManager] = None
        self.selected_config_path: Optional[str] = None
        self.run_rows: Dict[str, int] = {}
        self.run_errors: Dict[str, str] = {}
        self.run_logs: Dict[str, List[str]] = {}
        self.current_log_run_id: Optional[str] = None
        self.pending_run_plan: Optional[Dict[str, Any]] = None
        self.last_started_run_ids: List[str] = []
        self.saved_config_info: Dict[str, Dict[str, Any]] = {}
        self.init_ui()

    @staticmethod
    def _format_size(size_bytes: Optional[int]) -> str:
        if size_bytes is None:
            return ""
        try:
            size = float(size_bytes)
        except (TypeError, ValueError):
            return ""
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        return f"{size:.1f} {units[unit_index]}"

    def init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Run Queue")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        controls_layout = QHBoxLayout()

        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config)

        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(220)
        self.config_combo.addItem("Select saved config...", None)
        self.config_combo.currentIndexChanged.connect(self.on_saved_config_selected)

        start_run_btn = QPushButton("Start Run")
        start_run_btn.clicked.connect(self.start_run)

        clear_queue_btn = QPushButton("Clear Completed")
        clear_queue_btn.clicked.connect(self.clear_completed)

        controls_layout.addWidget(load_config_btn)
        controls_layout.addWidget(self.config_combo)
        controls_layout.addWidget(start_run_btn)
        controls_layout.addWidget(clear_queue_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.config_path_label = QLabel("No configuration selected")
        self.config_path_label.setStyleSheet("color: gray;")
        self.config_path_label.setWordWrap(True)
        layout.addWidget(self.config_path_label)

        configs_container = QHBoxLayout()
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(4)
        self.config_table.setHorizontalHeaderLabels(["Name", "Created", "Size", "Path"])
        self.config_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.config_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.config_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.config_table.verticalHeader().setVisible(False)
        self.config_table.horizontalHeader().setStretchLastSection(True)
        self.config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.config_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.config_table.itemSelectionChanged.connect(self.on_config_table_selection)
        self.config_table.itemDoubleClicked.connect(self.on_config_table_double_clicked)
        configs_container.addWidget(self.config_table)

        config_actions = QVBoxLayout()
        self.config_refresh_btn = QPushButton("Refresh List")
        self.config_refresh_btn.clicked.connect(self.refresh_config_list)
        config_actions.addWidget(self.config_refresh_btn)

        self.config_queue_btn = QPushButton("Queue Selected")
        self.config_queue_btn.clicked.connect(self.queue_selected_configs)
        config_actions.addWidget(self.config_queue_btn)

        config_actions.addStretch()
        configs_container.addLayout(config_actions)
        layout.addLayout(configs_container)

        self.run_table = QTableWidget()
        self.run_table.setColumnCount(7)
        self.run_table.setHorizontalHeaderLabels([
            "Run ID", "Status", "Progress", "Elapsed", "ETA", "Actions", "Details"
        ])
        self.run_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_table.itemSelectionChanged.connect(self.update_log_view)
        self.run_table.horizontalHeader().setStretchLastSection(True)
        self.run_table.verticalHeader().setVisible(False)
        layout.addWidget(self.run_table)

        log_label = QLabel("Run Logs")
        log_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(log_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Log output will appear here for the selected run.")
        self.log_view.setMinimumHeight(160)
        layout.addWidget(self.log_view)

        self.setLayout(layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_run_status)
        self.update_timer.start(1000)

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]):
        """Configure workspace manager reference."""
        self.workspace_manager = workspace_manager
        self.set_selected_config(None)
        self.run_table.setRowCount(0)
        self.run_rows.clear()
        self.run_errors.clear()
        self.run_logs.clear()
        self.current_log_run_id = None
        if hasattr(self, 'log_view'):
            self.log_view.clear()
            self.log_view.setPlaceholderText("Log output will appear here for the selected run.")
        if hasattr(self, 'config_table'):
            self.config_table.setRowCount(0)
        self.refresh_config_list()

    def set_process_manager(self, process_manager: Optional[ProcessManager]):
        """Attach process manager signals."""
        if process_manager is self.process_manager:
            return

        if self.process_manager is not None:
            try:
                self.process_manager.process_started.disconnect(self.on_process_started)
                self.process_manager.process_progress.disconnect(self.on_process_progress)
                self.process_manager.process_error.disconnect(self.on_process_error)
                self.process_manager.process_metric.disconnect(self.on_process_metric)
                self.process_manager.process_finished.disconnect(self.on_process_finished)
                self.process_manager.process_log.disconnect(self.on_process_log)
            except TypeError:
                pass

        self.process_manager = process_manager
        if process_manager is None:
            return

        process_manager.process_started.connect(self.on_process_started)
        process_manager.process_progress.connect(self.on_process_progress)
        process_manager.process_error.connect(self.on_process_error)
        process_manager.process_metric.connect(self.on_process_metric)
        process_manager.process_finished.connect(self.on_process_finished)
        process_manager.process_log.connect(self.on_process_log)

    def load_config(self):
        """Load configuration file."""
        start_dir = str(self.workspace_manager.configs_dir) if self.workspace_manager else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            start_dir,
            "JSON files (*.json);;TOML files (*.toml)"
        )
        if file_path:
            self.set_selected_config(file_path)

    def _default_run_plan(self) -> Dict[str, Any]:
        return {"run_type": "single", "run_label": None}

    def set_selected_config(self, config_path: Optional[str], run_plan: Optional[Dict[str, Any]] = None):
        """Set the selected configuration path for runs."""
        if not config_path:
            self.selected_config_path = None
            self.config_path_label.setText("No configuration selected")
            self.config_path_label.setToolTip("")
            if hasattr(self, 'config_combo'):
                self.config_combo.blockSignals(True)
                self.config_combo.setCurrentIndex(0)
                self.config_combo.blockSignals(False)
            if hasattr(self, 'config_table'):
                self.config_table.blockSignals(True)
                self.config_table.clearSelection()
                self.config_table.blockSignals(False)
            self.pending_run_plan = None if run_plan is None else run_plan
            return

        path_obj = Path(config_path)
        self.selected_config_path = config_path
        self.config_path_label.setText(f"Selected config: {path_obj.name}")
        self.config_path_label.setToolTip(config_path)
        if hasattr(self, 'config_combo'):
            idx = self.config_combo.findData(config_path)
            self.config_combo.blockSignals(True)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)
            else:
                self.config_combo.setCurrentIndex(0)
            self.config_combo.blockSignals(False)
        if hasattr(self, 'config_table'):
            self.config_table.blockSignals(True)
            matched = False
            for row in range(self.config_table.rowCount()):
                item = self.config_table.item(row, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == config_path:
                    self.config_table.selectRow(row)
                    matched = True
                    break
            if not matched:
                self.config_table.clearSelection()
            self.config_table.blockSignals(False)

        detected_run_plan = run_plan
        if detected_run_plan is None:
            try:
                with open(path_obj, "r") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    meta = payload.get("metadata")
                    if isinstance(meta, dict):
                        detected_run_plan = meta.get("run_plan")
            except Exception:
                detected_run_plan = None

        if detected_run_plan is not None:
            self.pending_run_plan = detected_run_plan
        elif self.pending_run_plan is None:
            self.pending_run_plan = self._default_run_plan()

        if self.pending_run_plan is not None:
            try:
                self._apply_run_plan_to_ui(self.pending_run_plan)
            except Exception:
                pass

    def _load_config_data(self, config_path: str) -> Dict[str, Any]:
        path_obj = Path(config_path)
        if path_obj.exists():
            with open(path_obj, "r") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "config" in data and "metadata" in data:
                return data["config"]
            return data
        if self.workspace_manager is not None:
            return self.workspace_manager.load_config(path_obj.name)
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    @staticmethod
    def _coerce_json_value(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _geometric_mean(series: pd.Series) -> float:
        valid = series[series > 0]
        if valid.empty:
            return float('nan')
        return float(np.exp(np.log(valid).mean()))

    def refresh_config_list(self):
        configs: List[Dict[str, Any]] = []
        if self.workspace_manager is not None:
            configs = self.workspace_manager.list_configs()
        self.saved_config_info = {}
        for cfg in configs:
            path_value = cfg.get("path")
            if path_value:
                self.saved_config_info[path_value] = cfg

        selected_path = self.selected_config_path

        if hasattr(self, 'config_combo'):
            self.config_combo.blockSignals(True)
            self.config_combo.clear()
            self.config_combo.addItem("Select saved config...", None)
            for cfg in configs:
                path_value = cfg.get("path")
                label = cfg.get("name") or (Path(path_value).name if path_value else "config")
                created = cfg.get("created")
                if created:
                    label = f"{label} ({created})"
                self.config_combo.addItem(label, path_value)
            self.config_combo.setEnabled(bool(configs))
            if selected_path:
                idx = self.config_combo.findData(selected_path)
                if idx >= 0:
                    self.config_combo.setCurrentIndex(idx)
            self.config_combo.blockSignals(False)

        if hasattr(self, 'config_table'):
            self.config_table.blockSignals(True)
            self.config_table.setRowCount(len(configs))
            selected_row = None
            for row, cfg in enumerate(configs):
                path_value = cfg.get("path", "")
                name = cfg.get("name") or (Path(path_value).name if path_value else "config")
                created = cfg.get("created") or ""
                size_text = self._format_size(cfg.get("size_bytes"))

                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.ItemDataRole.UserRole, path_value)
                name_item.setToolTip(path_value)
                created_item = QTableWidgetItem(created)
                size_item = QTableWidgetItem(size_text)
                path_item = QTableWidgetItem(path_value)
                path_item.setToolTip(path_value)

                self.config_table.setItem(row, 0, name_item)
                self.config_table.setItem(row, 1, created_item)
                self.config_table.setItem(row, 2, size_item)
                self.config_table.setItem(row, 3, path_item)

                if selected_path and path_value == selected_path and selected_row is None:
                    selected_row = row

            if selected_row is not None:
                self.config_table.selectRow(selected_row)
            else:
                self.config_table.clearSelection()
            self.config_table.blockSignals(False)

            # Update selection label manually when we restore selection
            if selected_row is not None:
                self.on_config_table_selection()

            if hasattr(self, 'config_queue_btn'):
                self.config_queue_btn.setEnabled(bool(configs))

        if self.workspace_manager is None and hasattr(self, 'config_queue_btn'):
            self.config_queue_btn.setEnabled(False)

    def on_saved_config_selected(self, index: int):
        if not hasattr(self, 'config_combo'):
            return
        path = self.config_combo.itemData(index) if index >= 0 else None
        if path:
            run_plan = None
            info = self.saved_config_info.get(path)
            if info is not None:
                run_plan = info.get("run_plan")
            self.set_selected_config(path, run_plan=run_plan)

    def start_run(self):
        """Start a new simulation run using the selected configuration."""
        if not self.selected_config_path:
            QMessageBox.warning(self, "No Config", "Load or save a configuration before starting a run.")
            return

        self.start_run_for_path(self.selected_config_path, show_message=True)

    def start_run_for_path(self, config_path: str, show_message: bool = True) -> Optional[str]:
        if self.process_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before starting runs.")
            return None

        self.selected_config_path = config_path
        plan = self.pending_run_plan or self._default_run_plan()
        run_type = plan.get("run_type", "single")
        run_label = plan.get("run_label")
        self.last_started_run_ids = []

        try:
            if run_type != "single":
                if run_type == "sweep":
                    run_ids = self.start_parameter_sweep(config_path, plan, show_message=show_message)
                    self.last_started_run_ids = run_ids
                    return run_ids[0] if run_ids else None
                if run_type == "parameter_estimation":
                    run_ids = self.start_parameter_estimation(config_path, plan, show_message=show_message)
                    self.last_started_run_ids = run_ids
                    return run_ids[0] if run_ids else None
                if run_type == "virtual_trial":
                    run_ids = self.start_virtual_trial(config_path, plan, show_message=show_message)
                    self.last_started_run_ids = run_ids
                    return run_ids[0] if run_ids else None
                if run_type == "virtual_bioequivalence":
                    run_ids = self.start_virtual_bioequivalence(config_path, plan, show_message=show_message)
                    self.last_started_run_ids = run_ids
                    return run_ids[0] if run_ids else None
                QMessageBox.information(
                    self,
                    "Run Type Not Yet Supported",
                    f"Run type '{run_type}' will be supported in an upcoming update.",
                )
                return None

            run_id = self.process_manager.start_simulation(
                config_path,
                run_type=run_type,
                label=run_label,
                metadata={"run_plan": plan},
            )
            self.last_started_run_ids = [run_id]
        except Exception as exc:
            QMessageBox.critical(self, "Run Error", f"Failed to start run: {exc}")
            return None

        self.run_logs[run_id] = []
        self.current_log_run_id = run_id
        self.add_run_row(run_id)
        if show_message:
            QMessageBox.information(self, "Run Started", f"Started run: {run_id}")
        self.pending_run_plan = self._default_run_plan()
        return run_id

    def start_parameter_sweep(self, config_path: str, plan: Dict[str, Any], show_message: bool = True) -> List[str]:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before starting runs.")
            return []

        try:
            config_data = self._load_config_data(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Sweep Error", f"Could not load config: {exc}")
            return []

        try:
            base_config = AppConfig.model_validate(config_data) if CONFIG_MODEL_AVAILABLE and AppConfig else None
        except Exception as exc:
            QMessageBox.critical(self, "Sweep Error", f"Configuration invalid: {exc}")
            return []

        if base_config is None:
            QMessageBox.critical(self, "Sweep Error", "Configuration model is unavailable for planning sweeps.")
            return []

        sweep_axes = plan.get("sweep_parameters") or {}
        if isinstance(sweep_axes, str):
            try:
                sweep_axes = json.loads(sweep_axes)
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Sweep Error", f"Invalid sweep parameters: {exc}")
                return []

        if not isinstance(sweep_axes, dict) or not sweep_axes:
            QMessageBox.warning(self, "Sweep Parameters", "No sweep parameters provided.")
            return []

        if app_api is None:
            QMessageBox.critical(self, "Sweep Error", "Simulation API is unavailable.")
            return []

        try:
            manifest = app_api.plan_simulation_manifest(base_config, sweep_axes)
        except Exception as exc:
            QMessageBox.critical(self, "Sweep Error", f"Failed to plan manifest: {exc}")
            return []

        if manifest.empty:
            QMessageBox.information(self, "Sweep", "Sweep manifest is empty; nothing to queue.")
            return []

        total_runs = len(manifest)
        label_base = plan.get("run_label") or Path(config_path).stem
        parent_run_id = self.workspace_manager.generate_run_id(prefix="sweep", label=label_base)
        parent_run_dir = self.workspace_manager.runs_dir / parent_run_id

        child_run_ids: List[str] = []
        completed_runs = 0
        if not parent_run_dir.exists():
            parent_metadata = {
                "run_plan": plan,
                "sweep_parameters": sweep_axes,
                "child_run_ids": [],
                "total_runs": total_runs,
                "config_path": config_path,
                "status": "group",
            }
            self.workspace_manager.create_run_directory(
                parent_run_id,
                run_type="sweep_group",
                label=label_base,
                parent_run_id=None,
                request_metadata=parent_metadata,
            )
            self.workspace_manager.update_run_status(
                parent_run_id,
                "group",
                child_run_ids=[],
                total_runs=total_runs,
                completed_runs=0,
            )
        else:
            existing_info = self.workspace_manager.get_run_info(parent_run_id) or {}
            existing_children = existing_info.get("child_run_ids") or []
            for child_id in existing_children:
                if child_id and str(child_id) not in child_run_ids:
                    child_run_ids.append(str(child_id))
            try:
                completed_runs = int(existing_info.get("completed_runs", 0) or 0)
            except Exception:
                completed_runs = 0

        queued_run_ids: List[str] = []
        for idx, row in manifest.iterrows():
            overrides = {}
            for param in sweep_axes.keys():
                if param in row:
                    overrides[param] = self._coerce_json_value(row[param])

            iteration_label = f"{label_base} #{idx + 1}"
            metadata = {
                "run_plan": plan,
                "parent_run_id": parent_run_id,
                "sweep_overrides": overrides,
                "manifest_row": int(idx),
                "manifest_run_id": self._coerce_json_value(row.get("run_id")) if "run_id" in row else None,
            }

            try:
                run_id = self.process_manager.start_simulation(
                    config_path,
                    run_type="sweep",
                    label=iteration_label,
                    parent_run_id=parent_run_id,
                    manifest_index=int(idx),
                    total_runs=total_runs,
                    metadata=metadata,
                    parameter_overrides=overrides,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Sweep Error", f"Failed to queue run #{idx + 1}: {exc}")
                continue

            self.run_logs[run_id] = []
            self.current_log_run_id = run_id
            self.add_run_row(run_id)
            queued_run_ids.append(run_id)
            if run_id not in child_run_ids:
                child_run_ids.append(run_id)
            self.workspace_manager.update_run_status(
                parent_run_id,
                "group",
                child_run_ids=child_run_ids,
                total_runs=total_runs,
                completed_runs=completed_runs,
            )

        if queued_run_ids and show_message:
            preview = ", ".join(queued_run_ids[:3])
            if len(queued_run_ids) > 3:
                preview += ", …"
            QMessageBox.information(
                self,
                "Sweep Queued",
                f"Queued {len(queued_run_ids)} sweep run(s) under parent {parent_run_id}: {preview}",
            )

        self.pending_run_plan = self._default_run_plan()
        return queued_run_ids

    def _load_app_config(self, config_path: str) -> Optional[AppConfig]:
        try:
            config_data = self._load_config_data(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", f"Could not load config: {exc}")
            return None

        try:
            base_config = AppConfig.model_validate(config_data) if CONFIG_MODEL_AVAILABLE and AppConfig else None
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", f"Configuration invalid: {exc}")
            return None

        if base_config is None:
            QMessageBox.critical(self, "Config Error", "Configuration model is unavailable.")
            return None
        return base_config

    def start_virtual_trial(self, config_path: str, plan: Dict[str, Any], show_message: bool = True) -> List[str]:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before starting runs.")
            return []

        base_config = self._load_app_config(config_path)
        if base_config is None:
            return []

        virtual_plan = plan.get("virtual_trial") or {}
        n_subjects = int(virtual_plan.get("n_subjects", 0))
        if n_subjects <= 0:
            QMessageBox.warning(self, "Virtual Trial", "Number of subjects must be greater than zero.")
            return []

        additional_products = virtual_plan.get("additional_products") or []
        base_product = base_config.product.ref if hasattr(base_config, "product") else None
        all_products = [base_product] if base_product else []
        for product in additional_products:
            if product and product not in all_products:
                all_products.append(product)

        try:
            tasks = app_api.plan_virtual_trial_tasks(
                base_config,
                n_subjects=n_subjects,
                base_seed=int(virtual_plan.get("base_seed", 1234)),
                products=all_products,
                apply_variability=bool(virtual_plan.get("apply_variability", True)),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Virtual Trial", f"Failed to plan virtual trial: {exc}")
            return []

        total_runs = len(tasks)
        if total_runs == 0:
            QMessageBox.information(self, "Virtual Trial", "No subjects to queue.")
            return []

        label_base = plan.get("run_label") or Path(config_path).stem
        parent_run_id = self.workspace_manager.generate_run_id(prefix="vt", label=label_base)
        parent_run_dir = self.workspace_manager.runs_dir / parent_run_id

        trial_settings = {
            "plan": virtual_plan,
            "tasks": tasks,
            "config_path": config_path,
            "total_runs": total_runs,
            "products": all_products,
        }

        if not parent_run_dir.exists():
            self.workspace_manager.create_run_directory(
                parent_run_id,
                run_type="virtual_trial_group",
                label=label_base,
                parent_run_id=None,
                request_metadata={
                    "run_plan": plan,
                    "virtual_trial_settings": trial_settings,
                    "child_run_ids": [],
                    "completed_runs": 0,
                },
            )

        child_run_ids: List[str] = []
        for index, task_spec in enumerate(tasks):
            subject_label = f"Subject {task_spec.get('subject_index', index) + 1}"
            try:
                run_id = self.process_manager.start_simulation(
                    config_path,
                    run_type="virtual_trial",
                    label=f"{label_base} · {subject_label}",
                    parent_run_id=parent_run_id,
                    manifest_index=index,
                    total_runs=total_runs,
                    metadata={
                        "run_plan": plan,
                        "virtual_trial_task": task_spec,
                    },
                    task_spec=task_spec,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Virtual Trial", f"Failed to queue subject {index + 1}: {exc}")
                continue

            child_run_ids.append(run_id)
            self.run_logs[run_id] = []
            self.current_log_run_id = run_id
            self.add_run_row(run_id)

        self.workspace_manager.update_run_status(
            parent_run_id,
            "group",
            child_run_ids=child_run_ids,
            total_runs=total_runs,
            completed_runs=0,
            virtual_trial_settings=trial_settings,
        )

        if child_run_ids and show_message:
            preview = ", ".join(child_run_ids[:3])
            if len(child_run_ids) > 3:
                preview += ", …"
            QMessageBox.information(
                self,
                "Virtual Trial Queued",
                f"Queued {len(child_run_ids)} virtual trial run(s) under parent {parent_run_id}: {preview}",
            )

        return child_run_ids

    def start_virtual_bioequivalence(self, config_path: str, plan: Dict[str, Any], show_message: bool = True) -> List[str]:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before starting runs.")
            return []

        base_config = self._load_app_config(config_path)
        if base_config is None:
            return []

        vbe_plan = plan.get("virtual_bioequivalence") or {}
        n_subjects = int(vbe_plan.get("n_subjects", 0))
        if n_subjects <= 0:
            QMessageBox.warning(self, "Virtual Bioequivalence", "Number of subjects must be greater than zero.")
            return []

        test_products = vbe_plan.get("test_products") or []
        if not test_products:
            QMessageBox.warning(self, "Virtual Bioequivalence", "Enter at least one test product.")
            return []

        reference_product = vbe_plan.get("reference_product") or base_config.product.ref

        try:
            tasks, study_metadata = app_api.plan_virtual_bioequivalence_tasks(
                base_config,
                n_subjects=n_subjects,
                test_products=test_products,
                reference_product=reference_product,
                base_seed=int(vbe_plan.get("base_seed", 1234)),
                apply_variability=bool(vbe_plan.get("apply_variability", True)),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Virtual Bioequivalence", f"Failed to plan VBE tasks: {exc}")
            return []

        total_runs = len(tasks)
        label_base = plan.get("run_label") or Path(config_path).stem
        parent_run_id = self.workspace_manager.generate_run_id(prefix="vbe", label=label_base)
        parent_run_dir = self.workspace_manager.runs_dir / parent_run_id

        vbe_settings = {
            "plan": vbe_plan,
            "tasks": tasks,
            "config_path": config_path,
            "study_metadata": study_metadata,
            "total_runs": total_runs,
        }

        if not parent_run_dir.exists():
            self.workspace_manager.create_run_directory(
                parent_run_id,
                run_type="virtual_bioequivalence_group",
                label=label_base,
                parent_run_id=None,
                request_metadata={
                    "run_plan": plan,
                    "virtual_bioequivalence_settings": vbe_settings,
                    "child_run_ids": [],
                    "completed_runs": 0,
                },
            )

        child_run_ids: List[str] = []
        for index, task_spec in enumerate(tasks):
            subject_label = f"Subject {task_spec.get('subject_index', index) + 1}"
            try:
                run_id = self.process_manager.start_simulation(
                    config_path,
                    run_type="virtual_bioequivalence",
                    label=f"{label_base} · {subject_label}",
                    parent_run_id=parent_run_id,
                    manifest_index=index,
                    total_runs=total_runs,
                    metadata={
                        "run_plan": plan,
                        "virtual_bioequivalence_task": task_spec,
                    },
                    task_spec=task_spec,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Virtual Bioequivalence", f"Failed to queue subject {index + 1}: {exc}")
                continue

            child_run_ids.append(run_id)
            self.run_logs[run_id] = []
            self.current_log_run_id = run_id
            self.add_run_row(run_id)

        self.workspace_manager.update_run_status(
            parent_run_id,
            "group",
            child_run_ids=child_run_ids,
            total_runs=total_runs,
            completed_runs=0,
            virtual_bioequivalence_settings=vbe_settings,
        )

        if child_run_ids and show_message:
            preview = ", ".join(child_run_ids[:3])
            if len(child_run_ids) > 3:
                preview += ", …"
            QMessageBox.information(
                self,
                "Virtual Bioequivalence Queued",
                f"Queued {len(child_run_ids)} VBE run(s) under parent {parent_run_id}: {preview}",
            )

        return child_run_ids

    @staticmethod
    def _parameter_entries_to_manifest(parameters: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        manifest: Dict[str, Dict[str, Any]] = {}
        for entry in parameters:
            path = entry.get("path")
            if not path:
                continue
            manifest[path] = {
                "delta": float(entry.get("delta", 0.1)),
                "mode": entry.get("mode", "relative"),
            }
        return manifest

    def _resolve_observed_series(self, observed: Mapping[str, Any], config_path: str) -> Dict[str, Any]:
        observed = observed or {}
        label = str(observed.get("label") or "observed")
        series = observed.get("series")
        if series:
            if not isinstance(series, Mapping):
                raise ValueError("Observed series JSON must provide time_s and values")
            times = [float(v) for v in series.get("time_s", series.get("time", []))]
            values = [float(v) for v in series.get("values", series.get("value", []))]
            if not times or len(times) != len(values):
                raise ValueError("Manual observed series must include matching time_s and values arrays")
            return {
                "time_s": times,
                "values": values,
                "label": label,
                "source_type": "manual",
            }

        csv_path_text = observed.get("path")
        if not csv_path_text:
            raise ValueError("Observed dataset requires either a CSV path or manual series")

        path_obj = Path(csv_path_text).expanduser()
        if not path_obj.is_absolute():
            config_dir = Path(config_path).resolve().parent
            candidate = (config_dir / path_obj).resolve()
            if candidate.exists():
                path_obj = candidate
            elif self.workspace_manager is not None:
                candidate_ws = (Path(self.workspace_manager.workspace_path) / path_obj).resolve()
                if candidate_ws.exists():
                    path_obj = candidate_ws
                else:
                    path_obj = candidate

        if not path_obj.exists():
            raise FileNotFoundError(f"Observed dataset not found: {path_obj}")

        time_col = str(observed.get("time_column") or "time_h")
        value_col = str(observed.get("value_column") or "conc")
        time_unit = str(observed.get("time_unit") or "h")

        dataset = app_api.load_observed_pk_csv(
            path_obj,
            time_col=time_col,
            conc_col=value_col,
            time_unit=time_unit,
            label=label,
        )

        return {
            "time_s": dataset.time_s.astype(float).tolist(),
            "values": dataset.concentration_ng_per_ml.astype(float).tolist(),
            "label": dataset.label,
            "source": str(path_obj),
            "time_unit": time_unit,
            "source_type": "csv",
        }

    def start_parameter_estimation(self, config_path: str, plan: Dict[str, Any], show_message: bool = True) -> List[str]:
        if self.workspace_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before starting runs.")
            return []

        try:
            config_data = self._load_config_data(config_path)
        except Exception as exc:
            QMessageBox.critical(self, "Estimation Error", f"Could not load config: {exc}")
            return []

        try:
            base_config = AppConfig.model_validate(config_data) if CONFIG_MODEL_AVAILABLE and AppConfig else None
        except Exception as exc:
            QMessageBox.critical(self, "Estimation Error", f"Configuration invalid: {exc}")
            return []

        if base_config is None:
            QMessageBox.critical(self, "Estimation Error", "Configuration model is unavailable for planning runs.")
            return []

        estimation = plan.get("estimation") or {}
        parameters = estimation.get("parameters") or []
        if not parameters:
            QMessageBox.warning(self, "Estimation Parameters", "Provide at least one parameter to calibrate.")
            return []

        param_map = self._parameter_entries_to_manifest(parameters)
        if not param_map:
            QMessageBox.warning(self, "Estimation Parameters", "Parameter entries must include config paths.")
            return []

        raw_targets = estimation.get("targets") or []
        resolved_targets: List[Dict[str, Any]] = []
        if not isinstance(raw_targets, list):
            QMessageBox.critical(self, "Estimation Targets", "Targets must be provided as a list")
            return []

        for target in raw_targets:
            if not isinstance(target, Mapping):
                continue
            metric = target.get("metric")
            if not metric:
                continue
            weight = float(target.get("weight", 1.0))
            observed_spec = target.get("observed") or {}
            loss = target.get("loss")

            try:
                if metric == "pk_concentration":
                    resolved_observed = self._resolve_observed_series(observed_spec, config_path)
                    resolved_target = {
                        "metric": metric,
                        "weight": weight,
                        "loss": loss or "sse",
                        "observed": resolved_observed,
                    }
                elif metric in {"pk_auc_0_t", "cfd_mmad", "cfd_gsd", "cfd_mt_fraction"}:
                    value = observed_spec.get("value")
                    if value is None:
                        raise ValueError("Scalar targets require an observed value")
                    resolved_target = {
                        "metric": metric,
                        "weight": weight,
                        "observed": {"value": float(value)},
                    }
                elif metric == "deposition_fraction":
                    regions = observed_spec.get("regions")
                    if not isinstance(regions, Mapping):
                        raise ValueError("Deposition fraction targets require a regions mapping")
                    resolved_regions: Dict[str, float] = {}
                    for region_key, region_value in regions.items():
                        resolved_regions[str(region_key)] = float(region_value)
                    resolved_target = {
                        "metric": metric,
                        "weight": weight,
                        "observed": {"regions": resolved_regions},
                    }
                else:
                    raise ValueError(f"Unsupported target metric '{metric}'")
            except Exception as exc:
                QMessageBox.critical(self, "Estimation Targets", f"Invalid target '{metric}': {exc}")
                return []

            resolved_targets.append(resolved_target)

        if not resolved_targets:
            QMessageBox.critical(self, "Estimation Targets", "Add at least one estimation target before queuing runs.")
            return []

        try:
            manifest = app_api.plan_parameter_estimation_runs(
                base_config,
                param_map,
                include_baseline=bool(estimation.get("include_baseline", True)),
                default_relative_step=float(estimation.get("default_relative_step", 0.1)),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Estimation Error", f"Failed to plan estimation manifest: {exc}")
            return []

        if manifest.empty:
            QMessageBox.information(self, "Estimation", "Parameter estimation manifest is empty; nothing to queue.")
            return []

        total_runs = len(manifest)
        label_base = plan.get("run_label") or Path(config_path).stem or "parameter_estimation"
        parent_run_id = self.workspace_manager.generate_run_id(prefix="pe", label=label_base)
        parent_run_dir = self.workspace_manager.runs_dir / parent_run_id

        manifest_records: List[Dict[str, Any]] = []
        parameter_paths = [str(entry.get("path")) for entry in parameters if entry.get("path")]

        for idx, row in manifest.iterrows():
            overrides: Dict[str, Any] = {}
            for path in parameter_paths:
                if path in row and not pd.isna(row[path]):
                    overrides[path] = self._coerce_json_value(row[path])
            record = {
                "index": int(idx),
                "run_id": self._coerce_json_value(row.get("run_id")),
                "parameter": self._coerce_json_value(row.get("parameter")),
                "direction": self._coerce_json_value(row.get("direction")),
                "value": self._coerce_json_value(row.get("value")),
                "overrides": overrides,
            }
            manifest_records.append(record)

        estimation_settings = {
            "parameters": parameters,
            "targets": resolved_targets,
            "include_baseline": bool(estimation.get("include_baseline", True)),
            "default_relative_step": estimation.get("default_relative_step", 0.1),
            "manifest": manifest_records,
        }

        existing_children: List[str] = []
        completed_runs = 0
        if parent_run_dir.exists():
            try:
                existing_info = self.workspace_manager.get_run_info(parent_run_id) or {}
            except Exception:
                existing_info = {}
            existing_children = [str(child) for child in existing_info.get("child_run_ids") or [] if child]
            try:
                completed_runs = int(existing_info.get("completed_runs", 0) or 0)
            except Exception:
                completed_runs = 0

        parent_metadata = {
            "run_plan": plan,
            "estimation_settings": estimation_settings,
            "child_run_ids": existing_children,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "config_path": config_path,
            "status": "group",
        }

        if not parent_run_dir.exists():
            self.workspace_manager.create_run_directory(
                parent_run_id,
                run_type="parameter_estimation_group",
                label=label_base,
                parent_run_id=None,
                request_metadata=parent_metadata,
            )
        else:
            self.workspace_manager.update_run_status(
                parent_run_id,
                "group",
                **parent_metadata,
            )

        child_run_ids: List[str] = list(existing_children)
        queued_run_ids: List[str] = []

        for record in manifest_records:
            idx = record["index"]
            direction = record.get("direction") or "candidate"
            parameter_name = record.get("parameter") or "parameter"
            iteration_label = f"{label_base} · {parameter_name} ({direction})"
            overrides = record.get("overrides", {}) if direction != "baseline" else {}
            metadata = {
                "run_plan": plan,
                "parent_run_id": parent_run_id,
                "manifest_row": idx,
                "manifest_run_id": record.get("run_id"),
                "estimation_parameter": parameter_name,
                "estimation_direction": direction,
                "estimation_value": record.get("value"),
                "estimation_overrides": overrides,
            }

            try:
                run_id = self.process_manager.start_simulation(
                    config_path,
                    run_type="parameter_estimation",
                    label=iteration_label,
                    parent_run_id=parent_run_id,
                    manifest_index=idx,
                    total_runs=total_runs,
                    metadata=metadata,
                    parameter_overrides=overrides,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Estimation Error", f"Failed to queue iteration #{idx + 1}: {exc}")
                continue

            self.run_logs[run_id] = []
            self.current_log_run_id = run_id
            self.add_run_row(run_id)
            queued_run_ids.append(run_id)
            if run_id not in child_run_ids:
                child_run_ids.append(run_id)

        self.workspace_manager.update_run_status(
            parent_run_id,
            "group",
            child_run_ids=child_run_ids,
            total_runs=total_runs,
            completed_runs=completed_runs,
        )

        if queued_run_ids and show_message:
            preview = ", ".join(queued_run_ids[:3])
            if len(queued_run_ids) > 3:
                preview += ", …"
            QMessageBox.information(
                self,
                "Parameter Estimation Queued",
                f"Queued {len(queued_run_ids)} calibration run(s) under parent {parent_run_id}: {preview}",
            )

        self.pending_run_plan = self._default_run_plan()
        return queued_run_ids

    def _finalize_parameter_estimation(self, parent_run_id: str) -> None:
        if self.workspace_manager is None:
            return

        parent_info = self.workspace_manager.get_run_info(parent_run_id)
        if not parent_info:
            return

        estimation_settings = parent_info.get("estimation_settings") or {}
        targets_config = estimation_settings.get("targets") or []
        child_ids = [str(child) for child in parent_info.get("child_run_ids") or [] if child]
        if not child_ids:
            return

        summary_records: List[Dict[str, Any]] = []
        timeseries_rows: List[Dict[str, Any]] = []
        regional_rows: List[Dict[str, Any]] = []
        aggregated_frames: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
        best_run_id: Optional[str] = None
        best_value: Optional[float] = None

        for child_id in child_ids:
            child_info = self.workspace_manager.get_run_info(child_id) or {}
            status = child_info.get("status")
            pe_payload = child_info.get("parameter_estimation") or {}

            record = {
                "run_id": child_id,
                "parameter": child_info.get("estimation_parameter"),
                "direction": child_info.get("estimation_direction"),
                "value": child_info.get("estimation_value"),
                "manifest_index": pe_payload.get("manifest_index") or child_info.get("manifest_index"),
                "status": status,
                "combined_objective": pe_payload.get("combined_objective"),
            }
            summary_records.append(record)

            evaluations = pe_payload.get("targets") or []
            for evaluation in evaluations:
                if not isinstance(evaluation, Mapping):
                    continue
                metric = evaluation.get("metric")
                metric_type = evaluation.get("type")
                if metric is None:
                    continue

                error_msg = evaluation.get("error")
                if error_msg:
                    record[f"{metric}_error"] = error_msg
                    continue

                if evaluation.get("weight") is not None:
                    try:
                        record[f"{metric}_weight"] = float(evaluation.get("weight"))
                    except Exception:
                        record[f"{metric}_weight"] = evaluation.get("weight")

                objective_value = evaluation.get("objective")
                if objective_value is not None:
                    try:
                        record[f"{metric}_objective"] = float(objective_value)
                    except Exception:
                        record[f"{metric}_objective"] = objective_value

                if metric_type == "scalar":
                    record[f"{metric}_predicted"] = evaluation.get("predicted")
                    record[f"{metric}_observed"] = evaluation.get("observed")
                    record[f"{metric}_residual"] = evaluation.get("residual")
                elif metric_type == "timeseries":
                    record[f"{metric}_loss"] = evaluation.get("loss")
                    record[f"{metric}_sse"] = evaluation.get("sse")
                    record[f"{metric}_mae"] = evaluation.get("mae")
                    record[f"{metric}_rmse"] = evaluation.get("rmse")
                    series = evaluation.get("series") or {}
                    time_s = series.get("time_s") or []
                    observed_vals = series.get("observed") or []
                    predicted_vals = series.get("predicted") or []
                    residual_vals = series.get("residual") or []
                    for idx, t_value in enumerate(time_s):
                        timeseries_rows.append({
                            "run_id": child_id,
                            "metric": metric,
                            "time_s": float(t_value),
                            "time_h": float(t_value) / 3600.0,
                            "observed": observed_vals[idx] if idx < len(observed_vals) else None,
                            "predicted": predicted_vals[idx] if idx < len(predicted_vals) else None,
                            "residual": residual_vals[idx] if idx < len(residual_vals) else None,
                        })
                elif metric_type == "regional":
                    per_region = evaluation.get("per_region") or []
                    for entry in per_region:
                        if not isinstance(entry, Mapping):
                            continue
                        regional_rows.append({
                            "run_id": child_id,
                            "metric": metric,
                            "region": entry.get("region"),
                            "observed": entry.get("observed"),
                            "predicted": entry.get("predicted"),
                            "residual": entry.get("residual"),
                        })

            child_results: Dict[str, pd.DataFrame] = {}
            try:
                child_results = self.workspace_manager.load_run_results(child_id)
            except Exception:
                child_results = {}

            for dataset_name, df in child_results.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                product_series = df.get("product")
                if isinstance(product_series, pd.Series) and not product_series.empty:
                    product_name = str(product_series.iloc[0])
                else:
                    product_name = str(next(iter((df.columns[df.columns.str.contains("product", case=False)])), "product"))
                product_role_series = df.get("product_role")
                product_role = None
                if isinstance(product_role_series, pd.Series) and not product_role_series.empty:
                    product_role = product_role_series.iloc[0]
                augmented = df.copy()
                augmented["run_id"] = child_id
                augmented["subject_index"] = subject_index
                if "product" not in augmented.columns:
                    augmented["product"] = product_name
                if product_role is not None and "product_role" not in augmented.columns:
                    augmented["product_role"] = product_role
                key = (product_name, dataset_name)
                aggregated_frames.setdefault(key, []).append(augmented)

            combined_objective = pe_payload.get("combined_objective")
            if combined_objective is None:
                total = 0.0
                has_any = False
                for evaluation in evaluations:
                    weighted = evaluation.get("weighted_objective")
                    if weighted is None:
                        continue
                    try:
                        total += float(weighted)
                        has_any = True
                    except Exception:
                        continue
                if has_any:
                    combined_objective = total

            if combined_objective is not None:
                try:
                    combined_float = float(combined_objective)
                except Exception:
                    combined_float = None
                if combined_float is not None and (best_value is None or combined_float < best_value):
                    best_value = combined_float
                    best_run_id = child_id

        for record in summary_records:
            record["is_best"] = (record.get("run_id") == best_run_id)

        results_dir = Path(parent_info.get("results_dir") or (self.workspace_manager.runs_dir / parent_run_id / "results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(summary_records)
        summary_path = results_dir / "parameter_estimation_summary.parquet"
        summary_df.to_parquet(summary_path, index=False)

        residual_df = pd.DataFrame(timeseries_rows) if timeseries_rows else pd.DataFrame(
            columns=["run_id", "metric", "time_s", "time_h", "observed", "predicted", "residual"]
        )
        residual_path = None
        if not residual_df.empty:
            residual_path = results_dir / "parameter_estimation_residuals.parquet"
            residual_df.to_parquet(residual_path, index=False)

        regional_df = pd.DataFrame(regional_rows) if regional_rows else pd.DataFrame(
            columns=["run_id", "metric", "region", "observed", "predicted", "residual"]
        )
        regional_path = None
        if not regional_df.empty:
            regional_path = results_dir / "parameter_estimation_regional.parquet"
            regional_df.to_parquet(regional_path, index=False)

        overlay_path = None
        if best_run_id and not residual_df.empty:
            overlay_rows: List[Dict[str, Any]] = []
            best_rows = residual_df[residual_df["run_id"] == best_run_id]
            for _, row in best_rows.iterrows():
                overlay_rows.append({
                    "run_id": best_run_id,
                    "series": f"observed_{row['metric']}",
                    "time_s": float(row["time_s"]),
                    "time_h": float(row["time_h"]),
                    "value": float(row["observed"]) if row["observed"] is not None else None,
                })
                overlay_rows.append({
                    "run_id": best_run_id,
                    "series": f"predicted_{row['metric']}",
                    "time_s": float(row["time_s"]),
                    "time_h": float(row["time_h"]),
                    "value": float(row["predicted"]) if row["predicted"] is not None else None,
                })
            if overlay_rows:
                overlay_df = pd.DataFrame(overlay_rows)
                overlay_path = results_dir / "parameter_estimation_overlay.parquet"
                overlay_df.to_parquet(overlay_path, index=False)

        for (product_name, dataset_name), frames in aggregated_frames.items():
            if not frames:
                continue
            combined_df = pd.concat(frames, ignore_index=True)
            product_slug = _sanitise_product_name(product_name)
            base_name = str(dataset_name)
            if base_name.startswith(f"{product_slug}__"):
                filename = f"{base_name}.parquet"
            else:
                filename = f"{product_slug}__{base_name}.parquet"
            path = results_dir / filename
            combined_df.to_parquet(path, index=False)

        summary_payload = {
            "best_run_id": best_run_id,
            "best_objective": self._coerce_json_value(best_value) if best_value is not None else None,
            "targets": targets_config,
            "records": [
                {key: self._coerce_json_value(value) for key, value in record.items()}
                for record in summary_records
            ],
            "summary_path": str(summary_path),
            "residual_path": str(residual_path) if residual_path else None,
            "overlay_path": str(overlay_path) if overlay_path else None,
            "regional_path": str(regional_path) if regional_path else None,
        }

        self.workspace_manager.update_run_status(
            parent_run_id,
            parent_info.get("status", "completed"),
            parameter_estimation_summary=summary_payload,
        )
        try:
            updated_info = self.workspace_manager.get_run_info(parent_run_id) or {}
        except Exception:
            updated_info = {}
        if updated_info:
            self.run_metadata[parent_run_id] = updated_info

    def _finalize_virtual_trial(self, parent_run_id: str) -> None:
        if self.workspace_manager is None:
            return

        parent_info = self.workspace_manager.get_run_info(parent_run_id)
        if not parent_info:
            return

        settings = parent_info.get("virtual_trial_settings") or {}
        child_ids = [str(child) for child in parent_info.get("child_run_ids") or [] if child]
        if not child_ids:
            return

        records: List[Dict[str, Any]] = []
        long_rows: List[Dict[str, Any]] = []

        for child_id in child_ids:
            child_info = self.workspace_manager.get_run_info(child_id) or {}
            products = child_info.get("products") or {}
            subject_index = child_info.get("subject_index")
            seed = child_info.get("seed")
            for product_name, payload in products.items():
                summary_metrics = payload.get("summary_metrics") or {}
                product_role = payload.get("role")
                record = {
                    "run_id": child_id,
                    "subject_index": subject_index,
                    "seed": seed,
                    "product": product_name,
                    "product_role": product_role,
                }
                for metric_name, value in summary_metrics.items():
                    key = f"metric:{metric_name}"
                    record[key] = self._coerce_json_value(value)
                    long_rows.append({
                        "run_id": child_id,
                        "subject_index": subject_index,
                        "product": product_name,
                        "product_role": product_role,
                        "metric": metric_name,
                        "value": value,
                    })
                records.append(record)

        results_dir = Path(parent_info.get("results_dir") or (self.workspace_manager.runs_dir / parent_run_id / "results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        subjects_df = pd.DataFrame(records)
        subjects_path = results_dir / "virtual_trial_subjects.parquet"
        subjects_df.to_parquet(subjects_path, index=False)

        summary_rows: List[Dict[str, Any]] = []
        long_df = pd.DataFrame(long_rows)
        if not long_df.empty:
            for (product, role, metric), group in long_df.groupby(["product", "product_role", "metric"]):
                values = group["value"].dropna().astype(float)
                if values.empty:
                    continue
                stats_entry = {
                    "product": product,
                    "product_role": role,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.count() > 1 else float("nan"),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
                positive = values[values > 0]
                if not positive.empty:
                    stats_entry["geomean"] = float(np.exp(np.log(positive).mean()))
                else:
                    stats_entry["geomean"] = float("nan")
                summary_rows.append(stats_entry)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / "virtual_trial_summary.parquet"
        summary_df.to_parquet(summary_path, index=False)

        summary_payload = {
            "subjects_path": str(subjects_path),
            "summary_path": str(summary_path),
            "n_subjects": len(child_ids),
        }

        self.workspace_manager.update_run_status(
            parent_run_id,
            parent_info.get("status", "completed"),
            virtual_trial_summary=summary_payload,
        )
        try:
            updated_info = self.workspace_manager.get_run_info(parent_run_id) or {}
        except Exception:
            updated_info = {}
        if updated_info:
            self.run_metadata[parent_run_id] = updated_info

    def _finalize_virtual_bioequivalence(self, parent_run_id: str) -> None:
        if self.workspace_manager is None:
            return

        parent_info = self.workspace_manager.get_run_info(parent_run_id)
        if not parent_info:
            return

        settings = parent_info.get("virtual_bioequivalence_settings") or {}
        study_metadata = settings.get("study_metadata") or {}
        reference_product = study_metadata.get("reference_product")
        test_products = study_metadata.get("test_products") or []
        child_ids = [str(child) for child in parent_info.get("child_run_ids") or [] if child]
        if not child_ids or not reference_product or not test_products:
            return

        subject_rows: List[Dict[str, Any]] = []
        long_rows: List[Dict[str, Any]] = []
        aggregated_frames: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

        for child_id in child_ids:
            child_info = self.workspace_manager.get_run_info(child_id) or {}
            products = child_info.get("products") or {}
            subject_index = child_info.get("subject_index")
            for product_name, payload in products.items():
                summary_metrics = payload.get("summary_metrics") or {}
                product_role = payload.get("role")
                record = {
                    "run_id": child_id,
                    "subject_index": subject_index,
                    "product": product_name,
                    "product_role": product_role,
                }
                for metric_name, value in summary_metrics.items():
                    record[f"metric:{metric_name}"] = self._coerce_json_value(value)
                    long_rows.append({
                        "run_id": child_id,
                        "subject_index": subject_index,
                        "product": product_name,
                        "product_role": product_role,
                        "metric": metric_name,
                        "value": value,
                    })
                subject_rows.append(record)

            child_results: Dict[str, pd.DataFrame] = {}
            try:
                child_results = self.workspace_manager.load_run_results(child_id)
            except Exception:
                child_results = {}

            for dataset_name, df in child_results.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                product_series = df.get("product")
                if isinstance(product_series, pd.Series) and not product_series.empty:
                    prod_name_for_ds = str(product_series.iloc[0])
                else:
                    prod_name_for_ds = product_name
                augmented = df.copy()
                augmented["run_id"] = child_id
                augmented["subject_index"] = subject_index
                if "product" not in augmented.columns:
                    augmented["product"] = prod_name_for_ds
                if product_role and "product_role" not in augmented.columns:
                    augmented["product_role"] = product_role
                key = (prod_name_for_ds, dataset_name)
                aggregated_frames.setdefault(key, []).append(augmented)

        results_dir = Path(parent_info.get("results_dir") or (self.workspace_manager.runs_dir / parent_run_id / "results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        subjects_df = pd.DataFrame(subject_rows)
        subjects_path = results_dir / "virtual_bioequivalence_subjects.parquet"
        subjects_df.to_parquet(subjects_path, index=False)

        long_df = pd.DataFrame(long_rows)
        summary_records: List[Dict[str, Any]] = []

        for metric_name in long_df["metric"].dropna().unique():
            metric_df = long_df[long_df["metric"] == metric_name]
            pivot = metric_df.pivot_table(index="subject_index", columns="product", values="value")
            if reference_product not in pivot.columns:
                continue
            reference_values = pivot[reference_product]
            for test_product in test_products:
                if test_product not in pivot.columns:
                    continue
                combined = pd.DataFrame({
                    "reference": reference_values,
                    "test": pivot[test_product],
                }).dropna()
                if combined.empty:
                    continue
                ref = combined["reference"].astype(float)
                test = combined["test"].astype(float)
                diffs = np.log(test) - np.log(ref)
                n = diffs.size
                mean_diff = float(diffs.mean())
                if n > 1:
                    std_diff = float(diffs.std(ddof=1))
                    try:
                        from scipy import stats  # type: ignore

                        t_value = float(stats.t.ppf(0.95, df=n - 1))
                    except Exception:
                        t_value = 1.645
                    se = std_diff / np.sqrt(n)
                    ci_lower = float(np.exp(mean_diff - t_value * se))
                    ci_upper = float(np.exp(mean_diff + t_value * se))
                else:
                    ci_lower = float("nan")
                    ci_upper = float("nan")
                gmr = float(np.exp(mean_diff))
                pass_flag = bool(ci_lower >= 0.8 and ci_upper <= 1.25) if np.isfinite(ci_lower) and np.isfinite(ci_upper) else False

                summary_records.append(
                    {
                        "metric": metric_name,
                        "test_product": test_product,
                        "reference_product": reference_product,
                        "gmr": gmr,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "pass_80_125": pass_flag,
                        "n": int(n),
                        "mean_log_diff": mean_diff,
                    }
                )

        summary_df = pd.DataFrame(summary_records)
        summary_path = results_dir / "virtual_bioequivalence_summary.parquet"
        summary_df.to_parquet(summary_path, index=False)

        product_summary_records: List[Dict[str, Any]] = []
        for (product, role, metric), group in long_df.groupby(["product", "product_role", "metric"]):
            values = group["value"].dropna().astype(float)
            if values.empty:
                continue
            positive = values[values > 0]
            product_summary_records.append(
                {
                    "product": product,
                    "product_role": role,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.count() > 1 else float("nan"),
                    "geomean": float(np.exp(np.log(positive).mean())) if not positive.empty else float("nan"),
                }
            )

        product_summary_df = pd.DataFrame(product_summary_records)
        product_summary_path = results_dir / "virtual_bioequivalence_product_summary.parquet"
        product_summary_df.to_parquet(product_summary_path, index=False)

        for (product_name, dataset_name), frames in aggregated_frames.items():
            if not frames:
                continue
            combined_df = pd.concat(frames, ignore_index=True)
            product_slug = _sanitise_product_name(product_name)
            base_name = str(dataset_name)
            if base_name.startswith(f"{product_slug}__"):
                filename = f"{base_name}.parquet"
            else:
                filename = f"{product_slug}__{base_name}.parquet"
            path = results_dir / filename
            combined_df.to_parquet(path, index=False)

        summary_payload = {
            "subjects_path": str(subjects_path),
            "summary_path": str(summary_path),
            "product_summary_path": str(product_summary_path),
            "reference_product": reference_product,
            "test_products": test_products,
        }

        self.workspace_manager.update_run_status(
            parent_run_id,
            parent_info.get("status", "completed"),
            virtual_bioequivalence_summary=summary_payload,
        )
        try:
            updated_info = self.workspace_manager.get_run_info(parent_run_id) or {}
        except Exception:
            updated_info = {}
        if updated_info:
            self.run_metadata[parent_run_id] = updated_info

    def queue_selected_configs(self):
        if self.process_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before queueing runs.")
            return

        selected_paths: List[str] = []
        if hasattr(self, 'config_table') and self.config_table.selectionModel() is not None:
            for index in self.config_table.selectionModel().selectedRows():
                item = self.config_table.item(index.row(), 0)
                if item is None:
                    continue
                path_value = item.data(Qt.ItemDataRole.UserRole)
                if path_value and path_value not in selected_paths:
                    selected_paths.append(path_value)

        if not selected_paths and self.selected_config_path:
            selected_paths.append(self.selected_config_path)

        if not selected_paths:
            QMessageBox.warning(self, "No Config", "Select at least one configuration to queue.")
            return

        queued_runs: List[str] = []
        for path_value in selected_paths:
            run_id = self.start_run_for_path(path_value, show_message=False)
            queued_runs.extend(self.last_started_run_ids)

        if queued_runs:
            preview = ", ".join(queued_runs[:3])
            if len(queued_runs) > 3:
                preview += ", …"
            QMessageBox.information(
                self,
                "Runs Queued",
                f"Queued {len(queued_runs)} run(s): {preview}"
            )

    def on_config_table_selection(self):
        if not hasattr(self, 'config_table') or self.config_table.selectionModel() is None:
            return
        indexes = self.config_table.selectionModel().selectedRows()
        if not indexes:
            return
        first_row = indexes[0].row()
        item = self.config_table.item(first_row, 0)
        if item is None:
            return
        path_value = item.data(Qt.ItemDataRole.UserRole)
        if path_value:
            self.set_selected_config(path_value)

    def on_config_table_double_clicked(self, _: QTableWidgetItem):
        self.queue_selected_configs()

    def add_run_row(self, run_id: str):
        """Add a new run entry to the table."""
        row = self.run_table.rowCount()
        self.run_table.insertRow(row)
        self.run_rows[run_id] = row

        self.run_table.setItem(row, 0, QTableWidgetItem(run_id))
        self.run_table.setItem(row, 1, QTableWidgetItem("Starting"))

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        self.run_table.setCellWidget(row, 2, progress_bar)

        self.run_table.setItem(row, 3, QTableWidgetItem("0s"))
        self.run_table.setItem(row, 4, QTableWidgetItem("--"))

        actions_widget = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda _, rid=run_id: self.cancel_run(rid))

        actions_layout.addWidget(cancel_btn)
        actions_widget.setLayout(actions_layout)

        self.run_table.setCellWidget(row, 5, actions_widget)
        self.run_table.setItem(row, 6, QTableWidgetItem("Queued"))
        self.run_table.selectRow(row)
        self.update_log_view()

    def cancel_run(self, run_id: str):
        """Cancel a running simulation."""
        if self.process_manager is None:
            return

        if self.process_manager.cancel_simulation(run_id):
            self.update_row_status(run_id, "Cancelling", "Cancelling...")
        else:
            QMessageBox.warning(self, "Cancel Run", f"Run {run_id} is not active.")

    def clear_completed(self):
        """Clear completed runs from queue."""
        rows_to_remove = []
        for run_id, row in self.run_rows.items():
            status_item = self.run_table.item(row, 1)
            status_text = status_item.text().lower() if status_item else ""
            if status_text in {"completed", "failed", "cancelled"}:
                rows_to_remove.append((row, run_id))

        for row, run_id in sorted(rows_to_remove, reverse=True):
            self.run_table.removeRow(row)
            self.run_rows.pop(run_id, None)
            self.run_errors.pop(run_id, None)
            self.run_logs.pop(run_id, None)
            if self.current_log_run_id == run_id:
                self.current_log_run_id = None

        self.reindex_rows()
        self.update_log_view()

    def reindex_rows(self):
        """Rebuild run_id to row index mapping."""
        new_mapping: Dict[str, int] = {}
        for row in range(self.run_table.rowCount()):
            run_item = self.run_table.item(row, 0)
            if run_item:
                new_mapping[run_item.text()] = row
        self.run_rows = new_mapping

    def update_run_status(self):
        """Update run status periodically."""
        if self.process_manager is None:
            return

        for run_id, row in list(self.run_rows.items()):
            info = self.process_manager.get_process_info(run_id)
            if info is None:
                continue

            elapsed_item = self.run_table.item(row, 3)
            if elapsed_item:
                elapsed_item.setText(f"{int(info.runtime_seconds)}s")

            status_item = self.run_table.item(row, 1)
            if status_item:
                status_item.setText(info.status.capitalize())

            details_item = self.run_table.item(row, 6)
            if details_item and info.last_message:
                details_item.setText(info.last_message)

            progress_bar = self.run_table.cellWidget(row, 2)
            if isinstance(progress_bar, QProgressBar):
                progress_bar.setValue(int(info.progress_pct))

    def get_selected_run_id(self) -> Optional[str]:
        selection_model = self.run_table.selectionModel()
        if selection_model is None:
            return None
        selection = selection_model.selectedRows()
        if not selection:
            return None
        row = selection[0].row()
        item = self.run_table.item(row, 0)
        return item.text() if item else None

    def load_logs_from_disk(self, run_id: str) -> List[str]:
        if self.workspace_manager is None:
            return []

        info = self.workspace_manager.get_run_info(run_id)
        if not info:
            self.run_logs.setdefault(run_id, [])
            return self.run_logs[run_id]

        log_files = info.get("log_files") or []
        lines: List[str] = []
        for log_file in log_files:
            path = Path(log_file)
            if not path.exists():
                continue
            try:
                with open(path, 'r') as handle:
                    for line in handle:
                        lines.append(line.rstrip())
            except Exception as exc:
                lines.append(f"[failed to read {path}: {exc}]")

        self.run_logs[run_id] = lines
        return lines

    def update_log_view(self):
        if not hasattr(self, 'log_view'):
            return

        run_id = self.get_selected_run_id()
        self.current_log_run_id = run_id

        if run_id is None:
            self.log_view.clear()
            self.log_view.setPlaceholderText("Log output will appear here for the selected run.")
            return

        lines = self.run_logs.get(run_id)
        if lines is None:
            lines = self.load_logs_from_disk(run_id)

        if not lines:
            self.log_view.setPlainText("No logs available for this run yet.")
        else:
            self.log_view.setPlainText("\n".join(lines))
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_view.setTextCursor(cursor)

    def on_process_log(self, run_id: str, message: str):
        lines = self.run_logs.setdefault(run_id, [])
        lines.append(message)
        if len(lines) > 2000:
            self.run_logs[run_id] = lines[-2000:]
            lines = self.run_logs[run_id]

        if hasattr(self, 'log_view') and self.current_log_run_id == run_id:
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(message + "\n")
            self.log_view.setTextCursor(cursor)

    def update_row_status(self, run_id: str, status: str, details: str):
        """Helper to set status and details for a run."""
        row = self.run_rows.get(run_id)
        if row is None:
            return

        status_item = self.run_table.item(row, 1)
        if status_item:
            status_item.setText(status)

        details_item = self.run_table.item(row, 6)
        if details_item:
            details_item.setText(details)

    def on_process_started(self, run_id: str):
        self.update_row_status(run_id, "Running", "Simulation started")

    def on_process_progress(self, run_id: str, pct: float, message: str):
        row = self.run_rows.get(run_id)
        if row is None:
            return

        progress_bar = self.run_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(int(pct))

        if message:
            details_item = self.run_table.item(row, 6)
            if details_item:
                details_item.setText(message)

    def on_process_metric(self, run_id: str, metric_name: str, value: float):
        details = f"{metric_name}: {value:.3g}"
        self.update_row_status(run_id, "Running", details)

    def on_process_error(self, run_id: str, error_message: str):
        self.run_errors[run_id] = error_message
        self.update_row_status(run_id, "Failed", error_message)
        QMessageBox.critical(self, "Run Error", f"Run {run_id} failed:\n{error_message}")

    def on_process_finished(self, run_id: str, status: str, runtime_seconds: float):
        row = self.run_rows.get(run_id)
        if row is None:
            return

        status_text = status.capitalize()
        elapsed_item = self.run_table.item(row, 3)
        if elapsed_item:
            elapsed_item.setText(f"{runtime_seconds:.1f}s")

        eta_item = self.run_table.item(row, 4)
        if eta_item:
            eta_item.setText("--")

        progress_bar = self.run_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar) and status == "completed":
            progress_bar.setValue(100)

        if status == "completed":
            details = f"Completed in {runtime_seconds:.1f}s"
        elif status == "cancelled":
            details = "Run cancelled"
        else:
            details = self.run_errors.get(run_id, "Run failed")

        self.update_row_status(run_id, status_text, details)
        if status in {"completed", "cancelled", "failed"}:
            self.run_errors.pop(run_id, None)

        if self.process_manager is not None:
            self.run_logs.setdefault(run_id, self.process_manager.get_process_logs(run_id))
        if self.current_log_run_id == run_id:
            self.update_log_view()

        if self.workspace_manager is not None:
            try:
                child_info = self.workspace_manager.get_run_info(run_id) or {}
            except Exception:
                child_info = {}
            parent_id = child_info.get("parent_run_id")
            if parent_id:
                parent_info = self.workspace_manager.get_run_info(parent_id) or {}
                existing_children = parent_info.get("child_run_ids") or []
                child_ids: List[str] = []
                for child in existing_children:
                    if child:
                        value = str(child)
                        if value not in child_ids:
                            child_ids.append(value)
                if run_id not in child_ids:
                    child_ids.append(run_id)

                total_runs_raw = parent_info.get("total_runs")
                try:
                    total_runs_int = int(total_runs_raw) if total_runs_raw is not None else None
                except Exception:
                    total_runs_int = None

                try:
                    completed_runs = int(parent_info.get("completed_runs", 0) or 0)
                except Exception:
                    completed_runs = 0

                if status == "completed":
                    completed_runs += 1

                parent_status = parent_info.get("status", "group")
                if total_runs_int is not None and completed_runs >= total_runs_int:
                    parent_status = "completed"

                self.workspace_manager.update_run_status(
                    parent_id,
                    parent_status,
                    child_run_ids=child_ids,
                    total_runs=total_runs_int if total_runs_int is not None else total_runs_raw,
                    completed_runs=completed_runs,
                    last_child_updated=run_id,
                )

                if parent_status == "completed":
                    parent_run_type = parent_info.get("run_type")
                    try:
                        if parent_run_type == "parameter_estimation_group":
                            self._finalize_parameter_estimation(parent_id)
                        elif parent_run_type == "virtual_trial_group":
                            self._finalize_virtual_trial(parent_id)
                        elif parent_run_type == "virtual_bioequivalence_group":
                            self._finalize_virtual_bioequivalence(parent_id)
                    except Exception as exc:
                        logger.warning("Failed to finalize parent run %s: %s", parent_id, exc)

        self.run_completed.emit(run_id, status)


class ResultsViewerTab(QWidget):
    """Results viewer tab with multi-run plotting and data export."""

    @staticmethod
    def _coerce_json_value(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _geometric_mean(series: pd.Series) -> float:
        valid = series[series > 0]
        if valid.empty:
            return float('nan')
        return float(np.exp(np.log(valid).mean()))

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.loaded_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.run_metadata: Dict[str, Dict[str, Any]] = {}
        self.active_run_id: Optional[str] = None
        self.dataset_plot_defaults = {
            "pk_curve": "pk_linear",
            "deposition_bins": "dep_fraction",
            "regional_auc": "regional_auc_bar",
            "pbpk_regional_timeseries": "pbpk_timeseries",
            "parameter_estimation_residuals": "parameter_estimation_residuals",
            "parameter_estimation_overlay": "parameter_estimation_overlay",
            "parameter_estimation_regional": "parameter_estimation_regional",
        }
        self.pbpk_combined_df: Optional[pd.DataFrame] = None
        self.view_mode_combo: Optional[QComboBox] = None
        self.run_summary_label: Optional[QLabel] = None
        self.raw_controls_widget: Optional[QWidget] = None
        self.study_placeholder_label: Optional[QLabel] = None
        self.study_controls_widget: Optional[QWidget] = None
        self.study_stage_combo: Optional[QComboBox] = None
        self.study_param_x_combo: Optional[QComboBox] = None
        self.study_param_y_combo: Optional[QComboBox] = None
        self.study_metric_combo: Optional[QComboBox] = None
        self.study_plot_type_combo: Optional[QComboBox] = None
        self.study_table: Optional[QTableWidget] = None
        self.study_records_df: Optional[pd.DataFrame] = None
        self.stage_metric_columns: Dict[str, List[str]] = {}
        self.study_stage_order: List[str] = []
        self.study_metric_filter_edit: Optional[QLineEdit] = None
        self.init_ui()

    @staticmethod
    def _split_dataset_name(name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not name:
            return None, None
        if "__" in name:
            prefix, suffix = name.split("__", 1)
            return suffix, prefix
        return name, None

    def _dataset_for_run(
        self,
        run_id: str,
        dataset_name: Optional[str],
        base_dataset: Optional[str],
        dataset_prefix: Optional[str],
    ) -> Optional[pd.DataFrame]:
        candidates: List[str] = []
        if dataset_prefix and base_dataset:
            candidates.append(f"{dataset_prefix}__{base_dataset}")
        if dataset_name:
            candidates.append(dataset_name)
        if base_dataset:
            candidates.append(base_dataset)

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            df = self.loaded_results.get(run_id, {}).get(candidate)
            if df is not None:
                return df
        return None

    @staticmethod
    def _format_pbpk_label(value: Any) -> str:
        if value is None or pd.isna(value):
            return "Unknown"
        return str(value).replace('_', ' ').title()

    @staticmethod
    def _ordered_unique(series: pd.Series) -> List[Any]:
        values: List[Any] = []
        for value in series:
            if pd.isna(value):
                continue
            if value not in values:
                values.append(value)
        return values

    @staticmethod
    def _merge_stage_order(existing: List[str], new_order: Iterable[str]) -> List[str]:
        merged: List[str] = list(existing)
        seen = set(existing)
        for stage in new_order:
            if not stage or stage in seen:
                continue
            merged.append(stage)
            seen.add(stage)
        return merged

    @staticmethod
    def _stage_display_label(stage_name: Optional[str]) -> str:
        if stage_name is None:
            return "All Stages"
        return STAGE_DISPLAY_NAMES.get(stage_name, stage_name.replace('_', ' ').title())

    def _format_metric_column_label(self, column: str) -> str:
        parts = column.split(":", 2)
        if len(parts) == 3:
            return f"{self._stage_display_label(parts[1])} · {parts[2]}"
        return parts[-1]

    @staticmethod
    def _list_selected_values(widget: QListWidget) -> List[Any]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in widget.selectedItems()]

    def _populate_list_widget(
        self,
        widget: QListWidget,
        label: QLabel,
        values: List[Any],
        previous_selection: Optional[List[Any]] = None,
    ) -> None:
        widget.blockSignals(True)
        widget.clear()
        previous_selection = previous_selection or []
        for value in values:
            item = QListWidgetItem(self._format_pbpk_label(value))
            item.setData(Qt.ItemDataRole.UserRole, value)
            widget.addItem(item)
            if value in previous_selection:
                item.setSelected(True)

        if widget.count() > 0:
            if not previous_selection:
                widget.item(0).setSelected(True)
            widget.setEnabled(True)
        else:
            widget.setEnabled(False)

        is_visible = widget.count() > 0
        widget.setVisible(is_visible)
        label.setVisible(is_visible)
        widget.blockSignals(False)

    def _get_selected_or_all(self, widget: QListWidget, available: List[Any]) -> List[Any]:
        if not available:
            return []
        if not widget.isVisible():
            return list(available)
        selected = [item.data(Qt.ItemDataRole.UserRole) for item in widget.selectedItems()]
        return selected if selected else list(available)

    def _metric_filter_text(self) -> str:
        if self.study_metric_filter_edit is None:
            return ""
        return self.study_metric_filter_edit.text().strip().lower()

    def _available_metric_columns(self, df: pd.DataFrame) -> List[str]:
        columns = [col for col in df.columns if col.startswith("metric:")]
        stage = self.study_stage_combo.currentData() if self.study_stage_combo else None
        if stage:
            columns = [col for col in columns if col.startswith(f"metric:{stage}:")]
        filter_text = self._metric_filter_text()
        if filter_text:
            columns = [col for col in columns if filter_text in col.lower()]
        return columns

    def init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Results Viewer")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        run_panel = QWidget()
        run_layout = QVBoxLayout()
        run_controls = QHBoxLayout()
        run_controls.addWidget(QLabel("Runs"))
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_runs)
        run_controls.addWidget(self.refresh_btn)
        self.export_btn = QPushButton("Export Selected Runs…")
        self.export_btn.clicked.connect(self.export_selected_runs)
        run_controls.addWidget(self.export_btn)
        run_controls.addStretch()
        run_layout.addLayout(run_controls)

        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.run_list.itemSelectionChanged.connect(self.on_run_selection_changed)
        run_layout.addWidget(self.run_list)
        run_panel.setLayout(run_layout)
        splitter.addWidget(run_panel)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout()

        self.status_label = QLabel("Select a run to view results.")
        self.status_label.setStyleSheet("color: gray;")
        detail_layout.addWidget(self.status_label)

        self.run_summary_label = QLabel("")
        self.run_summary_label.setStyleSheet("color: #555; font-size: 11px;")
        detail_layout.addWidget(self.run_summary_label)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Raw Results", "raw")
        self.view_mode_combo.addItem("Study Summary", "study")
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        mode_layout.addWidget(self.view_mode_combo)
        mode_layout.addStretch()
        detail_layout.addLayout(mode_layout)

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMinimumHeight(120)
        detail_layout.addWidget(self.metadata_text)

        self.raw_controls_widget = QWidget()
        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(self.dataset_combo)

        self.pbpk_controls_widget = QWidget()
        pbpk_controls_layout = QGridLayout()
        pbpk_controls_layout.setContentsMargins(0, 0, 0, 0)
        pbpk_controls_layout.setHorizontalSpacing(12)
        pbpk_controls_layout.setVerticalSpacing(4)

        def _configure_list(widget: QListWidget, minimum_width: int = 140) -> None:
            widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            widget.setMinimumWidth(minimum_width)
            widget.setMaximumHeight(110)
            widget.setAlternatingRowColors(True)

        self.pbpk_region_label = QLabel("Region")
        self.pbpk_region_list = QListWidget()
        _configure_list(self.pbpk_region_list)
        self.pbpk_region_list.itemSelectionChanged.connect(self.on_pbpk_control_changed)
        pbpk_controls_layout.addWidget(self.pbpk_region_label, 0, 0)
        pbpk_controls_layout.addWidget(self.pbpk_region_list, 1, 0)

        self.pbpk_compartment_label = QLabel("Compartment")
        self.pbpk_compartment_list = QListWidget()
        _configure_list(self.pbpk_compartment_list)
        self.pbpk_compartment_list.itemSelectionChanged.connect(self.on_pbpk_control_changed)
        pbpk_controls_layout.addWidget(self.pbpk_compartment_label, 0, 1)
        pbpk_controls_layout.addWidget(self.pbpk_compartment_list, 1, 1)

        self.pbpk_quantity_label = QLabel("Quantity")
        self.pbpk_quantity_list = QListWidget()
        _configure_list(self.pbpk_quantity_list)
        self.pbpk_quantity_list.itemSelectionChanged.connect(self.on_pbpk_control_changed)
        pbpk_controls_layout.addWidget(self.pbpk_quantity_label, 0, 2)
        pbpk_controls_layout.addWidget(self.pbpk_quantity_list, 1, 2)

        self.pbpk_binding_label = QLabel("Binding")
        self.pbpk_binding_list = QListWidget()
        _configure_list(self.pbpk_binding_list, minimum_width=120)
        self.pbpk_binding_list.itemSelectionChanged.connect(self.on_pbpk_control_changed)
        pbpk_controls_layout.addWidget(self.pbpk_binding_label, 0, 3)
        pbpk_controls_layout.addWidget(self.pbpk_binding_list, 1, 3)

        self.pbpk_controls_widget.setLayout(pbpk_controls_layout)
        dataset_layout.addWidget(self.pbpk_controls_widget)
        self.pbpk_controls_widget.hide()

        if MATPLOTLIB_AVAILABLE:
            dataset_layout.addSpacing(12)
            dataset_layout.addWidget(QLabel("Plot:"))
            self.plot_type_combo = QComboBox()
            self.plot_type_combo.addItem("None", "none")
            self.plot_type_combo.addItem("Plasma Concentration", "pk_linear")
            self.plot_type_combo.addItem("Plasma Concentration (log)", "pk_log")
            self.plot_type_combo.addItem("Regional Deposition (pmol)", "dep_pmol")
            self.plot_type_combo.addItem("Regional Deposition (fraction)", "dep_fraction")
            self.plot_type_combo.addItem("Deposition Heatmap (fraction)", "dep_heatmap")
            self.plot_type_combo.addItem("Regional AUC (stack)", "regional_auc_bar")
            self.plot_type_combo.addItem("PBPK Time Series", "pbpk_timeseries")
            self.plot_type_combo.addItem("PE Residuals", "parameter_estimation_residuals")
            self.plot_type_combo.addItem("PE Overlay", "parameter_estimation_overlay")
            self.plot_type_combo.addItem("PE Regional Residuals", "parameter_estimation_regional")
            self.plot_type_combo.currentIndexChanged.connect(self.update_plots)
            dataset_layout.addWidget(self.plot_type_combo)
        else:
            self.plot_type_combo = None

        dataset_layout.addStretch()
        self.raw_controls_widget.setLayout(dataset_layout)
        detail_layout.addWidget(self.raw_controls_widget)

        self.study_controls_widget = QWidget()
        study_controls_layout = QHBoxLayout()
        study_controls_layout.setContentsMargins(0, 0, 0, 0)
        study_controls_layout.addWidget(QLabel("Stage:"))
        self.study_stage_combo = QComboBox()
        self.study_stage_combo.currentIndexChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(self.study_stage_combo)
        study_controls_layout.addWidget(QLabel("Parameter X:"))
        self.study_param_x_combo = QComboBox()
        self.study_param_x_combo.currentIndexChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(self.study_param_x_combo)

        self.study_param_x_min = QDoubleSpinBox()
        self.study_param_x_min.setDecimals(6)
        self.study_param_x_min.valueChanged.connect(self.on_study_controls_changed)
        self.study_param_x_min.setEnabled(False)
        study_controls_layout.addWidget(QLabel("X Min:"))
        study_controls_layout.addWidget(self.study_param_x_min)

        self.study_param_x_max = QDoubleSpinBox()
        self.study_param_x_max.setDecimals(6)
        self.study_param_x_max.valueChanged.connect(self.on_study_controls_changed)
        self.study_param_x_max.setEnabled(False)
        study_controls_layout.addWidget(QLabel("X Max:"))
        study_controls_layout.addWidget(self.study_param_x_max)

        study_controls_layout.addWidget(QLabel("Parameter Y:"))
        self.study_param_y_combo = QComboBox()
        self.study_param_y_combo.currentIndexChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(self.study_param_y_combo)

        self.study_param_y_min = QDoubleSpinBox()
        self.study_param_y_min.setDecimals(6)
        self.study_param_y_min.valueChanged.connect(self.on_study_controls_changed)
        self.study_param_y_min.setEnabled(False)
        study_controls_layout.addWidget(QLabel("Y Min:"))
        study_controls_layout.addWidget(self.study_param_y_min)

        self.study_param_y_max = QDoubleSpinBox()
        self.study_param_y_max.setDecimals(6)
        self.study_param_y_max.valueChanged.connect(self.on_study_controls_changed)
        self.study_param_y_max.setEnabled(False)
        study_controls_layout.addWidget(QLabel("Y Max:"))
        study_controls_layout.addWidget(self.study_param_y_max)

        study_controls_layout.addWidget(QLabel("Metric:"))
        self.study_metric_combo = QComboBox()
        self.study_metric_combo.currentIndexChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(self.study_metric_combo)

        self.study_metric_filter_edit = QLineEdit()
        self.study_metric_filter_edit.setPlaceholderText("Filter metrics (e.g. gmr)")
        self.study_metric_filter_edit.textChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(QLabel("Filter:"))
        study_controls_layout.addWidget(self.study_metric_filter_edit)

        study_controls_layout.addWidget(QLabel("Plot:"))
        self.study_plot_type_combo = QComboBox()
        self.study_plot_type_combo.addItem("Scatter", "scatter")
        self.study_plot_type_combo.addItem("Line", "line")
        self.study_plot_type_combo.addItem("Heatmap", "heatmap")
        self.study_plot_type_combo.currentIndexChanged.connect(self.on_study_controls_changed)
        study_controls_layout.addWidget(self.study_plot_type_combo)

        study_controls_layout.addStretch()
        self.study_controls_widget.setLayout(study_controls_layout)
        detail_layout.addWidget(self.study_controls_widget)
        self.study_controls_widget.hide()

        self.results_table = QTableWidget()
        detail_layout.addWidget(self.results_table)

        self.study_table = QTableWidget()
        detail_layout.addWidget(self.study_table)
        self.study_table.hide()

        self.study_placeholder_label = QLabel("Study-level summaries will appear here once available.")
        self.study_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.study_placeholder_label.setStyleSheet("color: #777; font-size: 12px;")
        self.study_placeholder_label.hide()
        detail_layout.addWidget(self.study_placeholder_label)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(6, 4))
            self.canvas = FigureCanvas(self.figure)
            detail_layout.addWidget(self.canvas)
        else:
            placeholder = QLabel("Install matplotlib to enable plotting.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: gray;")
            detail_layout.addWidget(placeholder)

        detail_panel.setLayout(detail_layout)
        splitter.addWidget(detail_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)
        self.setLayout(layout)

        self.on_view_mode_changed()

    # --- Run selection and loading -------------------------------------------------

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]):
        self.workspace_manager = workspace_manager
        self.loaded_results.clear()
        self.run_metadata.clear()
        self.active_run_id = None
        self.refresh_runs()
        if self.view_mode_combo is not None:
            self.view_mode_combo.setCurrentIndex(0)

    def get_selected_runs(self) -> list[str]:
        selected_ids = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.run_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        ]

        expanded: List[str] = []
        seen = set()
        for run_id in selected_ids:
            if not run_id or run_id in seen:
                continue
            expanded.append(run_id)
            seen.add(run_id)
            metadata = self.run_metadata.get(run_id) or {}
            child_ids = metadata.get("child_run_ids") or []
            for child in child_ids:
                child_run = str(child)
                if child_run and child_run not in seen:
                    expanded.append(child_run)
                    seen.add(child_run)

        return expanded

    def refresh_runs(self, select_run_id: Optional[str] = None):
        self.run_list.blockSignals(True)
        self.run_list.clear()

        if self.workspace_manager is None:
            self.status_label.setText("Select a workspace to view results.")
            self.status_label.setStyleSheet("color: gray;")
            self.dataset_combo.clear()
            self.dataset_combo.setEnabled(False)
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.metadata_text.clear()
            if MATPLOTLIB_AVAILABLE:
                self.figure.clear()
                self.canvas.draw_idle()
            self.run_list.blockSignals(False)
            return

        runs = self.workspace_manager.list_runs()
        if not runs:
            self.status_label.setText("No runs have been recorded in this workspace.")
            self.status_label.setStyleSheet("color: gray;")
            self.run_list.blockSignals(False)
            return

        target_runs = []
        if select_run_id:
            target_runs.append(select_run_id)
        elif self.active_run_id:
            target_runs.append(self.active_run_id)

        target_set = set(target_runs)

        for info in runs:
            run_id = info.get("run_id") or Path(info.get("run_dir", "")).name
            status = info.get("status", "unknown")
            label = f"{run_id} ({status})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, run_id)
            self.run_metadata[run_id] = info
            self.run_list.addItem(item)
            if run_id in target_set:
                item.setSelected(True)

        if not target_runs and self.run_list.count() > 0:
            self.run_list.item(0).setSelected(True)

        self.run_list.blockSignals(False)
        self.on_run_selection_changed()

    def on_view_mode_changed(self):
        if self.view_mode_combo is None:
            return

        mode = self.view_mode_combo.currentData()
        is_study = mode == "study"

        if self.raw_controls_widget is not None:
            self.raw_controls_widget.setVisible(not is_study)
        if self.study_controls_widget is not None:
            self.study_controls_widget.setVisible(is_study)
        if self.study_table is not None:
            self.study_table.setVisible(is_study)
        self.results_table.setVisible(not is_study)
        if MATPLOTLIB_AVAILABLE:
            self.canvas.setVisible(True)
        if self.study_placeholder_label is not None:
            self.study_placeholder_label.setVisible(is_study)

        if is_study:
            self.update_study_view()
        else:
            dataset_name = self.dataset_combo.currentData()
            if dataset_name:
                self.display_dataset(dataset_name)

    def update_study_placeholder(self):
        if self.study_placeholder_label is None or not self.study_placeholder_label.isVisible():
            return

        metadata = self.run_metadata.get(self.active_run_id or "", {})
        run_type = metadata.get("run_type", "single")
        display_label = metadata.get("display_label") or metadata.get("label")
        stage_metrics = metadata.get("stage_metrics") or {}

        if not self.active_run_id:
            message = "Select a run to view study-level summaries."
        elif run_type == "single" and not stage_metrics:
            message = "Study summary view is available for multi-run workflows."
        else:
            message = "Use the controls above to explore stage metrics and aggregate plots."
            if display_label:
                message += f"\nActive label: {display_label}"

        self.study_placeholder_label.setText(message)

    # --- Study summary helpers -------------------------------------------------

    def collect_study_dataframe(self, run_ids: List[str]) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        stage_order_union: List[str] = []
        for run_id in run_ids:
            metadata = self.run_metadata.get(run_id, {}) or {}
            if (not metadata) and self.workspace_manager is not None:
                try:
                    metadata = self.workspace_manager.get_run_info(run_id) or {}
                except Exception:
                    metadata = {}
                if metadata:
                    self.run_metadata[run_id] = metadata
            run_type = metadata.get("run_type")
            overrides = metadata.get("parameter_overrides") or metadata.get("sweep_overrides") or {}
            summary_metrics = metadata.get("summary_metrics") or {}
            stage_metrics = metadata.get("stage_metrics") or {}

            stage_order_hint = metadata.get("stage_order") or []
            if not stage_order_hint and stage_metrics:
                stage_order_hint = list(stage_metrics.keys())
            stage_order_union = self._merge_stage_order(stage_order_union, stage_order_hint)

            record: Dict[str, Any] = {"run_id": run_id}
            label = metadata.get("display_label") or metadata.get("label")
            if label:
                record["label"] = label
            record["run_type"] = run_type

            for key, value in overrides.items():
                record[f"param:{key}"] = self._coerce_json_value(value)

            for key, value in summary_metrics.items():
                record[f"metric:overall:{key}"] = self._coerce_json_value(value)

            for stage, metrics in stage_metrics.items():
                if not isinstance(metrics, Mapping):
                    continue
                for key, value in metrics.items():
                    record[f"metric:{stage}:{key}"] = self._coerce_json_value(value)

            if run_type == "virtual_trial_group":
                summary_info = metadata.get("virtual_trial_summary") or {}
                summary_path = summary_info.get("summary_path")
                if summary_path:
                    summary_path = Path(summary_path)
                    if not summary_path.is_absolute():
                        results_dir = metadata.get("results_dir")
                        if results_dir:
                            summary_path = Path(results_dir) / summary_path
                    try:
                        summary_df = pd.read_parquet(summary_path)
                    except Exception:
                        summary_df = None
                    if summary_df is not None:
                        for _, row in summary_df.iterrows():
                            product = str(row.get("product") or "")
                            role = str(row.get("product_role") or "").strip()
                            metric_name = str(row.get("metric") or "")
                            prefix_parts = [part for part in (product, role, metric_name) if part]
                            if not prefix_parts:
                                continue
                            prefix = ":".join(prefix_parts)
                            for stat_key in ("count", "mean", "std", "geomean", "min", "max"):
                                if stat_key in row and pd.notna(row[stat_key]):
                                    record[f"metric:summary:{prefix}:{stat_key}"] = self._coerce_json_value(row[stat_key])
            elif run_type == "virtual_bioequivalence_group":
                summary_info = metadata.get("virtual_bioequivalence_summary") or {}
                summary_path = summary_info.get("summary_path")
                product_summary_path = summary_info.get("product_summary_path")
                if summary_path:
                    summary_path = Path(summary_path)
                    if not summary_path.is_absolute():
                        results_dir = metadata.get("results_dir")
                        if results_dir:
                            summary_path = Path(results_dir) / summary_path
                    try:
                        summary_df = pd.read_parquet(summary_path)
                    except Exception:
                        summary_df = None
                    if summary_df is not None:
                        for _, row in summary_df.iterrows():
                            metric_name = str(row.get("metric") or "")
                            product = str(row.get("test_product") or "")
                            prefix_parts = [part for part in (product, metric_name) if part]
                            if not prefix_parts:
                                continue
                            prefix = ":".join(prefix_parts)
                            for stat_key in ("gmr", "ci_lower", "ci_upper", "pass_80_125", "n"):
                                if stat_key in row and pd.notna(row[stat_key]):
                                    record[f"metric:summary:{prefix}:{stat_key}"] = self._coerce_json_value(row[stat_key])
                if product_summary_path:
                    product_summary_path = Path(product_summary_path)
                    if not product_summary_path.is_absolute():
                        results_dir = metadata.get("results_dir")
                        if results_dir:
                            product_summary_path = Path(results_dir) / product_summary_path
                    try:
                        product_summary_df = pd.read_parquet(product_summary_path)
                    except Exception:
                        product_summary_df = None
                    if product_summary_df is not None:
                        for _, row in product_summary_df.iterrows():
                            product = str(row.get("product") or "")
                            role = str(row.get("product_role") or "").strip()
                            metric_name = str(row.get("metric") or "")
                            prefix_parts = [part for part in (product, role, metric_name) if part]
                            if not prefix_parts:
                                continue
                            prefix = ":".join(prefix_parts)
                            for stat_key in ("mean", "std", "geomean", "count"):
                                if stat_key in row and pd.notna(row[stat_key]):
                                    record[f"metric:summary:{prefix}:{stat_key}"] = self._coerce_json_value(row[stat_key])

            if len(record) > 1:
                records.append(record)

        if not records:
            self.study_stage_order = []
            return pd.DataFrame()

        self.study_stage_order = stage_order_union
        return pd.DataFrame(records)

    def get_filtered_study_df(self) -> Optional[pd.DataFrame]:
        if self.study_records_df is None or self.study_records_df.empty:
            return None

        df = self.study_records_df.copy()

        param_x_col = self.study_param_x_combo.currentData() if self.study_param_x_combo else None
        if param_x_col and param_x_col in df.columns:
            series = pd.to_numeric(df[param_x_col], errors='coerce')
            df = df.assign(**{param_x_col: series})
            if self.study_param_x_min and self.study_param_x_max:
                low = min(self.study_param_x_min.value(), self.study_param_x_max.value())
                high = max(self.study_param_x_min.value(), self.study_param_x_max.value())
                df = df[(series >= low) & (series <= high)]

        param_y_col = self.study_param_y_combo.currentData() if self.study_param_y_combo else None
        if param_y_col and param_y_col in df.columns:
            series_y = pd.to_numeric(df[param_y_col], errors='coerce')
            df = df.assign(**{param_y_col: series_y})
            if self.study_param_y_min and self.study_param_y_max:
                low = min(self.study_param_y_min.value(), self.study_param_y_max.value())
                high = max(self.study_param_y_min.value(), self.study_param_y_max.value())
                df = df[(series_y >= low) & (series_y <= high)]

        return df

    def populate_study_controls(self):
        if self.study_records_df is None or self.study_records_df.empty:
            return

        param_columns = [col for col in self.study_records_df.columns if col.startswith("param:")]
        metric_columns = [col for col in self.study_records_df.columns if col.startswith("metric:")]

        stage_metrics_map: Dict[str, List[str]] = {}
        for column in metric_columns:
            parts = column.split(":", 2)
            if len(parts) == 3:
                stage_name, metric_key = parts[1], parts[2]
            elif len(parts) == 2:
                stage_name, metric_key = parts[1], parts[1]
            else:
                stage_name, metric_key = "overall", parts[-1]
            stage_metrics_map.setdefault(stage_name, []).append(column)

        ordered_map: Dict[str, List[str]] = {}
        for stage_name in self.study_stage_order:
            if stage_name in stage_metrics_map and stage_name not in ordered_map:
                ordered_map[stage_name] = stage_metrics_map[stage_name]
        for stage_name, columns in stage_metrics_map.items():
            if stage_name not in ordered_map and stage_name != "overall":
                ordered_map[stage_name] = columns
        if "overall" in stage_metrics_map:
            ordered_map["overall"] = stage_metrics_map["overall"]

        self.stage_metric_columns = ordered_map

        def _fill_combo(combo: QComboBox, values: List[str], include_none: bool = False):
            combo.blockSignals(True)
            combo.clear()
            if include_none:
                combo.addItem("None", userData=None)
            for value in values:
                combo.addItem(value.replace("param:", ""), userData=value)
            if combo.count() > 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        if self.study_param_x_combo is not None:
            _fill_combo(self.study_param_x_combo, param_columns, include_none=False)
        if self.study_param_y_combo is not None:
            _fill_combo(self.study_param_y_combo, param_columns, include_none=True)
        if self.study_stage_combo is not None:
            self.study_stage_combo.blockSignals(True)
            self.study_stage_combo.clear()
            self.study_stage_combo.addItem(self._stage_display_label(None), userData=None)
            for stage_name in self.stage_metric_columns.keys():
                display_stage = self._stage_display_label(stage_name)
                self.study_stage_combo.addItem(display_stage, userData=stage_name)
            self.study_stage_combo.blockSignals(False)

        self.update_metric_combo_for_stage()
        self.update_param_range_controls()

    def update_metric_combo_for_stage(self):
        if self.study_metric_combo is None:
            return

        stage = self.study_stage_combo.currentData() if self.study_stage_combo else None
        mapping = self.stage_metric_columns if hasattr(self, 'stage_metric_columns') else {}

        if stage:
            columns = list(mapping.get(stage, []))
        else:
            columns = []
            seen = set()
            for stage_name in getattr(self, 'study_stage_order', []):
                for col in mapping.get(stage_name, []):
                    if col not in seen:
                        columns.append(col)
                        seen.add(col)
            for stage_name, cols in mapping.items():
                for col in cols:
                    if col not in seen:
                        columns.append(col)
                        seen.add(col)

        filter_text = self._metric_filter_text()
        if filter_text:
            columns = [col for col in columns if filter_text in col.lower()]

        self.study_metric_combo.blockSignals(True)
        previous_selection = self.study_metric_combo.currentData()
        self.study_metric_combo.clear()
        self.study_metric_combo.addItem("All metrics", userData=None)

        for column in columns:
            parts = column.split(":", 2)
            label = parts[-1]
            if not stage and len(parts) == 3:
                label = f"{self._stage_display_label(parts[1])} · {parts[2]}"
            self.study_metric_combo.addItem(label, userData=column)
        if previous_selection is None:
            self.study_metric_combo.setCurrentIndex(0)
        else:
            idx = self.study_metric_combo.findData(previous_selection)
            if idx >= 0:
                self.study_metric_combo.setCurrentIndex(idx)
            elif self.study_metric_combo.count() > 0:
                self.study_metric_combo.setCurrentIndex(0)
        self.study_metric_combo.blockSignals(False)

    def update_param_range_controls(self):
        if self.study_records_df is None or self.study_records_df.empty:
            return

        param_x_col = self.study_param_x_combo.currentData() if self.study_param_x_combo else None
        if param_x_col:
            series_x = pd.to_numeric(self.study_records_df[param_x_col], errors='coerce').dropna()
            if not series_x.empty and self.study_param_x_min and self.study_param_x_max:
                min_val = float(series_x.min())
                max_val = float(series_x.max())
                self.study_param_x_min.blockSignals(True)
                self.study_param_x_max.blockSignals(True)
                self.study_param_x_min.setEnabled(True)
                self.study_param_x_max.setEnabled(True)
                self.study_param_x_min.setRange(min_val, max_val)
                self.study_param_x_max.setRange(min_val, max_val)
                self.study_param_x_min.setValue(min_val)
                self.study_param_x_max.setValue(max_val)
                self.study_param_x_min.blockSignals(False)
                self.study_param_x_max.blockSignals(False)
        elif self.study_param_x_min and self.study_param_x_max:
            self.study_param_x_min.setEnabled(False)
            self.study_param_x_max.setEnabled(False)

        param_y_col = self.study_param_y_combo.currentData() if self.study_param_y_combo else None
        if param_y_col:
            series_y = pd.to_numeric(self.study_records_df[param_y_col], errors='coerce').dropna()
            if not series_y.empty and self.study_param_y_min and self.study_param_y_max:
                min_val = float(series_y.min())
                max_val = float(series_y.max())
                self.study_param_y_min.blockSignals(True)
                self.study_param_y_max.blockSignals(True)
                self.study_param_y_min.setEnabled(True)
                self.study_param_y_max.setEnabled(True)
                self.study_param_y_min.setRange(min_val, max_val)
                self.study_param_y_max.setRange(min_val, max_val)
                self.study_param_y_min.setValue(min_val)
                self.study_param_y_max.setValue(max_val)
                self.study_param_y_min.blockSignals(False)
                self.study_param_y_max.blockSignals(False)
        elif self.study_param_y_min and self.study_param_y_max:
            self.study_param_y_min.setEnabled(False)
            self.study_param_y_max.setEnabled(False)

    def on_study_controls_changed(self):
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            sender = self.sender()
            if isinstance(sender, QComboBox):
                if sender is self.study_stage_combo:
                    self.update_metric_combo_for_stage()
                self.update_param_range_controls()
            elif sender is self.study_metric_filter_edit:
                self.update_metric_combo_for_stage()
            self.update_study_plot()
            self.update_study_table()

    def update_study_view(self):
        if self.view_mode_combo is None or self.view_mode_combo.currentData() != "study":
            return

        selected_runs = self.get_selected_runs()
        if not selected_runs:
            self.study_records_df = None
            if self.study_controls_widget is not None:
                self.study_controls_widget.hide()
            if self.study_table is not None:
                self.study_table.hide()
            if self.study_placeholder_label is not None:
                self.study_placeholder_label.show()
                self.study_placeholder_label.setText("Select runs to view study summaries.")
            if MATPLOTLIB_AVAILABLE:
                self.figure.clear()
                self.canvas.draw_idle()
            return

        df = self.collect_study_dataframe(selected_runs)
        self.study_records_df = df

        if df.empty:
            self.study_records_df = None
            if self.study_controls_widget is not None:
                self.study_controls_widget.hide()
            if self.study_table is not None:
                self.study_table.hide()
            if self.study_placeholder_label is not None:
                self.study_placeholder_label.show()
                self.study_placeholder_label.setText("No study metadata available for the selected runs.")
            if MATPLOTLIB_AVAILABLE:
                self.figure.clear()
                self.canvas.draw_idle()
            return

        if self.study_placeholder_label is not None:
            self.study_placeholder_label.hide()
        if self.study_controls_widget is not None:
            self.study_controls_widget.show()
        if self.study_table is not None:
            self.study_table.show()

        self.populate_study_controls()
        self.update_study_table()
        self.update_study_plot()

    def update_study_table(self):
        if self.study_table is None:
            return

        df = self.get_filtered_study_df()
        if df is None or df.empty:
            self.study_table.setRowCount(0)
            self.study_table.setColumnCount(0)
            return

        metric_columns = self._available_metric_columns(df)
        selected_metric = self.study_metric_combo.currentData() if self.study_metric_combo else None
        if selected_metric and selected_metric in metric_columns:
            metric_columns = [selected_metric]

        if not metric_columns:
            self.study_table.setRowCount(0)
            self.study_table.setColumnCount(0)
            return

        numeric_df = df[metric_columns].apply(pd.to_numeric, errors='coerce')
        if numeric_df.empty:
            self.study_table.setRowCount(0)
            self.study_table.setColumnCount(0)
            return

        describe_df = numeric_df.describe(percentiles=[0.05, 0.5, 0.95]).T
        describe_df.rename(columns={"50%": "median", "5%": "p05", "95%": "p95"}, inplace=True)
        describe_df["geomean"] = numeric_df.apply(self._geometric_mean, axis=0)
        describe_df.reset_index(inplace=True)
        describe_df.rename(columns={"index": "metric"}, inplace=True)
        describe_df["metric"] = describe_df["metric"].apply(self._format_metric_column_label)

        self.study_table.setRowCount(len(describe_df))
        self.study_table.setColumnCount(len(describe_df.columns))
        self.study_table.setHorizontalHeaderLabels([str(col) for col in describe_df.columns])

        for row_idx in range(len(describe_df)):
            for col_idx, column in enumerate(describe_df.columns):
                value = describe_df.iloc[row_idx, col_idx]
                display_value = "" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (int, float, np.floating)) else str(value)
                self.study_table.setItem(row_idx, col_idx, QTableWidgetItem(display_value))

        self.study_table.resizeColumnsToContents()

    def update_study_plot(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        plot_mode = self.study_plot_type_combo.currentData() if self.study_plot_type_combo else "scatter"
        param_x = self.study_param_x_combo.currentData() if self.study_param_x_combo else None
        param_y = self.study_param_y_combo.currentData() if self.study_param_y_combo else None
        metric_col = self.study_metric_combo.currentData() if self.study_metric_combo else None

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        df = self.get_filtered_study_df()

        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes, color='0.5')
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        available_metrics = self._available_metric_columns(df)
        if metric_col is None or metric_col not in available_metrics:
            if available_metrics:
                metric_col = available_metrics[0]
            else:
                ax.text(0.5, 0.5, "No metrics available for plotting", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
                self.canvas.draw_idle()
                return

        if param_x not in df.columns:
            ax.text(0.5, 0.5, "Select at least one parameter", ha='center', va='center', transform=ax.transAxes, color='0.5')
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        x = pd.to_numeric(df[param_x], errors='coerce')
        y_metric = pd.to_numeric(df[metric_col], errors='coerce')

        parts = metric_col.split(":", 2)
        if len(parts) == 3:
            stage_key, metric_name = parts[1], parts[2]
        else:
            stage_key, metric_name = "overall", parts[-1]
        stage_label = self._stage_display_label(stage_key)

        if plot_mode == "heatmap":
            if not param_y or param_y not in df.columns:
                ax.text(0.5, 0.5, "Select Parameter Y for heatmap", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                y_param = pd.to_numeric(df[param_y], errors='coerce')
                pivot = pd.concat([x.rename('x'), y_param.rename('y'), y_metric.rename('value')], axis=1).dropna()
                if pivot.empty:
                    ax.text(0.5, 0.5, "Not enough data for heatmap", ha='center', va='center', transform=ax.transAxes, color='0.5')
                    ax.set_axis_off()
                else:
                    grid = pivot.pivot_table(index='y', columns='x', values='value', aggfunc='mean')
                    im = ax.imshow(grid.values, aspect='auto', origin='lower')
                    ax.set_xticks(range(len(grid.columns)))
                    ax.set_xticklabels([f"{val:.3g}" for val in grid.columns])
                    ax.set_yticks(range(len(grid.index)))
                    ax.set_yticklabels([f"{val:.3g}" for val in grid.index])
                    ax.set_xlabel(param_x.replace("param:", ""))
                    ax.set_ylabel(param_y.replace("param:", ""))
                    ax.set_title(f"{stage_label} · {metric_name}")
                    self.figure.colorbar(im, ax=ax, label='Mean value')
        else:
            valid = pd.concat([x.rename('x'), y_metric.rename('y')], axis=1).dropna()
            if valid.empty:
                ax.text(0.5, 0.5, "No numeric data available for plot", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                if plot_mode == "line":
                    valid = valid.sort_values('x')
                    ax.plot(valid['x'], valid['y'], marker='o')
                else:
                    ax.scatter(valid['x'], valid['y'], alpha=0.8)
                ax.set_xlabel(param_x.replace("param:", ""))
                ax.set_ylabel(metric_name)
                ax.set_title(f"{stage_label} · {metric_name} vs {param_x.replace('param:', '')}")

        self.canvas.draw_idle()

    def on_run_selection_changed(self):
        selected_runs = self.get_selected_runs()
        if not selected_runs:
            self.active_run_id = None
            self.metadata_text.clear()
            self.dataset_combo.clear()
            self.dataset_combo.setEnabled(False)
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.status_label.setText("Select a run to view results.")
            self.status_label.setStyleSheet("color: gray;")
            if self.run_summary_label is not None:
                self.run_summary_label.setText("")
            if MATPLOTLIB_AVAILABLE:
                self.figure.clear()
                self.canvas.draw_idle()
            self.update_study_placeholder()
            return

        for run_id in selected_runs:
            if run_id not in self.loaded_results:
                self.load_run(run_id, make_active=False)

        self.set_active_run(selected_runs[0])
        self.update_plots()
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            self.update_study_view()

    def set_active_run(self, run_id: str):
        try:
            refreshed = self.workspace_manager.load_run_results(run_id)
        except Exception:
            if run_id not in self.loaded_results:
                self.load_run(run_id, make_active=False)
                refreshed = self.loaded_results.get(run_id)
            else:
                refreshed = self.loaded_results.get(run_id)
        else:
            self.loaded_results[run_id] = refreshed

        self.active_run_id = run_id
        metadata = self.run_metadata.get(run_id, {})
        run_type = metadata.get("run_type", "single")

        self.metadata_text.setPlainText(json.dumps(metadata, indent=2, default=str))
        self.status_label.setText(f"Loaded run {run_id} (status: {metadata.get('status', 'unknown')})")
        self.status_label.setStyleSheet("color: #2e7d32;")

        if self.run_summary_label is not None:
            display_label = metadata.get("display_label") or metadata.get("label")
            summary_parts = [f"Run type: {run_type}"]
            if display_label:
                summary_parts.append(f"Label: {display_label}")
            if run_type == "parameter_estimation_group":
                summary = metadata.get("parameter_estimation_summary") or {}
                best_run_id = summary.get("best_run_id")
                best_objective = summary.get("best_objective")
                if best_run_id:
                    try:
                        best_value = float(best_objective) if best_objective is not None else None
                    except Exception:
                        best_value = None
                    if best_value is not None:
                        summary_parts.append(f"Best: {best_run_id} (objective {best_value:.3g})")
                    else:
                        summary_parts.append(f"Best: {best_run_id}")
            elif run_type == "virtual_trial_group":
                summary = metadata.get("virtual_trial_summary") or {}
                n_subjects = summary.get("n_subjects")
                if n_subjects:
                    summary_parts.append(f"Subjects: {n_subjects}")
            elif run_type == "virtual_bioequivalence_group":
                summary = metadata.get("virtual_bioequivalence_summary") or {}
                ref_product = summary.get("reference_product")
                test_products = summary.get("test_products")
                if ref_product:
                    summary_parts.append(f"Reference: {ref_product}")
                if test_products:
                    summary_parts.append(f"Tests: {', '.join(test_products)}")
            self.run_summary_label.setText(" • ".join(summary_parts))
        self.update_study_placeholder()
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            self.update_study_view()

        datasets = sorted(self.loaded_results.get(run_id, {}).keys())
        run_type = metadata.get("run_type", "single")
        if run_type == "parameter_estimation_group":
            priority = [
                "parameter_estimation_summary",
                "parameter_estimation_residuals",
                "parameter_estimation_overlay",
                "parameter_estimation_regional",
            ]
        elif run_type == "virtual_trial_group":
            priority = [
                "virtual_trial_summary",
                "virtual_trial_subjects",
            ]
        elif run_type == "virtual_bioequivalence_group":
            priority = [
                "virtual_bioequivalence_summary",
                "virtual_bioequivalence_product_summary",
                "virtual_bioequivalence_subjects",
            ]
        else:
            priority = []
        if priority:
            ordered = [name for name in priority if name in datasets]
            ordered.extend([name for name in datasets if name not in ordered])
            datasets = ordered
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        if datasets:
            aggregated_names: List[str] = []
            added_any = False
            summary_datasets = {
                "virtual_trial_summary",
                "virtual_bioequivalence_summary",
                "virtual_bioequivalence_product_summary",
                "virtual_bioequivalence_subjects",
            }
            for name in datasets:
                if name in summary_datasets:
                    continue
                df = self.loaded_results[run_id][name]
                base_dataset, dataset_prefix = self._split_dataset_name(name)
                if dataset_prefix:
                    label = f"Aggregated {dataset_prefix} · {base_dataset} ({len(df)} rows)"
                    aggregated_names.append(name)
                else:
                    label = f"{name} ({len(df)} rows)"
                self.dataset_combo.addItem(label, userData=name)
                added_any = True
            if not added_any:
                self.dataset_combo.addItem("No raw datasets", userData=None)
                self.dataset_combo.setEnabled(False)
            else:
                self.dataset_combo.setEnabled(True)

                default_index = 0
                if aggregated_names:
                    first_aggregated = aggregated_names[0]
                    idx = self.dataset_combo.findData(first_aggregated)
                    if idx >= 0:
                        default_index = idx
                self.dataset_combo.setCurrentIndex(default_index)
        else:
            self.dataset_combo.addItem("No datasets", userData=None)
            self.dataset_combo.setEnabled(False)
        self.dataset_combo.blockSignals(False)

        dataset_name = self.dataset_combo.currentData()
        if dataset_name:
            self.display_dataset(dataset_name)
        else:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.update_plots()

    def load_run(self, run_id: str, make_active: bool = True):
        if self.workspace_manager is None:
            return

        info = self.workspace_manager.get_run_info(run_id)
        if info is None:
            QMessageBox.warning(self, "Run Not Found", f"Run {run_id} could not be located.")
            return

        try:
            results = self.workspace_manager.load_run_results(run_id)
        except FileNotFoundError:
            results = {}
        except Exception as exc:
            QMessageBox.critical(self, "Results Error", f"Failed to load results for {run_id}: {exc}")
            results = {}

        self.loaded_results[run_id] = results
        self.run_metadata[run_id] = info

        if make_active:
            self.set_active_run(run_id)

    # --- Dataset display and plotting ---------------------------------------------

    def on_dataset_changed(self, index: int):
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            self.update_study_placeholder()
            return
        dataset_name = self.dataset_combo.itemData(index)
        if not dataset_name:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.update_plots()
            return

        base_dataset, _ = self._split_dataset_name(dataset_name)

        if base_dataset != "pbpk_regional_timeseries":
            self.pbpk_controls_widget.hide()

        if self.plot_type_combo and self.plot_type_combo.currentData() == "none":
            default_plot = self.dataset_plot_defaults.get(base_dataset)
            if default_plot:
                idx = self.plot_type_combo.findData(default_plot)
                if idx >= 0:
                    self.plot_type_combo.setCurrentIndex(idx)

        self.display_dataset(dataset_name)

    def populate_pbpk_controls(self):
        selected_runs = self.get_selected_runs()
        if not selected_runs and self.active_run_id:
            selected_runs = [self.active_run_id]

        run_frames = []
        for run_id in selected_runs:
            df = self.loaded_results.get(run_id, {}).get("pbpk_regional_timeseries")
            if df is not None and not df.empty:
                run_frames.append(df)

        if not run_frames:
            self.pbpk_combined_df = None
            self.pbpk_controls_widget.hide()
            return None

        combined = pd.concat(run_frames, ignore_index=True)
        self.pbpk_combined_df = combined

        required_columns = {"region", "compartment", "quantity", "time_h", "value"}
        if not required_columns.issubset(set(combined.columns)):
            self.status_label.setText(
                "PBPK results need regeneration to enable the detailed viewer. Re-run the simulation with the updated package."
            )
            self.status_label.setStyleSheet("color: #f57c00;")
            self.pbpk_controls_widget.hide()
            return None

        prev_regions = self._list_selected_values(self.pbpk_region_list)
        prev_compartments = self._list_selected_values(self.pbpk_compartment_list)
        prev_quantities = self._list_selected_values(self.pbpk_quantity_list)
        prev_bindings = self._list_selected_values(self.pbpk_binding_list)

        regions = self._ordered_unique(combined["region"])
        compartments = self._ordered_unique(combined["compartment"])
        quantities = self._ordered_unique(combined["quantity"])
        bindings = self._ordered_unique(combined["binding"]) if "binding" in combined.columns else []

        self._populate_list_widget(self.pbpk_region_list, self.pbpk_region_label, regions, prev_regions)
        self._populate_list_widget(self.pbpk_compartment_list, self.pbpk_compartment_label, compartments, prev_compartments)
        self._populate_list_widget(self.pbpk_quantity_list, self.pbpk_quantity_label, quantities, prev_quantities)

        if bindings:
            self._populate_list_widget(self.pbpk_binding_list, self.pbpk_binding_label, bindings, prev_bindings)
        else:
            self.pbpk_binding_list.blockSignals(True)
            self.pbpk_binding_list.clear()
            self.pbpk_binding_list.blockSignals(False)
            self.pbpk_binding_list.setVisible(False)
            self.pbpk_binding_list.setEnabled(False)
            self.pbpk_binding_label.setVisible(False)

        self.pbpk_controls_widget.show()
        return combined

    def on_pbpk_control_changed(self):
        if self.dataset_combo.currentData() == "pbpk_regional_timeseries":
            self.display_dataset("pbpk_regional_timeseries")

    def display_dataset(self, dataset_name: str):
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            self.update_study_placeholder()
            return
        if self.active_run_id is None:
            return

        base_dataset, dataset_prefix = self._split_dataset_name(dataset_name)

        if base_dataset == "pbpk_regional_timeseries":
            combined = self.populate_pbpk_controls()
            if combined is None:
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                self.update_plots()
                return

            df = self.loaded_results.get(self.active_run_id, {}).get(dataset_name)
            if df is not None and not df.empty:
                filtered = df.copy()

                regions_available = combined["region"].dropna().unique().tolist()
                regions_selected = self._get_selected_or_all(self.pbpk_region_list, regions_available)
                if regions_selected:
                    filtered = filtered[filtered["region"].isin(regions_selected)]

                compartments_available = combined["compartment"].dropna().unique().tolist()
                compartments_selected = self._get_selected_or_all(self.pbpk_compartment_list, compartments_available)
                if compartments_selected:
                    filtered = filtered[filtered["compartment"].isin(compartments_selected)]

                quantities_available = combined["quantity"].dropna().unique().tolist()
                quantities_selected = self._get_selected_or_all(self.pbpk_quantity_list, quantities_available)
                if quantities_selected:
                    filtered = filtered[filtered["quantity"].isin(quantities_selected)]

                if "binding" in combined.columns:
                    bindings_available = combined["binding"].dropna().unique().tolist()
                    bindings_selected = self._get_selected_or_all(self.pbpk_binding_list, bindings_available) if bindings_available else []
                    if bindings_selected:
                        filtered = filtered[filtered["binding"].isin(bindings_selected)]

                filtered = filtered.sort_values(["time_h", "time_s"], kind="mergesort")
                df = filtered if not filtered.empty else None
            else:
                df = None
        else:
            self.pbpk_controls_widget.hide()
            df = self.loaded_results.get(self.active_run_id, {}).get(dataset_name)

        if df is None or df.empty:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.update_plots()
            return

        max_rows = 200
        subset = df.head(max_rows)

        self.results_table.setRowCount(len(subset))
        self.results_table.setColumnCount(len(subset.columns))
        self.results_table.setHorizontalHeaderLabels([str(col) for col in subset.columns])

        for row_idx in range(len(subset)):
            for col_idx, column in enumerate(subset.columns):
                value = subset.iloc[row_idx, col_idx]
                display_value = "" if pd.isna(value) else str(value)
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(display_value))

        if len(df) > max_rows:
            remaining = len(df) - max_rows
            self.status_label.setText(
                f"Showing first {max_rows} of {len(df)} rows for {dataset_name}. {remaining} additional rows not displayed."
            )
            self.status_label.setStyleSheet("color: #f57c00;")
        else:
            if dataset_name == "parameter_estimation_summary":
                summary_meta = (self.run_metadata.get(self.active_run_id or "", {}).get("parameter_estimation_summary") or {})
                best_run = summary_meta.get("best_run_id") or "n/a"
                best_objective = summary_meta.get("best_objective")
                if isinstance(best_objective, (int, float)):
                    best_text = f"Best: {best_run} (objective {best_objective:.4g})"
                else:
                    best_text = f"Best: {best_run}"
                self.status_label.setText(f"Parameter estimation summary across {len(df)} runs. {best_text}.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_residuals":
                run_count = df["run_id"].nunique() if "run_id" in df.columns else 0
                self.status_label.setText(f"Residual samples for {run_count} run(s), {len(df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_regional":
                run_count = df["run_id"].nunique() if "run_id" in df.columns else 0
                self.status_label.setText(f"Regional residuals for {run_count} run(s), {len(df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_overlay":
                series_count = df["series"].nunique() if "series" in df.columns else 0
                self.status_label.setText(f"Overlay dataset with {series_count} series and {len(df)} points.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_trial_summary":
                self.status_label.setText(f"Virtual trial summary across {len(df)} product/metric rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_trial_subjects":
                run_count = df["run_id"].nunique() if "run_id" in df.columns else 0
                self.status_label.setText(f"Virtual trial subjects: {run_count} run(s), {len(df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_summary":
                self.status_label.setText(f"VBE summary metrics across {len(df)} entries.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_subjects":
                run_count = df["run_id"].nunique() if "run_id" in df.columns else 0
                self.status_label.setText(f"VBE subject metrics for {run_count} run(s), {len(df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_product_summary":
                self.status_label.setText(f"Product-level summary across {len(df)} entries.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            else:
                self.status_label.setText(f"Displaying {len(df)} rows for {dataset_name}.")
                self.status_label.setStyleSheet("color: #2e7d32;")

        self.update_plots()

    def update_plots(self):
        if self.view_mode_combo and self.view_mode_combo.currentData() == "study":
            return
        if not MATPLOTLIB_AVAILABLE:
            return

        dataset_name = self.dataset_combo.currentData()
        base_dataset, dataset_prefix = self._split_dataset_name(dataset_name)
        plot_mode = self.plot_type_combo.currentData() if self.plot_type_combo else "none"
        self.figure.clear()

        if plot_mode in (None, "none"):
            self.canvas.draw_idle()
            return

        selected_runs = self.get_selected_runs()
        if not selected_runs and self.active_run_id:
            selected_runs = [self.active_run_id]

        if not selected_runs:
            self.canvas.draw_idle()
            return

        ax = self.figure.add_subplot(111)

        if plot_mode in {"pk_linear", "pk_log"} and base_dataset == "pk_curve":
            for run_id in selected_runs:
                df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
                if df is None or df.empty or not {"t", "plasma_conc"}.issubset(df.columns):
                    continue
                df_sorted = df.sort_values("t")
                if dataset_prefix:
                    label = dataset_prefix if len(selected_runs) == 1 else f"{run_id}:{dataset_prefix}"
                else:
                    label = run_id
                ax.plot(df_sorted["t"], df_sorted["plasma_conc"], label=label)

            if not ax.has_data():
                ax.text(0.5, 0.5, "No PK data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                ax.set_title("Plasma Concentration")
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration")
                if plot_mode == "pk_log":
                    ax.set_yscale('log')
                ax.legend()

        elif plot_mode in {"dep_pmol", "dep_fraction"} and base_dataset == "deposition_bins":
            value_column = "amount_pmol" if plot_mode == "dep_pmol" else "fraction_of_dose"
            ylabel = "Amount (pmol)" if plot_mode == "dep_pmol" else "Fraction of dose"

            regions = set()
            aggregated: Dict[str, pd.Series] = {}
            for run_id in selected_runs:
                df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
                if df is None or value_column not in df.columns:
                    continue
                series = df.groupby("region")[value_column].sum()
                label = dataset_prefix if dataset_prefix else run_id
                if dataset_prefix and len(selected_runs) > 1:
                    label = f"{run_id}:{dataset_prefix}"
                aggregated[label] = series
                regions.update(series.index)

            if not aggregated:
                ax.text(0.5, 0.5, "No deposition data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                regions = sorted(regions)
                x = np.arange(len(regions))
                bar_width = 0.8 / max(len(aggregated), 1)
                for idx, (run_id, series) in enumerate(aggregated.items()):
                    values = series.reindex(regions, fill_value=0)
                    offset = x - 0.4 + bar_width / 2 + idx * bar_width
                    ax.bar(offset, values.values, width=bar_width, label=run_id)
                ax.set_xticks(x)
                ax.set_xticklabels(regions)
                ax.set_xlabel("Region")
                ax.set_ylabel(ylabel)
                ax.set_title("Regional Deposition")
                ax.legend()

        elif plot_mode == "dep_heatmap" and base_dataset == "deposition_bins":
            if dataset_prefix:
                run_id = selected_runs[0]
                df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
                run_label = dataset_prefix if len(selected_runs) == 1 else f"{run_id}:{dataset_prefix}"
            else:
                run_id = selected_runs[0]
                df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
                run_label = run_id
            if df is None or "fraction_of_dose" not in df.columns:
                ax.text(0.5, 0.5, "No deposition data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                pivot = df.pivot_table(index="region", columns="particle_um", values="fraction_of_dose", aggfunc="sum")
                if pivot.empty:
                    ax.text(0.5, 0.5, "No deposition data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                    ax.set_axis_off()
                else:
                    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
                    ax.set_xticks(np.arange(len(pivot.columns)))
                    ax.set_xticklabels([str(x) for x in pivot.columns])
                    ax.set_yticks(np.arange(len(pivot.index)))
                    ax.set_yticklabels(pivot.index)
                    ax.set_xlabel("Particle size (μm)")
                    ax.set_ylabel("Region")
                    ax.set_title(f"Deposition Heatmap ({run_label})")
                    self.figure.colorbar(im, ax=ax, label='Fraction of dose')

        elif plot_mode == "regional_auc_bar" and base_dataset == "regional_auc":
            datasets = []
            regions = set()
            for run_id in selected_runs:
                df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
                if df is None or df.empty:
                    continue
                label = dataset_prefix if dataset_prefix else run_id
                if dataset_prefix and len(selected_runs) > 1:
                    label = f"{run_id}:{dataset_prefix}"
                datasets.append((label, df))
                regions.update(df["region"].dropna().unique())
            if not datasets:
                ax.text(0.5, 0.5, "No regional AUC data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                regions = sorted(regions)
                x = np.arange(len(regions))
                bar_width = 0.8 / max(len(datasets), 1)
                metric_specs = [
                    ("auc_elf", "ELF"),
                    ("auc_epithelium_pmol_h_per_ml", "Epithelium"),
                    ("auc_tissue_pmol_h_per_ml", "Tissue"),
                    ("auc_epithelium_tissue_pmol_h_per_ml", "Epi+Tissue"),
                    ("auc_epithelium_unbound_pmol_h_per_ml", "Epithelium Unbound"),
                    ("auc_tissue_unbound_pmol_h_per_ml", "Tissue Unbound"),
                    ("auc_epithelium_tissue_unbound_pmol_h_per_ml", "Epi+Tissue Unbound")
                ]

                available_metrics = []
                for column, label in metric_specs:
                    for _, df in datasets:
                        if column in df.columns and not df[column].fillna(0).eq(0).all():
                            available_metrics.append((column, label))
                            break

                if not available_metrics:
                    available_metrics = [("auc_elf", "ELF")]

                for idx, (run_id, df) in enumerate(datasets):
                    regional_df = df.set_index("region")
                    offsets = x - 0.4 + bar_width / 2 + idx * bar_width
                    bottom = np.zeros(len(regions))
                    for column, label in available_metrics:
                        if column not in regional_df.columns:
                            continue
                        values = regional_df[column].reindex(regions).fillna(0).to_numpy(dtype=float)
                        if np.allclose(values, 0):
                            continue
                        ax.bar(
                            offsets,
                            values,
                            width=bar_width,
                            bottom=bottom,
                            label=f"{run_id} - {label}"
                        )
                        bottom += values

                ax.set_xticks(x)
                ax.set_xticklabels(regions)
                ax.set_xlabel("Region")
                ax.set_ylabel("Regional AUC (pmol·h/mL)")
                ax.set_title("Regional AUC (stacked)")
                ax.legend()
        elif plot_mode == "parameter_estimation_residuals":
            df = self.loaded_results.get(self.active_run_id, {}).get("parameter_estimation_residuals")
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No residual data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                child_runs = [rid for rid in selected_runs if rid != self.active_run_id]
                if not child_runs:
                    child_runs = df["run_id"].dropna().unique().tolist()
                for child_id in child_runs:
                    subset = df[df["run_id"] == child_id]
                    if subset.empty:
                        continue
                    ax.plot(subset["time_h"], subset["residual"], marker='o', label=child_id)
                ax.axhline(0.0, color='black', linewidth=1, linestyle='--')
                ax.set_xlabel("Time (h)")
                ax.set_ylabel("Residual (ng/mL)")
                ax.set_title("Parameter Estimation Residuals")
                ax.legend()

        elif plot_mode == "parameter_estimation_regional":
            df = self.loaded_results.get(self.active_run_id, {}).get("parameter_estimation_regional")
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No regional residual data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                child_runs = [rid for rid in selected_runs if rid != self.active_run_id]
                if not child_runs:
                    child_runs = df["run_id"].dropna().unique().tolist()
                regions = sorted(df["region"].dropna().unique())
                if not regions:
                    ax.text(0.5, 0.5, "No regions found", ha='center', va='center', transform=ax.transAxes, color='0.5')
                    ax.set_axis_off()
                else:
                    x = np.arange(len(regions))
                    bar_width = 0.8 / max(len(child_runs), 1)
                    for idx, child_id in enumerate(child_runs):
                        subset = df[df["run_id"] == child_id].set_index("region")
                        values = subset["residual"].reindex(regions).fillna(0.0).astype(float)
                        offset = x - 0.4 + (idx + 0.5) * bar_width
                        ax.bar(offset, values.values, width=bar_width, label=child_id)
                    ax.axhline(0.0, color='black', linewidth=1, linestyle='--')
                    ax.set_xticks(x)
                    ax.set_xticklabels(regions, rotation=30, ha='right')
                    ax.set_xlabel("Region")
                    ax.set_ylabel("Residual (predicted - observed)")
                    ax.set_title("Deposition Fraction Residuals")
                    ax.legend()

        elif plot_mode == "parameter_estimation_overlay":
            df = self.loaded_results.get(self.active_run_id, {}).get("parameter_estimation_overlay")
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No overlay data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                for series, subset in df.groupby("series"):
                    subset_sorted = subset.sort_values("time_h")
                    if "concentration_ng_ml" in subset_sorted.columns:
                        values = subset_sorted["concentration_ng_ml"]
                    elif "value" in subset_sorted.columns:
                        values = subset_sorted["value"]
                    else:
                        values = subset_sorted.get("observed", pd.Series(dtype=float))
                    ax.plot(
                        subset_sorted["time_h"],
                        values,
                        marker='o' if "observed" in str(series).lower() else None,
                        label=str(series)
                    )
                ax.set_xlabel("Time (h)")
                ax.set_ylabel("Value")
                ax.set_title("Observed vs Predicted (Best Run)")
                ax.legend()

        elif plot_mode == "pbpk_timeseries":
            if not self.pbpk_controls_widget.isVisible():
                ax.text(0.5, 0.5, "Select a PBPK dataset", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                combined = self.pbpk_combined_df
                if combined is None or combined.empty:
                    ax.text(0.5, 0.5, "No PBPK data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                    ax.set_axis_off()
                else:
                    regions_available = combined["region"].dropna().unique().tolist()
                    compartments_available = combined["compartment"].dropna().unique().tolist()
                    quantities_available = combined["quantity"].dropna().unique().tolist()
                    bindings_available = combined["binding"].dropna().unique().tolist() if "binding" in combined.columns else []

                    regions_selected = self._get_selected_or_all(self.pbpk_region_list, regions_available)
                    compartments_selected = self._get_selected_or_all(self.pbpk_compartment_list, compartments_available)
                    quantities_selected = self._get_selected_or_all(self.pbpk_quantity_list, quantities_available)
                    bindings_selected = self._get_selected_or_all(self.pbpk_binding_list, bindings_available) if bindings_available else []

                    if not regions_selected or not compartments_selected or not quantities_selected:
                        ax.text(0.5, 0.5, "Select region, compartment, and quantity", ha='center', va='center', transform=ax.transAxes, color='0.5')
                        ax.set_axis_off()
                    else:
                        multi_dim = {
                            "region": len(regions_selected) > 1,
                            "compartment": len(compartments_selected) > 1,
                            "quantity": len(quantities_selected) > 1,
                            "binding": len(bindings_selected) > 1 if bindings_available else False,
                        }

                        units_seen: List[str] = []
                        for run_id in selected_runs:
                            run_df = self.loaded_results.get(run_id, {}).get("pbpk_regional_timeseries")
                            if run_df is None or run_df.empty:
                                continue

                            run_filtered = run_df[
                                run_df["region"].isin(regions_selected)
                                & run_df["compartment"].isin(compartments_selected)
                                & run_df["quantity"].isin(quantities_selected)
                            ]

                            if bindings_available:
                                active_bindings = bindings_selected if bindings_selected else bindings_available
                                run_filtered = run_filtered[run_filtered["binding"].isin(active_bindings)]

                            if run_filtered.empty:
                                continue

                            grouping_columns = ["region", "compartment", "quantity"]
                            if bindings_available and "binding" in run_filtered.columns:
                                grouping_columns.append("binding")

                            unique_combos = run_filtered[grouping_columns].drop_duplicates()

                            for _, combo_row in unique_combos.iterrows():
                                subset = run_filtered
                                for col in grouping_columns:
                                    value = combo_row[col]
                                    if pd.isna(value):
                                        subset = subset[subset[col].isna()]
                                    else:
                                        subset = subset[subset[col] == value]
                                if subset.empty:
                                    continue

                                subset = subset.sort_values(["time_h", "time_s"], kind="mergesort")
                                units_column = subset["units"].dropna().unique()
                                if len(units_column):
                                    units_seen.append(units_column[0])

                                label_parts = [run_id]
                                for col in ["region", "compartment", "quantity", "binding"]:
                                    if col in subset.columns and multi_dim.get(col, False):
                                        label_parts.append(self._format_pbpk_label(combo_row.get(col)))

                                line_label = " | ".join(label_parts)
                                ax.plot(subset["time_h"], subset["value"], label=line_label)

                        if not ax.has_data():
                            ax.text(0.5, 0.5, "No PBPK data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                            ax.set_axis_off()
                        else:
                            region_label = "Multiple Regions" if multi_dim.get("region", False) else self._format_pbpk_label(regions_selected[0])
                            compartment_label = "Multiple Compartments" if multi_dim.get("compartment", False) else self._format_pbpk_label(compartments_selected[0])
                            quantity_label = "Multiple Quantities" if multi_dim.get("quantity", False) else self._format_pbpk_label(quantities_selected[0])
                            title = f"PBPK Time Series - {region_label} / {compartment_label} ({quantity_label})"
                            if bindings_available:
                                if multi_dim.get("binding", False):
                                    binding_label = "Multiple Binding States"
                                elif bindings_selected:
                                    binding_label = self._format_pbpk_label(bindings_selected[0])
                                else:
                                    binding_label = self._format_pbpk_label(bindings_available[0]) if bindings_available else "Total"
                                title += f" [{binding_label}]"
                            ax.set_title(title)
                            ax.set_xlabel("Time (h)")
                            unit_text = next((u for u in units_seen if u), None)
                            ax.set_ylabel(f"Value ({unit_text})" if unit_text else "Value")
                            ax.legend()

        self.canvas.draw_idle()

    # --- Export -------------------------------------------------------------------

    def export_selected_runs(self):
        if self.workspace_manager is None:
            QMessageBox.warning(self, "No Workspace", "Select a workspace before exporting runs.")
            return

        runs = self.get_selected_runs()
        if not runs:
            QMessageBox.warning(self, "No Selection", "Select at least one run to export.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select export directory")
        if not target_dir:
            return

        exported = []
        errors = []
        for run_id in runs:
            run_dir = self.workspace_manager.runs_dir / run_id
            if not run_dir.exists():
                errors.append(f"Run {run_id} directory not found.")
                continue
            try:
                base_name = Path(target_dir) / run_id
                archive_path = shutil.make_archive(str(base_name), 'zip', run_dir)
                exported.append(Path(archive_path).name)
            except Exception as exc:
                errors.append(f"Failed to export {run_id}: {exc}")

        if exported:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(exported)} run(s) to {target_dir}."
            )
        if errors:
            QMessageBox.warning(self, "Export Issues", "\n".join(errors))


class LogsTab(QWidget):
    """Logs and diagnostics tab."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("Logs & Diagnostics")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Log viewer
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlainText("Log output will appear here...")

        layout.addWidget(self.log_text)
        self.setLayout(layout)


class LMPMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.process_manager: Optional[ProcessManager] = None
        self.init_ui()
        self.setup_menu()

    def init_ui(self):
        self.setWindowTitle("LMP - Lung Modeling Platform")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.workspace_tab = WorkspaceTab()
        self.api_products_tab = APIProductsTab()
        self.population_tab = PopulationTab()
        self.study_designer_tab = StudyDesignerTab()
        self.run_queue_tab = RunQueueTab()
        self.results_tab = ResultsViewerTab()
        self.logs_tab = LogsTab()

        # Add tabs
        self.tab_widget.addTab(self.workspace_tab, "Home / Workspace")
        self.tab_widget.addTab(self.api_products_tab, "API & Products")
        self.tab_widget.addTab(self.population_tab, "Population")
        self.tab_widget.addTab(self.study_designer_tab, "Study Designer")
        self.tab_widget.addTab(self.run_queue_tab, "Run Queue")
        self.tab_widget.addTab(self.results_tab, "Results Viewer")
        self.tab_widget.addTab(self.logs_tab, "Logs & Diagnostics")

        layout.addWidget(self.tab_widget)

        # Connect signals
        self.workspace_tab.workspace_changed.connect(self.on_workspace_changed)
        self.api_products_tab.config_updated.connect(self.on_catalog_config_updated)
        self.population_tab.config_updated.connect(self.on_catalog_config_updated)
        self.study_designer_tab.config_ready.connect(self.on_config_ready)
        self.run_queue_tab.run_completed.connect(self.on_run_completed)

    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
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

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About LMP", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def on_workspace_changed(self, workspace_path: str):
        """Handle workspace change."""
        try:
            self.workspace_manager = WorkspaceManager(workspace_path)
            self.process_manager = ProcessManager(self.workspace_manager)
            self.api_products_tab.set_workspace_manager(self.workspace_manager)
            self.population_tab.set_workspace_manager(self.workspace_manager)
            self.study_designer_tab.set_workspace_manager(self.workspace_manager)
            self.run_queue_tab.set_workspace_manager(self.workspace_manager)
            self.run_queue_tab.set_process_manager(self.process_manager)
            self.results_tab.set_workspace_manager(self.workspace_manager)
            self.logs_tab.log_text.append(f"Workspace set to: {workspace_path}")
        except Exception as e:
            QMessageBox.critical(self, "Workspace Error", f"Could not set workspace: {str(e)}")
            self.workspace_manager = None
            self.process_manager = None
            self.api_products_tab.set_workspace_manager(None)
            self.population_tab.set_workspace_manager(None)
            self.study_designer_tab.set_workspace_manager(None)
            self.results_tab.set_workspace_manager(None)

    def on_catalog_config_updated(self, category: str, data: Dict[str, Any]):
        """Handle config update from catalog tabs."""
        self.logs_tab.log_text.append(f"Updated {category} configuration: {data.get('name', 'Unknown')}")
        # Update study designer with the new data
        self.study_designer_tab.update_from_catalog(category, data)

    def on_config_ready(self, payload: Dict[str, Any]):
        """Handle config ready signal from study designer."""
        config_path = payload.get("config_path") if isinstance(payload, dict) else None
        run_plan = payload.get("run_plan") if isinstance(payload, dict) else None
        if config_path:
            self.run_queue_tab.set_selected_config(config_path, run_plan=run_plan)
            self.logs_tab.log_text.append(f"Configuration ready for simulation: {config_path}")
        else:
            self.logs_tab.log_text.append("Configuration ready for simulation")
        # Switch to run queue tab
        self.tab_widget.setCurrentWidget(self.run_queue_tab)

    def on_run_completed(self, run_id: str, status: str):
        self.logs_tab.log_text.append(f"Run {run_id} finished with status: {status}")

        if self.workspace_manager is None:
            return

        target_run_id = run_id
        if status == "completed":
            try:
                run_info = self.workspace_manager.get_run_info(run_id) or {}
            except Exception:
                run_info = {}
            parent_id = run_info.get("parent_run_id")
            if parent_id:
                target_run_id = parent_id

        self.results_tab.refresh_runs(select_run_id=target_run_id)

        if status == "completed":
            self.results_tab.load_run(run_id)
            self.tab_widget.setCurrentWidget(self.results_tab)

    def show_about(self):
        """Show about dialog."""
        catalog_status = "Available" if CATALOG_AVAILABLE else "Not Available"
        QMessageBox.about(
            self, "About LMP",
            "LMP - Lung Modeling Platform\n\n"
            "Version: 0.1.0\n"
            "GUI Framework: PySide6\n"
            f"Catalog Integration: {catalog_status}\n\n"
            "A modular PBPK simulation pipeline for lung modeling."
        )


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("LMP")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("LMP Team")

    # Create and show main window
    window = LMPMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
