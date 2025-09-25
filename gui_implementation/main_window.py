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
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterable, Mapping, Tuple, Set, Sequence, Callable, TYPE_CHECKING
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
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QProgressBar,
    QTextEdit, QSplitter, QGroupBox, QFormLayout, QComboBox, QSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QScrollArea, QHeaderView,
    QAbstractItemView, QGridLayout, QDoubleSpinBox, QStackedLayout,
    QPlainTextEdit, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from PySide6.QtGui import QFont, QAction, QTextCursor, QDoubleValidator, QKeySequence

from workspace_manager import WorkspaceManager
from process_manager import ProcessManager
from population_tab_widget import PopulationTabWidget

# Import app_api for catalog integration
sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
try:
    from lmp_pkg import app_api
    from lmp_pkg.config import AppConfig, check_catalog_coverage
    from lmp_pkg.domain.entities import Product
    test_entries = app_api.list_catalog_entries("subject")
    CATALOG_AVAILABLE = True
    CONFIG_MODEL_AVAILABLE = True
except Exception:
    CATALOG_AVAILABLE = False
    CONFIG_MODEL_AVAILABLE = False
    app_api = None
    AppConfig = None
    check_catalog_coverage = None
    Product = None

try:
    from lmp_pkg.catalog.builtin_loader import BuiltinDataLoader
except Exception:
    BuiltinDataLoader = None

CATALOG_ROOT = Path(__file__).parent.parent / "lmp_pkg" / "src" / "lmp_pkg" / "catalog" / "builtin"


STAGE_DISPLAY_NAMES = {
    "cfd": "CFD",
    "deposition": "Deposition",
    "pbbm": "PBPK",
    "pbpk": "PBPK",
    "pk": "PK",
    "iv_pk": "IV PK",
    "gi_pk": "GI PK",
    "vbe": "VBE",
    "analysis": "Analysis",
    "analysis_bioequivalence": "Bioequivalence",
    "overall": "Overall",
}


PK_PARAM_PLACEHOLDERS = (
    "clearance_L_h",
    "volume_central_L",
    "volume_peripheral_L",
    "volume_peripheral1_L",
    "volume_peripheral2_L",
    "cl_distribution_L_h",
    "cl_distribution1_L_h",
    "cl_distribution2_L_h",
)


logger = logging.getLogger(__name__)


def _sanitise_product_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name.strip())
    safe = safe.strip("-")
    return safe or name.replace(" ", "_")


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


def _to_plain_data(value: Any) -> Any:
    """Convert Pydantic models or nested containers into plain Python data."""

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="python")
        except Exception:
            return value.model_dump()

    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(sub_value) for key, sub_value in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]

    return value


class SpreadsheetWidget(QTableWidget):
    """Simple spreadsheet-style table with paste and copy support."""

    def __init__(self, headers: Sequence[str], parent: Optional[QWidget] = None):
        super().__init__(0, len(headers), parent)
        self._base_headers = list(headers)
        self._update_headers(len(headers))
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def add_empty_row(self) -> int:
        row = self.rowCount()
        self.insertRow(row)
        for col in range(self.columnCount()):
            if self.item(row, col) is None:
                self.setItem(row, col, QTableWidgetItem(""))
        return row

    def clear_rows(self, keep_one: bool = True) -> None:
        self.setRowCount(0)
        if keep_one:
            self.add_empty_row()

    def set_data(self, rows: Sequence[Sequence[Any]]) -> None:
        self.setRowCount(0)
        max_cols = max((len(row) for row in rows), default=self.columnCount())
        self._ensure_column_count(max_cols)
        for row_data in rows:
            row_index = self.add_empty_row()
            for col_index, value in enumerate(row_data):
                text = "" if value is None else str(value)
                self.item(row_index, col_index).setText(text)

    def get_non_empty_rows(self) -> List[List[str]]:
        rows: List[List[str]] = []
        for row in range(self.rowCount()):
            row_values: List[str] = []
            has_content = False
            for col in range(self.columnCount()):
                item = self.item(row, col)
                text = item.text().strip() if item is not None else ""
                row_values.append(text)
                if text:
                    has_content = True
            if has_content:
                rows.append(row_values)
        return rows

    # ------------------------------------------------------------------
    # Copy / Paste support
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.matches(QKeySequence.StandardKey.Paste):
            self._paste_from_clipboard()
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_to_clipboard()
            return
        super().keyPressEvent(event)

    def _paste_from_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return

        rows = [row for row in text.replace("\r", "").split("\n") if row]
        if not rows:
            return

        start_row = self.currentRow()
        if start_row < 0:
            start_row = self.rowCount()
        start_col = self.currentColumn()
        if start_col < 0:
            start_col = 0

        for r_offset, row_text in enumerate(rows):
            columns = row_text.split("	")
            target_row = start_row + r_offset
            while target_row >= self.rowCount():
                self.add_empty_row()
            self._ensure_column_count(start_col + len(columns))
            for c_offset, cell_text in enumerate(columns):
                target_col = start_col + c_offset
                if self.item(target_row, target_col) is None:
                    self.setItem(target_row, target_col, QTableWidgetItem(""))
                self.item(target_row, target_col).setText(cell_text)

    def _copy_to_clipboard(self) -> None:
        ranges = self.selectedRanges()
        if not ranges:
            return
        selected_range = ranges[0]
        rows: List[str] = []
        for row in range(selected_range.topRow(), selected_range.bottomRow() + 1):
            values: List[str] = []
            for col in range(selected_range.leftColumn(), selected_range.rightColumn() + 1):
                item = self.item(row, col)
                values.append(item.text() if item is not None else "")
            rows.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(rows))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_column_count(self, count: int) -> None:
        if count <= self.columnCount():
            return
        current_headers = list(self._base_headers)
        while len(current_headers) < count:
            current_headers.append(f"Value {len(current_headers)}")
        self.setColumnCount(count)
        self._update_headers(count, headers=current_headers)
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                if self.item(row, col) is None:
                    self.setItem(row, col, QTableWidgetItem(""))

    def _update_headers(self, count: int, headers: Optional[Sequence[str]] = None) -> None:
        header_values = list(headers or self._base_headers)
        if len(header_values) < count:
            header_values.extend([f"Value {idx}" for idx in range(len(header_values), count)])
        self.setHorizontalHeaderLabels(header_values[:count])




TREE_KEY_ROLE = Qt.ItemDataRole.UserRole
TREE_TYPE_ROLE = Qt.ItemDataRole.UserRole + 1


class ParameterTreeWidget(QTreeWidget):
    """Tree-based editor that exposes nested dictionaries and lists for editing."""

    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        self.setHeaderLabels(["Parameter", "Value"])
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Public API

    def set_data(self, data: Mapping[str, Any]) -> None:
        self.blockSignals(True)
        self.clear()
        root = self.invisibleRootItem()
        for key, value in data.items():
            self._add_item(root, key, value)
        self.expandToDepth(0)
        self.blockSignals(False)

    def get_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        root = self.invisibleRootItem()
        for index in range(root.childCount()):
            child = root.child(index)
            key = child.data(TREE_KEY_ROLE, 0) or child.text(0)
            result[str(key)] = self._collect(child)
        return result

    def clear_data(self) -> None:
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)

    def expand_all(self) -> None:
        self.expandAll()

    def collapse_all(self) -> None:
        self.collapseAll()

    def apply_filter(self, pattern: str) -> None:
        pattern = pattern.strip().lower()
        root = self.invisibleRootItem()
        for index in range(root.childCount()):
            child = root.child(index)
            self._apply_filter(child, pattern)
        if not pattern:
            self.expandToDepth(0)

    # ------------------------------------------------------------------
    # Internal helpers

    def _add_item(self, parent: QTreeWidgetItem, key: Any, value: Any) -> QTreeWidgetItem:
        if isinstance(value, dict):
            item = QTreeWidgetItem(parent, [str(key), ""])
            item.setData(0, TREE_KEY_ROLE, str(key))
            item.setData(0, TREE_TYPE_ROLE, "dict")
            for sub_key, sub_value in value.items():
                self._add_item(item, sub_key, sub_value)
            item.setExpanded(True)
            return item

        if isinstance(value, list):
            item = QTreeWidgetItem(parent, [str(key), ""])
            item.setData(0, TREE_KEY_ROLE, str(key))
            item.setData(0, TREE_TYPE_ROLE, "list")
            for index, element in enumerate(value):
                child = self._add_item(item, f"[{index}]", element)
                child.setData(0, TREE_KEY_ROLE, index)
                if isinstance(element, (dict, list)):
                    # child already assigned its own type via recursion
                    pass
                else:
                    child.setData(0, TREE_TYPE_ROLE, "list_item")
            item.setExpanded(True)
            return item

        display = _format_cell_value(value)
        item = QTreeWidgetItem(parent, [str(key), display])
        item.setData(0, TREE_KEY_ROLE, str(key))
        item.setData(0, TREE_TYPE_ROLE, "value")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

    def _collect(self, item: QTreeWidgetItem) -> Any:
        node_type = item.data(0, TREE_TYPE_ROLE)

        if node_type in {"value", "list_item"}:
            return _parse_cell_value(item.text(1))

        if node_type == "list":
            values: List[Tuple[int, Any]] = []
            for idx in range(item.childCount()):
                child = item.child(idx)
                key = child.data(0, TREE_KEY_ROLE)
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    index = idx
                values.append((index, self._collect(child)))
            values.sort(key=lambda pair: pair[0])
            return [value for _, value in values]

        # default dict
        data: Dict[str, Any] = {}
        for idx in range(item.childCount()):
            child = item.child(idx)
            key = child.data(0, TREE_KEY_ROLE)
            if isinstance(key, int):
                key = str(key)
            data[str(key)] = self._collect(child)
        return data

    def _apply_filter(self, item: QTreeWidgetItem, pattern: str) -> bool:
        if not pattern:
            item.setHidden(False)
            for idx in range(item.childCount()):
                self._apply_filter(item.child(idx), pattern)
            return True

        text_match = pattern in item.text(0).lower() or pattern in item.text(1).lower()
        child_match = False
        for idx in range(item.childCount()):
            if self._apply_filter(item.child(idx), pattern):
                child_match = True

        visible = text_match or child_match
        item.setHidden(not visible)
        if visible and child_match:
            item.setExpanded(True)
        return visible

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        # Ensure value column always reflects edits consistently; placeholder for future validation
        if column != 1:
            return


class ParameterTreePanel(QWidget):
    """Convenience wrapper around ParameterTreeWidget with filter and expand controls."""

    def __init__(self, placeholder: str = "No parameters available."):
        super().__init__()
        self.placeholder_text = placeholder
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search parameters…")
        expand_btn = QPushButton("Expand")
        collapse_btn = QPushButton("Collapse")

        controls.addWidget(filter_label)
        controls.addWidget(self.filter_edit)
        controls.addWidget(expand_btn)
        controls.addWidget(collapse_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.placeholder_label = QLabel(self.placeholder_text)
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #777; font-style: italic;")

        self.tree = ParameterTreeWidget()

        self.stack = QStackedLayout()

        placeholder_container = QWidget()
        placeholder_layout = QVBoxLayout()
        placeholder_layout.addStretch()
        placeholder_layout.addWidget(self.placeholder_label)
        placeholder_layout.addStretch()
        placeholder_container.setLayout(placeholder_layout)

        self.stack.addWidget(placeholder_container)
        self.stack.addWidget(self.tree)
        layout.addLayout(self.stack)

        self.setLayout(layout)

        expand_btn.clicked.connect(self.tree.expand_all)
        collapse_btn.clicked.connect(self.tree.collapse_all)
        self.filter_edit.textChanged.connect(self.tree.apply_filter)

    # ------------------------------------------------------------------
    # Public API

    def set_data(self, data: Mapping[str, Any]) -> None:
        if not data:
            self.clear()
            return
        self.stack.setCurrentIndex(1)
        self.filter_edit.blockSignals(True)
        self.filter_edit.clear()
        self.filter_edit.blockSignals(False)
        self.tree.set_data(data)

    def get_data(self) -> Dict[str, Any]:
        if self.stack.currentIndex() == 0:
            return {}
        return self.tree.get_data()

    def clear(self) -> None:
        self.tree.clear_data()
        self.stack.setCurrentIndex(0)
        self.filter_edit.clear()

    def setEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        super().setEnabled(enabled)
        if not enabled:
            self.filter_edit.clear()




class ParameterPathSelector(QWidget):
    """Inline widget with a path line edit and picker button."""

    def __init__(self, initial: str, picker: Callable[["ParameterPathSelector"], None]):
        super().__init__()
        self._picker = picker
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(initial)
        self.line_edit.setPlaceholderText("e.g. pk.params.clearance_L_h")
        picker_btn = QPushButton("…")
        picker_btn.setFixedWidth(28)
        picker_btn.clicked.connect(self._on_pick)
        layout.addWidget(self.line_edit)
        layout.addWidget(picker_btn)
        self.setLayout(layout)

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, value: str) -> None:
        self.line_edit.setText(value)

    def _on_pick(self) -> None:
        if self._picker:
            self._picker(self)


class ParameterPickerDialog(QDialog):
    """Modal dialog allowing users to select a parameter path from a tree."""

    def __init__(self, data: Mapping[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select Parameter")
        self.resize(520, 600)
        layout = QVBoxLayout(self)
        self.panel = ParameterTreePanel("No parameters available.")
        self.panel.set_data(data)
        layout.addWidget(self.panel)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.panel.tree.itemDoubleClicked.connect(lambda _item, _column: self._accept_selection(double_click=True))
        self.selected_path: Optional[str] = None

    def preselect_path(self, path: str) -> None:
        if not path:
            return
        tokens = self._tokenise_path(path)
        item = self._find_item(tokens)
        if item is not None:
            self.panel.tree.setCurrentItem(item)
            self.panel.tree.scrollToItem(item)

    def _accept_selection(self, double_click: bool = False) -> None:
        item = self.panel.tree.currentItem()
        if item is None or item.data(0, TREE_TYPE_ROLE) not in {"value", "list_item"}:
            if not double_click:
                QMessageBox.warning(self, "Select Parameter", "Choose a parameter value from the tree.")
            return
        self.selected_path = self._resolve_path(item)
        self.accept()

    def _resolve_path(self, item: QTreeWidgetItem) -> str:
        parts: List[str] = []
        current = item
        while current and current.parent() is not None:
            key = current.data(0, TREE_KEY_ROLE)
            parts.append(str(key))
            current = current.parent()
        if current and current.data(0, TREE_KEY_ROLE) is not None:
            parts.append(str(current.data(0, TREE_KEY_ROLE)))
        parts.reverse()
        path = ""
        for token in parts:
            cleaned = token.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                path += cleaned
            elif cleaned.isdigit():
                path += f"[{cleaned}]"
            else:
                if path and not path.endswith(']'):
                    path += '.'
                elif path and path.endswith(']'):
                    path += '.'
                path += cleaned
        return path

    def _find_item(self, tokens: Sequence[str]) -> Optional[QTreeWidgetItem]:
        parent = self.panel.tree.invisibleRootItem()
        current = None
        for token in tokens:
            normalized = token.strip()
            if normalized.startswith('[') and normalized.endswith(']'):
                normalized = normalized[1:-1]
            match = None
            for idx in range(parent.childCount()):
                child = parent.child(idx)
                key = child.data(0, TREE_KEY_ROLE)
                key_str = str(key)
                if key_str == token or key_str == normalized or key_str == f"[{normalized}]":
                    match = child
                    break
                if isinstance(key, int) and normalized.isdigit() and int(normalized) == key:
                    match = child
                    break
            if match is None:
                return None
            current = match
            parent = current
        return current

    @staticmethod
    def _tokenise_path(path: str) -> List[str]:
        path = path.strip()
        if not path:
            return []
        tokens: List[str] = []
        buffer = ''
        i = 0
        while i < len(path):
            char = path[i]
            if char == '.':
                if buffer:
                    tokens.append(buffer)
                    buffer = ''
                i += 1
                continue
            if char == '[':
                if buffer:
                    tokens.append(buffer)
                    buffer = ''
                end = path.find(']', i)
                if end == -1:
                    break
                tokens.append(path[i:end + 1])
                i = end + 1
                continue
            buffer += char
            i += 1
        if buffer:
            tokens.append(buffer)
        return tokens


class ProductAPIEditor(QWidget):
    """Inline editor for configuring APIs associated with a product."""

    COLUMN_HEADERS = ["API", "Dose (µg)", "USP Depo (%)", "MMAD (µm)", "GSD"]
    FIELD_MAP = {
        1: "dose_ug",
        2: "usp_depo_fraction",
        3: "mmad",
        4: "gsd",
    }

    def __init__(self):
        super().__init__()
        self.available_apis: List[str] = []
        self._row_names: Dict[int, Optional[str]] = {}
        self._row_defaults: Dict[int, Dict[str, Any]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Number of APIs:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 10)
        self.count_spin.valueChanged.connect(self._on_count_changed)
        controls.addWidget(self.count_spin)
        controls.addStretch()
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(self.COLUMN_HEADERS))
        self.table.setHorizontalHeaderLabels(self.COLUMN_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(self.COLUMN_HEADERS)):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        note = QLabel("Select APIs and provide overrides. Leave numeric fields blank to keep defaults.")
        note.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(note)

        self.setLayout(layout)

    def set_available_apis(self, api_names: Iterable[str]) -> None:
        names = sorted({name for name in api_names if name})
        self.available_apis = names
        for row in range(self.table.rowCount()):
            combo = self._api_combo(row)
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select API...")
            combo.addItems(self.available_apis)
            if current and current in self.available_apis:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def set_entries(self, entries: Iterable[Dict[str, Any]]) -> None:
        entries = list(entries or [])
        desired = len(entries)
        self._set_row_count(desired)
        self.count_spin.blockSignals(True)
        self.count_spin.setValue(desired)
        self.count_spin.blockSignals(False)

        self._row_names = {row: None for row in range(self.table.rowCount())}
        self._row_defaults = {row: {} for row in range(self.table.rowCount())}

        for row, entry in enumerate(entries):
            data = self._normalise_entry(entry)
            combo = self._api_combo(row)
            if combo is None:
                continue

            slot_name = entry.get("slot_name") or data.get("slot_name") or data.get("name")
            if slot_name:
                self._row_names[row] = str(slot_name)

            target_name = data.get("ref") or data.get("name")
            if target_name and target_name not in self.available_apis:
                updated = set(self.available_apis)
                updated.add(str(target_name))
                self.set_available_apis(sorted(updated))
            if target_name and target_name in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(str(target_name))
            else:
                combo.setCurrentIndex(0)

            defaults: Dict[str, Any] = {}
            for col, field in self.FIELD_MAP.items():
                editor = self._value_editor(row, col)
                if editor is None:
                    continue
                value = data.get(field)
                if value is not None and value != "":
                    editor.setText(str(value))
                    defaults[field] = value
                else:
                    editor.clear()
            if data.get("dose_pg") is not None:
                defaults.setdefault("dose_pg", data["dose_pg"])
            if data.get("dose_ug") is not None:
                defaults.setdefault("dose_ug", data["dose_ug"])
            self._row_defaults[row] = defaults

    def get_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            combo = self._api_combo(row)
            if combo is None:
                continue
            name = combo.currentText()
            if not name or name == "Select API...":
                continue

            slot_name = self._row_names.get(row) or name
            if not slot_name:
                continue

            entry_payload: Dict[str, Any] = {"name": str(slot_name)}
            if name:
                entry_payload["ref"] = str(name)

            defaults = self._row_defaults.get(row, {})

            for col, field in self.FIELD_MAP.items():
                editor = self._value_editor(row, col)
                if editor is None:
                    continue
                text = editor.text().strip()
                if not text:
                    if field == "dose_ug":
                        if "dose_pg" not in entry_payload and "dose_pg" in defaults:
                            entry_payload["dose_pg"] = defaults["dose_pg"]
                        if "dose_ug" not in entry_payload and "dose_ug" in defaults:
                            entry_payload["dose_ug"] = defaults["dose_ug"]
                    else:
                        if field in defaults:
                            entry_payload[field] = defaults[field]
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if field == "dose_ug":
                    entry_payload["dose_ug"] = value
                    entry_payload["dose_pg"] = value * 1_000_000.0
                else:
                    entry_payload[field] = value

            if "dose_pg" not in entry_payload and "dose_pg" in defaults:
                entry_payload["dose_pg"] = defaults["dose_pg"]
            if "dose_ug" not in entry_payload and "dose_ug" in defaults:
                entry_payload["dose_ug"] = defaults["dose_ug"]

            entries.append(entry_payload)
        return entries

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.count_spin.blockSignals(True)
        self.count_spin.setValue(0)
        self.count_spin.blockSignals(False)
        self._row_names.clear()
        self._row_defaults.clear()

    # ------------------------------------------------------------------
    # Internal helpers

    def _on_count_changed(self, value: int) -> None:
        self._set_row_count(value)

    def _set_row_count(self, count: int) -> None:
        current = self.table.rowCount()
        if count == current:
            return
        if count < 0:
            count = 0
        if count < current:
            for row in range(count, current):
                self._row_names.pop(row, None)
                self._row_defaults.pop(row, None)
        self.table.setRowCount(count)
        for row in range(current, count):
            self._setup_row(row)
            self._row_names.setdefault(row, None)
            self._row_defaults.setdefault(row, {})

    def _setup_row(self, row: int) -> None:
        combo = QComboBox()
        combo.addItem("Select API...")
        combo.addItems(self.available_apis)
        self.table.setCellWidget(row, 0, combo)

        validator = QDoubleValidator(bottom=-1e9, top=1e9, decimals=6)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        placeholders = {
            1: "e.g. 100",
            2: "e.g. 45",
            3: "e.g. 1.8",
            4: "e.g. 2.0",
        }

        for col in range(1, len(self.COLUMN_HEADERS)):
            editor = QLineEdit()
            editor.setValidator(validator)
            editor.setPlaceholderText(placeholders.get(col, ""))
            self.table.setCellWidget(row, col, editor)

    def _api_combo(self, row: int) -> Optional[QComboBox]:
        widget = self.table.cellWidget(row, 0)
        return widget if isinstance(widget, QComboBox) else None

    def _value_editor(self, row: int, col: int) -> Optional[QLineEdit]:
        widget = self.table.cellWidget(row, col)
        return widget if isinstance(widget, QLineEdit) else None

    @staticmethod
    def _normalise_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if not isinstance(entry, dict):
            return {}
        data: Dict[str, Any] = {}

        sources: List[Dict[str, Any]] = []
        overrides = entry.get("overrides")
        if isinstance(overrides, dict):
            sources.append(overrides)
        sources.append(entry)

        for source in sources:
            for key in ("name", "ref", "dose_pg", "dose_ug", "usp_depo_fraction", "mmad", "gsd"):
                if key not in source:
                    continue
                value = source[key]
                if value is None:
                    continue
                data[key] = value

        if "dose_ug" not in data and "dose_pg" in data:
            try:
                data["dose_ug"] = float(data["dose_pg"]) / 1_000_000.0
            except (TypeError, ValueError):
                pass
        if "dose_pg" not in data and "dose_ug" in data:
            try:
                data["dose_pg"] = float(data["dose_ug"]) * 1_000_000.0
            except (TypeError, ValueError):
                pass

        name = data.get("name") or entry.get("name") or entry.get("ref")
        if name:
            data["name"] = str(name)
        if "ref" not in data and name:
            data["ref"] = str(name)
        return data


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
                "api_overrides": [],
            },
            "product": {
                "base_ref": None,
                "base_data": {},
                "variability_base": {},
                "current_id": None,
                "loading": False,
                "overrides": {},
                "variability_overrides": {},
                "api_overrides": [],
            },
        }
        self.init_ui()
        self.refresh_catalog_options("api")
        self.refresh_catalog_options("product")
        self._refresh_available_api_names()

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
        parameters_panel = ParameterTreePanel("Select a catalog entry to view parameters.")
        tab_widget.addTab(parameters_panel, "Parameters")
        variability_panel = ParameterTreePanel("No variability parameters available for this entry.")
        tab_widget.addTab(variability_panel, "Variability")
        group_layout.addWidget(tab_widget)

        api_editor: Optional[ProductAPIEditor] = None
        if category == "product":
            api_editor = ProductAPIEditor()
            api_editor.setEnabled(False)
            group_layout.addWidget(api_editor)

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

        editor_payload = {
            "group": group,
            "saved_combo": saved_combo,
            "delete_btn": delete_btn,
            "name_edit": name_edit,
            "base_combo": base_combo,
            "parameters_panel": parameters_panel,
            "variability_panel": variability_panel,
            "tab_widget": tab_widget,
            "save_btn": save_btn,
            "save_as_btn": save_as_btn,
            "revert_btn": revert_btn,
        }
        if api_editor is not None:
            editor_payload["api_editor"] = api_editor
        return editor_payload

    # ------------------------------------------------------------------
    # Workspace integration

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]) -> None:
        self.workspace_manager = workspace_manager
        for category in ("api", "product"):
            self.refresh_saved_entries(category)
            self._update_button_states(category)
        self._refresh_available_api_names()

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
        if category == "api":
            self._refresh_available_api_names()

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
                "api_overrides": [],
            })
            editor["parameters_panel"].clear()
            editor["variability_panel"].clear()
            api_editor = editor.get("api_editor")
            if api_editor is not None:
                api_editor.clear()
                api_editor.setEnabled(False)
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
            "api_overrides": [],
        })

        self.populate_tables(category, overrides=None, variability_overrides=None)
        editor["name_edit"].setPlaceholderText(ref)

        saved_combo = editor["saved_combo"]
        saved_combo.blockSignals(True)
        saved_combo.setCurrentIndex(0)
        saved_combo.blockSignals(False)
        self._update_button_states(category)
        if category == "product":
            api_editor = editor.get("api_editor")
            if api_editor is not None:
                api_editor.setEnabled(True)

    def populate_tables(
        self,
        category: str,
        overrides: Optional[Dict[str, Any]],
        variability_overrides: Optional[Dict[str, Any]],
    ) -> None:
        state = self._state[category]
        editor = self._editors[category]

        overrides = overrides or {}
        variability_overrides = variability_overrides or {}

        api_override_entries: List[Dict[str, Any]] = []
        if isinstance(overrides, dict) and "apis" in overrides:
            apis_value = overrides.get("apis")
            if isinstance(apis_value, dict):
                api_override_entries = [
                    {
                        "name": key,
                        "slot_name": key,
                        **(value if isinstance(value, dict) else {}),
                    }
                    for key, value in apis_value.items()
                ]
            elif isinstance(apis_value, list):
                api_override_entries = apis_value
            overrides = {k: v for k, v in overrides.items() if k != "apis"}

        resolved = _apply_overrides(state["base_data"], overrides)
        parameters_payload = {k: v for k, v in resolved.items() if k != "apis"}
        editor["parameters_panel"].set_data(parameters_payload)
        state["overrides"] = copy.deepcopy(overrides) if overrides else {}
        state["api_overrides"] = copy.deepcopy(api_override_entries)

        if category == "product":
            api_editor = editor.get("api_editor")
            base_apis = []
            if isinstance(state["base_data"], dict):
                base_apis = state["base_data"].get("apis") or []
            merged_apis = self._merge_api_entries(base_apis, api_override_entries)
            if api_editor is not None:
                api_editor.set_entries(merged_apis)
        elif category != "product":
            api_editor = editor.get("api_editor")
            if api_editor is not None:
                api_editor.clear()
                api_editor.setEnabled(False)

        variability_base = state.get("variability_base") or {}
        if variability_base:
            resolved_var = _apply_overrides(variability_base, variability_overrides)
            editor["variability_panel"].set_data(resolved_var)
            state["variability_overrides"] = copy.deepcopy(variability_overrides) if variability_overrides else {}
            self._set_variability_tab_enabled(category, True)
        else:
            editor["variability_panel"].clear()
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
        if category == "api":
            self._refresh_available_api_names()

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

    def _refresh_available_api_names(self) -> None:
        editor = self._editors.get("product")
        if not editor:
            return
        names: Set[str] = set(self._catalog_names.get("api", []))
        saved_info: List[Dict[str, Any]] = []
        for entry in self._saved_entries.get("api", []):
            ref = entry.get("ref") or entry.get("name")
            display = entry.get("name") or entry.get("display_name") or ref
            if ref:
                names.add(str(ref))
            if display:
                names.add(str(display))
            saved_info.append({"name": display, "ref": ref})
            overrides = entry.get("overrides")
            if isinstance(overrides, dict):
                candidate = overrides.get("name")
                if candidate:
                    names.add(str(candidate))
                    saved_info.append({"name": candidate, "ref": ref})
        api_editor = editor.get("api_editor")
        if api_editor is not None:
            api_editor.set_available_apis(sorted(names))
        self._notify_saved_api_names(saved_info)

    def _notify_saved_api_names(self, api_info: List[Dict[str, Any]]) -> None:
        if hasattr(self, "population_tab") and isinstance(api_info, list):
            try:
                self.population_tab.update_saved_apis(api_info)
            except Exception:
                pass

    @staticmethod
    def _merge_api_entries(
        base_entries: Optional[Iterable[Dict[str, Any]]],
        override_entries: Optional[Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        base_list = APIProductsTab._prepare_base_api_entries(base_entries)

        overrides_raw = []
        for entry in (override_entries or []):
            normalized = APIProductsTab._normalise_api_entry(entry)
            if normalized:
                overrides_raw.append(normalized)

        if not overrides_raw:
            return base_list

        merged: List[Dict[str, Any]] = []
        for idx, override in enumerate(overrides_raw):
            if idx < len(base_list):
                base_entry = base_list[idx]
                combined = dict(base_entry)
                for key, value in override.items():
                    if value is None:
                        continue
                    combined[key] = value
                slot_name = base_entry.get("slot_name") or base_entry.get("name")
                if slot_name:
                    combined["name"] = slot_name
                    combined.setdefault("slot_name", slot_name)
            else:
                combined = dict(override)
                if "slot_name" not in combined and combined.get("name"):
                    combined["slot_name"] = combined["name"]
            merged.append(combined)
        return merged

    def _apply_category_config(self, category: str, payload: Mapping[str, Any]) -> None:
        ref = payload.get("ref")
        if not ref:
            return

        editor = self._editors[category]
        state = self._state[category]

        base_combo = editor["base_combo"]
        base_combo.blockSignals(True)
        if base_combo.findData(ref) == -1:
            base_combo.addItem(ref, ref)
        base_combo.setCurrentIndex(base_combo.findData(ref))
        base_combo.blockSignals(False)

        state["loading"] = False
        self.on_base_changed(category)

        overrides = payload.get("overrides") if isinstance(payload, Mapping) else None
        variability_overrides = None
        if isinstance(payload, Mapping):
            variability_overrides = payload.get("variability") or payload.get("variability_overrides")

        self.populate_tables(category, overrides, variability_overrides)
        editor["name_edit"].setText(payload.get("name", ""))
        state["current_id"] = None
        self._update_button_states(category)

    def apply_config(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            return

        for category in ("api", "product"):
            payload = config.get(category)
            if isinstance(payload, Mapping):
                try:
                    self._apply_category_config(category, payload)
                except Exception as exc:
                    logger.warning("apply catalog config failed", category=category, error=str(exc))

        self._refresh_available_api_names()

    @staticmethod
    def _normalise_api_entry(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(entry, tuple) and len(entry) == 2:
            key, payload = entry
            base_payload: Dict[str, Any] = {"name": key, "ref": key}
            if hasattr(payload, "model_dump"):
                base_payload.update(payload.model_dump(exclude_none=True))
            elif isinstance(payload, Mapping):
                base_payload.update(payload)
            return ProductAPIEditor._normalise_entry(base_payload)

        if isinstance(entry, Mapping):
            return ProductAPIEditor._normalise_entry(dict(entry))

        return {}

    @staticmethod
    def _prepare_base_api_entries(entries: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if entries is None:
            return []

        if isinstance(entries, Mapping):
            prepared: List[Dict[str, Any]] = []
            for key, payload in entries.items():
                base_payload: Dict[str, Any] = {"name": key, "ref": key}
                if hasattr(payload, "model_dump"):
                    base_payload.update(payload.model_dump(exclude_none=True))
                elif isinstance(payload, Mapping):
                    base_payload.update(payload)
                normalized = ProductAPIEditor._normalise_entry(base_payload)
                if normalized.get("name"):
                    normalized.setdefault("slot_name", normalized["name"])
                prepared.append(normalized)
            return prepared

        prepared_list: List[Dict[str, Any]] = []
        for entry in entries:
            normalized = APIProductsTab._normalise_api_entry(entry)
            if normalized.get("name"):
                normalized.setdefault("slot_name", normalized["name"])
            prepared_list.append(normalized)
        return prepared_list

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
        parameters = editor["parameters_panel"].get_data()
        overrides = _compute_overrides(base_data, parameters)
        # Remove None overrides to avoid wiping defaults unless explicit
        overrides = {k: v for k, v in overrides.items() if v is not None}

        if category == "product":
            api_editor = editor.get("api_editor")
            api_entries = api_editor.get_entries() if api_editor is not None else []
            api_map: Dict[str, Dict[str, Any]] = {}
            for entry in api_entries:
                slot = entry.get("name") or entry.get("slot")
                if not slot:
                    continue
                payload = {k: v for k, v in entry.items() if k not in {"name", "slot", "slot_name"}}
                if not payload:
                    continue
                api_map[str(slot)] = payload
            if api_map:
                overrides["apis"] = api_map
            elif "apis" in overrides:
                overrides.pop("apis")
            state["api_overrides"] = api_entries

        variability_base = state.get("variability_base") or {}
        variability_overrides: Dict[str, Any] = {}
        if variability_base:
            variability_values = editor["variability_panel"].get_data()
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
        if category == "api":
            self._refresh_available_api_names()

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
        variability_panel = editor["variability_panel"]
        index = tab_widget.indexOf(variability_panel)
        if index != -1:
            tab_widget.setTabEnabled(index, enabled)
            if not enabled:
                variability_panel.clear()


class PopulationTab(QWidget):
    """Wrapper around the detailed population editor widget with quick shortcuts."""

    config_updated = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.population_widget = PopulationTabWidget()
        self.population_widget.config_updated.connect(self._route_config_update)

        self.shortcut_combos: Dict[str, QComboBox] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._build_shortcuts_group())
        layout.addWidget(self.population_widget)
        self.setLayout(layout)
        self._refresh_shortcuts()

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]) -> None:
        self.workspace_manager = workspace_manager
        self._refresh_shortcuts()
        self._sync_saved_api_entries()

    def update_saved_apis(
        self,
        saved_api_info: Iterable[Dict[str, Any]],
        *,
        replace_existing: bool = False,
    ) -> None:
        """Forward saved API metadata to the detailed population widget."""
        if hasattr(self.population_widget, "update_saved_apis"):
            self.population_widget.update_saved_apis(
                saved_api_info,
                replace_existing=replace_existing,
            )

    def _build_shortcuts_group(self) -> QGroupBox:
        group = QGroupBox("Quick Load")
        grid = QGridLayout()
        entries = [
            ("Subject", "subject"),
            ("Maneuver", "maneuver"),
            ("Lung Geometry", "lung_geometry"),
            ("GI Tract", "gi_tract"),
        ]
        for row, (label, category) in enumerate(entries):
            combo = QComboBox()
            combo.addItem(f"Select {label.lower()}...", None)
            load_btn = QPushButton("Load")
            load_btn.clicked.connect(lambda _=False, cat=category: self._activate_shortcut(cat))
            self.shortcut_combos[category] = combo
            grid.addWidget(QLabel(label + ":"), row, 0)
            grid.addWidget(combo, row, 1)
            grid.addWidget(load_btn, row, 2)
        group.setLayout(grid)
        return group

    def _refresh_shortcuts(self) -> None:
        categories = {
            "subject": "subject",
            "maneuver": "inhalation",
            "lung_geometry": "lung_geometry",
            "gi_tract": "gi_tract",
        }
        for category, combo in self.shortcut_combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(f"Select {category.replace('_', ' ')}...", None)
            for name in self._list_builtin_names(categories[category]):
                combo.addItem(f"Builtin: {name}", {"source": "builtin", "ref": name})
            if self.workspace_manager is not None:
                try:
                    entries = self.workspace_manager.list_catalog_entries(category)
                except Exception:
                    entries = []
                for entry in entries:
                    identifier = entry.get("id") or entry.get("name") or entry.get("ref")
                    combo.addItem(f"Workspace: {identifier}", {"source": "workspace", "id": entry.get("id")})
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def _sync_saved_api_entries(self) -> None:
        if not hasattr(self.population_widget, "update_saved_apis"):
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

        self.update_saved_apis(api_info, replace_existing=True)

    @staticmethod
    def _list_builtin_names(subdir: str) -> List[str]:
        target_dir = CATALOG_ROOT / subdir
        if not target_dir.exists():
            return []
        names = [path.stem for path in target_dir.glob('*.toml') if not path.stem.startswith('Variability_')]
        return sorted(names)

    def apply_config(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            return

        subject_payload = config.get("subject")
        if isinstance(subject_payload, Mapping) and subject_payload.get("ref"):
            try:
                self.population_widget.load_subject_entry(
                    subject_payload.get("ref"),
                    overrides=subject_payload.get("overrides"),
                )
            except Exception as exc:
                logger.warning("apply subject config failed", error=str(exc))

        maneuver_payload = config.get("maneuver")
        if isinstance(maneuver_payload, Mapping) and maneuver_payload.get("ref"):
            try:
                self.population_widget.load_maneuver_entry(
                    maneuver_payload.get("ref"),
                    overrides=maneuver_payload.get("overrides"),
                )
            except Exception as exc:
                logger.warning("apply maneuver config failed", error=str(exc))

        lung_payload = config.get("deposition")
        if isinstance(lung_payload, Mapping):
            geometry_ref = (
                lung_payload.get("lung_geometry_ref")
                or lung_payload.get("lung_geometry")
                or lung_payload.get("geometry_ref")
            )
            if geometry_ref:
                try:
                    self.population_widget.load_lung_geometry_entry(str(geometry_ref))
                except Exception as exc:
                    logger.warning("apply lung geometry failed", error=str(exc))

        gi_payload = config.get("gi_tract") or config.get("gi")
        if isinstance(gi_payload, Mapping) and gi_payload.get("ref"):
            try:
                self.population_widget.load_gi_entry(
                    gi_payload.get("ref"),
                    overrides=gi_payload.get("overrides"),
                )
            except Exception as exc:
                logger.warning("apply gi config failed", error=str(exc))

    def _activate_shortcut(self, category: str) -> None:
        combo = self.shortcut_combos.get(category)
        if combo is None:
            return
        payload = combo.currentData()
        if not payload:
            return
        source = payload.get("source")
        ref = payload.get("ref") or payload.get("id")
        if source == "builtin" and ref:
            self._load_builtin_entry(category, ref)
        elif source == "workspace" and self.workspace_manager is not None and ref:
            try:
                entry = self.workspace_manager.load_catalog_entry(category, ref)
            except Exception:
                entry = None
            if entry:
                self._load_workspace_entry(category, entry)
        combo.setCurrentIndex(0)

    def _load_builtin_entry(self, category: str, ref: str) -> None:
        if category == "subject":
            self.population_widget.load_subject_entry(ref)
        elif category == "maneuver":
            self.population_widget.load_maneuver_entry(ref)
        elif category == "lung_geometry":
            self.population_widget.load_lung_geometry_entry(ref)
        elif category == "gi_tract":
            self.population_widget.load_gi_entry(ref)

    def _load_workspace_entry(self, category: str, entry: Dict[str, Any]) -> None:
        ref = entry.get("ref") or entry.get("name") or entry.get("id")
        overrides = copy.deepcopy(entry.get("overrides") or {})
        variability = entry.get("variability_overrides")
        if variability:
            overrides = overrides or {}
            overrides["variability"] = variability
        if category == "subject" and ref:
            self.population_widget.load_subject_entry(ref, overrides)
        elif category == "maneuver" and ref:
            self.population_widget.load_maneuver_entry(ref, overrides)
        elif category == "lung_geometry" and ref:
            self.population_widget.load_lung_geometry_entry(ref, overrides)
        elif category == "gi_tract" and ref:
            self.population_widget.load_gi_entry(ref, overrides)

    def _route_config_update(self, category: str, data: Dict[str, Any]) -> None:
        self.config_updated.emit(category, data)
        self._refresh_shortcuts()


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
        self.catalog_category_map: Dict[str, str] = {
            "subject": "subject",
            "gi": "gi_tract",
            "lung": "lung_geometry",
            "api": "api",
            "product": "product",
            "maneuver": "maneuver",
        }
        self.workspace_catalog_entries: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in self.catalog_category_map.keys()
        }
        self.builtin_catalog_entries: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in self.catalog_category_map.keys()
        }
        self.all_catalog_entries: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in self.catalog_category_map.keys()
        }
        self.selected_entities: Dict[str, Optional[Dict[str, Any]]] = {
            key: None for key in self.catalog_category_map.keys()
        }
        self.entity_combos: Dict[str, QComboBox] = {}
        self.entity_placeholder: Dict[str, str] = {
            "subject": "Configure subjects in Population tab",
            "gi": "Configure GI physiology in Population tab",
            "lung": "Configure lung physiology in Population tab",
            "maneuver": "Configure maneuvers in Population tab",
            "api": "Active API is derived from the selected product",
            "product": "Configure products in API & Products tab",
        }
        self.entity_display: Dict[str, str] = {
            "subject": "subject",
            "gi": "GI physiology",
            "lung": "lung physiology",
            "api": "API",
            "product": "product",
            "maneuver": "maneuver",
        }
        self.subject_modes: Dict[str, str] = {"subject": "model", "gi": "model", "lung": "model", "maneuver": "model"}
        self.subject_manual_edits: Dict[str, QPlainTextEdit] = {}
        self.subject_panels: Dict[str, QWidget] = {}
        self.subject_summary_labels: Dict[str, QLabel] = {}

        self.available_models: Dict[str, List[str]] = {}
        if app_api is not None:
            try:
                self.available_models = app_api.list_available_models() or {}
            except Exception as exc:
                logger.warning("list available models failed", error=str(exc))
                self.available_models = {}

        self.stage_definitions: List[Dict[str, Any]] = [
            {
                "key": "deposition",
                "label": "Lung Deposition",
                "families": ["deposition"],
                "pipeline_stage": "deposition",
                "administrations": {"inhalation"},
                "default_enabled": True,
            },
            {
                "key": "pbbm",
                "label": "Lung PBPK",
                "families": ["lung_pbbm"],
                "pipeline_stage": "pbbm",
                "administrations": {"inhalation"},
                "default_enabled": True,
            },
            {
                "key": "gi_pk",
                "label": "GI PK",
                "families": ["gi_pk"],
                "pipeline_stage": "gi_pk",
                "administrations": {"po", "inhalation"},
                "default_enabled": True,
                "parameters": [
                    {
                        "name": "formulation",
                        "label": "Formulation",
                        "type": "choice",
                        "options": [
                            ("immediate_release", "Immediate Release"),
                            ("enteric_coated", "Enteric Coated"),
                            ("extended_release", "Extended Release"),
                        ],
                        "default": "immediate_release",
                    },
                ],
            },
            {
                "key": "pk",
                "label": "Systemic PK",
                "families": ["systemic_pk"],
                "pipeline_stage": "pk",
                "administrations": {"po", "inhalation"},
                "default_enabled": True,
            },
            {
                "key": "iv_pk",
                "label": "Systemic PK (IV)",
                "families": ["iv_pk"],
                "pipeline_stage": "iv_pk",
                "administrations": {"iv"},
                "default_enabled": True,
                "parameters": [
                    {"name": "iv_dose_duration_h", "label": "Infusion Duration (h)", "type": "double", "default": 0.0, "min": 0.0, "max": 48.0, "step": 0.1},
                ],
            },
        ]
        self.stage_controls: Dict[str, Dict[str, Any]] = {}
        self.stage_modes: Dict[str, str] = {definition["key"]: "model" for definition in self.stage_definitions}
        self.stage_pipeline_lookup: Dict[str, str] = {
            definition["pipeline_stage"]: definition["key"] for definition in self.stage_definitions
        }
        self.init_ui()
        self.populate_catalog_defaults()

    def init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Study Designer")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        scroll = QScrollArea()
        form_widget = QWidget()
        form_layout = QVBoxLayout()

        form_layout.addWidget(self._build_study_details_group())
        form_layout.addWidget(self._build_administration_group())
        form_layout.addWidget(self._build_subject_configuration_group())
        form_layout.addWidget(self._build_product_configuration_group())
        form_layout.addWidget(self._build_run_configuration_group())
        form_layout.addWidget(self._build_stage_configuration_group())

        for category in self.entity_combos.keys():
            self._set_combo_placeholder(category)

        # Sweep configuration (basic)
        self.sweep_group = QGroupBox("Parameter Sweep")
        sweep_layout = QVBoxLayout()

        self.sweep_enabled = QCheckBox("Enable Parameter Sweep")
        sweep_layout.addWidget(self.sweep_enabled)

        sweep_help = QLabel(
            "One parameter per row. Paste data directly from a spreadsheet; columns after the first are interpreted as sweep values."
        )
        sweep_help.setWordWrap(True)
        sweep_help.setStyleSheet("color: #666; font-size: 11px;")
        sweep_layout.addWidget(sweep_help)

        self.sweep_table = SpreadsheetWidget([
            "Parameter Path",
            "Value 1",
            "Value 2",
            "Value 3",
            "Value 4",
        ])
        self.sweep_table.setMinimumHeight(160)
        self.sweep_table.setEnabled(False)
        self.sweep_table.add_empty_row()
        sweep_layout.addWidget(self.sweep_table)

        sweep_button_row = QHBoxLayout()
        self.sweep_select_param_btn = QPushButton("Select Parameter…")
        self.sweep_select_param_btn.clicked.connect(self.select_sweep_parameter)
        self.sweep_add_row_btn = QPushButton("Add Row")
        self.sweep_add_row_btn.clicked.connect(lambda: self.sweep_table.add_empty_row())
        self.sweep_remove_row_btn = QPushButton("Remove Selected")
        self.sweep_remove_row_btn.clicked.connect(self.remove_selected_sweep_rows)
        sweep_button_row.addWidget(self.sweep_select_param_btn)
        sweep_button_row.addWidget(self.sweep_add_row_btn)
        sweep_button_row.addWidget(self.sweep_remove_row_btn)
        sweep_button_row.addStretch()
        sweep_layout.addLayout(sweep_button_row)

        self.sweep_enabled.toggled.connect(self._on_sweep_enabled_changed)
        self._on_sweep_enabled_changed(False)
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

        self.observed_series_table = SpreadsheetWidget(["Time", "Value"])
        self.observed_series_table.setMinimumHeight(160)
        self.observed_series_table.add_empty_row()
        observed_form.addRow("Observed series:", self.observed_series_table)

        observed_button_row = QHBoxLayout()
        load_series_btn = QPushButton("Load Series…")
        load_series_btn.clicked.connect(self.load_observed_series)
        clear_series_btn = QPushButton("Clear Series")
        clear_series_btn.clicked.connect(self.clear_observed_series)
        observed_button_row.addWidget(load_series_btn)
        observed_button_row.addWidget(clear_series_btn)
        observed_button_row.addStretch()
        observed_form.addRow("", observed_button_row)

        self.observed_load_button = load_series_btn
        self.observed_clear_button = clear_series_btn

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
            value_label = QLabel("Observed")
            value_spin = QDoubleSpinBox()
            value_spin.setDecimals(6)
            value_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
            weight_label = QLabel("Weight")
            weight_spin = QDoubleSpinBox()
            weight_spin.setDecimals(4)
            weight_spin.setRange(0.0, 1_000_000.0)
            weight_spin.setValue(1.0)
            scalar_layout.addWidget(checkbox, row, 0)
            scalar_layout.addWidget(value_label, row, 1)
            scalar_layout.addWidget(value_spin, row, 2)
            scalar_layout.addWidget(weight_label, row, 3)
            scalar_layout.addWidget(weight_spin, row, 4)
            return checkbox, value_spin, weight_spin, value_label, weight_label

        (
            self.pk_auc_checkbox,
            self.pk_auc_value_spin,
            self.pk_auc_weight_spin,
            self.pk_auc_value_label,
            self.pk_auc_weight_label,
        ) = _make_scalar_row(0, "PK AUC0_t")

        (
            self.cfd_mmad_checkbox,
            self.cfd_mmad_value_spin,
            self.cfd_mmad_weight_spin,
            self.cfd_mmad_value_label,
            self.cfd_mmad_weight_label,
        ) = _make_scalar_row(1, "CFD MMAD (um)")

        (
            self.cfd_gsd_checkbox,
            self.cfd_gsd_value_spin,
            self.cfd_gsd_weight_spin,
            self.cfd_gsd_value_label,
            self.cfd_gsd_weight_label,
        ) = _make_scalar_row(2, "CFD GSD")

        (
            self.cfd_mt_checkbox,
            self.cfd_mt_value_spin,
            self.cfd_mt_weight_spin,
            self.cfd_mt_value_label,
            self.cfd_mt_weight_label,
        ) = _make_scalar_row(3, "CFD MT fraction")

        scalar_group.setLayout(scalar_layout)
        self.scalar_target_group = scalar_group
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
        self.deposition_group = deposition_group
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
        self.on_administration_changed()

    def _build_study_details_group(self) -> QWidget:
        group = QGroupBox("Study Details")
        layout = QFormLayout()

        self.study_name_edit = QLineEdit()
        self.study_name_edit.setPlaceholderText("Study or configuration name")
        layout.addRow("Config Name:", self.study_name_edit)

        group.setLayout(layout)
        return group

    def _build_administration_group(self) -> QWidget:
        group = QGroupBox("Administration")
        layout = QFormLayout()

        self.administration_combo = QComboBox()
        self.administration_combo.addItem("Intravenous (IV)", userData="iv")
        self.administration_combo.addItem("Oral (PO)", userData="po")
        self.administration_combo.addItem("Inhalation", userData="inhalation")
        self.administration_combo.currentIndexChanged.connect(self.on_administration_changed)
        layout.addRow("Route:", self.administration_combo)

        help_label = QLabel("Administration route controls which subject attributes and stages are available.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addRow(help_label)

        group.setLayout(layout)
        return group

    def _create_subject_attribute_panel(self, category: str, title: str) -> QWidget:
        group = QGroupBox(title)
        vbox = QVBoxLayout()

        tabs = QTabWidget()
        tabs.setObjectName(f"{category}_subject_tabs")

        model_widget = QWidget()
        model_form = QFormLayout()
        combo = QComboBox()
        combo.setEditable(False)
        combo.currentIndexChanged.connect(
            lambda _=0, cat=category: self.on_entity_selection_changed(cat)
        )
        summary_label = QLabel("Select a configuration from the catalog or workspace.")
        summary_label.setWordWrap(True)
        model_form.addRow("Reference:", combo)
        model_form.addRow("Details:", summary_label)
        model_widget.setLayout(model_form)

        manual_widget = QWidget()
        manual_layout = QVBoxLayout()
        manual_edit = QPlainTextEdit()
        manual_edit.setPlaceholderText("Manual overrides (JSON).")
        manual_layout.addWidget(manual_edit)
        manual_widget.setLayout(manual_layout)

        tabs.addTab(model_widget, "Model")
        tabs.addTab(manual_widget, "Manual")
        tabs.currentChanged.connect(
            lambda index, cat=category: self.on_subject_mode_changed(cat, index)
        )

        self.entity_combos[category] = combo
        self.subject_manual_edits[category] = manual_edit
        self.subject_summary_labels[category] = summary_label
        self.subject_panels[category] = group

        vbox.addWidget(tabs)
        group.setLayout(vbox)
        return group

    def _build_subject_configuration_group(self) -> QWidget:
        group = QGroupBox("Subject Configuration")
        layout = QVBoxLayout()

        layout.addWidget(self._create_subject_attribute_panel("subject", "Demographics"))
        layout.addWidget(self._create_subject_attribute_panel("gi", "GI Physiology"))
        layout.addWidget(self._create_subject_attribute_panel("lung", "Lung Physiology"))
        layout.addWidget(self._create_subject_attribute_panel("maneuver", "Inhalation Maneuver"))

        group.setLayout(layout)
        return group

    def _build_product_configuration_group(self) -> QWidget:
        group = QGroupBox("Products")
        layout = QFormLayout()

        self.product_ref_combo = QComboBox()
        self.product_ref_combo.setEditable(False)
        self.product_ref_combo.currentIndexChanged.connect(
            lambda _=0, cat="product": self.on_entity_selection_changed(cat)
        )
        self.product_info_label = QLabel("Select a saved product configuration.")
        self.product_info_label.setWordWrap(True)

        layout.addRow("Product Reference:", self.product_ref_combo)
        layout.addRow("Product Details:", self.product_info_label)

        self.api_info_label = QLabel("Active API is derived from the selected product.")
        self.api_info_label.setWordWrap(True)
        layout.addRow("Active API:", self.api_info_label)

        group.setLayout(layout)

        self.entity_combos.update({
            "product": self.product_ref_combo,
        })
        return group

    def _build_run_configuration_group(self) -> QWidget:
        group = QGroupBox("Simulation Run")
        layout = QFormLayout()

        self.run_type_combo = QComboBox()
        for value, label, _ in self.run_type_definitions:
            self.run_type_combo.addItem(label, userData=value)
        self.run_type_combo.currentIndexChanged.connect(self.on_run_type_changed)
        layout.addRow("Run Type:", self.run_type_combo)

        self.run_label_edit = QLineEdit()
        self.run_label_edit.setPlaceholderText("Friendly run label (optional)")
        layout.addRow("Run Label:", self.run_label_edit)

        self.run_type_help_label = QLabel()
        self.run_type_help_label.setWordWrap(True)
        self.run_type_help_label.setStyleSheet("color: #555; font-size: 11px;")
        layout.addRow(self.run_type_help_label)

        self.run_type_notice_label = QLabel()
        self.run_type_notice_label.setWordWrap(True)
        self.run_type_notice_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addRow(self.run_type_notice_label)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999_999)
        self.seed_spin.setValue(123)
        layout.addRow("Seed:", self.seed_spin)

        group.setLayout(layout)
        return group

    def _build_stage_configuration_group(self) -> QWidget:
        group = QGroupBox("Stages and Models")
        layout = QVBoxLayout()

        info_label = QLabel("Toggle stages and configure models for the selected administration route.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info_label)

        for definition in self.stage_definitions:
            row_widget = self._create_stage_row(definition)
            layout.addWidget(row_widget)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_stage_row(self, definition: Dict[str, Any]) -> QWidget:
        row_widget = QWidget()
        row_layout = QVBoxLayout()

        include_checkbox = QCheckBox(definition["label"])
        include_checkbox.setChecked(definition.get("default_enabled", True))
        include_checkbox.toggled.connect(
            lambda checked, key=definition["key"]: self._on_stage_enabled_changed(key, checked)
        )

        tabs = QTabWidget()
        tabs.setObjectName(f"stage_tabs_{definition['key']}")

        model_widget = QWidget()
        model_form = QFormLayout()
        model_combo = QComboBox()
        families = definition.get("families", [])
        models: List[str] = []
        for family in families:
            models.extend(self.available_models.get(family, []))
        if not models:
            models = ["default"]
        seen: Set[str] = set()
        for model_name in models:
            if model_name in seen:
                continue
            seen.add(model_name)
            model_combo.addItem(model_name, userData=model_name)
        model_form.addRow("Model:", model_combo)

        param_widgets: Dict[str, QWidget] = {}
        for param in definition.get("parameters", []) or []:
            name = param.get("name") or param.get("label")
            if not name:
                continue
            label_text = param.get("label") or name
            param_type = param.get("type", "double")
            widget: QWidget
            if param_type == "double":
                spin = QDoubleSpinBox()
                spin.setDecimals(int(param.get("decimals", 6)))
                spin.setRange(float(param.get("min", -1e12)), float(param.get("max", 1e12)))
                spin.setSingleStep(float(param.get("step", 0.1)))
                spin.setValue(float(param.get("default", 0.0)))
                widget = spin
            elif param_type == "int":
                spin_i = QSpinBox()
                spin_i.setRange(int(param.get("min", -1_000_000)), int(param.get("max", 1_000_000)))
                spin_i.setSingleStep(int(param.get("step", 1)))
                spin_i.setValue(int(param.get("default", 0)))
                widget = spin_i
            elif param_type == "choice":
                combo_widget = QComboBox()
                options = param.get("options", [])
                default_value = param.get("default")
                selected_index = 0
                for idx, option in enumerate(options):
                    if isinstance(option, (list, tuple)) and len(option) >= 2:
                        value, display = option[0], option[1]
                    else:
                        value = option
                        display = option
                    combo_widget.addItem(str(display), userData=value)
                    if default_value is not None and value == default_value:
                        selected_index = idx
                combo_widget.setCurrentIndex(selected_index)
                widget = combo_widget
            else:
                line_edit = QLineEdit()
                default_text = param.get("default")
                if default_text is not None:
                    line_edit.setText(str(default_text))
                widget = line_edit

            param_widgets[name] = widget
            model_form.addRow(f"{label_text}:", widget)

        model_widget.setLayout(model_form)

        manual_widget = QWidget()
        manual_layout = QVBoxLayout()
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_widget.setLayout(manual_layout)

        table_stage_keys = {"cfd", "deposition"}
        manual_widgets: List[QWidget] = []
        if definition["key"] in table_stage_keys:
            manual_edit = SpreadsheetWidget(["Field", "Value"])
            manual_edit.setMinimumHeight(140)
            manual_edit.add_empty_row()
            manual_edit.itemChanged.connect(lambda _item, key=definition["key"]: self._clear_stage_manual_payload(key))
            manual_layout.addWidget(manual_edit)

            load_json_btn = QPushButton("Load JSON…")
            load_json_btn.setFixedWidth(120)
            load_json_btn.clicked.connect(
                lambda _=False, key=definition["key"], table=manual_edit: self.load_stage_manual_json(key, table)
            )
            select_param_btn = QPushButton("Select Parameter…")
            select_param_btn.setFixedWidth(150)
            select_param_btn.clicked.connect(
                lambda _=False, key=definition["key"], table=manual_edit: self.select_stage_manual_parameter(key, table)
            )

            button_row = QHBoxLayout()
            button_row.addWidget(select_param_btn)
            button_row.addWidget(load_json_btn)
            button_row.addStretch()
            manual_layout.addLayout(button_row)
            manual_widget_type = "table"
            manual_loaded_payload = None
            manual_widgets = [manual_edit, select_param_btn, load_json_btn]
        else:
            manual_edit = QPlainTextEdit()
            manual_edit.setPlaceholderText("Manual data overrides or file references.")
            manual_layout.addWidget(manual_edit)
            manual_widget_type = "text"
            manual_loaded_payload = None
            manual_widgets = [manual_edit]

        tabs.addTab(model_widget, "Model")
        tabs.addTab(manual_widget, "Manual")
        tabs.currentChanged.connect(
            lambda index, key=definition["key"]: self.on_stage_mode_changed(key, index)
        )

        row_layout.addWidget(include_checkbox)
        row_layout.addWidget(tabs)
        row_widget.setLayout(row_layout)

        self.stage_controls[definition["key"]] = {
            "definition": definition,
            "checkbox": include_checkbox,
            "model_combo": model_combo,
            "manual_edit": manual_edit,
            "manual_widget_type": manual_widget_type,
            "manual_loaded_payload": manual_loaded_payload,
            "tabs": tabs,
            "container": row_widget,
            "param_widgets": param_widgets,
            "manual_widgets": manual_widgets,
        }

        self._on_stage_enabled_changed(definition["key"], include_checkbox.isChecked())

        return row_widget

    def current_administration(self) -> str:
        if not hasattr(self, "administration_combo"):
            return "iv"
        value = self.administration_combo.currentData()
        return value or "iv"

    def on_subject_mode_changed(self, category: str, tab_index: int) -> None:
        self.subject_modes[category] = "model" if tab_index == 0 else "manual"

    def on_stage_mode_changed(self, stage_key: str, tab_index: int) -> None:
        self.stage_modes[stage_key] = "model" if tab_index == 0 else "manual"
        self._update_stage_manual_controls_state(stage_key)

    def _on_stage_enabled_changed(self, stage_key: str, enabled: bool) -> None:
        controls = self.stage_controls.get(stage_key)
        if not controls:
            return
        tabs = controls.get("tabs")
        if isinstance(tabs, QTabWidget):
            tabs.setEnabled(enabled)
        self._update_stage_manual_controls_state(stage_key, stage_enabled=enabled)

    def _update_stage_manual_controls_state(self, stage_key: str, stage_enabled: Optional[bool] = None) -> None:
        controls = self.stage_controls.get(stage_key)
        if not controls:
            return
        checkbox = controls.get("checkbox")
        if stage_enabled is None and isinstance(checkbox, QCheckBox):
            stage_enabled = checkbox.isChecked()
        if stage_enabled is None:
            stage_enabled = True

        manual_widgets = controls.get("manual_widgets") or []
        for widget in manual_widgets:
            if isinstance(widget, QWidget):
                widget.setEnabled(stage_enabled)

    def on_administration_changed(self) -> None:
        admin = self.current_administration()

        subject_visibility = {
            "subject": True,
            "gi": admin in {"po", "inhalation"},
            "lung": admin == "inhalation",
            "maneuver": admin == "inhalation",
        }
        for category, panel in self.subject_panels.items():
            panel.setVisible(subject_visibility.get(category, False))

        for definition in self.stage_definitions:
            controls = self.stage_controls.get(definition["key"])
            if not controls:
                continue
            visible = admin in definition.get("administrations", set())
            container = controls["container"]
            container.setVisible(visible)
            checkbox: QCheckBox = controls["checkbox"]
            checkbox.blockSignals(True)
            checkbox.setChecked(visible and definition.get("default_enabled", True))
            checkbox.blockSignals(False)
            self._on_stage_enabled_changed(definition["key"], checkbox.isChecked())

        for category in self.entity_combos.keys():
            if category not in self.selected_entities or self.selected_entities[category] is not None:
                continue
            self._set_combo_placeholder(category)

        self.update_stage_parameters_from_product(route=admin)
        self._update_parameter_estimation_visibility()

    def _resolve_active_product_payload(self) -> Optional[Dict[str, Any]]:
        if app_api is None or Product is None:
            return None

        entry = self.selected_entities.get("product")
        if not isinstance(entry, Mapping):
            return None

        ref = entry.get("ref") or entry.get("name")
        overrides = copy.deepcopy(entry.get("overrides") or {})

        base_payload: Dict[str, Any] = {}
        if ref:
            try:
                base_payload = app_api.get_catalog_entry("product", ref) or {}
            except Exception as exc:  # pragma: no cover - catalog read failure
                logger.debug("load product catalog entry failed: %s", exc)

        if overrides:
            base_payload = _apply_overrides(base_payload, overrides)

        return base_payload or None

    def _auto_select_api_from_product(self) -> None:
        product_entry = self.selected_entities.get("product")
        if not isinstance(product_entry, Mapping):
            self.selected_entities["api"] = None
            label = getattr(self, "api_info_label", None)
            if isinstance(label, QLabel):
                label.setText("Select a product configuration to derive API details.")
            return

        product_payload = self._resolve_active_product_payload()
        api_ref: Optional[str] = None
        api_overrides: Dict[str, Any] = {}

        raw_api_payload: Dict[str, Any] = {}

        if isinstance(product_payload, Mapping):
            apis = product_payload.get("apis")
            if isinstance(apis, Mapping) and apis:
                for slot, payload in apis.items():
                    if isinstance(payload, Mapping):
                        candidate_ref = payload.get("ref") or slot
                        if candidate_ref:
                            api_ref = str(candidate_ref)
                            raw_api_payload = copy.deepcopy(payload)
                            api_overrides = copy.deepcopy(payload)
                            break

        if not api_ref:
            previous_api = self.selected_entities.get("api")
            if isinstance(previous_api, Mapping):
                api_ref = previous_api.get("ref") or previous_api.get("name")
                api_overrides = copy.deepcopy(previous_api.get("overrides") or {})

        if not api_ref:
            self.selected_entities["api"] = None
            label = getattr(self, "api_info_label", None)
            if isinstance(label, QLabel):
                label.setText("No API resolved from selected product.")
            return

        if isinstance(api_overrides, Mapping):
            api_overrides = {
                key: value for key, value in api_overrides.items() if key != "ref"
            }
        else:
            api_overrides = {}

        synthetic_api = {
            "ref": api_ref,
            "name": api_ref,
            "overrides": api_overrides,
            "source": "product",
        }
        self.selected_entities["api"] = synthetic_api
        self._update_entity_summary("api")
        label = getattr(self, "api_info_label", None)
        if isinstance(label, QLabel):
            info_parts = [f"Ref: {api_ref}"]
            if isinstance(raw_api_payload, Mapping):
                numeric_pairs = [
                    f"{key}={value}"
                    for key, value in raw_api_payload.items()
                    if isinstance(value, (int, float))
                ]
                if numeric_pairs:
                    info_parts.append(", ".join(numeric_pairs))
            label.setText("; ".join(info_parts))

        self.update_stage_parameters_from_product()

    def _apply_stage_selection(self, stages: Sequence[str], route: Optional[str]) -> None:
        stage_set = {str(stage) for stage in stages}
        for definition in self.stage_definitions:
            if route and route not in definition.get("administrations", set()):
                continue
            controls = self.stage_controls.get(definition["key"])
            if not controls:
                continue
            container = controls.get("container")
            if container is not None and not container.isVisible():
                continue
            checkbox: QCheckBox = controls["checkbox"]
            desired = definition["pipeline_stage"] in stage_set
            if checkbox.isChecked() != desired:
                checkbox.blockSignals(True)
                checkbox.setChecked(desired)
                checkbox.blockSignals(False)

    def _apply_stage_override_values(self, overrides: Mapping[str, Mapping[str, Any]]) -> None:
        for stage_key, payload in overrides.items():
            controls = self.stage_controls.get(stage_key)
            if not controls:
                continue

            if self.stage_modes.get(stage_key, "model") != "model":
                continue

            container = controls.get("container")
            if container is not None and not container.isVisible():
                continue

            tabs: QTabWidget = controls.get("tabs")
            if tabs is not None and tabs.currentIndex() != 0:
                tabs.blockSignals(True)
                tabs.setCurrentIndex(0)
                tabs.blockSignals(False)

            model_value = payload.get("model")
            if model_value:
                model_combo: QComboBox = controls.get("model_combo")
                if model_combo is not None:
                    idx = model_combo.findData(model_value)
                    if idx < 0:
                        idx = model_combo.findText(str(model_value))
                    if idx < 0:
                        model_combo.addItem(str(model_value), userData=model_value)
                        idx = model_combo.findData(model_value)
                    if idx >= 0:
                        model_combo.blockSignals(True)
                        model_combo.setCurrentIndex(idx)
                        model_combo.blockSignals(False)

            params = payload.get("params") or {}
            param_widgets = controls.get("param_widgets", {})
            for name, value in params.items():
                widget = param_widgets.get(name)
                if widget is None or value is None:
                    continue
                if isinstance(widget, QDoubleSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(float(value))
                    widget.blockSignals(False)
                elif isinstance(widget, QSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(int(value))
                    widget.blockSignals(False)
                elif isinstance(widget, QComboBox):
                    idx = widget.findData(value)
                    if idx < 0:
                        idx = widget.findText(str(value))
                    if idx < 0:
                        widget.addItem(str(value), userData=value)
                        idx = widget.findData(value)
                    if idx >= 0:
                        widget.blockSignals(True)
                        widget.setCurrentIndex(idx)
                        widget.blockSignals(False)
                elif isinstance(widget, QLineEdit):
                    widget.blockSignals(True)
                    widget.setText(str(value))
                    widget.blockSignals(False)

    def update_stage_parameters_from_product(self, route: Optional[str] = None) -> None:
        if app_api is None or Product is None:
            return

        admin_route = route or self.current_administration()
        product_payload = self._resolve_active_product_payload()
        if not product_payload:
            return

        api_name = self._current_api_reference()
        if not api_name:
            return

        try:
            product_model = Product.model_validate(product_payload)
        except Exception as exc:  # pragma: no cover - validation failure
            logger.debug("product validation failed for stage defaults: %s", exc)
            return

        product_route = product_model.route
        if product_route and route is None and product_route != self.current_administration():
            self._set_administration_ui(product_route)
            return

        effective_route = route or product_route or self.current_administration()

        stage_list = product_model.get_route_stage_list(effective_route)
        if stage_list:
            self._apply_stage_selection(stage_list, effective_route)

        overrides = product_model.build_stage_overrides(effective_route, api_name=api_name)
        if overrides:
            self._apply_stage_override_values(overrides)
        self._update_parameter_estimation_visibility()

    def _update_parameter_estimation_visibility(self) -> None:
        if not hasattr(self, "scalar_target_group"):
            return

        route = self.current_administration()
        inhalation = route == "inhalation"

        cfd_rows = [
            (
                self.cfd_mmad_checkbox,
                self.cfd_mmad_value_spin,
                self.cfd_mmad_weight_spin,
                self.cfd_mmad_value_label,
                self.cfd_mmad_weight_label,
            ),
            (
                self.cfd_gsd_checkbox,
                self.cfd_gsd_value_spin,
                self.cfd_gsd_weight_spin,
                self.cfd_gsd_value_label,
                self.cfd_gsd_weight_label,
            ),
            (
                self.cfd_mt_checkbox,
                self.cfd_mt_value_spin,
                self.cfd_mt_weight_spin,
                self.cfd_mt_value_label,
                self.cfd_mt_weight_label,
            ),
        ]

        for widgets in cfd_rows:
            for widget in widgets:
                widget.setVisible(inhalation)
        if not inhalation:
            for row in cfd_rows:
                checkbox = row[0]
                checkbox.setChecked(False)

        if hasattr(self, "deposition_group"):
            self.deposition_group.setVisible(inhalation)
            if not inhalation and self.deposition_fraction_enable.isChecked():
                self.deposition_fraction_enable.setChecked(False)

    def _stage_model_value(self, stage_key: str, default: str) -> str:
        controls = self.stage_controls.get(stage_key)
        if not controls:
            return default
        combo: QComboBox = controls["model_combo"]
        value = combo.currentData()
        if not value or value == "default":
            value = combo.currentText()
        return str(value) if value else default

    def _manual_json_for_category(self, category: str) -> Dict[str, Any]:
        edit = self.subject_manual_edits.get(category)
        label = self.entity_display.get(category, category)
        if edit is None:
            return {}
        text = edit.toPlainText().strip()
        if not text:
            raise ValueError(f"Provide manual data for {label} or switch back to model selection.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manual data for {label} must be valid JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"Manual data for {label} must be a JSON object containing at least a 'ref' value.")
        return dict(data)

    def _build_entity_payload(self, category: str, *, required: bool, allow_manual: bool) -> Optional[Dict[str, Any]]:
        mode = self.subject_modes.get(category, "model") if category in self.subject_modes else "model"
        if allow_manual and mode == "manual":
            manual_payload = self._manual_json_for_category(category)
            payload: Dict[str, Any] = {}
            ref_value = manual_payload.get("ref") or manual_payload.get("reference")
            if ref_value:
                payload["ref"] = str(ref_value)
            overrides = manual_payload.get("overrides")
            if overrides is None:
                manual_copy = manual_payload.copy()
                manual_copy.pop("ref", None)
                manual_copy.pop("reference", None)
                if manual_copy:
                    overrides = manual_copy
            if overrides:
                if not isinstance(overrides, Mapping):
                    raise ValueError(f"Manual overrides for {self.entity_display.get(category, category)} must be a dictionary")
                payload["overrides"] = copy.deepcopy(overrides)
            if not payload:
                if required:
                    raise ValueError(f"Provide a reference for {self.entity_display.get(category, category)} in manual mode.")
                return None
            return payload

        payload = self._entity_config_payload(category, required=required)
        return payload if payload else None

    def _set_administration_ui(self, value: str) -> None:
        if not hasattr(self, "administration_combo"):
            return
        idx = self.administration_combo.findData(value)
        if idx < 0 or self.administration_combo.currentIndex() == idx:
            return
        self.administration_combo.blockSignals(True)
        self.administration_combo.setCurrentIndex(idx)
        self.administration_combo.blockSignals(False)
        self.on_administration_changed()

    def _entry_identifier(self, entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        return entry.get("display_id") or entry.get("id") or entry.get("ref")

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
        self._update_entity_summary(category)
        if category == "product":
            self._auto_select_api_from_product()

    def _populate_combo_from_saved(
        self,
        category: str,
        entries: List[Dict[str, Any]],
        select_id: Optional[str] = None,
        select_ref: Optional[str] = None,
    ) -> None:
        combo = self.entity_combos.get(category)
        if not combo:
            return

        combo.blockSignals(True)
        combo.clear()

        placeholder = self.entity_placeholder.get(category, "Select entry")
        combo.addItem(placeholder, None)

        if not entries:
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            self.selected_entities[category] = None
            return

        target_index = 0
        for idx, entry in enumerate(entries, start=1):
            identifier = self._entry_identifier(entry)
            ref_value = entry.get("ref")
            source = entry.get("source", "workspace")
            display_name = entry.get("name") or ref_value or identifier or "(unnamed)"
            if source == "builtin":
                label = f"Builtin: {display_name}"
            else:
                label = f"Workspace: {display_name}"
            combo.addItem(label, copy.deepcopy(entry))

            if select_id and identifier == select_id:
                target_index = idx
            elif select_ref and ref_value and str(ref_value) == str(select_ref) and target_index == 0:
                target_index = idx

        combo.setCurrentIndex(target_index)
        combo.blockSignals(False)
        self.on_entity_selection_changed(category)

    def on_entity_selection_changed(self, category: str) -> None:
        combo = self.entity_combos.get(category)
        if combo:
            data = combo.currentData()
            if isinstance(data, dict):
                self.selected_entities[category] = copy.deepcopy(data)
            else:
                self.selected_entities[category] = None
        self._update_entity_summary(category)
        if category == "product":
            self._update_parameter_estimation_visibility()
            self._auto_select_api_from_product()
        elif category == "api":
            self.update_stage_parameters_from_product()

    def _set_combo_value(self, combo: QComboBox, value: Optional[str]) -> None:
        if combo is None or value is None:
            return
        value_str = str(value)
        combo.blockSignals(True)
        idx = combo.findText(value_str)
        if idx == -1:
            combo.addItem(value_str)
            idx = combo.findText(value_str)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _collect_workspace_bindings(self) -> Dict[str, str]:
        bindings: Dict[str, str] = {}
        for category, entry in self.selected_entities.items():
            if isinstance(entry, Mapping) and entry.get("source") == "workspace" and entry.get("id"):
                bindings[category] = str(entry["id"])
        return bindings

    def _apply_entity_payload(self, category: str, payload: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(payload, Mapping) or not payload.get("ref"):
            self._set_combo_placeholder(category)
            return

        entry = {
            "id": payload.get("id"),
            "ref": payload.get("ref"),
            "name": payload.get("name") or payload.get("ref"),
            "overrides": copy.deepcopy(payload.get("overrides") or {}),
            "variability_overrides": copy.deepcopy(payload.get("variability") or payload.get("variability_overrides") or {}),
            "source": payload.get("source", "config"),
        }

        combo = self.entity_combos.get(category)
        if combo is not None:
            combo.blockSignals(True)
            combo.clear()
            display = entry.get("name") or entry.get("ref") or "(unnamed)"
            combo.addItem(display, copy.deepcopy(entry))
            combo.setCurrentIndex(0)
            combo.blockSignals(False)

        self.selected_entities[category] = copy.deepcopy(entry)
        self._update_entity_summary(category)

    def _select_entity(
        self,
        category: str,
        binding_id: Optional[str],
        payload: Optional[Mapping[str, Any]],
    ) -> None:
        entries = list(self.all_catalog_entries.get(category, []))

        target_id = binding_id
        target_ref = None
        if isinstance(payload, Mapping):
            target_ref = payload.get("ref")
            if target_id is None:
                target_id = payload.get("id") or payload.get("display_id")

            identifier = target_id or (f"config::{target_ref}" if target_ref else None)
            if identifier and not any(self._entry_identifier(entry) == identifier for entry in entries):
                synthetic = {
                    "id": identifier,
                    "display_id": identifier,
                    "ref": target_ref,
                    "name": payload.get("name") or target_ref or identifier,
                    "overrides": copy.deepcopy(payload.get("overrides") or {}),
                    "variability_overrides": copy.deepcopy(payload.get("variability") or payload.get("variability_overrides") or {}),
                    "source": payload.get("source", "config"),
                }
                entries.append(synthetic)
        self.all_catalog_entries[category] = entries
        self._populate_combo_from_saved(
            category,
            entries,
            select_id=target_id,
            select_ref=target_ref,
        )

        # Merge overrides/variability into selected entity for accuracy
        if isinstance(payload, Mapping):
            selected = self.selected_entities.get(category)
            if isinstance(selected, dict):
                selected["overrides"] = copy.deepcopy(payload.get("overrides") or {})
                selected["variability_overrides"] = copy.deepcopy(
                    payload.get("variability") or payload.get("variability_overrides") or {}
                )
                selected.setdefault("ref", payload.get("ref"))
                selected.setdefault("name", payload.get("name") or payload.get("ref"))
                selected.setdefault("source", payload.get("source", selected.get("source", "config")))
                self._update_entity_summary(category)
                if category == "product":
                    self._update_parameter_estimation_visibility()

    def _update_entity_summary(self, category: str) -> None:
        if category in self.subject_summary_labels:
            label = self.subject_summary_labels.get(category)
        elif category == "api":
            label = getattr(self, "api_info_label", None)
        elif category == "product":
            label = getattr(self, "product_info_label", None)
        elif category == "maneuver":
            label = self.subject_summary_labels.get("maneuver")
        else:
            label = None
        if label is None:
            return

        entry = self.selected_entities.get(category)
        if not isinstance(entry, Mapping):
            label.setText("Select a configuration from the workspace or builtin catalog.")
            return

        source = entry.get("source", "workspace")
        name = entry.get("name") or entry.get("ref") or entry.get("display_id") or "(unnamed)"
        ref_value = entry.get("ref") or "—"
        overrides = entry.get("overrides") or {}
        override_keys = sorted(overrides.keys())
        variability = entry.get("variability_overrides") or {}
        variability_keys = sorted(variability.keys())

        lines = [
            f"Source: {source}",
            f"Reference: {ref_value}",
        ]

        if override_keys:
            lines.append(f"Overrides: {', '.join(override_keys)}")
        else:
            lines.append("Overrides: none")

        if variability_keys:
            lines.append(f"Variability: {', '.join(variability_keys)}")

        if category == "product":
            api_overrides = overrides.get("apis") if isinstance(overrides, Mapping) else None
            if api_overrides:
                if isinstance(api_overrides, Mapping):
                    api_summary = ", ".join(sorted(str(key) for key in api_overrides.keys()))
                elif isinstance(api_overrides, list):
                    api_summary = ", ".join(
                        str(item.get("name") or item.get("ref") or idx + 1)
                        for idx, item in enumerate(api_overrides)
                        if isinstance(item, Mapping)
                    )
                else:
                    api_summary = str(api_overrides)
                lines.append(f"APIs: {api_summary}")
            elif ref_value and CATALOG_AVAILABLE and app_api is not None:
                try:
                    base_product = app_api.get_catalog_entry("product", ref_value)
                except Exception:
                    base_product = None
                if base_product is not None:
                    base_payload = {}
                    try:
                        base_payload = base_product.model_dump()
                    except Exception:
                        try:
                            base_payload = dict(base_product)
                        except Exception:
                            base_payload = {}
                    base_apis = base_payload.get("apis")
                    if isinstance(base_apis, Mapping):
                        api_summary = ", ".join(sorted(str(key) for key in base_apis.keys()))
                        if api_summary:
                            lines.append(f"Catalog APIs: {api_summary}")
                    elif isinstance(base_apis, list):
                        api_summary = ", ".join(
                            str(item.get("name") or item.get("ref") or idx + 1)
                            for idx, item in enumerate(base_apis)
                            if isinstance(item, Mapping)
                        )
                        if api_summary:
                            lines.append(f"Catalog APIs: {api_summary}")

        label.setText("\n".join(lines))

    def apply_config(
        self,
        config: Mapping[str, Any],
        *,
        run_plan: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(config, Mapping):
            return

        run_section = config.get("run")
        if isinstance(run_section, Mapping):
            stages = run_section.get("stages")
            stage_set = set()
            if isinstance(stages, list):
                stage_set = {str(stage) for stage in stages}
            elif stages is not None:
                stage_set = {str(stages)}

            if stage_set:
                if "iv_pk" in stage_set and "deposition" not in stage_set and "pbbm" not in stage_set and "gi_pk" not in stage_set:
                    self._set_administration_ui("iv")
                elif "deposition" in stage_set or "pbbm" in stage_set:
                    self._set_administration_ui("inhalation")
                elif "gi_pk" in stage_set:
                    self._set_administration_ui("po")

            for pipeline_stage, stage_key in self.stage_pipeline_lookup.items():
                controls = self.stage_controls.get(stage_key)
                if not controls:
                    continue
                checkbox: QCheckBox = controls["checkbox"]
                checkbox.blockSignals(True)
                checkbox.setChecked(pipeline_stage in stage_set)
                checkbox.blockSignals(False)

            stage_override_map = run_section.get("stage_overrides") if isinstance(run_section, Mapping) else {}
            if isinstance(stage_override_map, Mapping):
                for stage_key, payload in stage_override_map.items():
                    controls = self.stage_controls.get(stage_key)
                    if not controls:
                        continue
                    tabs: QTabWidget = controls["tabs"]
                    manual_edit: QPlainTextEdit = controls["manual_edit"]
                    model_combo: QComboBox = controls["model_combo"]
                    param_widgets = controls.get("param_widgets", {})
                    if isinstance(payload, Mapping):
                        model_value = payload.get("model")
                        params_payload = payload.get("params") if isinstance(payload.get("params"), Mapping) else None
                        extra_keys = set(payload.keys()) - {"model", "params"}

                        if extra_keys:
                            manual_edit.setPlainText(json.dumps(payload, indent=2))
                            tabs.setCurrentIndex(1)
                        else:
                            if model_value is not None:
                                self._set_combo_value(model_combo, model_value)
                            if params_payload:
                                for name, value in params_payload.items():
                                    widget = param_widgets.get(name)
                                    if widget is None:
                                        continue
                                    if isinstance(widget, QDoubleSpinBox):
                                        try:
                                            widget.setValue(float(value))
                                        except (TypeError, ValueError):
                                            pass
                                    elif isinstance(widget, QSpinBox):
                                        try:
                                            widget.setValue(int(value))
                                        except (TypeError, ValueError):
                                            pass
                                    elif isinstance(widget, QComboBox):
                                        idx = widget.findData(value)
                                        if idx < 0:
                                            idx = widget.findText(str(value))
                                        if idx >= 0:
                                            widget.setCurrentIndex(idx)
                                    elif isinstance(widget, QLineEdit):
                                        widget.setText(str(value))
                            tabs.setCurrentIndex(0)
                    else:
                        try:
                            manual_edit.setPlainText(json.dumps(payload, indent=2))
                        except TypeError:
                            manual_edit.setPlainText(str(payload))
                        tabs.setCurrentIndex(1)

            seed = run_section.get("seed")
            if seed is not None:
                try:
                    self.seed_spin.setValue(int(seed))
                except (TypeError, ValueError):
                    pass

        deposition = config.get("deposition")
        if isinstance(deposition, Mapping):
            combo = self.stage_controls.get("deposition", {}).get("model_combo")
            if combo is not None:
                self._set_combo_value(combo, deposition.get("model"))

        pbbm = config.get("pbbm")
        if isinstance(pbbm, Mapping):
            combo = self.stage_controls.get("pbbm", {}).get("model_combo")
            if combo is not None:
                self._set_combo_value(combo, pbbm.get("model"))

        pk = config.get("pk")
        if isinstance(pk, Mapping):
            combo = self.stage_controls.get("pk", {}).get("model_combo")
            if combo is not None:
                self._set_combo_value(combo, pk.get("model"))

        gi_payload = config.get("gi_tract") or config.get("gi")
        if isinstance(gi_payload, Mapping):
            binding_id = bindings.get("gi") if isinstance(bindings, Mapping) else None
            self._select_entity("gi", binding_id, gi_payload)

        lung_payload = None
        if isinstance(deposition, Mapping):
            lung_geometry_ref = (
                deposition.get("lung_geometry_ref")
                or deposition.get("lung_geometry")
                or deposition.get("geometry_ref")
            )
            lung_overrides = deposition.get("lung_geometry_overrides") or deposition.get("lung_geometry_override")
            if lung_geometry_ref or lung_overrides:
                lung_payload = {}
                if lung_geometry_ref:
                    lung_payload["ref"] = lung_geometry_ref
                if isinstance(lung_overrides, Mapping) and lung_overrides:
                    lung_payload["overrides"] = copy.deepcopy(lung_overrides)
        if lung_payload:
            binding_id = bindings.get("lung") if isinstance(bindings, Mapping) else None
            self._select_entity("lung", binding_id, lung_payload)

        study_section = config.get("study")
        if isinstance(study_section, Mapping):
            label = study_section.get("study_label") or study_section.get("name")
            if label:
                self.study_name_edit.setText(str(label))

        bindings = {}
        if run_plan is not None and isinstance(run_plan, Mapping):
            bindings = run_plan.get("workspace_bindings") or {}

        entity_payloads: Dict[str, Optional[Mapping[str, Any]]] = {
            "subject": config.get("subject"),
            "gi": gi_payload,
            "lung": {"ref": lung_geometry_ref} if lung_geometry_ref else None,
            "api": config.get("api"),
            "product": config.get("product"),
            "maneuver": config.get("maneuver"),
        }

        for category, payload in entity_payloads.items():
            if payload is None:
                continue
            binding_id = None
            if isinstance(bindings, Mapping):
                binding_id = bindings.get(category)
            self._select_entity(category, binding_id, payload)

        if isinstance(bindings, Mapping):
            for category, binding_id in bindings.items():
                if category in entity_payloads:
                    continue
                if category in self.entity_combos:
                    self._select_entity(category, binding_id, None)

        if run_plan is not None:
            try:
                self._apply_run_plan_to_ui(run_plan)
            except Exception:
                pass
            label_text = run_plan.get("run_label")
            if label_text:
                self.run_label_edit.setText(str(label_text))
            config_name = run_plan.get("config_name")
            if config_name:
                self.study_name_edit.setText(str(config_name))

    def reload_workspace_catalog_entries(self) -> None:
        categories = tuple(self.catalog_category_map.items())

        if self.workspace_manager is None:
            self.populate_catalog_defaults()
            self._refresh_entity_combos()
            return

        for category, catalog_name in categories:
            try:
                raw_entries = self.workspace_manager.list_catalog_entries(catalog_name)
            except Exception as exc:
                logger.warning(
                    "list catalog entries failed",
                    category=catalog_name,
                    error=str(exc),
                )
                raw_entries = []

            processed: List[Dict[str, Any]] = []
            for entry in raw_entries or []:
                if not isinstance(entry, Mapping):
                    continue
                prepared = copy.deepcopy(entry)
                prepared["source"] = "workspace"
                prepared.setdefault("ref", prepared.get("name"))
                prepared.setdefault("display_id", prepared.get("id") or prepared.get("ref"))
                processed.append(prepared)

            self.workspace_catalog_entries[category] = processed

        self._refresh_entity_combos()

    def _refresh_entity_combos(self) -> None:
        for category in self.catalog_category_map.keys():
            combo = self.entity_combos.get(category)
            if combo is None:
                continue
            combined: List[Dict[str, Any]] = []
            for entry in self.workspace_catalog_entries.get(category, []):
                combined.append(copy.deepcopy(entry))
            for entry in self.builtin_catalog_entries.get(category, []):
                combined.append(copy.deepcopy(entry))

            self.all_catalog_entries[category] = combined

            selected_entry = self.selected_entities.get(category)
            target_id = self._entry_identifier(selected_entry) if isinstance(selected_entry, Mapping) else None
            target_ref = selected_entry.get("ref") if isinstance(selected_entry, Mapping) else None

            self._populate_combo_from_saved(
                category,
                combined,
                select_id=target_id,
                select_ref=target_ref,
            )

    def _entity_config_payload(self, category: str, *, required: bool = True) -> Dict[str, Any]:
        entry = self.selected_entities.get(category)
        label = self.entity_display.get(category, category)
        if not entry:
            if required:
                raise ValueError(
                    f"Select a {label} from its configuration tab before building the configuration."
                )
            return {}

        ref = entry.get("ref")
        if not ref:
            if required:
                raise ValueError(f"Saved {label} entry is missing a catalog reference.")
            return {}

        payload: Dict[str, Any] = {"ref": ref}
        overrides = entry.get("overrides") or {}
        if overrides:
            processed = copy.deepcopy(overrides)
            if category == "product":
                processed = self._normalise_product_overrides(ref, processed)
            payload["overrides"] = processed
        return payload

    def _normalise_product_overrides(self, base_ref: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(overrides)
        api_entries = result.get("apis")
        if not isinstance(api_entries, list):
            return result

        slot_names: List[str] = []
        if base_ref and CATALOG_AVAILABLE and app_api is not None:
            try:
                base_product = app_api.get_catalog_entry("product", base_ref) or {}
                base_apis = base_product.get("apis")
                if isinstance(base_apis, dict):
                    slot_names = list(base_apis.keys())
                elif isinstance(base_apis, list):
                    slot_names = [
                        str(item.get("name") or item.get("ref"))
                        for item in base_apis
                        if isinstance(item, dict) and (item.get("name") or item.get("ref"))
                    ]
            except Exception:
                slot_names = []

        normalised_entries: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(api_entries):
            if not isinstance(entry, dict):
                continue
            data = ProductAPIEditor._normalise_entry(entry)

            slot_name = None
            if slot_names and idx < len(slot_names):
                slot_name = slot_names[idx]
            slot_name = slot_name or entry.get("slot_name") or data.get("name") or entry.get("name") or entry.get("ref") or f"API_{idx + 1}"
            slot_key = str(slot_name)

            payload: Dict[str, Any] = {}

            ref_value = entry.get("ref") or data.get("ref")
            if ref_value:
                payload["ref"] = str(ref_value)

            dose_pg = data.get("dose_pg")
            if dose_pg is not None:
                try:
                    payload["dose_pg"] = float(dose_pg)
                except (TypeError, ValueError):
                    pass

            dose_ug = data.get("dose_ug")
            if dose_ug is not None:
                try:
                    payload["dose_ug"] = float(dose_ug)
                except (TypeError, ValueError):
                    pass

            for key in ("usp_depo_fraction", "mmad", "gsd"):
                value = data.get(key)
                if value is None:
                    continue
                try:
                    payload[key] = float(value)
                except (TypeError, ValueError):
                    continue

            if payload:
                normalised_entries[slot_key] = payload

        if normalised_entries:
            result["apis"] = normalised_entries
        return result

    def populate_catalog_defaults(self):
        """Populate reference selectors with catalog data when available."""
        categories = tuple(self.catalog_category_map.keys())

        if not CATALOG_AVAILABLE or app_api is None:
            for category in categories:
                self.builtin_catalog_entries[category] = []
            if self.workspace_manager is None:
                for category in categories:
                    combo = self.entity_combos.get(category)
                    if combo is not None:
                        self._set_combo_placeholder(category)
            return

        try:
            for category, catalog_name in self.catalog_category_map.items():
                try:
                    refs = app_api.list_catalog_entries(catalog_name) or []
                except Exception:
                    refs = []
                builtin_entries: List[Dict[str, Any]] = []
                for ref in refs:
                    entry = {
                        "id": f"builtin::{ref}",
                        "name": ref,
                        "ref": ref,
                        "overrides": {},
                        "source": "builtin",
                        "display_id": f"builtin::{ref}",
                    }
                    builtin_entries.append(entry)
                self.builtin_catalog_entries[category] = builtin_entries
        except Exception as exc:
            logger.warning("Catalog population failed", error=str(exc))

        if self.workspace_manager is None:
            self._refresh_entity_combos()

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
            self._on_sweep_enabled_changed(is_sweep)

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
        rows = self.sweep_table.get_non_empty_rows() if hasattr(self, "sweep_table") else []
        if not rows:
            return {}

        sweep_params: Dict[str, Any] = {}
        for idx, row in enumerate(rows, start=1):
            if not row:
                continue
            path = row[0].strip() if len(row) > 0 else ""
            if not path:
                raise ValueError(f"Sweep row {idx} is missing a parameter path")

            values: List[Any] = []
            sweep_expression: Optional[Dict[str, Any]] = None
            for raw_value in row[1:]:
                text = raw_value.strip() if raw_value is not None else ""
                if not text:
                    continue
                try:
                    parsed_expression = self._parse_sweep_expression(text)
                except ValueError as exc:
                    raise ValueError(f"Sweep row {idx}: {exc}") from exc

                if parsed_expression is not None:
                    if sweep_expression is not None or values:
                        raise ValueError(
                            f"Sweep row {idx} mixes expression syntax with explicit values; choose one style."
                        )
                    sweep_expression = parsed_expression
                    continue

                literal_value = self._coerce_literal(text)
                values.append(literal_value)

            if sweep_expression is not None:
                sweep_params[path] = sweep_expression
                continue

            if not values:
                raise ValueError(f"Provide at least one value for sweep parameter '{path}'")

            if len(values) == 1 and isinstance(values[0], Mapping) and "@sweep" in values[0]:
                sweep_params[path] = values[0]
            else:
                sweep_params[path] = values if len(values) > 1 else values[0]

        return sweep_params

    def _on_sweep_enabled_changed(self, enabled: bool) -> None:
        if hasattr(self, "sweep_table"):
            self.sweep_table.setEnabled(enabled)
        if hasattr(self, "sweep_add_row_btn"):
            self.sweep_add_row_btn.setEnabled(enabled)
        if hasattr(self, "sweep_remove_row_btn"):
            self.sweep_remove_row_btn.setEnabled(enabled)
        if hasattr(self, "sweep_select_param_btn"):
            self.sweep_select_param_btn.setEnabled(enabled)

    def remove_selected_sweep_rows(self) -> None:
        if not hasattr(self, "sweep_table"):
            return
        selection_model = self.sweep_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = {index.row() for index in selection_model.selectedRows()}
        if not selected_rows:
            selected_rows = {index.row() for index in selection_model.selectedIndexes()}
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < self.sweep_table.rowCount():
                self.sweep_table.removeRow(row)
        if self.sweep_table.rowCount() == 0:
            self.sweep_table.add_empty_row()

    def select_sweep_parameter(self) -> None:
        if not hasattr(self, "sweep_table"):
            return
        row = self.sweep_table.currentRow()
        if row < 0:
            if self.sweep_table.rowCount() == 0:
                row = self.sweep_table.add_empty_row()
            else:
                row = 0
        if self.sweep_table.rowCount() == 0:
            row = self.sweep_table.add_empty_row()
        if self.sweep_table.item(row, 0) is None:
            self.sweep_table.setItem(row, 0, QTableWidgetItem(""))
        initial_path = self.sweep_table.item(row, 0).text() if self.sweep_table.item(row, 0) else ""
        selected = self._open_parameter_picker_dialog(initial_path)
        if selected:
            self.sweep_table.item(row, 0).setText(selected)
            self.sweep_table.setCurrentCell(row, 0)

    def _choose_parameter_path(self, selector: "ParameterPathSelector") -> None:
        selected = self._open_parameter_picker_dialog(selector.text())
        if selected:
            selector.setText(selected)

    def _parameter_tree_source_config(
        self,
        *,
        allow_manual_fallback: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        app_config: Optional[AppConfig] = None
        config_dict: Optional[Dict[str, Any]] = None

        if self.current_config:
            config_dict = copy.deepcopy(self.current_config)
            if CONFIG_MODEL_AVAILABLE and AppConfig is not None:
                try:
                    app_config = AppConfig.model_validate(config_dict)
                except Exception as exc:  # pragma: no cover - validation diagnostics
                    logger.debug("validate cached config failed", exc_info=exc)
                    app_config = None

        build_error: Optional[str] = None
        if config_dict is None or app_config is None:
            original_modes = self.stage_modes.copy()
            try:
                if allow_manual_fallback and any(mode == "manual" for mode in original_modes.values()):
                    for key, mode in original_modes.items():
                        if mode == "manual":
                            self.stage_modes[key] = "model"

                if CONFIG_MODEL_AVAILABLE and AppConfig is not None:
                    app_config = self.build_app_config()
                    config_dict = app_config.model_dump(mode="python")
                else:
                    config_dict = self.build_config()
            except Exception as exc:
                logger.debug("Parameter tree snapshot failed", exc_info=exc)
                build_error = str(exc)
                config_dict = None
            finally:
                self.stage_modes.clear()
                self.stage_modes.update(original_modes)

        if config_dict is None:
            return None, build_error or "Unable to build configuration"

        if app_config is None and CONFIG_MODEL_AVAILABLE and AppConfig is not None:
            try:
                app_config = AppConfig.model_validate(config_dict)
            except Exception as exc:  # pragma: no cover
                logger.debug("late AppConfig validation failed", exc_info=exc)
                app_config = None

        tree_config: Dict[str, Any] = copy.deepcopy(config_dict)

        if app_api is not None:
            try:
                self._augment_tree_with_catalog_defaults(tree_config)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("augment catalog defaults failed", exc_info=exc)

            if app_config is not None:
                try:
                    hydrated = app_api.hydrate_config(app_config)
                except Exception as exc:  # pragma: no cover - hydration failures
                    logger.debug("hydrate config failed", exc_info=exc)
                else:
                    resolved = tree_config.setdefault("_resolved_entities", {})
                    for key, value in hydrated.items():
                        if key == "config":
                            continue
                        resolved[key] = _to_plain_data(value)

        self._inject_stage_placeholders_for_tree(tree_config)

        return tree_config, None

    def _open_parameter_picker_dialog(
        self,
        initial_path: str = "",
        *,
        allow_manual_fallback: bool = True,
    ) -> Optional[str]:
        config_dict, error_message = self._parameter_tree_source_config(
            allow_manual_fallback=allow_manual_fallback
        )
        if not config_dict:
            details = f"\n\nDetails: {error_message}" if error_message else ""
            QMessageBox.warning(
                self,
                "Select Parameter",
                "Unable to build configuration for parameter selection."
                + details,
            )
            return None

        dialog = ParameterPickerDialog(config_dict, self)
        if initial_path:
            dialog.preselect_path(initial_path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_path
        return None

    @staticmethod
    def _ensure_tree_path(root: Dict[str, Any], segments: Sequence[str]) -> Dict[str, Any]:
        current = root
        for segment in segments:
            node = current.get(segment)
            if not isinstance(node, dict):
                node = {}
                current[segment] = node
            current = node
        return current

    def _augment_tree_with_catalog_defaults(self, tree_config: Dict[str, Any]) -> None:
        if app_api is None:
            return

        entity_paths: Dict[str, Tuple[str, ...]] = {
            "subject": ("subject", "overrides"),
            "api": ("api", "overrides"),
            "product": ("product", "overrides"),
            "maneuver": ("maneuver", "overrides"),
            "gi": ("gi_tract", "overrides"),
            "lung": ("deposition", "lung_geometry_overrides"),
        }

        for category, path in entity_paths.items():
            entry = self.selected_entities.get(category)
            if not isinstance(entry, Mapping):
                continue
            ref = entry.get("ref") or entry.get("name")
            if not ref:
                continue

            catalog_key = self.catalog_category_map.get(category, category)
            if category == "lung":
                catalog_key = "lung_geometry"
            elif category == "gi":
                catalog_key = "gi_tract"

            try:
                base_payload = app_api.get_catalog_entry(catalog_key, ref)
            except Exception as exc:  # pragma: no cover - catalog issues
                logger.debug(
                    "load catalog entry failed",
                    category=catalog_key,
                    ref=ref,
                    error=str(exc),
                )
                base_payload = None

            if not isinstance(base_payload, Mapping):
                continue

            overrides = copy.deepcopy(entry.get("overrides") or {})
            if overrides:
                base_payload = _apply_overrides(base_payload, overrides)

            target_parent = self._ensure_tree_path(tree_config, path[:-1])
            target = target_parent.setdefault(path[-1], {})
            if not isinstance(target, dict):
                target_parent[path[-1]] = {}
                target = target_parent[path[-1]]

            for key, value in base_payload.items():
                target.setdefault(str(key), _to_plain_data(value))

    def _current_api_reference(self) -> Optional[str]:
        entry = self.selected_entities.get("api")
        if isinstance(entry, Mapping):
            ref = entry.get("ref") or entry.get("name")
            if ref:
                return str(ref)

        product_payload = self._resolve_active_product_payload()
        if isinstance(product_payload, Mapping):
            apis = product_payload.get("apis")
            if isinstance(apis, Mapping):
                for slot, payload in apis.items():
                    if isinstance(payload, Mapping):
                        ref = payload.get("ref") or slot
                        if ref:
                            return str(ref)
                # fallback to first key if values are not dicts
                for slot in apis.keys():
                    if slot:
                        return str(slot)
        return None

    def _inject_stage_placeholders_for_tree(self, tree_config: Dict[str, Any]) -> None:
        run_section = tree_config.setdefault("run", {})
        overrides_section = run_section.setdefault("stage_overrides", {})

        # Pre-seed with product defaults where available
        product_payload = self._resolve_active_product_payload()
        api_ref = self._current_api_reference()
        if Product is not None and isinstance(product_payload, Mapping):
            try:
                product_model = Product.model_validate(product_payload)
            except Exception as exc:  # pragma: no cover - validation diagnostics
                logger.debug("product validation failed for tree placeholders", exc_info=exc)
            else:
                route = self.current_administration()
                stage_defaults = product_model.build_stage_overrides(route, api_name=api_ref)
                for stage_key, payload in stage_defaults.items():
                    stage_entry = overrides_section.setdefault(stage_key, {})
                    for key, value in payload.items():
                        if isinstance(value, Mapping):
                            branch = stage_entry.setdefault(key, {})
                            for sub_key, sub_value in value.items():
                                branch.setdefault(str(sub_key), _to_plain_data(sub_value))
                        else:
                            stage_entry.setdefault(str(key), _to_plain_data(value))

        for stage_key, controls in self.stage_controls.items():
            stage_entry = overrides_section.setdefault(stage_key, {})

            combo: Optional[QComboBox] = controls.get("model_combo")
            if isinstance(combo, QComboBox):
                model_value = combo.currentData() or combo.currentText()
                if model_value:
                    stage_entry.setdefault("model", model_value)

            param_widgets = controls.get("param_widgets") or {}
            if not param_widgets:
                params_entry = stage_entry.setdefault("params", {})
            else:
                params_entry = stage_entry.setdefault("params", {})
                for name in param_widgets.keys():
                    params_entry.setdefault(str(name), None)

            if stage_key in {"pk", "iv_pk", "gi_pk"}:
                for placeholder in PK_PARAM_PLACEHOLDERS:
                    params_entry.setdefault(placeholder, None)

    def select_stage_manual_parameter(self, stage_key: str, table: SpreadsheetWidget) -> None:
        controls = self.stage_controls.get(stage_key)
        if controls is None:
            return
        checkbox = controls.get("checkbox")
        if isinstance(checkbox, QCheckBox) and not checkbox.isChecked():
            QMessageBox.information(
                self,
                "Stage Disabled",
                "Enable the stage before selecting manual override parameters.",
            )
            return

        if table.rowCount() == 0:
            table.add_empty_row()
        row = table.currentRow()
        if row < 0:
            row = 0

        if table.item(row, 0) is None:
            table.setItem(row, 0, QTableWidgetItem(""))

        initial_path = table.item(row, 0).text().strip()
        selected = self._open_parameter_picker_dialog(initial_path, allow_manual_fallback=True)
        if not selected:
            return

        table.item(row, 0).setText(selected)

        if table.columnCount() < 2:
            table._ensure_column_count(2)
        if table.item(row, 1) is None:
            table.setItem(row, 1, QTableWidgetItem(""))
        table.setCurrentCell(row, 1)

    def _clear_stage_manual_payload(self, stage_key: str) -> None:
        controls = self.stage_controls.get(stage_key)
        if controls is not None:
            controls["manual_loaded_payload"] = None

    def load_stage_manual_json(self, stage_key: str, table: SpreadsheetWidget) -> None:
        try:
            start_dir = str(self.workspace_manager.workspace_path) if self.workspace_manager is not None else str(Path.cwd())
        except Exception:
            start_dir = str(Path.cwd())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Stage Overrides",
            start_dir,
            "JSON files (*.json);;All files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            QMessageBox.warning(self, "Load JSON", f"Failed to load overrides: {exc}")
            return

        controls = self.stage_controls.get(stage_key)
        if controls is None:
            return
        controls["manual_loaded_payload"] = payload

        if isinstance(payload, Mapping):
            rows = []
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    value_text = json.dumps(value)
                else:
                    value_text = str(value)
                rows.append([str(key), value_text])
            if rows:
                table.set_data(rows)
                table.add_empty_row()
            else:
                table.clear_rows()
        else:
            QMessageBox.information(
                self,
                "Load JSON",
                "Loaded JSON is not an object; it will be used as-is when the configuration is built.",
            )

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

    def clear_observed_series(self) -> None:
        if hasattr(self, "observed_series_table"):
            self.observed_series_table.clear_rows()

    def load_observed_series(self) -> None:
        if not hasattr(self, "observed_series_table"):
            return

        time_column = self.observed_time_col_edit.text().strip() or "time"
        value_column = self.observed_value_col_edit.text().strip() or "value"

        if self.workspace_manager is not None:
            start_dir = str(self.workspace_manager.workspace_path)
        else:
            start_dir = str(Path.cwd())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Observed Series",
            start_dir,
            "CSV files (*.csv *.tsv);;All files (*)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            raise ValueError(f"Failed to load observed series: {exc}") from exc

        if time_column not in df.columns or value_column not in df.columns:
            columns = ", ".join(df.columns)
            raise ValueError(
                f"Columns '{time_column}' or '{value_column}' not found in observed series file. Available: {columns}"
            )

        times = df[time_column].tolist()
        values = df[value_column].tolist()
        rows = [[str(t), str(v)] for t, v in zip(times, values)]
        self.observed_series_table.set_data(rows)
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

        path_selector = ParameterPathSelector(path, picker=self._choose_parameter_path)
        self.parameter_table.setCellWidget(row, 1, path_selector)

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
            rows: List[List[str]] = []
            for path_value, sweep_values in params.items():
                values = sweep_values if isinstance(sweep_values, (list, tuple)) else [sweep_values]
                row = [str(path_value)] + [str(v) for v in values]
                rows.append(row)
            if hasattr(self, "sweep_table"):
                if rows:
                    self.sweep_table.set_data(rows)
                else:
                    self.sweep_table.clear_rows()
            self.sweep_enabled.setChecked(bool(rows))
            self._on_sweep_enabled_changed(bool(rows))
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
        self.clear_observed_series()
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
                        series_dict = self._normalise_observed_payload(series)
                        rows = [[str(t), str(v)] for t, v in zip(series_dict.get("time_s", []), series_dict.get("values", []))]
                        if rows:
                            self.observed_series_table.set_data(rows)
                    except Exception:
                        self.clear_observed_series()
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
    def _parse_sweep_expression(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return None

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", stripped)
        if not match:
            return None

        func_name = match.group(1).lower()
        arg_body = match.group(2).strip()

        args: List[str] = []
        if arg_body:
            buffer: List[str] = []
            depth = 0
            for char in arg_body:
                if char == ',' and depth == 0:
                    argument = ''.join(buffer).strip()
                    if argument:
                        args.append(argument)
                    buffer = []
                    continue
                if char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth = max(0, depth - 1)
                buffer.append(char)
            argument = ''.join(buffer).strip()
            if argument:
                args.append(argument)

        def _parse_literal(arg: str) -> Any:
            lowered = arg.lower()
            if lowered in {"none", "null"}:
                return None
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            try:
                if any(ch in arg for ch in (".", "e", "E")):
                    return float(arg)
                return int(arg)
            except ValueError:
                return arg

        parsed_args = [_parse_literal(arg) for arg in args]

        def _require_length(expected: Sequence[int]) -> None:
            if len(parsed_args) not in expected:
                if len(expected) == 1:
                    raise ValueError(
                        f"Expression '{stripped}' expects {expected[0]} argument(s); received {len(parsed_args)}"
                    )
                expected_str = ", ".join(str(num) for num in expected)
                raise ValueError(
                    f"Expression '{stripped}' expects {expected_str} argument(s); received {len(parsed_args)}"
                )

        spec: Dict[str, Any]

        if func_name == "range":
            _require_length({2, 3})
            start = float(parsed_args[0])
            stop = float(parsed_args[1])
            step = float(parsed_args[2]) if len(parsed_args) == 3 else 1.0
            if step == 0:
                raise ValueError("Range step must be non-zero")
            spec = {
                "@sweep": {
                    "type": "range",
                    "start": start,
                    "stop": stop,
                    "step": step,
                }
            }
            return spec

        if func_name == "linspace":
            _require_length({3, 4})
            start = float(parsed_args[0])
            stop = float(parsed_args[1])
            count = int(parsed_args[2])
            if count <= 0:
                raise ValueError("linspace requires a positive sample count")
            endpoint = True if len(parsed_args) == 3 else bool(parsed_args[3])
            spec = {
                "@sweep": {
                    "type": "linspace",
                    "start": start,
                    "stop": stop,
                    "count": count,
                    "endpoint": endpoint,
                }
            }
            return spec

        if func_name == "logspace":
            _require_length({3, 4})
            start = float(parsed_args[0])
            stop = float(parsed_args[1])
            count = int(parsed_args[2])
            if count <= 0:
                raise ValueError("logspace requires a positive sample count")
            base = float(parsed_args[3]) if len(parsed_args) == 4 else 10.0
            spec = {
                "@sweep": {
                    "type": "logspace",
                    "start": start,
                    "stop": stop,
                    "count": count,
                    "base": base,
                }
            }
            return spec

        if func_name in {"normal", "lognormal"}:
            _require_length({2, 3})
            mean = float(parsed_args[0])
            sigma = float(parsed_args[1])
            samples = int(parsed_args[2]) if len(parsed_args) == 3 else None
            if samples is not None and samples <= 0:
                raise ValueError(f"{func_name} requires samples > 0 when specified")
            payload = {
                "type": func_name,
                "mean": mean,
                "sigma": sigma,
            }
            if samples is not None:
                payload["samples"] = samples
            return {"@sweep": payload}

        if func_name == "uniform":
            _require_length({2, 3})
            low = float(parsed_args[0])
            high = float(parsed_args[1])
            if high <= low:
                raise ValueError("uniform requires high > low")
            samples = int(parsed_args[2]) if len(parsed_args) == 3 else None
            if samples is not None and samples <= 0:
                raise ValueError("uniform requires samples > 0 when specified")
            payload = {
                "type": "uniform",
                "low": low,
                "high": high,
            }
            if samples is not None:
                payload["samples"] = samples
            return {"@sweep": payload}

        if func_name == "triangular":
            _require_length({3, 4})
            left = float(parsed_args[0])
            mode = float(parsed_args[1])
            right = float(parsed_args[2])
            if not (left <= mode <= right):
                raise ValueError("triangular requires left <= mode <= right")
            samples = int(parsed_args[3]) if len(parsed_args) == 4 else None
            if samples is not None and samples <= 0:
                raise ValueError("triangular requires samples > 0 when specified")
            payload = {
                "type": "triangular",
                "left": left,
                "mode": mode,
                "right": right,
            }
            if samples is not None:
                payload["samples"] = samples
            return {"@sweep": payload}

        return None

    @staticmethod
    def _coerce_literal(text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            return ""
        lowered = stripped.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if stripped.startswith(("[", "{", """)) and stripped.endswith(("]", "}", """)):
            try:
                return json.loads(stripped)
            except Exception:
                pass
        try:
            if any(ch in stripped for ch in (".", "e", "E")):
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped

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

            if not isinstance(step_widget, QDoubleSpinBox):
                continue

            if isinstance(path_widget, ParameterPathSelector):
                path_value = path_widget.text().strip()
            elif isinstance(path_widget, QLineEdit):
                path_value = path_widget.text().strip()
            else:
                continue
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
        manual_series = None
        if hasattr(self, "observed_series_table"):
            series_rows = self.observed_series_table.get_non_empty_rows()
            if series_rows:
                times: List[float] = []
                values: List[float] = []
                for idx, row in enumerate(series_rows, start=1):
                    if len(row) < 2:
                        raise ValueError(f"Observed series row {idx} must include time and value")
                    time_text, value_text = row[0], row[1]
                    if not time_text or not value_text:
                        raise ValueError(f"Observed series row {idx} must include time and value")
                    try:
                        times.append(float(time_text))
                        values.append(float(value_text))
                    except ValueError as exc:
                        raise ValueError(f"Observed series row {idx} contains non-numeric data: {exc}") from exc
                if times and values:
                    manual_series = {"time_s": times, "values": values}

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

        active_stages: List[str] = []
        for definition in self.stage_definitions:
            controls = self.stage_controls.get(definition["key"])
            if not controls:
                continue
            if not controls["container"].isVisible():
                continue
            checkbox: QCheckBox = controls["checkbox"]
            if checkbox.isChecked():
                active_stages.append(definition["pipeline_stage"])
        if active_stages:
            plan["stages"] = active_stages

        if plan["run_type"] == "sweep":
            plan["sweep_parameters"] = self.get_sweep_parameters() if self.sweep_enabled.isChecked() else {}
        elif plan["run_type"] == "parameter_estimation":
            plan["estimation"] = self.build_parameter_estimation_plan()
        elif plan["run_type"] == "virtual_trial":
            plan["virtual_trial"] = self.build_virtual_trial_plan()
        elif plan["run_type"] == "virtual_bioequivalence":
            plan["virtual_bioequivalence"] = self.build_virtual_bioequivalence_plan()

        bindings = self._collect_workspace_bindings()
        if bindings:
            plan["workspace_bindings"] = bindings

        return plan

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]):
        """Assign workspace manager for saving configurations."""
        self.workspace_manager = workspace_manager
        self.populate_catalog_defaults()
        self.reload_workspace_catalog_entries()

    def build_config(self) -> Dict[str, Any]:
        """Build configuration dictionary from form inputs."""
        admin = self.current_administration()
        stages: List[str] = []
        stage_overrides_payload: Dict[str, Dict[str, Any]] = {}
        for definition in self.stage_definitions:
            controls = self.stage_controls.get(definition["key"])
            if not controls:
                continue
            if not controls["container"].isVisible():
                continue
            checkbox: QCheckBox = controls["checkbox"]
            if not checkbox.isChecked():
                continue

            stages.append(definition["pipeline_stage"])
            mode = self.stage_modes.get(definition["key"], "model")
            if mode == "manual":
                editor_type = controls.get("manual_widget_type", "text")
                manual_payload = None
                if editor_type == "table":
                    table_widget = controls.get("manual_edit")
                    rows = table_widget.get_non_empty_rows() if isinstance(table_widget, SpreadsheetWidget) else []
                    if rows:
                        payload_dict: Dict[str, Any] = {}
                        for idx, row in enumerate(rows, start=1):
                            key = row[0].strip() if row else ""
                            if not key:
                                raise ValueError(f"Manual row {idx} for {definition['label']} is missing a field name")
                            value_text = row[1] if len(row) > 1 else ""
                            payload_dict[key] = self._coerce_literal(value_text)
                        manual_payload = payload_dict
                        controls["manual_loaded_payload"] = None
                    else:
                        manual_payload = controls.get("manual_loaded_payload")
                        if manual_payload is None:
                            raise ValueError(f"Provide manual data for {definition['label']} or switch to model mode.")
                else:
                    manual_widget = controls.get("manual_edit")
                    manual_text = manual_widget.toPlainText().strip() if isinstance(manual_widget, QPlainTextEdit) else ""
                    if not manual_text:
                        raise ValueError(f"Provide manual data for {definition['label']} or switch to model mode.")
                    try:
                        manual_payload = json.loads(manual_text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Manual data for {definition['label']} must be valid JSON: {exc}") from exc
                    if not isinstance(manual_payload, Mapping):
                        raise ValueError(f"Manual data for {definition['label']} must be a JSON object")

                stage_overrides_payload[definition["key"]] = dict(manual_payload)
            else:
                model_value = self._stage_model_value(
                    definition["key"],
                    default=(definition.get("families") or ["default"])[0],
                )
                stage_payload: Dict[str, Any] = {"model": model_value}
                params_payload: Dict[str, Any] = {}
                for name, widget in controls.get("param_widgets", {}).items():
                    if isinstance(widget, QDoubleSpinBox):
                        params_payload[name] = float(widget.value())
                    elif isinstance(widget, QSpinBox):
                        params_payload[name] = int(widget.value())
                    elif isinstance(widget, QComboBox):
                        value = widget.currentData()
                        if value is None:
                            value = widget.currentText()
                        params_payload[name] = value
                    elif isinstance(widget, QLineEdit):
                        text = widget.text().strip()
                        if text:
                            params_payload[name] = text
                if params_payload:
                    stage_payload["params"] = params_payload
                stage_overrides_payload[definition["key"]] = stage_payload

        if "pk" in stages and "pbbm" not in stages:
            stages = [stage for stage in stages if stage != "pk"]
            stage_overrides_payload.pop("pk", None)

        if not stages:
            raise ValueError("Select at least one simulation stage for the chosen administration route.")

        config = {
            "run": {
                "stages": stages,
                "seed": self.seed_spin.value(),
                "threads": 1,
                "enable_numba": False,
                "artifact_dir": "results"
            },
            "deposition": {
                "model": self._stage_model_value("deposition", default="clean_lung"),
                "particle_grid": "medium"
            },
            "pbbm": {
                "model": self._stage_model_value("pbbm", default="numba"),
                "epi_layers": [2, 2, 1, 1]
            },
            "pk": {
                "model": self._stage_model_value("pk", default="pk_3c")
            }
        }

        subject_payload = self._build_entity_payload("subject", required=True, allow_manual=True)
        if not subject_payload:
            raise ValueError("Select subject demographics before building the configuration.")
        config["subject"] = subject_payload

        api_payload = self._build_entity_payload("api", required=True, allow_manual=True)
        if not api_payload:
            raise ValueError("Select an API configuration or provide manual API data.")
        config["api"] = api_payload

        api_overrides: Dict[str, Any] = {}
        if isinstance(api_payload, Mapping):
            api_overrides = api_payload.get("overrides") or {}
        if api_overrides and isinstance(subject_payload, Mapping):
            subject_overrides = subject_payload.get("overrides")
            if isinstance(subject_overrides, Mapping):
                subject_pk_overrides = subject_overrides.get("pk")
                if isinstance(subject_pk_overrides, dict):
                    for param_key in PK_PARAM_PLACEHOLDERS:
                        if param_key in api_overrides and param_key not in subject_pk_overrides:
                            subject_pk_overrides[param_key] = api_overrides[param_key]

        product_payload = self._build_entity_payload("product", required=True, allow_manual=True)
        if not product_payload:
            raise ValueError("Select a product configuration or provide manual product data.")
        config["product"] = product_payload

        if admin == "inhalation":
            maneuver_payload = self._build_entity_payload("maneuver", required=True, allow_manual=True)
            if not maneuver_payload:
                raise ValueError("Select an inhalation maneuver before building the configuration.")
            config["maneuver"] = maneuver_payload
        else:
            maneuver_payload = self._build_entity_payload("maneuver", required=False, allow_manual=True)
            if maneuver_payload:
                config["maneuver"] = maneuver_payload

        gi_payload = self._build_entity_payload("gi", required=False, allow_manual=True)
        if gi_payload:
            config["gi_tract"] = gi_payload

        lung_payload = self._build_entity_payload("lung", required=False, allow_manual=True)
        if lung_payload:
            lung_ref = lung_payload.get("ref")
            if lung_ref:
                config.setdefault("deposition", {})["lung_geometry_ref"] = lung_ref
            overrides = copy.deepcopy(lung_payload.get("overrides") or {})
            if overrides:
                config.setdefault("deposition", {})["lung_geometry_overrides"] = overrides

        for key in ("deposition", "pbbm", "pk"):
            manual_payload = stage_overrides_payload.pop(key, None)
            if manual_payload:
                base_payload = config.get(key, {})
                if base_payload:
                    config[key] = _apply_overrides(base_payload, manual_payload)
                else:
                    config[key] = copy.deepcopy(manual_payload)

        if stage_overrides_payload:
            filtered_stage_overrides = {
                key: value
                for key, value in stage_overrides_payload.items()
                if key not in {"deposition", "pbbm", "pk"}
            }
            if filtered_stage_overrides:
                config.setdefault("run", {})["stage_overrides"] = filtered_stage_overrides

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
        internal_category = category
        if internal_category not in self.workspace_catalog_entries:
            for local_key, catalog_name in self.catalog_category_map.items():
                if catalog_name == category:
                    internal_category = local_key
                    break

        category = internal_category
        if category not in self.workspace_catalog_entries:
            return

        entry_id = data.get("id") or data.get("ref")
        if not entry_id:
            QMessageBox.warning(
                self,
                "Catalog",
                f"Could not determine identifier for {category} selection.",
            )
            return

        entries = self.workspace_catalog_entries.get(category, [])
        if data.get("deleted"):
            entries = [
                entry for entry in entries
                if self._entry_identifier(entry) != entry_id
            ]
            self.workspace_catalog_entries[category] = entries
            self._refresh_entity_combos()
            friendly = self.entity_display.get(category, category)
            display_name = data.get("name") or data.get("ref") or entry_id
            QMessageBox.information(
                self,
                "Removed",
                f"Removed saved {friendly} '{display_name}'.",
            )
            return

        entries = self.workspace_catalog_entries.get(category, [])
        replaced = False
        for idx, entry in enumerate(entries):
            if self._entry_identifier(entry) == entry_id:
                updated = copy.deepcopy(data)
                updated["source"] = "workspace"
                updated.setdefault("display_id", updated.get("id") or updated.get("ref"))
                entries[idx] = updated
                replaced = True
                break
        if not replaced:
            new_entry = copy.deepcopy(data)
            new_entry["source"] = "workspace"
            new_entry.setdefault("display_id", new_entry.get("id") or new_entry.get("ref"))
            entries.append(new_entry)
        self.workspace_catalog_entries[category] = entries

        self._refresh_entity_combos()

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
        self.main_window: Optional['LMPMainWindow'] = None
        self.selected_config_path: Optional[str] = None
        self.run_rows: Dict[str, int] = {}
        self.run_errors: Dict[str, str] = {}
        self.run_logs: Dict[str, List[str]] = {}
        self.current_log_run_id: Optional[str] = None
        self.pending_run_plan: Optional[Dict[str, Any]] = None
        self.last_started_run_ids: List[str] = []
        self.saved_config_info: Dict[str, Dict[str, Any]] = {}
        self._last_config_data: Optional[Dict[str, Any]] = None
        self._last_config_metadata: Optional[Dict[str, Any]] = None
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

        self.apply_to_gui_btn = QPushButton("Apply to Designer")
        self.apply_to_gui_btn.setEnabled(False)
        self.apply_to_gui_btn.clicked.connect(self.apply_selected_to_gui)

        self.parallel_label = QLabel("Parallel slots:")
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 64)
        self.parallel_spin.setValue(4)
        self.parallel_spin.valueChanged.connect(self.on_parallel_changed)

        start_run_btn = QPushButton("Start Run")
        start_run_btn.clicked.connect(self.start_run)

        clear_queue_btn = QPushButton("Clear Completed")
        clear_queue_btn.clicked.connect(self.clear_completed)

        controls_layout.addWidget(load_config_btn)
        controls_layout.addWidget(self.config_combo)
        controls_layout.addWidget(self.apply_to_gui_btn)
        controls_layout.addWidget(self.parallel_label)
        controls_layout.addWidget(self.parallel_spin)
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

        preview_group = QGroupBox("Configuration Preview")
        preview_layout = QVBoxLayout()
        self.config_preview = QPlainTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setPlaceholderText(
            "Select or load a configuration to preview it. Use 'Apply to Designer' when you want to overwrite the Study Designer tabs."
        )
        preview_layout.addWidget(self.config_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

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
                self.process_manager.process_queued.disconnect(self.on_process_queued)
                self.process_manager.queue_positions_updated.disconnect(self.on_queue_positions_updated)
            except TypeError:
                pass

        self.process_manager = process_manager

        if hasattr(self, 'parallel_spin') and isinstance(self.parallel_spin, QSpinBox):
            self.parallel_spin.blockSignals(True)
            if process_manager is None:
                self.parallel_spin.setValue(1)
                self.parallel_spin.setEnabled(False)
            else:
                self.parallel_spin.setValue(process_manager.max_parallel_processes)
                self.parallel_spin.setEnabled(True)
            self.parallel_spin.blockSignals(False)

        if process_manager is None:
            return

        process_manager.process_started.connect(self.on_process_started)
        process_manager.process_progress.connect(self.on_process_progress)
        process_manager.process_error.connect(self.on_process_error)
        process_manager.process_metric.connect(self.on_process_metric)
        process_manager.process_finished.connect(self.on_process_finished)
        process_manager.process_log.connect(self.on_process_log)
        process_manager.process_queued.connect(self.on_process_queued)
        process_manager.queue_positions_updated.connect(self.on_queue_positions_updated)

    def set_main_window(self, main_window: Optional['LMPMainWindow']) -> None:
        self.main_window = main_window

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
            self.set_selected_config(file_path, apply_to_ui=False)

    def _default_run_plan(self) -> Dict[str, Any]:
        return {"run_type": "single", "run_label": None}

    def set_selected_config(
        self,
        config_path: Optional[str],
        run_plan: Optional[Dict[str, Any]] = None,
        *,
        apply_to_ui: bool = True,
    ) -> None:
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
            self._last_config_data = None
            self._update_config_preview(None, None)
            if hasattr(self, 'apply_to_gui_btn'):
                self.apply_to_gui_btn.setEnabled(False)
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
        config_name_hint: Optional[str] = None
        if detected_run_plan is None:
            try:
                with open(path_obj, "r") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    meta = payload.get("metadata")
                    if isinstance(meta, dict):
                        self._last_config_metadata = meta
                        config_name_hint = meta.get("config_name")
                        detected_run_plan = meta.get("run_plan")
                else:
                    self._last_config_metadata = None
            except Exception:
                detected_run_plan = None
                self._last_config_metadata = None
        else:
            self._last_config_metadata = None

        if isinstance(detected_run_plan, Mapping):
            detected_run_plan = dict(detected_run_plan)
            if config_name_hint is None and self._last_config_metadata:
                config_name_hint = self._last_config_metadata.get("config_name")
            if config_name_hint is None:
                config_name_hint = path_obj.name
            detected_run_plan.setdefault("config_name", config_name_hint)
        elif config_name_hint is not None:
            detected_run_plan = {
                "run_type": "single",
                "run_label": None,
                "config_name": config_name_hint,
            }

        if detected_run_plan is not None:
            self.pending_run_plan = detected_run_plan
        elif self.pending_run_plan is None:
            self.pending_run_plan = self._default_run_plan()

        if self.pending_run_plan is not None:
            try:
                self._apply_run_plan_to_ui(self.pending_run_plan)
            except Exception:
                pass

        config_data: Optional[Dict[str, Any]] = None
        try:
            config_data = self._load_config_data(config_path)
        except Exception as exc:
            logger.warning("config preview load failed", config=config_path, error=str(exc))

        if isinstance(config_data, dict):
            self._last_config_data = config_data
        else:
            self._last_config_data = None

        self._update_config_preview(config_path, self._last_config_data)
        if hasattr(self, 'apply_to_gui_btn'):
            self.apply_to_gui_btn.setEnabled(self._last_config_data is not None)

        if apply_to_ui and self.main_window is not None and self._last_config_data is not None:
            plan_for_ui = self.pending_run_plan or self._default_run_plan()
            try:
                self.main_window.apply_config_from_run_queue(self._last_config_data, run_plan=plan_for_ui)
            except Exception as exc:
                logger.warning("apply config to ui failed", config=config_path, error=str(exc))

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

    def _update_config_preview(
        self,
        config_path: Optional[str],
        config_data: Optional[Mapping[str, Any]],
    ) -> None:
        if not hasattr(self, 'config_preview') or self.config_preview is None:
            return

        if not config_data:
            self.config_preview.clear()
            if config_path:
                self.config_preview.setPlainText(f"No preview available for {config_path}.")
            else:
                self.config_preview.setPlainText("Load or select a configuration to preview its contents.")
            return

        preview_text = self._format_config_preview(config_path, config_data)
        self.config_preview.setPlainText(preview_text)

    def _format_config_preview(
        self,
        config_path: Optional[str],
        config_data: Mapping[str, Any],
    ) -> str:
        lines: List[str] = []
        if config_path:
            lines.append(f"File: {config_path}")

        run_section = config_data.get("run")
        if isinstance(run_section, Mapping):
            stages = run_section.get("stages")
            if isinstance(stages, list):
                stages_text = ", ".join(str(stage) for stage in stages)
            else:
                stages_text = str(stages)
            seed = run_section.get("seed")
            lines.append("Run Settings:")
            lines.append(f"  stages: {stages_text}")
            if seed is not None:
                lines.append(f"  seed: {seed}")
            threads = run_section.get("threads")
            if threads is not None:
                lines.append(f"  threads: {threads}")

        for section_name in ("subject", "api", "product", "maneuver"):
            payload = config_data.get(section_name)
            if not isinstance(payload, Mapping):
                continue
            ref_value = payload.get("ref") or "(missing ref)"
            lines.append(f"{section_name.title()}:")
            lines.append(f"  ref: {ref_value}")
            overrides = payload.get("overrides")
            if isinstance(overrides, Mapping) and overrides:
                keys = ", ".join(sorted(str(key) for key in overrides.keys()))
                lines.append(f"  overrides: {keys}")
            variability = payload.get("variability") or payload.get("variability_overrides")
            if isinstance(variability, Mapping) and variability:
                vkeys = ", ".join(sorted(str(key) for key in variability.keys()))
                lines.append(f"  variability: {vkeys}")

        plan = self.pending_run_plan
        if isinstance(plan, Mapping):
            run_type = plan.get("run_type")
            label = plan.get("run_label")
            lines.append("Run Plan:")
            if run_type:
                lines.append(f"  type: {run_type}")
            if label:
                lines.append(f"  label: {label}")
            bindings = plan.get("workspace_bindings")
            if isinstance(bindings, Mapping) and bindings:
                binding_text = ", ".join(f"{key}→{value}" for key, value in bindings.items())
                lines.append(f"  workspace bindings: {binding_text}")

        if not lines:
            return "No preview content available."
        return "\n".join(lines)

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
            self.set_selected_config(path, run_plan=run_plan, apply_to_ui=False)
        else:
            self.set_selected_config(None, apply_to_ui=False)

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
        self._refresh_row_queue_status(run_id)
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
            self._refresh_row_queue_status(run_id)
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
            self._refresh_row_queue_status(run_id)
            self._refresh_row_queue_status(run_id)

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
            self._refresh_row_queue_status(run_id)
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

    def apply_selected_to_gui(self) -> None:
        if self.main_window is None:
            if self.isVisible():
                QMessageBox.warning(self, "Apply Config", "Unable to reach main window context.")
            return

        if self._last_config_data is None:
            if self.isVisible():
                QMessageBox.information(
                    self,
                    "Apply Config",
                    "Select or load a configuration before applying it to the Study Designer.",
                )
            return

        run_plan: Optional[Dict[str, Any]] = None
        if self.selected_config_path:
            info = self.saved_config_info.get(self.selected_config_path)
            if isinstance(info, Mapping):
                run_plan = info.get("run_plan")
        if run_plan is None:
            run_plan = self.pending_run_plan or self._default_run_plan()

        try:
            self.main_window.apply_config_from_run_queue(self._last_config_data, run_plan=run_plan)
            if self.isVisible():
                QMessageBox.information(
                    self,
                    "Apply Config",
                    "Configuration applied to the Study Designer and related tabs.",
                )
            if hasattr(self.main_window, 'tab_widget') and self.main_window.tab_widget is not None:
                self.main_window.tab_widget.setCurrentWidget(self.main_window.study_designer_tab)
        except Exception as exc:
            if self.isVisible():
                QMessageBox.critical(
                    self,
                    "Apply Config",
                    f"Failed to apply configuration: {exc}",
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
            self.set_selected_config(path_value, apply_to_ui=False)

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

    def _queued_position(self, run_id: str) -> Optional[int]:
        if not self.process_manager:
            return None
        pending = getattr(self.process_manager, "pending_requests", [])
        for idx, (request, _) in enumerate(pending):
            if request.run_id == run_id:
                return idx
        return None

    def _refresh_row_queue_status(self, run_id: str) -> None:
        if not self.process_manager or self.process_manager.is_run_active(run_id):
            return
        position = self._queued_position(run_id)
        if position is None:
            return
        message = f"Waiting for slot (position {position + 1})"
        self.update_row_status(run_id, "Queued", message)

    def on_process_queued(self, run_id: str, position: int) -> None:
        message = f"Waiting for slot (position {position + 1})"
        self.update_row_status(run_id, "Queued", message)

    def on_queue_positions_updated(self) -> None:
        if not self.process_manager:
            return
        for request, _ in getattr(self.process_manager, "pending_requests", []):
            self._refresh_row_queue_status(request.run_id)

    def on_parallel_changed(self, value: int) -> None:
        if not self.process_manager:
            return
        try:
            self.process_manager.set_max_parallel_processes(value)
        except ValueError as exc:
            QMessageBox.warning(self, "Parallel Slots", str(exc))
            self.parallel_spin.blockSignals(True)
            self.parallel_spin.setValue(self.process_manager.max_parallel_processes)
            self.parallel_spin.blockSignals(False)

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
        self.current_table_full_df: Optional[pd.DataFrame] = None
        self.current_table_subset_df: Optional[pd.DataFrame] = None
        self.current_table_dataset: Optional[str] = None
        self.study_table_df: Optional[pd.DataFrame] = None
        self.init_ui()

    @staticmethod
    def _split_dataset_name(name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not name:
            return None, None
        if "__" in name:
            prefix, suffix = name.split("__", 1)
            return suffix, prefix
        return name, None

    def _build_group_datasets(self, metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Aggregate child run datasets so group runs expose raw results."""

        if self.workspace_manager is None or not metadata:
            return {}

        run_type = metadata.get("run_type", "")
        group_types = {
            "parameter_estimation_group",
            "virtual_trial_group",
            "virtual_bioequivalence_group",
            "sweep_group",
        }
        if run_type not in group_types:
            return {}

        raw_child_ids = metadata.get("child_run_ids") or []
        child_ids: List[str] = []
        seen_children: Set[str] = set()
        for run_id in raw_child_ids:
            child = str(run_id)
            if not child or child in seen_children:
                continue
            seen_children.add(child)
            child_ids.append(child)
        if not child_ids:
            return {}

        summary_lookup: Dict[str, Dict[str, Any]] = {}
        if run_type == "parameter_estimation_group":
            summary = metadata.get("parameter_estimation_summary") or {}
            for record in summary.get("records", []) or []:
                if not isinstance(record, dict):
                    continue
                child_run = record.get("run_id")
                if child_run:
                    summary_lookup[str(child_run)] = record

        aggregated_frames: Dict[Tuple[str, Optional[str]], List[pd.DataFrame]] = {}

        for child_id in child_ids:
            child_results = self.loaded_results.get(child_id)
            if child_results is None:
                try:
                    child_results = self.workspace_manager.load_run_results(child_id)
                except FileNotFoundError:
                    logger.warning(
                        "Group dataset child results missing for parent %s (child %s)",
                        metadata.get("run_id"),
                        child_id,
                    )
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Group dataset child results error for parent %s (child %s): %s",
                        metadata.get("run_id"),
                        child_id,
                        exc,
                    )
                    continue

            if not isinstance(child_results, dict):
                continue

            child_meta = self.run_metadata.get(child_id)
            if child_meta is None:
                try:
                    child_meta = self.workspace_manager.get_run_info(child_id) or {}
                except Exception:  # pragma: no cover - workspace I/O guard
                    child_meta = {}

            base_columns: Dict[str, Any] = {"source_run_id": child_id}
            if isinstance(child_meta, dict):
                label = child_meta.get("display_label") or child_meta.get("label")
                if label:
                    base_columns["source_label"] = label
                manifest_index = child_meta.get("manifest_index")
                if manifest_index is not None and "manifest_index" not in base_columns:
                    base_columns["manifest_index"] = manifest_index
                manifest_run_id = child_meta.get("manifest_run_id")
                if manifest_run_id and "manifest_run_id" not in base_columns:
                    base_columns["manifest_run_id"] = manifest_run_id
                child_run_type = child_meta.get("run_type")
                if child_run_type:
                    base_columns["source_run_type"] = child_run_type
            else:
                child_meta = {}

            if run_type == "parameter_estimation_group":
                base_columns.update(
                    {
                        "estimation_parameter": child_meta.get("estimation_parameter"),
                        "estimation_direction": child_meta.get("estimation_direction"),
                        "estimation_value": child_meta.get("estimation_value"),
                    }
                )
                overrides = child_meta.get("estimation_overrides") or child_meta.get("parameter_overrides")
                if overrides and "estimation_overrides" not in base_columns:
                    try:
                        base_columns["estimation_overrides"] = json.dumps(overrides, sort_keys=True)
                    except TypeError:
                        base_columns["estimation_overrides"] = str(overrides)
                record = summary_lookup.get(child_id)
                if record:
                    for key in ("objective", "sse", "mae", "rmse", "is_best"):
                        if key in record and record[key] is not None:
                            base_columns[f"summary_{key}"] = record[key]
            elif run_type in {"virtual_trial_group", "virtual_bioequivalence_group"}:
                base_columns.update(
                    {
                        "subject_index": child_meta.get("subject_index"),
                        "subject_name": child_meta.get("subject_name"),
                        "seed": child_meta.get("seed"),
                        "api_name": child_meta.get("api_name"),
                    }
                )
            elif run_type == "sweep_group":
                base_columns["sweep_index"] = child_meta.get("manifest_index")
                overrides = child_meta.get("sweep_overrides") or child_meta.get("parameter_overrides")
                if overrides:
                    try:
                        base_columns["sweep_overrides"] = json.dumps(overrides, sort_keys=True)
                    except TypeError:
                        base_columns["sweep_overrides"] = str(overrides)

            for dataset_name, frame in child_results.items():
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    continue

                base_dataset, dataset_prefix = self._split_dataset_name(dataset_name)
                augmented = frame.copy()

                # Attach run-level metadata columns when they are not already provided.
                for column_name, value in base_columns.items():
                    if value is None or column_name in augmented.columns:
                        continue
                    augmented[column_name] = value

                if dataset_prefix and "dataset_prefix" not in augmented.columns:
                    augmented["dataset_prefix"] = dataset_prefix

                if run_type in {"virtual_trial_group", "virtual_bioequivalence_group"} and dataset_prefix:
                    products = child_meta.get("products")
                    if isinstance(products, dict):
                        product_info = products.get(dataset_prefix)
                        if isinstance(product_info, dict):
                            role = product_info.get("role")
                            if role and "product_role" not in augmented.columns:
                                augmented["product_role"] = role

                key = (base_dataset, dataset_prefix)
                aggregated_frames.setdefault(key, []).append(augmented)

        combined_results: Dict[str, pd.DataFrame] = {}
        for (base_dataset, dataset_prefix), frames in aggregated_frames.items():
            if not frames:
                continue
            try:
                combined = pd.concat(frames, ignore_index=True, sort=False)
            except Exception:  # pragma: no cover - fallback for unusual frames
                combined = pd.concat(frames, ignore_index=True)

            dataset_key = base_dataset if dataset_prefix is None else f"{dataset_prefix}__{base_dataset}"
            if not combined.empty:
                combined_results[dataset_key] = combined

        return combined_results

    def _iter_plot_frames(
        self,
        run_id: str,
        dataset_name: Optional[str],
        base_dataset: Optional[str],
        dataset_prefix: Optional[str],
    ) -> List[Tuple[Tuple[str, Optional[str]], str, pd.DataFrame]]:
        """Yield (key, label, frame) tuples for plotting, expanding grouped data."""

        df = self._dataset_for_run(run_id, dataset_name, base_dataset, dataset_prefix)
        if df is None or df.empty:
            return []

        frames: List[Tuple[Tuple[str, Optional[str]], str, pd.DataFrame]] = []

        if "source_run_id" in df.columns:
            grouped = df.groupby("source_run_id")
            for source_run_id, subset in grouped:
                if subset.empty:
                    continue
                source_id = str(source_run_id) if source_run_id is not None else run_id
                label_hint = subset.iloc[0].get("source_label") if not subset.empty else None
                label = str(label_hint or source_id)
                if dataset_prefix:
                    label = f"{label} · {dataset_prefix}"
                frames.append(((source_id, dataset_prefix), label, subset))
        else:
            label = f"{run_id} · {dataset_prefix}" if dataset_prefix else run_id
            frames.append(((run_id, dataset_prefix), label, df))

        return frames

    def _collect_plot_frames(
        self,
        run_ids: Sequence[str],
        dataset_name: Optional[str],
        base_dataset: Optional[str],
        dataset_prefix: Optional[str],
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Collect unique plot frames across selected runs."""

        collected: List[Tuple[str, pd.DataFrame]] = []
        seen_keys: Set[Tuple[str, Optional[str]]] = set()

        for run_id in run_ids:
            for key, label, frame in self._iter_plot_frames(run_id, dataset_name, base_dataset, dataset_prefix):
                if frame is None or frame.empty:
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                collected.append((label, frame))

        return collected

    def _clear_table_state(self) -> None:
        self.current_table_full_df = None
        self.current_table_subset_df = None
        self.current_table_dataset = None
        self.study_table_df = None
        self._update_table_action_state()

    def _update_table_action_state(self) -> None:
        has_raw = (
            self.view_mode_combo is not None
            and self.view_mode_combo.currentData() == "raw"
            and self.current_table_subset_df is not None
            and not self.current_table_subset_df.empty
        )
        has_study = (
            self.view_mode_combo is not None
            and self.view_mode_combo.currentData() == "study"
            and self.study_table_df is not None
            and not self.study_table_df.empty
        )

        enabled = has_raw or has_study
        if hasattr(self, "copy_table_btn") and isinstance(self.copy_table_btn, QPushButton):
            self.copy_table_btn.setEnabled(enabled)
        if hasattr(self, "export_table_btn") and isinstance(self.export_table_btn, QPushButton):
            self.export_table_btn.setEnabled(enabled)

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

        table_actions_layout = QHBoxLayout()
        table_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.copy_table_btn = QPushButton("Copy Table")
        self.copy_table_btn.clicked.connect(self.copy_visible_table)
        self.export_table_btn = QPushButton("Export CSV…")
        self.export_table_btn.clicked.connect(self.export_visible_table)
        table_actions_layout.addWidget(self.copy_table_btn)
        table_actions_layout.addWidget(self.export_table_btn)
        table_actions_layout.addStretch()
        self.table_actions_widget = QWidget()
        self.table_actions_widget.setLayout(table_actions_layout)
        detail_layout.addWidget(self.table_actions_widget)

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
        self._update_table_action_state()

    # --- Run selection and loading -------------------------------------------------

    def set_workspace_manager(self, workspace_manager: Optional[WorkspaceManager]):
        self.workspace_manager = workspace_manager
        self.loaded_results.clear()
        self.run_metadata.clear()
        self.active_run_id = None
        self._clear_table_state()
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
            self._clear_table_state()
            self.run_list.blockSignals(False)
            return

        runs = self.workspace_manager.list_runs()
        if not runs:
            self.status_label.setText("No runs have been recorded in this workspace.")
            self.status_label.setStyleSheet("color: gray;")
            self._clear_table_state()
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
        self._update_table_action_state()

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
            self.study_table_df = None
            self._update_table_action_state()
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
            self.study_table_df = None
            self._update_table_action_state()
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
        self.study_table_df = None
        if df is None or df.empty:
            self.study_table.setRowCount(0)
            self.study_table.setColumnCount(0)
            self._update_table_action_state()
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
        self.study_table_df = describe_df.copy()

        self.study_table.setRowCount(len(describe_df))
        self.study_table.setColumnCount(len(describe_df.columns))
        self.study_table.setHorizontalHeaderLabels([str(col) for col in describe_df.columns])

        for row_idx in range(len(describe_df)):
            for col_idx, column in enumerate(describe_df.columns):
                value = describe_df.iloc[row_idx, col_idx]
                display_value = "" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (int, float, np.floating)) else str(value)
                self.study_table.setItem(row_idx, col_idx, QTableWidgetItem(display_value))

        self.study_table.resizeColumnsToContents()
        self._update_table_action_state()

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
            self._clear_table_state()
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

        group_results = self._build_group_datasets(info)
        for name, dataframe in group_results.items():
            results.setdefault(name, dataframe)

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

        self.current_table_dataset = dataset_name
        self.current_table_full_df = None
        self.current_table_subset_df = None

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
            self._update_table_action_state()
            return

        full_df = df.reset_index(drop=True)
        self.current_table_full_df = full_df

        max_rows = 200
        subset = full_df.head(max_rows)
        self.current_table_subset_df = subset.copy()

        self.results_table.setRowCount(len(subset))
        self.results_table.setColumnCount(len(subset.columns))
        self.results_table.setHorizontalHeaderLabels([str(col) for col in subset.columns])

        for row_idx in range(len(subset)):
            for col_idx, column in enumerate(subset.columns):
                value = subset.iloc[row_idx, col_idx]
                display_value = "" if pd.isna(value) else str(value)
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(display_value))

        if len(full_df) > max_rows:
            remaining = len(full_df) - max_rows
            self.status_label.setText(
                f"Showing first {max_rows} of {len(full_df)} rows for {dataset_name}. {remaining} additional rows not displayed."
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
                self.status_label.setText(f"Parameter estimation summary across {len(full_df)} runs. {best_text}.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_residuals":
                run_count = full_df["run_id"].nunique() if "run_id" in full_df.columns else 0
                self.status_label.setText(f"Residual samples for {run_count} run(s), {len(full_df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_regional":
                run_count = full_df["run_id"].nunique() if "run_id" in full_df.columns else 0
                self.status_label.setText(f"Regional residuals for {run_count} run(s), {len(full_df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "parameter_estimation_overlay":
                series_count = full_df["series"].nunique() if "series" in full_df.columns else 0
                self.status_label.setText(f"Overlay dataset with {series_count} series and {len(full_df)} points.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_trial_summary":
                self.status_label.setText(f"Virtual trial summary across {len(full_df)} product/metric rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_trial_subjects":
                run_count = full_df["run_id"].nunique() if "run_id" in full_df.columns else 0
                self.status_label.setText(f"Virtual trial subjects: {run_count} run(s), {len(full_df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_summary":
                self.status_label.setText(f"VBE summary metrics across {len(full_df)} entries.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_subjects":
                run_count = full_df["run_id"].nunique() if "run_id" in full_df.columns else 0
                self.status_label.setText(f"VBE subject metrics for {run_count} run(s), {len(full_df)} rows.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            elif dataset_name == "virtual_bioequivalence_product_summary":
                self.status_label.setText(f"Product-level summary across {len(full_df)} entries.")
                self.status_label.setStyleSheet("color: #2e7d32;")
            else:
                self.status_label.setText(f"Displaying {len(full_df)} rows for {dataset_name}.")
                self.status_label.setStyleSheet("color: #2e7d32;")

        self.update_plots()
        self._update_table_action_state()

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
            frames = self._collect_plot_frames(selected_runs, dataset_name, base_dataset, dataset_prefix)
            plotted = False
            for label, df in frames:
                if df is None or df.empty or not {"t", "plasma_conc"}.issubset(df.columns):
                    continue
                df_sorted = df.sort_values("t")
                ax.plot(df_sorted["t"], df_sorted["plasma_conc"], label=label)
                plotted = True

            if not plotted:
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
            frames = self._collect_plot_frames(selected_runs, dataset_name, base_dataset, dataset_prefix)
            for label, df in frames:
                if df is None or value_column not in df.columns:
                    continue
                series = df.groupby("region")[value_column].sum()
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
            frames = self._collect_plot_frames(selected_runs, dataset_name, base_dataset, dataset_prefix)
            df = frames[0][1] if frames else None
            run_label = frames[0][0] if frames else None
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
            frames = self._collect_plot_frames(selected_runs, dataset_name, base_dataset, dataset_prefix)
            regions = set()
            for _, df in frames:
                regions.update(df["region"].dropna().unique())

            if not frames:
                ax.text(0.5, 0.5, "No regional AUC data", ha='center', va='center', transform=ax.transAxes, color='0.5')
                ax.set_axis_off()
            else:
                regions = sorted(regions)
                x = np.arange(len(regions))
                bar_width = 0.8 / max(len(frames), 1)
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
                for column, metric_label in metric_specs:
                    for _, df in frames:
                        if column in df.columns and not df[column].fillna(0).eq(0).all():
                            available_metrics.append((column, metric_label))
                            break

                if not available_metrics:
                    available_metrics = [("auc_elf", "ELF")]

                for idx, (frame_label, df) in enumerate(frames):
                    regional_df = df.set_index("region")
                    offsets = x - 0.4 + bar_width / 2 + idx * bar_width
                    bottom = np.zeros(len(regions))
                    for column, metric_label in available_metrics:
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
                            label=f"{frame_label} - {metric_label}"
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

    def copy_visible_table(self) -> None:
        mode = self.view_mode_combo.currentData() if self.view_mode_combo is not None else "raw"
        if mode == "study":
            df = self.study_table_df
            context_label = "study summary"
        else:
            df = self.current_table_subset_df
            context_label = self.current_table_dataset or "results"

        if df is None or df.empty:
            QMessageBox.information(self, "Copy Table", "No table data is available to copy.")
            return

        export_df = df.copy().fillna("")
        clipboard_text = export_df.to_csv(sep="\t", index=False).rstrip("\n")
        QApplication.clipboard().setText(clipboard_text)

        row_count = len(export_df)
        self.status_label.setText(f"Copied {row_count} row(s) from {context_label} to clipboard.")
        self.status_label.setStyleSheet("color: #2e7d32;")

    def export_visible_table(self) -> None:
        mode = self.view_mode_combo.currentData() if self.view_mode_combo is not None else "raw"
        if mode == "study":
            df = self.study_table_df
            base_name = f"{self.active_run_id or 'study'}_summary"
        else:
            df = self.current_table_full_df
            base_name = self.current_table_dataset or "results"

        if df is None or df.empty:
            QMessageBox.information(self, "Export Table", "No table data is available to export.")
            return

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name) or "results"

        start_dir = Path.cwd()
        if self.workspace_manager is not None:
            try:
                start_dir = self.workspace_manager.workspace_path
            except Exception:
                start_dir = Path.cwd()

        default_path = start_dir / f"{safe_name}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Table",
            str(default_path),
            "CSV files (*.csv);;All files (*)"
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".csv":
            path_obj = path_obj.with_suffix(".csv")

        export_df = df.copy().fillna("")

        try:
            export_df.to_csv(path_obj, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Export Table", f"Failed to export table: {exc}")
            return

        row_count = len(export_df)
        self.status_label.setText(f"Exported {row_count} row(s) to {path_obj}.")
        self.status_label.setStyleSheet("color: #2e7d32;")

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
        self.run_queue_tab.set_main_window(self)
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

    def apply_config_from_run_queue(
        self,
        config_data: Mapping[str, Any],
        run_plan: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Apply a loaded configuration to the relevant tabs."""
        if not isinstance(config_data, Mapping):
            return

        try:
            self.study_designer_tab.apply_config(config_data, run_plan=run_plan)
        except Exception as exc:
            logger.warning("apply study designer failed", error=str(exc))

        try:
            if hasattr(self.population_tab, "apply_config"):
                self.population_tab.apply_config(config_data)
        except Exception as exc:
            logger.warning("apply population tab failed", error=str(exc))

        try:
            if hasattr(self.api_products_tab, "apply_config"):
                self.api_products_tab.apply_config(config_data)
        except Exception as exc:
            logger.warning("apply api/product tab failed", error=str(exc))

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
