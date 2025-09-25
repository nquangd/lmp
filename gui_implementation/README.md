# LMP GUI Implementation

This directory contains a GUI implementation for the LMP (Lung Modeling Platform) based on the design plan in `../gui/gui_plan.txt`.

## Architecture

The GUI follows the plan's modular design with PySide6 as the Qt framework:

### Core Components

1. **gui_worker.py** - Worker process CLI that runs simulations in separate processes
2. **workspace_manager.py** - Manages workspace directory structure and run metadata
3. **process_manager.py** - Handles spawning and monitoring worker processes
4. **main_window.py** - Main GUI application with tabbed interface

### GUI Tabs

1. **Home/Workspace** - Workspace selection and project management
2. **Catalog & Libraries** - Browse available subjects, APIs, products, and maneuvers
3. **Study Designer** - Create simulation configurations with form-based UI
4. **Run Queue** - Start simulations and monitor progress
5. **Results Viewer** - View and analyze simulation results with exports and plots
6. **Logs & Diagnostics** - Live log viewing and debugging (placeholder)

## Features Implemented

✅ **Process Model**
- Worker processes launched via QProcess
- JSONL progress protocol for real-time updates
- Graceful cancellation support

✅ **Workspace Management**
- Deterministic directory structure
- Run metadata tracking
- Config saving and loading

✅ **Configuration UI**
- Form-based study designer
- Catalog integration (placeholder)
- Parameter sweep support

✅ **Manual Overrides**
- Manual deposition editors expose a parameter tree picker for override keys
- Stage controls enable/disable manual widgets based on administration visibility

✅ **Results Viewer**
- Dataset filtering with PBPK selectors
- Copy-to-clipboard and CSV export for the currently visible table

### Sweep Expressions

The sweep grid accepts either explicit value lists or helper expressions. Examples:

- `range(0, 10, 2)` – 0, 2, 4, 6, 8
- `linspace(0, 1, 5)` – five evenly spaced samples between 0 and 1
- `normal(0, 1, 50)` – request 50 samples from a normal distribution (engine support required)
- `uniform(5, 25, 10)` – ten draws from a uniform distribution

Use a single expression or explicit values per parameter row; leave the remaining columns blank when providing an expression.

✅ **Run Management**
- Queue-based execution
- Progress monitoring
- Status tracking

## Running the GUI

### Prerequisites

Make sure PySide6 is installed:
```bash
pip install PySide6
```

### Launch Options

1. **Using the launcher script** (recommended):
```bash
python run_launcher.py
```

2. **Direct execution**:
```bash
python main_window.py
```

3. **Testing the worker**:
```bash
python gui_worker.py run --run-id test_001 --config ../lmp_pkg/examples/basic.toml --workspace ./test_workspace
```

## Directory Structure

When a workspace is created, it follows this structure:
```
workspace/
├── workspace.json          # Workspace metadata
├── configs/                # Saved configurations
│   └── config_*.json
├── runs/                   # Individual run directories
│   └── run_*/
│       ├── metadata.json   # Run metadata
│       ├── config.json     # Exact config used
│       ├── logs/           # Log files
│       ├── results/        # Result parquet files
│       └── artifacts/      # Other outputs
└── logs/                   # Global logs
```

## Integration with LMP Package

The GUI integrates with the existing `lmp_pkg` through:

- **app_api.py** - Uses the main API functions for simulation execution
- **Configuration loading** - Compatible with existing TOML configs
- **Catalog system** - Accesses built-in subjects, APIs, products
- **Result processing** - Uses `convert_results_to_dataframes()`

## Progress Protocol

Worker processes communicate via JSONL events:

```json
{"event": "started", "run_id": "run_001", "config": "/path/to/config.json"}
{"event": "progress", "pct": 25.0, "message": "Running deposition stage"}
{"event": "metric", "name": "AUC", "value": 123.45, "units": "ng*h/mL"}
{"event": "checkpoint", "path": "/path/to/results.parquet", "stage": "pk"}
{"event": "completed", "run_id": "run_001", "runtime": 45.2}
{"event": "error", "message": "Validation failed", "details": "..."}
```

## Limitations & TODOs

### Current Limitations
- Results viewer relies on Matplotlib fallback; interactive overlays and comparisons are pending
- Catalog integration uses placeholder data
- No pyqtgraph integration yet for high-frequency plots
- No SLURM integration yet
- Limited error handling in some areas

### Next Steps
1. Integrate real catalog data via `app_api.list_catalog_entries()`
2. Replace Matplotlib fallback with pyqtgraph for interactive plotting and overlays
3. Add comparative run analytics (overlays, delta metrics)
4. Implement live log tailing
5. Improve inline validation and error messaging
6. Package as standalone executable

## Example Usage

1. **Start the GUI**: `python run_launcher.py`
2. **Create workspace**: Go to Home tab, click "Create New"
3. **Design study**: Go to Study Designer, configure parameters
4. **Save config**: Click "Save Config" to store in workspace
5. **Run simulation**: Go to Run Queue, click "Start Run"
6. **Monitor progress**: Watch progress bars and logs

## Technical Details

### Thread Safety
- Worker processes run in separate OS processes (not threads)
- GUI updates happen on main thread via Qt signals
- No shared memory between GUI and workers

### Error Handling
- Worker errors reported via JSONL protocol
- Process crashes detected and reported
- Graceful degradation on missing components

### Performance
- Up to 4 parallel worker processes (configurable)
- Lazy loading of results data
- Minimal memory footprint for long-running GUI

## Testing

To test individual components:

```bash
# Test workspace manager
python -c "from workspace_manager import WorkspaceManager; wm = WorkspaceManager('./test_ws'); print(wm.list_runs())"

# Test worker process
python gui_worker.py validate ../lmp_pkg/examples/basic.toml

# Test GUI without backend
python main_window.py  # Most features work as placeholders
```

This implementation provides a solid foundation for the LMP GUI and can be extended with additional features as needed.
