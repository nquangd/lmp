"""Process Manager for GUI Worker Processes.

Handles spawning, monitoring, and communication with worker processes
that run simulations in the background.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List, Union
from datetime import datetime
import structlog

from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtWidgets import QMessageBox

from workspace_manager import WorkspaceManager

LMP_SRC = Path(__file__).parent.parent / "lmp_pkg" / "src"
if str(LMP_SRC) not in sys.path:
    sys.path.insert(0, str(LMP_SRC))

from lmp_pkg.run_types import RunRequest, RunType

logger = structlog.get_logger()


class ProcessInfo:
    """Information about a running worker process."""

    def __init__(
        self,
        run_id: str,
        process: QProcess,
        config_path: str,
        *,
        run_type: RunType,
        label: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ):
        self.run_id = run_id
        self.process = process
        self.config_path = config_path
        self.started_at = datetime.now()
        self.progress_pct = 0.0
        self.status = "starting"
        self.last_message = ""
        self.metrics: Dict[str, float] = {}
        self.runtime_seconds = 0.0
        self.error_message: Optional[str] = None
        self.log_lines: List[str] = []
        self.run_type = run_type
        self.label = label
        self.parent_run_id = parent_run_id


class ProcessManager(QObject):
    """Manages worker processes for GUI simulations."""

    # Signals
    process_started = Signal(str)  # run_id
    process_finished = Signal(str, str, float)  # run_id, status, runtime_seconds
    process_progress = Signal(str, float, str)  # run_id, progress_pct, message
    process_error = Signal(str, str)  # run_id, error_message
    process_metric = Signal(str, str, float)  # run_id, metric_name, value
    process_log = Signal(str, str)  # run_id, log_line

    def __init__(self, workspace_manager: WorkspaceManager):
        super().__init__()
        self.workspace_manager = workspace_manager
        self.active_processes: Dict[str, ProcessInfo] = {}
        self.max_parallel_processes = 4
        self.completed_logs: Dict[str, List[str]] = {}

        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_processes)
        self.update_timer.start(1000)  # Update every second

    def start_simulation(
        self,
        config_path: str,
        run_type: Union[RunType, str] = RunType.SINGLE,
        label: Optional[str] = None,
        run_id: Optional[str] = None,
        manifest_index: Optional[int] = None,
        total_runs: Optional[int] = None,
        parent_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None,
        task_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a simulation worker process.

        Args:
            config_path: Path to configuration file
            run_type: Requested run orchestration type
            label: Optional human-readable label for the run
            run_id: Optional run ID (generated if not provided)
            manifest_index: Index in manifest for sweep runs
            total_runs: Total runs in sweep
            parent_run_id: Optional parent run grouping identifier
            metadata: Additional metadata to persist with the run

        Returns:
            Run ID of started process

        Raises:
            RuntimeError: If too many processes are running
        """
        request = self._build_run_request(
            config_path=config_path,
            run_type=run_type,
            label=label,
            run_id=run_id,
            manifest_index=manifest_index,
            total_runs=total_runs,
            parent_run_id=parent_run_id,
            metadata=metadata or {},
            parameter_overrides=parameter_overrides,
            task_spec=task_spec,
        )

        return self._start_from_request(request)

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_run_request(
        self,
        *,
        config_path: str,
        run_type: Union[RunType, str],
        label: Optional[str],
        run_id: Optional[str],
        manifest_index: Optional[int],
        total_runs: Optional[int],
        parent_run_id: Optional[str],
        metadata: Dict[str, Any],
        parameter_overrides: Optional[Dict[str, Any]],
        task_spec: Optional[Dict[str, Any]],
    ) -> RunRequest:
        if not isinstance(run_type, RunType):
            run_type = RunType(run_type)

        request = RunRequest(
            config_path=config_path,
            run_type=run_type,
            run_id=run_id,
            parent_run_id=parent_run_id,
            manifest_index=manifest_index,
            total_runs=total_runs,
            metadata=metadata,
            parameter_overrides=parameter_overrides,
            task_spec=task_spec,
        ).with_label(label)

        return request

    def _start_from_request(self, request: RunRequest) -> str:
        if len(self.active_processes) >= self.max_parallel_processes:
            raise RuntimeError(f"Maximum {self.max_parallel_processes} processes already running")

        label_slug = request.label

        if request.run_id is None:
            request.run_id = self.workspace_manager.generate_run_id(
                prefix=request.run_type.prefix,
                label=label_slug,
            )

        run_id = request.run_id

        logger.info(
            "Starting simulation process",
            run_id=run_id,
            config=request.config_path,
            run_type=request.run_type.value,
            label=request.display_label or request.label,
        )

        run_dir = self.workspace_manager.create_run_directory(
            run_id,
            run_type=request.run_type.value,
            label=request.display_label or request.label,
            parent_run_id=request.parent_run_id,
            request_metadata=request.as_metadata(),
        )

        process = QProcess()
        process.setProgram("python3")

        gui_worker_path = Path(__file__).parent / "gui_worker.py"
        args = [
            str(gui_worker_path),
            "run",
            "--run-id",
            run_id,
            "--config",
            request.config_path,
            "--workspace",
            str(self.workspace_manager.workspace_path),
            "--run-type",
            request.run_type.value,
        ]

        if request.display_label:
            args.extend(["--run-label", request.display_label])
        elif request.label:
            args.extend(["--run-label", request.label])

        if request.parent_run_id:
            args.extend(["--parent-run-id", request.parent_run_id])

        if request.manifest_index is not None:
            args.extend(["--manifest-index", str(request.manifest_index)])
        if request.total_runs is not None:
            args.extend(["--total-runs", str(request.total_runs)])
        if request.parameter_overrides:
            try:
                overrides_payload = json.dumps(request.parameter_overrides)
            except TypeError:
                overrides_payload = json.dumps(request.parameter_overrides, default=str)
            args.extend(["--parameter-overrides", overrides_payload])

        if request.task_spec:
            try:
                task_payload = json.dumps(request.task_spec)
            except TypeError:
                task_payload = json.dumps(request.task_spec, default=str)
            args.extend(["--task-spec", task_payload])

        process.setArguments(args)
        process.setWorkingDirectory(str(run_dir))

        process.readyReadStandardOutput.connect(lambda: self._handle_stdout(run_id))
        process.readyReadStandardError.connect(lambda: self._handle_stderr(run_id))
        process.finished.connect(
            lambda exit_code, exit_status: self._handle_finished(run_id, exit_code, exit_status)
        )

        process.start()

        if not process.waitForStarted(3000):
            raise RuntimeError(f"Failed to start worker process: {process.errorString()}")

        process_info = ProcessInfo(
            run_id,
            process,
            request.config_path,
            run_type=request.run_type,
            label=request.display_label or request.label,
            parent_run_id=request.parent_run_id,
        )
        self.active_processes[run_id] = process_info

        self.workspace_manager.update_run_status(
            run_id,
            "running",
            started_at=datetime.now().isoformat(),
            run_type=request.run_type.value,
            label=request.label,
            display_label=request.display_label or request.label,
            parent_run_id=request.parent_run_id,
            parameter_overrides=request.parameter_overrides,
        )

        self.process_started.emit(run_id)
        logger.info("Simulation process started", run_id=run_id, pid=process.processId())

        return run_id

    def _append_log(self, run_id: str, message: str):
        message = message.rstrip()
        if not message:
            return

        if run_id in self.active_processes:
            info = self.active_processes[run_id]
            info.log_lines.append(message)
            if len(info.log_lines) > 2000:
                info.log_lines = info.log_lines[-2000:]
        else:
            log_list = self.completed_logs.setdefault(run_id, [])
            log_list.append(message)
            if len(log_list) > 2000:
                self.completed_logs[run_id] = log_list[-2000:]

        self.process_log.emit(run_id, message)

    def get_process_logs(self, run_id: str) -> List[str]:
        if run_id in self.active_processes:
            return list(self.active_processes[run_id].log_lines)
        return list(self.completed_logs.get(run_id, []))

    def cancel_simulation(self, run_id: str, force: bool = False) -> bool:
        """Cancel a running simulation.

        Args:
            run_id: Run ID to cancel
            force: Whether to force kill (SIGKILL) vs graceful termination

        Returns:
            True if cancellation was initiated
        """
        if run_id not in self.active_processes:
            logger.warning("Cannot cancel unknown process", run_id=run_id)
            return False

        process_info = self.active_processes[run_id]
        process = process_info.process

        if process.state() == QProcess.ProcessState.NotRunning:
            logger.warning("Process not running", run_id=run_id)
            return False

        logger.info("Cancelling simulation", run_id=run_id, force=force)

        if force:
            process.kill()  # SIGKILL
        else:
            process.terminate()  # SIGTERM

        # Update status
        process_info.status = "cancelling"
        self.workspace_manager.update_run_status(run_id, "cancelled")

        return True

    def get_process_info(self, run_id: str) -> Optional[ProcessInfo]:
        """Get information about a process."""
        return self.active_processes.get(run_id)

    def list_active_processes(self) -> List[ProcessInfo]:
        """List all active processes."""
        return list(self.active_processes.values())

    def _handle_stdout(self, run_id: str):
        """Handle stdout from worker process (JSONL progress)."""
        if run_id not in self.active_processes:
            return

        process_info = self.active_processes[run_id]
        process = process_info.process

        # Read all available data
        data = process.readAllStandardOutput().data().decode('utf-8')

        # Process each line as JSONL
        for line in data.strip().split('\n'):
            raw = line.strip()
            if not raw:
                continue

            try:
                event = json.loads(raw)
                self._handle_progress_event(run_id, event)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in worker output", run_id=run_id, line=raw, error=str(e))
                self._append_log(run_id, f"[worker] {raw}")

    def _handle_stderr(self, run_id: str):
        """Handle stderr from worker process."""
        if run_id not in self.active_processes:
            return

        process_info = self.active_processes[run_id]
        process = process_info.process

        # Read stderr data
        data = process.readAllStandardError().data().decode('utf-8')
        if data.strip():
            logger.warning("Worker stderr", run_id=run_id, stderr=data)
            process_info.error_message = data
            for line in data.splitlines():
                self._append_log(run_id, f"[stderr] {line}")

    def _handle_progress_event(self, run_id: str, event: Dict[str, Any]):
        """Handle a progress event from worker."""
        if run_id not in self.active_processes:
            return

        process_info = self.active_processes[run_id]
        event_type = event.get("event")
        log_message: Optional[str] = None

        if event_type == "started":
            process_info.status = "running"
            process_info.last_message = "Simulation started"
            logger.info("Worker started", run_id=run_id)
            log_message = "Worker started"

        elif event_type == "progress":
            pct = event.get("pct", 0.0)
            message = event.get("message", "")
            process_info.progress_pct = pct
            process_info.last_message = message
            self.process_progress.emit(run_id, pct, message)
            logger.debug("Worker progress", run_id=run_id, pct=pct, message=message)
            log_message = f"Progress {pct:.1f}%" if not message else f"Progress {pct:.1f}% - {message}"

        elif event_type == "metric":
            metric_name = event.get("name", "")
            value = event.get("value", 0.0)
            process_info.metrics[metric_name] = value
            self.process_metric.emit(run_id, metric_name, value)
            logger.info("Worker metric", run_id=run_id, metric=metric_name, value=value)
            log_message = f"Metric {metric_name}: {value}"

        elif event_type == "checkpoint":
            path = event.get("path", "")
            stage = event.get("stage", "")
            logger.info("Worker checkpoint", run_id=run_id, path=path, stage=stage)
            log_message = f"Checkpoint ({stage}): {path}"

        elif event_type == "completed":
            runtime = event.get("runtime", 0.0)
            process_info.status = "completed"
            process_info.runtime_seconds = runtime
            process_info.last_message = f"Completed in {runtime:.2f}s"
            logger.info("Worker completed", run_id=run_id, runtime=runtime)
            log_message = f"Completed in {runtime:.2f}s"

        elif event_type == "error":
            error_msg = event.get("message", "Unknown error")
            details = event.get("details", "")
            process_info.status = "failed"
            process_info.error_message = f"{error_msg}\n{details}" if details else error_msg
            self.process_error.emit(run_id, process_info.error_message)
            logger.error("Worker error", run_id=run_id, error=error_msg, details=details)
            log_message = f"Error: {error_msg}"
            if details:
                log_message = f"{log_message} ({details})"

        if log_message:
            self._append_log(run_id, log_message)

    def _handle_finished(self, run_id: str, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle worker process finished."""
        if run_id not in self.active_processes:
            return

        process_info = self.active_processes[run_id]
        success = (exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit)

        if success and process_info.status != "completed":
            process_info.status = "completed"

        if not success:
            if process_info.status == "cancelling":
                process_info.status = "cancelled"
            else:
                process_info.status = "failed"
                if process_info.error_message is None:
                    process_info.error_message = f"Process exited with code {exit_code}"

        # Calculate runtime
        runtime = (datetime.now() - process_info.started_at).total_seconds()
        process_info.runtime_seconds = runtime

        # Update workspace
        self.workspace_manager.update_run_status(
            run_id,
            process_info.status,
            runtime_seconds=runtime,
            exit_code=exit_code,
            completed_at=datetime.now().isoformat()
        )

        self._append_log(run_id, f"Process exited (code {exit_code})")
        self.completed_logs[run_id] = list(process_info.log_lines)

        # Emit signal with final status and runtime
        self.process_finished.emit(run_id, process_info.status, runtime)

        logger.info("Worker process finished",
                   run_id=run_id,
                   success=success,
                   exit_code=exit_code,
                   runtime=runtime)

        # Remove from active processes
        del self.active_processes[run_id]

    def update_processes(self):
        """Periodic update of process information."""
        current_time = datetime.now()

        for run_id, process_info in list(self.active_processes.items()):
            # Update runtime
            process_info.runtime_seconds = (current_time - process_info.started_at).total_seconds()

            # Check if process is still alive
            if process_info.process.state() == QProcess.ProcessState.NotRunning:
                # Process ended but we haven't handled it yet
                if process_info.status not in ["completed", "failed", "cancelled"]:
                    logger.warning("Process ended unexpectedly", run_id=run_id)
                    process_info.status = "failed"
                    process_info.error_message = "Process ended unexpectedly"

    def cleanup_finished_processes(self):
        """Clean up any processes that have finished but are still tracked."""
        for run_id in list(self.active_processes.keys()):
            process_info = self.active_processes[run_id]
            if process_info.process.state() == QProcess.ProcessState.NotRunning:
                if process_info.status in ["completed", "failed", "cancelled"]:
                    logger.info("Cleaning up finished process", run_id=run_id)
                    del self.active_processes[run_id]

    def set_max_parallel_processes(self, max_processes: int):
        """Set maximum number of parallel processes."""
        if max_processes < 1:
            raise ValueError("Max processes must be at least 1")

        self.max_parallel_processes = max_processes
        logger.info("Max parallel processes set", max_processes=max_processes)
