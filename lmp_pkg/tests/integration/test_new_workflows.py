"""Integration tests for newly added workflows."""

from __future__ import annotations

import pytest

from lmp_pkg.engine.workflow import get_workflow


def test_pe_pk_iv_workflow_load():
    """Test that the pe_pk_iv workflow can be loaded."""
    workflow = get_workflow("pe_pk_iv")
    assert workflow.name == "pe_pk_iv"
    assert workflow.stages == ["pk"]


def test_pe_gi_oral_workflow_load():
    """Test that the pe_gi_oral workflow can be loaded."""
    workflow = get_workflow("pe_gi_oral")
    assert workflow.name == "pe_gi_oral"
    assert workflow.stages == ["gi", "pk"]


def test_pe_full_pipeline_workflow_load():
    """Test that the pe_full_pipeline workflow can be loaded."""
    workflow = get_workflow("pe_full_pipeline")
    assert workflow.name == "pe_full_pipeline"
    assert workflow.stages == ["cfd", "deposition", "pbbm", "gi", "pk"]


def test_sa_full_pipeline_workflow_load():
    """Test that the sa_full_pipeline workflow can be loaded."""
    workflow = get_workflow("sa_full_pipeline")
    assert workflow.name == "sa_full_pipeline"
    assert workflow.stages == ["cfd", "deposition", "pbbm", "gi", "pk"]
