"""Snapshot tests."""

import json
import os

import pytest

from crusoe_nemotron_harness.snap import RunSnapshot


def test_first_run_writes_fixture(tmp_path) -> None:
    snap = RunSnapshot()
    snap.record("prompt", {"text": "hi"})
    snap.record("response", {"text": "hello"})
    fixture = tmp_path / "first.json"
    result = snap.compare(fixture)
    assert result.ok
    assert fixture.exists()
    loaded = json.loads(fixture.read_text())
    assert len(loaded) == 2
    assert loaded[0]["kind"] == "prompt"


def test_matching_run_passes_clean(tmp_path) -> None:
    snap1 = RunSnapshot()
    snap1.record("prompt", {"text": "hi"})
    fixture = tmp_path / "match.json"
    snap1.compare(fixture)

    snap2 = RunSnapshot()
    snap2.record("prompt", {"text": "hi"})
    result = snap2.compare(fixture)
    assert result.ok
    assert result.diff == ""


def test_mismatch_returns_diff(tmp_path) -> None:
    snap1 = RunSnapshot()
    snap1.record("prompt", {"text": "hi"})
    fixture = tmp_path / "diff.json"
    snap1.compare(fixture)

    snap2 = RunSnapshot()
    snap2.record("prompt", {"text": "different"})
    result = snap2.compare(fixture)
    assert not result.ok
    assert "different" in result.diff
    assert "hi" in result.diff


def test_update_kwarg_overwrites_fixture(tmp_path) -> None:
    snap1 = RunSnapshot()
    snap1.record("prompt", {"text": "old"})
    fixture = tmp_path / "upd.json"
    snap1.compare(fixture)

    snap2 = RunSnapshot()
    snap2.record("prompt", {"text": "new"})
    result = snap2.compare(fixture, update=True)
    assert result.ok
    loaded = json.loads(fixture.read_text())
    assert loaded[0]["payload"]["text"] == "new"


def test_env_var_triggers_update(tmp_path, monkeypatch) -> None:
    snap1 = RunSnapshot()
    snap1.record("prompt", {"text": "old"})
    fixture = tmp_path / "env.json"
    snap1.compare(fixture)

    monkeypatch.setenv("CRUSOE_HARNESS_UPDATE", "1")
    snap2 = RunSnapshot()
    snap2.record("prompt", {"text": "new"})
    result = snap2.compare(fixture)
    assert result.ok
    loaded = json.loads(fixture.read_text())
    assert loaded[0]["payload"]["text"] == "new"
