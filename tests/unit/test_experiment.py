"""Tests for the experiment record store."""

from pathlib import Path

from deephedging.experiment import ExperimentRecord, append_record, load_records
from deephedging.training.trainer import TrainConfig, TrainResult


def _record(name: str) -> ExperimentRecord:
    return ExperimentRecord.from_run(
        name=name,
        config=TrainConfig(n_iterations=3, batch_paths=8, seed=1),
        setup={"market": "gbm", "sigma": 0.2, "alpha": 0.95},
        result=TrainResult(losses=[3.0, 2.0, 1.0]),
        duration_seconds=1.5,
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "experiments.jsonl"
    original = _record("first")
    append_record(path, original)
    append_record(path, _record("second"))
    loaded = load_records(path)
    assert [record.name for record in loaded] == ["first", "second"]
    assert loaded[0].losses == [3.0, 2.0, 1.0]
    assert loaded[0].train_config["batch_paths"] == 8
    assert loaded[0].setup["sigma"] == 0.2
    assert loaded[0].duration_seconds == 1.5


def test_provenance_identifies_the_build(tmp_path: Path) -> None:
    record = _record("prov")
    for key in ("commit", "torch", "cuda", "device", "platform", "created_utc"):
        assert record.provenance[key]
    assert len(record.provenance["commit"]) in (7, 40) or record.provenance["commit"] == "unknown"


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    assert load_records(tmp_path / "absent.jsonl") == []


def test_file_is_ascii(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    append_record(path, _record("ascii"))
    raw = path.read_bytes()
    assert raw.decode("ascii")
