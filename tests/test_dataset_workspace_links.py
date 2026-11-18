from pathlib import Path

import utils
def test_linked_nested_dataset_supports_recursive_discovery_and_relative_paths(tmp_path):
    source = tmp_path / "source"
    nested = source / "experiment" / "measurements"
    nested.mkdir(parents=True)
    csv = nested / "cooling.csv"
    csv.write_text("time,value\n1,2\n", encoding="utf-8")
    destination = tmp_path / "input"
    destination.mkdir()
    utils.copytree(source, destination, use_symlinks=True)
    found = list(destination.rglob("cooling.csv"))
    assert len(found) == 1
    assert found[0].resolve().relative_to(destination.resolve()) == Path("experiment/measurements/cooling.csv")
    assert found[0].read_bytes() == csv.read_bytes()
    assert found[0].samefile(csv)
    assert not (destination / "experiment").is_symlink()


def test_dataset_link_falls_back_to_copy_when_filesystem_rejects_links(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n", encoding="utf-8")
    destination = tmp_path / "input"
    destination.mkdir()

    def unavailable(*args):
        raise OSError("Cross-device link")

    monkeypatch.setattr(utils.os, "link", unavailable)
    utils.copytree(source, destination)
    target = destination / source.name
    assert target.read_bytes() == source.read_bytes()
    assert not target.is_symlink()
    assert not target.samefile(source)


def test_copy_mode_keeps_source_independent(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/a.csv").write_text("original", encoding="utf-8")
    destination = tmp_path / "input"
    destination.mkdir()
    utils.copytree(source, destination, use_symlinks=False)
    (destination / "nested/a.csv").write_text("changed", encoding="utf-8")
    assert (source / "nested/a.csv").read_text(encoding="utf-8") == "original"


def test_migration_reads_old_directory_symlink_targets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "values.csv").write_text("a,b\n", encoding="utf-8")
    (source / "nested").symlink_to(data, target_is_directory=True)
    destination = tmp_path / "input"
    destination.mkdir()
    utils.copytree(source, destination)
    assert [file.name for file in destination.rglob("*.csv")] == ["values.csv"]
    assert not (destination / "nested").is_symlink()
