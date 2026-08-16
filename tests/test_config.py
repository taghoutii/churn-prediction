from pathlib import Path

from churn.config import PROJECT_ROOT, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_FILES


def test_project_paths_are_path_objects():
    for path in [PROJECT_ROOT, RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        assert isinstance(path, Path)


def test_raw_files_defined_under_raw_dir():
    assert RAW_FILES
    for path in RAW_FILES.values():
        assert isinstance(path, Path)
        assert path.parent == RAW_DIR
