from pathlib import Path

from mycode.tools.ignore import is_ignored_path, is_low_relevance_path, is_sensitive_path


def test_is_ignored_path_returns_false_for_normal_file(tmp_path: Path) -> None:
    path = tmp_path / "mycode" / "tools" / "glob.py"

    assert is_ignored_path(path, tmp_path) is False


def test_is_ignored_path_returns_true_for_excluded_directory(tmp_path: Path) -> None:
    path = tmp_path / ".venv" / "Lib" / "site-packages" / "pkg.py"

    assert is_ignored_path(path, tmp_path) is True


def test_is_ignored_path_returns_true_for_nested_excluded_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mycode" / "__pycache__" / "glob.cpython-311.pyc"

    assert is_ignored_path(path, tmp_path) is True


def test_is_ignored_path_returns_true_for_path_outside_root(tmp_path: Path) -> None:
    path = tmp_path.parent / "outside.py"

    assert is_ignored_path(path, tmp_path) is True


def test_is_ignored_path_uses_custom_excluded_directory_names(tmp_path: Path) -> None:
    path = tmp_path / "vendor" / "pkg.py"

    assert is_ignored_path(path, tmp_path, frozenset({"vendor"})) is True


def test_is_low_relevance_path_returns_true_for_reference_material(
    tmp_path: Path,
) -> None:
    path = tmp_path / "docs" / "reference" / "external_project" / "app.py"

    assert is_low_relevance_path(path, tmp_path) is True
    assert is_ignored_path(path, tmp_path) is False


def test_is_low_relevance_path_returns_false_for_main_project_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mycode" / "tools" / "glob.py"

    assert is_low_relevance_path(path, tmp_path) is False


def test_is_sensitive_path_returns_true_for_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"

    assert is_sensitive_path(path, tmp_path) is True
    assert is_ignored_path(path, tmp_path) is True


def test_is_sensitive_path_returns_true_for_env_variant(tmp_path: Path) -> None:
    path = tmp_path / ".env.local"

    assert is_sensitive_path(path, tmp_path) is True
    assert is_ignored_path(path, tmp_path) is True


def test_is_sensitive_path_returns_false_for_env_template(tmp_path: Path) -> None:
    path = tmp_path / ".env.example"

    assert is_sensitive_path(path, tmp_path) is False
    assert is_ignored_path(path, tmp_path) is False


def test_is_sensitive_path_returns_true_for_key_file(tmp_path: Path) -> None:
    path = tmp_path / "private.pem"

    assert is_sensitive_path(path, tmp_path) is True
    assert is_ignored_path(path, tmp_path) is True
