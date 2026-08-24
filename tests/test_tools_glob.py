from pathlib import Path

import pytest

from mycode.tools import GlobArgs, GlobTool, Workspace


def test_glob_finds_files_by_pattern(tmp_path: Path) -> None:
    write_file(tmp_path / "mycode" / "cli.py")
    write_file(tmp_path / "mycode" / "tools" / "base.py")
    write_file(tmp_path / "tests" / "test_cli.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "mycode/**/*.py"})

    assert result.ok is True
    assert result.content == "mycode/cli.py\nmycode/tools/base.py"
    assert result.metadata == {
        "pattern": "mycode/**/*.py",
        "result_count": 2,
        "total_matches": 2,
        "filtered_count": 0,
        "filtered_reasons": {},
        "max_results": 100,
        "truncated": False,
    }


def test_glob_returns_stable_sorted_results(tmp_path: Path) -> None:
    write_file(tmp_path / "b.py")
    write_file(tmp_path / "a.py")
    write_file(tmp_path / "c.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "*.py"})

    assert result.ok is True
    assert result.content == "a.py\nb.py\nc.py"


def test_glob_returns_only_files(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    write_file(tmp_path / "pkg" / "__init__.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "pkg*"})

    assert result.ok is True
    assert result.content == "No files matched."
    assert result.metadata["result_count"] == 0
    assert result.metadata["total_matches"] == 0


def test_glob_skips_ignored_directories(tmp_path: Path) -> None:
    write_file(tmp_path / "src" / "app.py")
    write_file(tmp_path / ".venv" / "Lib" / "site-packages" / "pkg.py")
    write_file(tmp_path / "node_modules" / "pkg" / "index.py")
    write_file(tmp_path / "src" / "__pycache__" / "app.pyc")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "**/*"})

    assert result.ok is True
    assert result.content == "src/app.py"
    assert result.metadata["result_count"] == 1
    assert result.metadata["total_matches"] == 1
    assert result.metadata["filtered_count"] == 3
    assert result.metadata["filtered_reasons"] == {"ignored": 3}


def test_glob_skips_sensitive_files(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py")
    write_file(tmp_path / ".env")
    write_file(tmp_path / ".env.local")
    write_file(tmp_path / "private.pem")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "**/*"})

    assert result.ok is True
    assert result.content == "app.py"
    assert result.metadata["result_count"] == 1
    assert result.metadata["total_matches"] == 1
    assert result.metadata["filtered_count"] == 3
    assert result.metadata["filtered_reasons"] == {"ignored": 3}


def test_glob_direct_run_still_filters_explicit_sensitive_file(tmp_path: Path) -> None:
    write_file(tmp_path / ".env")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": ".env"})

    assert result.ok is True
    assert result.content == "No files matched. Some matches were filtered by workspace relevance or safety rules."
    assert result.metadata["filtered_reasons"] == {"ignored": 1}


def test_glob_skips_low_relevance_directories_in_broad_scan(
    tmp_path: Path,
) -> None:
    write_file(tmp_path / "mycode" / "app.py")
    write_file(tmp_path / "examples" / "demo.py")
    write_file(tmp_path / "vendor" / "package.py")
    write_file(tmp_path / "docs" / "reference" / "external_project.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "**/*.py"})

    assert result.ok is True
    assert result.content == "mycode/app.py"
    assert result.metadata["result_count"] == 1
    assert result.metadata["total_matches"] == 1
    assert result.metadata["filtered_count"] == 3
    assert result.metadata["filtered_reasons"] == {"low_relevance": 3}


def test_glob_can_access_low_relevance_file_by_explicit_path(
    tmp_path: Path,
) -> None:
    write_file(tmp_path / "examples" / "demo.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "examples/demo.py"})

    assert result.ok is True
    assert result.content == "examples/demo.py"
    assert result.metadata["filtered_count"] == 0


def test_glob_limits_results_and_marks_truncated(tmp_path: Path) -> None:
    write_file(tmp_path / "a.py")
    write_file(tmp_path / "b.py")
    write_file(tmp_path / "c.py")
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "*.py", "max_results": 2})

    assert result.ok is True
    assert result.content == "a.py\nb.py"
    assert result.metadata["result_count"] == 2
    assert result.metadata["total_matches"] == 3
    assert result.metadata["truncated"] is True


def test_glob_returns_success_for_no_matches(tmp_path: Path) -> None:
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "*.py"})

    assert result.ok is True
    assert result.content == "No files matched."
    assert result.metadata["result_count"] == 0
    assert result.metadata["total_matches"] == 0
    assert result.metadata["truncated"] is False


def test_glob_rejects_absolute_pattern(tmp_path: Path) -> None:
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": str(tmp_path / "*.py")})

    assert result.ok is False
    assert result.error == f"Glob pattern must be relative: {tmp_path / '*.py'}"


def test_glob_rejects_parent_traversal_pattern(tmp_path: Path) -> None:
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "../*.py"})

    assert result.ok is False
    assert result.error == "Glob pattern must not contain '..': ../*.py"


def test_glob_rejects_invalid_arguments(tmp_path: Path) -> None:
    tool = GlobTool(workspace=Workspace(tmp_path))

    result = tool.run({"pattern": "*.py", "max_results": 0})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_glob_schema_describes_arguments() -> None:
    schema = GlobTool(workspace=Workspace(Path.cwd())).get_schema()

    assert schema["name"] == "glob"
    assert schema["parameters"]["properties"]["pattern"]["type"] == "string"
    assert schema["parameters"]["properties"]["max_results"]["default"] == 100


def test_glob_args_defaults() -> None:
    args = GlobArgs(pattern="*.py")

    assert args.pattern == "*.py"
    assert args.max_results == 100


def test_glob_skips_symlink_that_resolves_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    write_file(outside)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    symlink_path = workspace_root / "outside.py"
    try:
        symlink_path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is not available: {error}")

    tool = GlobTool(workspace=Workspace(workspace_root))

    result = tool.run({"pattern": "*.py"})

    assert result.ok is True
    assert result.content == "No files matched."


def write_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
