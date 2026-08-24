from pathlib import Path

from mycode.tools import GrepArgs, GrepTool, Workspace


def test_grep_finds_query_in_utf8_files(tmp_path: Path) -> None:
    write_file(tmp_path / "mycode" / "tools" / "read_file.py", "class ReadFileTool:\n")
    write_file(tmp_path / "tests" / "test_read_file.py", "ReadFileTool()\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "ReadFileTool", "path_pattern": "**/*.py"})

    assert result.ok is True
    assert result.content == (
        "mycode/tools/read_file.py:1 | class ReadFileTool:\n"
        "tests/test_read_file.py:1 | ReadFileTool()"
    )
    assert result.metadata == {
        "query": "ReadFileTool",
        "path_pattern": "**/*.py",
        "case_sensitive": False,
        "result_count": 2,
        "max_results": 100,
        "truncated": False,
        "searched_files": 2,
        "skipped_files": 0,
        "filtered_count": 0,
        "filtered_reasons": {},
    }


def test_grep_supports_gbk_files(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_bytes("中文内容\n".encode("gbk"))
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "中文", "path_pattern": "*.txt"})

    assert result.ok is True
    assert result.content == "note.txt:1 | 中文内容"
    assert result.metadata["searched_files"] == 1


def test_grep_is_case_insensitive_by_default(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", "ToolResult\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "toolresult", "path_pattern": "*.py"})

    assert result.ok is True
    assert result.content == "app.py:1 | ToolResult"


def test_grep_can_be_case_sensitive(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", "ToolResult\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run(
        {"query": "toolresult", "path_pattern": "*.py", "case_sensitive": True}
    )

    assert result.ok is True
    assert result.content == "No matches found."
    assert result.metadata["result_count"] == 0


def test_grep_respects_path_pattern(tmp_path: Path) -> None:
    write_file(tmp_path / "mycode" / "app.py", "needle\n")
    write_file(tmp_path / "docs" / "note.md", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "docs/**/*.md"})

    assert result.ok is True
    assert result.content == "docs/note.md:1 | needle"
    assert result.metadata["searched_files"] == 1


def test_grep_skips_ignored_directories(tmp_path: Path) -> None:
    write_file(tmp_path / "src" / "app.py", "needle\n")
    write_file(tmp_path / ".venv" / "Lib" / "pkg.py", "needle\n")
    write_file(tmp_path / "node_modules" / "pkg" / "index.py", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "**/*.py"})

    assert result.ok is True
    assert result.content == "src/app.py:1 | needle"
    assert result.metadata["searched_files"] == 1
    assert result.metadata["filtered_count"] == 2
    assert result.metadata["filtered_reasons"] == {"ignored": 2}


def test_grep_skips_sensitive_files(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", "needle\n")
    write_file(tmp_path / ".env", "OPENAI_API_KEY=needle\n")
    write_file(tmp_path / ".env.local", "TOKEN=needle\n")
    write_file(tmp_path / "private.key", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "**/*"})

    assert result.ok is True
    assert result.content == "app.py:1 | needle"
    assert result.metadata["searched_files"] == 1
    assert result.metadata["filtered_count"] == 3
    assert result.metadata["filtered_reasons"] == {"ignored": 3}


def test_grep_direct_run_still_filters_explicit_sensitive_file(tmp_path: Path) -> None:
    write_file(tmp_path / ".env", "OPENAI_API_KEY=needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": ".env"})

    assert result.ok is True
    assert result.content == "No matches found."
    assert result.metadata["filtered_reasons"] == {"ignored": 1}


def test_grep_skips_low_relevance_directories_in_broad_scan(
    tmp_path: Path,
) -> None:
    write_file(tmp_path / "mycode" / "app.py", "needle\n")
    write_file(tmp_path / "examples" / "demo.py", "needle\n")
    write_file(tmp_path / "vendor" / "package.py", "needle\n")
    write_file(tmp_path / "docs" / "reference" / "external_project.py", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "**/*.py"})

    assert result.ok is True
    assert result.content == "mycode/app.py:1 | needle"
    assert result.metadata["searched_files"] == 1
    assert result.metadata["filtered_count"] == 3
    assert result.metadata["filtered_reasons"] == {"low_relevance": 3}


def test_grep_can_search_low_relevance_file_by_explicit_path(
    tmp_path: Path,
) -> None:
    write_file(tmp_path / "examples" / "demo.py", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "examples/demo.py"})

    assert result.ok is True
    assert result.content == "examples/demo.py:1 | needle"
    assert result.metadata["filtered_count"] == 0


def test_grep_skips_binary_and_unsupported_text_files(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", "needle\n")
    (tmp_path / "binary.bin").write_bytes(b"needle\x00binary")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xff\xff")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "*"})

    assert result.ok is True
    assert result.content == "app.py:1 | needle"
    assert result.metadata["searched_files"] == 1
    assert result.metadata["skipped_files"] == 2


def test_grep_limits_results_and_marks_truncated(tmp_path: Path) -> None:
    write_file(tmp_path / "a.py", "needle\nneedle\n")
    write_file(tmp_path / "b.py", "needle\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "*.py", "max_results": 2})

    assert result.ok is True
    assert result.content == "a.py:1 | needle\na.py:2 | needle"
    assert result.metadata["result_count"] == 2
    assert result.metadata["truncated"] is True


def test_grep_returns_success_for_no_matches(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", "hello\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "*.py"})

    assert result.ok is True
    assert result.content == "No matches found."
    assert result.metadata["result_count"] == 0
    assert result.metadata["truncated"] is False


def test_grep_rejects_absolute_path_pattern(tmp_path: Path) -> None:
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": str(tmp_path / "*.py")})

    assert result.ok is False
    assert result.error == f"Path pattern must be relative: {tmp_path / '*.py'}"


def test_grep_rejects_parent_traversal_path_pattern(tmp_path: Path) -> None:
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "../*.py"})

    assert result.ok is False
    assert result.error == "Path pattern must not contain '..': ../*.py"


def test_grep_rejects_invalid_arguments(tmp_path: Path) -> None:
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "", "path_pattern": "*.py"})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_grep_schema_describes_arguments() -> None:
    schema = GrepTool(workspace=Workspace(Path.cwd())).get_schema()

    assert schema["name"] == "grep"
    assert schema["parameters"]["properties"]["query"]["type"] == "string"
    assert schema["parameters"]["properties"]["path_pattern"]["default"] == "**/*"
    assert schema["parameters"]["properties"]["case_sensitive"]["default"] is False
    assert schema["parameters"]["properties"]["max_results"]["default"] == 100


def test_grep_args_defaults() -> None:
    args = GrepArgs(query="needle")

    assert args.query == "needle"
    assert args.path_pattern == "**/*"
    assert args.case_sensitive is False
    assert args.max_results == 100


def test_grep_truncates_long_matching_lines(tmp_path: Path) -> None:
    write_file(tmp_path / "app.py", f"needle {'a' * 300}\n")
    tool = GrepTool(workspace=Workspace(tmp_path))

    result = tool.run({"query": "needle", "path_pattern": "*.py"})

    assert result.ok is True
    assert result.content.startswith("app.py:1 | needle ")
    assert result.content.endswith("...")
    assert len(result.content.split(" | ", 1)[1]) == 240


def write_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
