from pathlib import Path

from mycode.tools import ReadFileArgs, ReadFileTool, Workspace


def test_read_file_reads_text_file_with_line_numbers(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("print('hello')\nprint('world')\n", encoding="utf-8")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "sample.py"})

    assert result.ok is True
    assert result.content == (
        "File: sample.py\n"
        "Lines: 1-2 / 2\n"
        "Has more: no\n\n"
        "1 | print('hello')\n"
        "2 | print('world')"
    )
    assert result.metadata == {
        "path": "sample.py",
        "encoding": "utf-8",
        "start_line": 1,
        "end_line": 2,
        "total_lines": 2,
        "has_more": False,
        "next_start_line": None,
    }


def test_read_file_reads_requested_line_window(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "sample.txt", "start_line": 2, "max_lines": 2})

    assert result.ok is True
    assert result.content == (
        "File: sample.txt\n"
        "Lines: 2-3 / 4\n"
        "Has more: yes\n"
        "Next start line: 4\n\n"
        "2 | two\n"
        "3 | three"
    )
    assert result.metadata["start_line"] == 2
    assert result.metadata["end_line"] == 3
    assert result.metadata["total_lines"] == 4
    assert result.metadata["has_more"] is True
    assert result.metadata["next_start_line"] == 4


def test_read_file_returns_empty_content_when_start_line_is_past_eof(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("one\ntwo\n", encoding="utf-8")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "sample.txt", "start_line": 10, "max_lines": 2})

    assert result.ok is True
    assert result.content == (
        "File: sample.txt\n"
        "Lines: none (requested start: 10) / 2\n"
        "Has more: no"
    )
    assert result.metadata["start_line"] == 10
    assert result.metadata["end_line"] == 9
    assert result.metadata["total_lines"] == 2
    assert result.metadata["has_more"] is False
    assert result.metadata["next_start_line"] is None


def test_read_file_supports_gbk_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_bytes("你好\n".encode("gbk"))
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "note.txt"})

    assert result.ok is True
    assert result.content.endswith("\n\n1 | 你好")
    assert result.metadata["encoding"] == "gbk"


def test_read_file_preserves_utf8_chinese_text(tmp_path: Path) -> None:
    file_path = tmp_path / "README.md"
    file_path.write_text("这是一个本地运行的项目\n", encoding="utf-8")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "README.md"})

    assert result.ok is True
    assert result.content.endswith("\n\n1 | 这是一个本地运行的项目")
    assert result.metadata["encoding"] == "utf-8"


def test_read_file_rejects_path_outside_workspace(tmp_path: Path) -> None:
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "../outside.txt"})

    assert result.ok is False
    assert result.error == "Path is outside workspace: ../outside.txt"
    assert result.metadata == {"path": "../outside.txt"}


def test_read_file_returns_failure_for_missing_file(tmp_path: Path) -> None:
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "missing.txt"})

    assert result.ok is False
    assert result.error == "File not found: missing.txt"
    assert result.metadata == {"path": "missing.txt"}


def test_read_file_returns_failure_for_directory(tmp_path: Path) -> None:
    directory = tmp_path / "src"
    directory.mkdir()
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "src"})

    assert result.ok is False
    assert result.error == "Path is not a file: src"
    assert result.metadata == {"path": "src"}


def test_read_file_rejects_sensitive_file(tmp_path: Path) -> None:
    file_path = tmp_path / ".env"
    file_path.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": ".env"})

    assert result.ok is False
    assert result.error == "Refusing to read sensitive file: .env"
    assert result.metadata == {"path": ".env", "reason": "sensitive_file"}


def test_read_file_permission_request_targets_path(tmp_path: Path) -> None:
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    request = tool.build_permission_request(ReadFileArgs(path="README.md"))

    assert request.tool_name == "read_file"
    assert request.capability == "read"
    assert request.action == "read_file"
    assert request.target == "README.md"
    assert request.arguments == {
        "path": "README.md",
        "start_line": 1,
        "max_lines": 200,
    }


def test_read_file_rejects_file_with_nul_byte(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"hello\x00world")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "binary.bin"})

    assert result.ok is False
    assert result.error == "File is not a supported text file: binary.bin"
    assert result.metadata == {"path": "binary.bin", "reason": "nul_byte"}


def test_read_file_rejects_unsupported_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\xff\xff\xff")
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "binary.bin"})

    assert result.ok is False
    assert result.error == "File is not a supported text file: binary.bin"
    assert result.metadata == {
        "path": "binary.bin",
        "supported_encodings": ["utf-8", "gbk"],
    }


def test_read_file_rejects_invalid_arguments(tmp_path: Path) -> None:
    tool = ReadFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "sample.txt", "start_line": 0})

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_read_file_schema_describes_arguments() -> None:
    schema = ReadFileTool(workspace=Workspace(Path.cwd())).get_schema()

    assert schema["name"] == "read_file"
    assert schema["parameters"]["properties"]["path"] == {"type": "string"}
    assert schema["parameters"]["properties"]["start_line"]["default"] == 1
    assert schema["parameters"]["properties"]["max_lines"]["default"] == 200


def test_read_file_args_defaults() -> None:
    args = ReadFileArgs(path="sample.txt")

    assert args.path == "sample.txt"
    assert args.start_line == 1
    assert args.max_lines == 200
