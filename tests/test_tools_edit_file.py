from pathlib import Path

from mycode.permissions import ConfirmationRequest, ConfirmationResult
from mycode.tools import EditFileArgs, EditFileTool, ToolRegistry, Workspace


def test_edit_file_schema_describes_arguments(tmp_path: Path) -> None:
    schema = EditFileTool(workspace=Workspace(tmp_path)).get_schema()

    assert schema["name"] == "edit_file"
    assert schema["parameters"]["properties"]["path"]["type"] == "string"
    assert schema["parameters"]["properties"]["old_text"]["type"] == "string"
    assert schema["parameters"]["properties"]["new_text"] == {"type": "string"}


def test_edit_file_args_accept_empty_new_text() -> None:
    args = EditFileArgs(path="notes.txt", old_text="remove me", new_text="")

    assert args.path == "notes.txt"
    assert args.old_text == "remove me"
    assert args.new_text == ""


def test_edit_file_rejects_empty_old_text_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Invalid tool arguments"


def test_edit_file_permission_request_targets_path(tmp_path: Path) -> None:
    tool = EditFileTool(workspace=Workspace(tmp_path))

    request = tool.build_permission_request(
        EditFileArgs(path="notes.txt", old_text="old", new_text="new")
    )

    assert request.tool_name == "edit_file"
    assert request.capability == "write"
    assert request.action == "edit_file"
    assert request.target == "notes.txt"
    assert request.arguments == {
        "path": "notes.txt",
        "old_text": "old",
        "new_text": "new",
    }


def test_edit_file_direct_run_requires_registry(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("old", encoding="utf-8")
    tool = EditFileTool(workspace=Workspace(tmp_path))

    result = tool.run({"path": "notes.txt", "old_text": "old", "new_text": "new"})

    assert result.ok is False
    assert result.error == "edit_file must be run through ToolRegistry.run_tool()."
    assert file_path.read_text(encoding="utf-8") == "old"


def test_edit_file_requires_confirmation_by_default(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello old", encoding="utf-8")
    registry = ToolRegistry.from_tools([EditFileTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_status"] == "ask"
    assert result.metadata["permission_reason"] == "requires_confirmation"
    assert result.metadata["operation"] == "replace"
    assert result.metadata["path_scope"] == "inside_workspace"
    assert file_path.read_text(encoding="utf-8") == "hello old"


def test_edit_file_replaces_unique_text_after_confirmation(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("alpha old omega", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is True
    assert result.content == (
        "Edited file: notes.txt\n\n"
        "Changed context (lines 1-1):\n"
        "1 | alpha new omega"
    )
    assert file_path.read_text(encoding="utf-8") == "alpha new omega"
    assert result.metadata["path"] == "notes.txt"
    assert result.metadata["operation"] == "replace"
    assert result.metadata["match_count"] == 1
    assert result.metadata["encoding"] == "utf-8"
    assert result.metadata["old_text_chars"] == 3
    assert result.metadata["new_text_chars"] == 3
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["line_start"] == 1
    assert result.metadata["line_end"] == 1
    assert result.metadata["snippet_truncated"] is False
    assert result.metadata["snippet"] == "1 | alpha new omega"


def test_edit_file_matches_lf_old_text_against_crlf_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_bytes(b"before\r\nalpha\r\nold\r\nomega\r\nafter\r\n")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {
            "path": "notes.txt",
            "old_text": "alpha\nold\nomega",
            "new_text": "alpha\nnew\nomega",
        },
    )

    assert result.ok is True
    assert file_path.read_bytes() == b"before\r\nalpha\r\nnew\r\nomega\r\nafter\r\n"
    assert result.metadata["match_count"] == 1
    assert result.metadata["exact_match_count"] == 0
    assert result.metadata["newline_normalized_match_count"] == 1
    assert result.metadata["newline_normalized_match"] is True


def test_edit_file_returns_changed_context_snippet_not_entire_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "\n".join(
            [
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5 old",
                "line 6",
                "line 7",
                "line 8",
                "line 9",
            ]
        ),
        encoding="utf-8",
    )
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "line 5 old", "new_text": "line 5 new"},
    )

    assert result.ok is True
    assert "5 | line 5 new" in result.content
    assert "1 | line 1" not in result.content
    assert result.metadata["line_start"] == 2
    assert result.metadata["line_end"] == 8
    assert result.metadata["snippet_truncated"] is True
    assert result.metadata["snippet"] == "\n".join(
        [
            "2 | line 2",
            "3 | line 3",
            "4 | line 4",
            "5 | line 5 new",
            "6 | line 6",
            "7 | line 7",
            "8 | line 8",
        ]
    )


def test_edit_file_can_delete_unique_text(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("keep remove keep", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": " remove", "new_text": ""},
    )

    assert result.ok is True
    assert file_path.read_text(encoding="utf-8") == "keep keep"


def test_edit_file_fails_when_old_text_is_missing(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "missing", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Target text not found."
    assert result.metadata["match_count"] == 0
    assert result.metadata["confirmation_status"] == "approved"
    assert result.metadata["path_scope"] == "inside_workspace"
    assert file_path.read_text(encoding="utf-8") == "hello"


def test_edit_file_fails_when_old_text_is_not_unique(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("old and old", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Target text is not unique."
    assert result.metadata["match_count"] == 2
    assert file_path.read_text(encoding="utf-8") == "old and old"


def test_edit_file_rejects_non_unique_newline_normalized_match(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_bytes(b"alpha\r\nold\r\nalpha\r\nold\r\n")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "notes.txt", "old_text": "alpha\nold", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Target text is not unique."
    assert result.metadata["match_count"] == 2
    assert result.metadata["exact_match_count"] == 0
    assert result.metadata["newline_normalized_match_count"] == 2
    assert file_path.read_bytes() == b"alpha\r\nold\r\nalpha\r\nold\r\n"


def test_edit_file_fails_for_missing_file(tmp_path: Path) -> None:
    confirmer = ApprovingConfirmer()
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=confirmer,
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "missing.txt", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "File not found: missing.txt"
    assert result.metadata["permission_status"] == "deny"
    assert result.metadata["target_exists"] is False
    assert confirmer.requests == []


def test_edit_file_denies_directory_target(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "src", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Path is not a file: src"


def test_edit_file_rejects_file_with_nul_byte(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"hello\x00old")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "binary.bin", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "File is not a supported text file: binary.bin"
    assert result.metadata["reason"] == "nul_byte"


def test_edit_file_rejects_unsupported_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\xff\xff\xff")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "binary.bin", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "File is not a supported text file: binary.bin"
    assert result.metadata["supported_encodings"] == ["utf-8", "gbk"]


def test_edit_file_preserves_gbk_encoding(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_bytes("你好，旧内容\n".encode("gbk"))
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "note.txt", "old_text": "旧内容", "new_text": "新内容"},
    )

    assert result.ok is True
    assert file_path.read_bytes().decode("gbk") == "你好，新内容\n"
    assert result.metadata["encoding"] == "gbk"


def test_edit_file_requires_confirmation_for_sensitive_path(tmp_path: Path) -> None:
    file_path = tmp_path / ".env"
    file_path.write_text("VALUE=old\n", encoding="utf-8")
    registry = ToolRegistry.from_tools([EditFileTool(Workspace(tmp_path))])

    result = registry.run_tool(
        "edit_file",
        {"path": ".env", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is False
    assert result.error == "Confirmation is not available."
    assert result.metadata["permission_reason"] == "sensitive_path"
    assert result.metadata["path_scope"] == "sensitive_path"
    assert file_path.read_text(encoding="utf-8") == "VALUE=old\n"


def test_edit_file_edits_sensitive_path_after_confirmation(tmp_path: Path) -> None:
    file_path = tmp_path / ".env"
    file_path.write_text("VALUE=old\n", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(tmp_path))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": ".env", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is True
    assert file_path.read_text(encoding="utf-8") == "VALUE=new\n"
    assert result.metadata["path_scope"] == "sensitive_path"
    assert "VALUE=new" not in result.content
    assert "snippet" not in result.metadata
    assert result.metadata["snippet_suppressed"] is True
    assert result.metadata["line_start"] == 1
    assert result.metadata["line_end"] == 1


def test_edit_file_edits_outside_workspace_after_confirmation(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("old", encoding="utf-8")
    registry = ToolRegistry.from_tools(
        [EditFileTool(Workspace(workspace_root))],
        confirmer=ApprovingConfirmer(),
    )

    result = registry.run_tool(
        "edit_file",
        {"path": "../outside.txt", "old_text": "old", "new_text": "new"},
    )

    assert result.ok is True
    assert outside_path.read_text(encoding="utf-8") == "new"
    assert result.metadata["path"] == outside_path.as_posix()
    assert result.metadata["path_scope"] == "outside_workspace"


class ApprovingConfirmer:
    def __init__(self) -> None:
        self.requests: list[ConfirmationRequest] = []

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        self.requests.append(request)
        return ConfirmationResult.approved()
