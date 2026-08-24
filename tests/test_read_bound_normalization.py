from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from mycode.artifacts import (
    MAX_ARTIFACT_READ_CHARS,
    ReadArtifactArgs,
    ReadArtifactTool,
    ToolResultArtifactStore,
)
from mycode.context_budget import parse_tool_result_content
from mycode.tools import (
    GlobArgs,
    GlobTool,
    GrepArgs,
    GrepTool,
    ReadFileArgs,
    ReadFileTool,
    Workspace,
)
from mycode.tools.glob import MAX_RESULTS_LIMIT as GLOB_MAX_RESULTS_LIMIT
from mycode.tools.grep import MAX_RESULTS_LIMIT as GREP_MAX_RESULTS_LIMIT
from mycode.tools.read_file import MAX_LINES_LIMIT


READ_BOUND_CASES = [
    (ReadFileArgs, {"path": "sample.txt"}, "max_lines", MAX_LINES_LIMIT),
    (GrepArgs, {"query": "needle"}, "max_results", GREP_MAX_RESULTS_LIMIT),
    (GlobArgs, {"pattern": "*.py"}, "max_results", GLOB_MAX_RESULTS_LIMIT),
    (
        ReadArtifactArgs,
        {"artifact_path": "C:/artifact.txt"},
        "max_chars",
        MAX_ARTIFACT_READ_CHARS,
    ),
]


@pytest.mark.parametrize(
    ("args_model", "required", "field_name", "limit"),
    READ_BOUND_CASES,
)
@pytest.mark.parametrize(
    ("offset", "expected_offset"),
    [(0, 0), (-1, -1), (1, 0), (5000, 0)],
)
def test_read_bound_accepts_safe_integers_and_clamps_only_overflow(
    args_model: type[BaseModel],
    required: dict[str, object],
    field_name: str,
    limit: int,
    offset: int,
    expected_offset: int,
) -> None:
    requested = limit + offset
    expected = limit + expected_offset

    args = args_model.model_validate({**required, field_name: requested})

    assert getattr(args, field_name) == expected


@pytest.mark.parametrize(
    ("args_model", "required", "field_name", "limit"),
    READ_BOUND_CASES,
)
@pytest.mark.parametrize("invalid", [0, -1, True, "100", 1.0, None, [100]])
def test_read_bound_rejects_non_positive_or_non_integer_values(
    args_model: type[BaseModel],
    required: dict[str, object],
    field_name: str,
    limit: int,
    invalid: object,
) -> None:
    del limit
    with pytest.raises(ValidationError):
        args_model.model_validate({**required, field_name: invalid})


@pytest.mark.parametrize(
    ("args_model", "required", "field_name", "limit"),
    READ_BOUND_CASES,
)
def test_read_bound_still_rejects_unknown_fields(
    args_model: type[BaseModel],
    required: dict[str, object],
    field_name: str,
    limit: int,
) -> None:
    with pytest.raises(ValidationError):
        args_model.model_validate(
            {
                **required,
                field_name: limit + 1,
                "unknown_alias": limit,
            }
        )


def test_grep_bound_normalization_does_not_infer_missing_query() -> None:
    with pytest.raises(ValidationError):
        GrepArgs.model_validate({"max_results": GREP_MAX_RESULTS_LIMIT + 1})


def test_read_file_uses_normalized_bound_in_permission_and_result(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("one\ntwo\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))
    args = tool.parse_arguments(
        {"path": "sample.txt", "max_lines": MAX_LINES_LIMIT + 500}
    )

    request = tool.build_permission_request(args)
    result = tool.run(
        {"path": "sample.txt", "max_lines": MAX_LINES_LIMIT + 500}
    )

    assert request.arguments["max_lines"] == MAX_LINES_LIMIT
    assert result.ok is True
    assert result.metadata["max_lines"] == MAX_LINES_LIMIT


def test_grep_uses_normalized_bound_in_permission_and_result(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    tool = GrepTool(Workspace(tmp_path))
    raw_args = {
        "query": "needle",
        "path_pattern": "*.py",
        "max_results": GREP_MAX_RESULTS_LIMIT + 500,
    }
    args = tool.parse_arguments(raw_args)

    request = tool.build_permission_request(args)
    result = tool.run(raw_args)

    assert request.arguments["max_results"] == GREP_MAX_RESULTS_LIMIT
    assert result.ok is True
    assert result.metadata["max_results"] == GREP_MAX_RESULTS_LIMIT


def test_glob_uses_normalized_bound_in_permission_and_result(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("", encoding="utf-8")
    tool = GlobTool(Workspace(tmp_path))
    raw_args = {
        "pattern": "*.py",
        "max_results": GLOB_MAX_RESULTS_LIMIT + 500,
    }
    args = tool.parse_arguments(raw_args)

    request = tool.build_permission_request(args)
    result = tool.run(raw_args)

    assert request.arguments["max_results"] == GLOB_MAX_RESULTS_LIMIT
    assert result.ok is True
    assert result.metadata["max_results"] == GLOB_MAX_RESULTS_LIMIT


def test_read_artifact_clamps_real_slice_and_reports_canonical_bound(
    tmp_path: Path,
) -> None:
    original = "x" * (MAX_ARTIFACT_READ_CHARS + 100)
    store = ToolResultArtifactStore(
        root=tmp_path / "artifacts",
        threshold_chars=1,
    )
    reference = store.externalize(
        tool_name="read_file",
        tool_call_id="call-large",
        content=original,
    )
    artifact_path = str(
        parse_tool_result_content(reference).metadata["artifact_path"]
    )
    tool = ReadArtifactTool(store.root)
    raw_args = {
        "artifact_path": artifact_path,
        "max_chars": MAX_ARTIFACT_READ_CHARS + 3000,
    }
    args = tool.parse_arguments(raw_args)

    request = tool.build_permission_request(args)
    result = tool.run(raw_args)

    assert request.arguments["max_chars"] == MAX_ARTIFACT_READ_CHARS
    assert result.ok is True
    assert result.content == original[:MAX_ARTIFACT_READ_CHARS]
    assert result.metadata["max_chars"] == MAX_ARTIFACT_READ_CHARS
    assert result.metadata["truncated"] is True


@pytest.mark.parametrize(
    ("tool", "field_name", "limit"),
    [
        (ReadFileTool(Workspace(Path.cwd())), "max_lines", MAX_LINES_LIMIT),
        (GrepTool(Workspace(Path.cwd())), "max_results", GREP_MAX_RESULTS_LIMIT),
        (GlobTool(Workspace(Path.cwd())), "max_results", GLOB_MAX_RESULTS_LIMIT),
        (
            ReadArtifactTool(Path.cwd() / ".artifacts"),
            "max_chars",
            MAX_ARTIFACT_READ_CHARS,
        ),
    ],
)
def test_read_bound_schema_keeps_existing_hard_maximum(
    tool: object,
    field_name: str,
    limit: int,
) -> None:
    schema = tool.get_schema()

    assert schema["parameters"]["properties"][field_name]["maximum"] == limit
