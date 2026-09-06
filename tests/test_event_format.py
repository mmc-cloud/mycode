import pytest

from mycode.event_format import summarize_event_content, summarize_tool_arguments


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        (
            "read_file",
            {"path": "mycode/runner.py", "start_line": 1, "max_lines": 200},
            "path='mycode/runner.py', start_line=1, max_lines=200",
        ),
        (
            "read_artifact",
            {"artifact_path": "artifacts/tool.txt", "offset_chars": 40, "max_chars": 8000},
            "artifact_path='artifacts/tool.txt', offset_chars=40, max_chars=8000",
        ),
        (
            "grep",
            {"query": "private query", "path_pattern": "mycode/**/*.py", "case_sensitive": False, "max_results": 50},
            "query_chars=13, path_pattern='mycode/**/*.py', case_sensitive=False, max_results=50",
        ),
        (
            "glob",
            {"pattern": "tests/**/*.py", "max_results": 100},
            "pattern='tests/**/*.py', max_results=100",
        ),
        (
            "write_file",
            {"path": "result.txt", "content": "private body"},
            "path='result.txt', content_chars=12",
        ),
        (
            "edit_file",
            {"path": "result.txt", "old_text": "old", "new_text": "new text"},
            "path='result.txt', old_text_chars=3, new_text_chars=8",
        ),
        (
            "run_command",
            {"command": ["uv", "run", "pytest"], "cwd": ".", "timeout_seconds": 30, "max_output_chars": 4000},
            "command='uv run pytest', cwd='.', timeout_seconds=30, max_output_chars=4000",
        ),
        (
            "run_validation",
            {"command": ["python", "-m", "pytest"], "cwd": "tests", "timeout_seconds": 60},
            "command='python -m pytest', cwd='tests', timeout_seconds=60",
        ),
        (
            "delegate_task",
            {"role": "tester", "objective": "verify tests", "context": "current diff", "scope_paths": ["tests"]},
            "role='tester', objective_chars=12, context_chars=12, scope_path_count=1",
        ),
    ],
)
def test_summarize_tool_arguments_preserves_shared_diagnostic_rules(
    name: str,
    arguments: dict[str, object],
    expected: str,
) -> None:
    assert summarize_tool_arguments(name, arguments) == expected


def test_summarize_event_content_collapses_and_truncates() -> None:
    assert summarize_event_content("first\n  second") == "first second"
    assert summarize_event_content("abcdefgh", max_chars=6) == "abc..."


def test_summarize_tool_arguments_hides_large_or_search_text_bodies() -> None:
    hidden = "PRIVATE BODY"
    summaries = (
        summarize_tool_arguments("grep", {"query": hidden}),
        summarize_tool_arguments("write_file", {"content": hidden}),
        summarize_tool_arguments(
            "edit_file", {"old_text": hidden, "new_text": hidden}
        ),
        summarize_tool_arguments(
            "delegate_task", {"objective": hidden, "context": hidden}
        ),
    )

    assert all(hidden not in summary for summary in summaries)
