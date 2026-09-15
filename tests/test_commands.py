import pytest

from mycode.presentation.commands import (
    COMMAND_SPECS,
    CommandParseError,
    ParsedCommand,
    parse_slash_command,
)


def test_command_specs_define_all_canonical_commands() -> None:
    assert [spec.name for spec in COMMAND_SPECS] == [
        "help",
        "new",
        "sessions",
        "resume",
        "context",
        "compact",
        "exit",
    ]
    assert all(spec.usage and spec.description for spec in COMMAND_SPECS)
    specs_by_name = {spec.name: spec for spec in COMMAND_SPECS}
    assert specs_by_name["sessions"].description == "List sessions in the current project."
    assert specs_by_name["exit"].description == "Exit the current interactive runtime."
    assert specs_by_name["exit"].aliases == ("quit",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", ParsedCommand("help")),
        ("/new", ParsedCommand("new")),
        ("/sessions", ParsedCommand("sessions")),
        ("/resume abc123", ParsedCommand("resume", ("abc123",))),
        ("/context", ParsedCommand("context")),
        ("/compact", ParsedCommand("compact")),
        ("/exit", ParsedCommand("exit")),
    ],
)
def test_parse_slash_command_returns_structured_command(
    text: str, expected: ParsedCommand
) -> None:
    assert parse_slash_command(text) == expected


def test_parse_slash_command_normalizes_alias_case_and_outer_whitespace() -> None:
    assert parse_slash_command("  /QuIt  ") == ParsedCommand("exit")
    assert parse_slash_command(" /EXIT ") == ParsedCommand("exit")


@pytest.mark.parametrize("text", ["hello", "  hello /exit  ", "", "   "])
def test_normal_text_returns_none(text: str) -> None:
    assert parse_slash_command(text) is None


def test_unknown_slash_command_returns_none() -> None:
    assert parse_slash_command("/unknown") is None


@pytest.mark.parametrize("text", ["/resume", "/resume a b", "/exit abc"])
def test_known_command_argument_errors_raise_usage_error(text: str) -> None:
    with pytest.raises(CommandParseError) as exc_info:
        parse_slash_command(text)

    assert exc_info.value.usage in {"/resume <session_id>", "/exit"}
