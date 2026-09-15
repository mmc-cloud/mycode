"""Definitions and parsing for MyCode's shared slash commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """Metadata describing one canonical slash command."""

    name: str
    aliases: tuple[str, ...]
    usage: str
    description: str


@dataclass(frozen=True)
class ParsedCommand:
    """A recognized command with its whitespace-separated arguments."""

    name: str
    args: tuple[str, ...] = ()


class CommandParseError(ValueError):
    """Raised when a registered command receives invalid arguments."""

    def __init__(self, command: str, usage: str) -> None:
        self.command = command
        self.usage = usage
        super().__init__(f"Invalid arguments for /{command}; usage: {usage}")


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("help", (), "/help", "Show available slash commands."),
    CommandSpec("new", (), "/new", "Start a new session."),
    CommandSpec(
        "sessions",
        (),
        "/sessions",
        "List sessions in the current project.",
    ),
    CommandSpec("resume", (), "/resume <session_id>", "Resume a session."),
    CommandSpec("context", (), "/context", "Show current context information."),
    CommandSpec("compact", (), "/compact", "Compact the current conversation."),
    CommandSpec(
        "exit",
        ("quit",),
        "/exit",
        "Exit the current interactive runtime.",
    ),
)

_COMMANDS_BY_NAME: dict[str, CommandSpec] = {}
for _spec in COMMAND_SPECS:
    _COMMANDS_BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _COMMANDS_BY_NAME[_alias] = _spec


def parse_slash_command(text: str) -> ParsedCommand | None:
    """Parse a registered slash command, or return ``None`` for normal input.

    Command names and aliases are case-insensitive. Arguments are split on
    whitespace only; command-specific validation is intentionally limited to
    the current contract.
    """

    parts = text.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None

    command_token = parts[0][1:].lower()
    spec = _COMMANDS_BY_NAME.get(command_token)
    if spec is None:
        return None

    args = tuple(parts[1:])
    if spec.name == "resume":
        if len(args) != 1:
            raise CommandParseError(spec.name, spec.usage)
    elif args:
        raise CommandParseError(spec.name, spec.usage)

    return ParsedCommand(name=spec.name, args=args)


__all__ = [
    "COMMAND_SPECS",
    "CommandParseError",
    "CommandSpec",
    "ParsedCommand",
    "parse_slash_command",
]
