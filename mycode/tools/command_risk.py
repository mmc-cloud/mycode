from dataclasses import dataclass
from pathlib import Path


HIGH_RISK_COMMANDS = {
    "dd",
    "del",
    "diskpart",
    "erase",
    "format",
    "mkfs",
    "rd",
    "rm",
    "rmdir",
    "shutdown",
    "stop-computer",
}
SYSTEM_COMMANDS = {
    "chmod",
    "chown",
    "kill",
    "killall",
    "mount",
    "net",
    "netsh",
    "reboot",
    "reg",
    "sc",
    "set-executionpolicy",
    "sudo",
    "takeown",
    "taskkill",
}
NETWORK_COMMANDS = {
    "curl",
    "wget",
    "invoke-restmethod",
    "invoke-webrequest",
}
INSTALL_COMMANDS = {
    "install",
    "add",
    "sync",
}
INSPECT_COMMANDS = {
    "cat",
    "diff",
    "dir",
    "file",
    "find",
    "findstr",
    "get-childitem",
    "grep",
    "head",
    "ls",
    "rg",
    "sed",
    "select-string",
    "stat",
    "strings",
    "tail",
    "type",
    "wc",
    "xxd",
}
TEST_COMMANDS = {
    "pytest",
}
PYTHON_COMMANDS = {"python", "python3", "py"}
POWERSHELL_COMMANDS = {"powershell", "pwsh"}
CMD_COMMANDS = {"cmd"}
POSIX_SHELL_COMMANDS = {"bash", "dash", "ksh", "sh", "zsh"}
DANGEROUS_TEXT_MARKERS = {
    ".unlink",
    " dd ",
    " del ",
    " diskpart ",
    " erase ",
    " git clean ",
    " git checkout ",
    " git push --force",
    " git restore ",
    " git reset ",
    " kill ",
    " killall ",
    " os.remove",
    " os.unlink",
    " rd ",
    " reboot ",
    " remove-item",
    " rm ",
    " rmdir ",
    " shutil.rmtree",
    " stop-process",
    " taskkill ",
}


@dataclass(frozen=True)
class CommandRiskAnalysis:
    category: str
    risk: str
    decision: str
    reason: str


def analyze_command_risk(command: list[str]) -> CommandRiskAnalysis:
    executable = _normalized_executable(command)
    lowered_parts = [part.lower() for part in command]

    if executable in CMD_COMMANDS:
        return _analyze_cmd_command(lowered_parts)

    if executable in POWERSHELL_COMMANDS:
        return _analyze_powershell_command(lowered_parts)

    if executable in POSIX_SHELL_COMMANDS:
        return _analyze_posix_shell_command(lowered_parts)

    if executable in PYTHON_COMMANDS:
        return _analyze_python_command(lowered_parts)

    if executable == "uv":
        return _analyze_uv_command(lowered_parts)

    if executable in HIGH_RISK_COMMANDS:
        return CommandRiskAnalysis(
            category="delete_or_destructive",
            risk="high",
            decision="deny",
            reason=f"Destructive command is blocked: {executable}",
        )

    if executable == "git":
        return _analyze_git_command(lowered_parts)

    if _is_test_command(executable, lowered_parts):
        return CommandRiskAnalysis(
            category="test",
            risk="low",
            decision="ask",
            reason="Command appears to run tests or validation.",
        )

    if _is_install_command(executable, lowered_parts):
        return CommandRiskAnalysis(
            category="install_dependency",
            risk="medium",
            decision="ask",
            reason="Command may install, add, or synchronize dependencies.",
        )

    if executable in NETWORK_COMMANDS:
        return CommandRiskAnalysis(
            category="network_access",
            risk="medium",
            decision="ask",
            reason="Command may access the network.",
        )

    if executable in SYSTEM_COMMANDS:
        return CommandRiskAnalysis(
            category="system_operation",
            risk="high",
            decision="deny",
            reason=f"System-level command is blocked: {executable}",
        )

    if executable in INSPECT_COMMANDS:
        return CommandRiskAnalysis(
            category="inspect",
            risk="low",
            decision="ask",
            reason="Command appears to inspect local state.",
        )

    return CommandRiskAnalysis(
        category="unknown",
        risk="medium",
        decision="ask",
        reason="Command risk is unknown and requires confirmation.",
    )


def _analyze_git_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    action_index = _git_action_index(lowered_parts)
    if action_index is None:
        if any(part in {"--help", "--version"} for part in lowered_parts[1:]):
            return CommandRiskAnalysis(
                category="inspect",
                risk="low",
                decision="ask",
                reason="Git help/version command appears to inspect local state.",
            )
        return CommandRiskAnalysis(
            category="unknown",
            risk="medium",
            decision="ask",
            reason="Git command risk is unknown and requires confirmation.",
        )

    git_parts = lowered_parts[action_index:]
    git_action = git_parts[0]
    if git_action == "reset":
        return CommandRiskAnalysis(
            category="git_reset",
            risk="high",
            decision="deny",
            reason="Git reset is blocked because it can discard work.",
        )
    if git_action == "clean":
        return CommandRiskAnalysis(
            category="delete_or_destructive",
            risk="high",
            decision="deny",
            reason="Git clean is blocked because it can delete untracked files.",
        )
    if git_action == "restore":
        return CommandRiskAnalysis(
            category="git_restore",
            risk="high",
            decision="deny",
            reason="Git restore is blocked because it can discard work.",
        )
    if git_action == "checkout":
        return CommandRiskAnalysis(
            category="git_checkout",
            risk="high",
            decision="deny",
            reason="Git checkout is blocked because it can discard work.",
        )
    if git_action == "push":
        if _git_push_is_force(git_parts):
            return CommandRiskAnalysis(
                category="git_force_push",
                risk="high",
                decision="deny",
                reason="Git force push is blocked because it can overwrite remote history.",
            )
        return CommandRiskAnalysis(
            category="network_access",
            risk="medium",
            decision="ask",
            reason="Git push may access the network or change repository state.",
        )
    if git_action in {"clone", "fetch", "pull"}:
        return CommandRiskAnalysis(
            category="network_access",
            risk="medium",
            decision="ask",
            reason=f"Git {git_action} may access the network or change repository state.",
        )
    if git_action == "switch":
        return CommandRiskAnalysis(
            category="git_switch",
            risk="medium",
            decision="ask",
            reason="Git switch may change working tree state and requires confirmation.",
        )
    if git_action in {"status", "diff", "log", "show"}:
        return CommandRiskAnalysis(
            category="inspect",
            risk="low",
            decision="ask",
            reason=f"Git {git_action} appears to inspect repository state.",
        )
    if git_action == "branch" and _git_branch_is_inspect(git_parts):
        return CommandRiskAnalysis(
            category="inspect",
            risk="low",
            decision="ask",
            reason="Git branch appears to inspect repository state.",
        )

    return CommandRiskAnalysis(
        category="git_operation",
        risk="medium",
        decision="ask",
        reason=f"Git {git_action} may change repository state and requires confirmation.",
    )


def _git_action_index(lowered_parts: list[str]) -> int | None:
    value_options = {
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--html-path",
        "--namespace",
        "--super-prefix",
        "--work-tree",
        "-c",
        "-C",
    }
    flag_options = {
        "--bare",
        "--help",
        "--literal-pathspecs",
        "--no-optional-locks",
        "--no-pager",
        "--no-replace-objects",
        "--version",
        "-p",
    }
    value_option_prefixes = tuple(f"{option}=" for option in value_options)

    index = 1
    while index < len(lowered_parts):
        part = lowered_parts[index]
        if part == "--":
            index += 1
            continue
        if part in value_options:
            index += 2
            continue
        if part.startswith(value_option_prefixes):
            index += 1
            continue
        if part in flag_options:
            index += 1
            continue
        if part.startswith("-"):
            index += 1
            continue

        return index

    return None


def _git_push_is_force(git_parts: list[str]) -> bool:
    force_options = {
        "--force",
        "--force-if-includes",
        "--force-with-lease",
        "-f",
    }
    return any(
        part in force_options
        or part.startswith("--force-with-lease=")
        or part.startswith("--force-if-includes=")
        for part in git_parts[1:]
    )


def _git_branch_is_inspect(lowered_parts: list[str]) -> bool:
    inspect_options = {
        "--all",
        "--contains",
        "--format",
        "--list",
        "--merged",
        "--no-contains",
        "--no-merged",
        "--points-at",
        "--remotes",
        "--show-current",
        "-a",
        "-r",
        "-v",
        "-vv",
    }
    return len(lowered_parts) == 2 or all(
        part.startswith("--sort=")
        or part.startswith("--format=")
        or part in inspect_options
        for part in lowered_parts[2:]
    )


def _is_test_command(executable: str, lowered_parts: list[str]) -> bool:
    if executable in TEST_COMMANDS:
        return True

    if executable in PYTHON_COMMANDS and lowered_parts[1:3] == ["-m", "pytest"]:
        return True

    if executable == "uv" and lowered_parts[1:3] == ["run", "pytest"]:
        return True

    return False


def _analyze_cmd_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    inner_parts = _parts_after_first_switch(lowered_parts, {"/c", "/k"})
    if _contains_dangerous_text(inner_parts):
        return _wrapped_dangerous_command("cmd")

    if inner_parts:
        inner_executable = _normalized_executable(inner_parts)
        if inner_executable in HIGH_RISK_COMMANDS:
            return CommandRiskAnalysis(
                category="delete_or_destructive",
                risk="high",
                decision="deny",
                reason=f"Destructive command is blocked inside cmd: {inner_executable}",
            )
        if inner_executable in SYSTEM_COMMANDS:
            return CommandRiskAnalysis(
                category="system_operation",
                risk="high",
                decision="deny",
                reason=f"System-level command is blocked inside cmd: {inner_executable}",
            )

    return CommandRiskAnalysis(
        category="shell_wrapper",
        risk="high",
        decision="ask",
        reason="cmd wrapper command requires confirmation.",
    )


def _analyze_powershell_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    command_parts = _parts_after_first_switch(
        lowered_parts,
        {"-command", "-c", "/c"},
    )
    if _contains_dangerous_text(command_parts or lowered_parts):
        return _wrapped_dangerous_command(lowered_parts[0])

    return CommandRiskAnalysis(
        category="shell_wrapper",
        risk="high",
        decision="ask",
        reason="PowerShell wrapper command requires confirmation.",
    )


def _analyze_posix_shell_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    command_parts = _parts_after_first_switch(lowered_parts, {"-c"})
    if _contains_dangerous_text(command_parts or lowered_parts):
        return _wrapped_dangerous_command(lowered_parts[0])

    return CommandRiskAnalysis(
        category="shell_wrapper",
        risk="high",
        decision="ask",
        reason="POSIX shell wrapper command requires confirmation.",
    )


def _analyze_python_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    if _is_python_version_command(lowered_parts):
        return CommandRiskAnalysis(
            category="inspect",
            risk="low",
            decision="ask",
            reason="Python version command appears to inspect local state.",
        )

    if lowered_parts[1:3] == ["-m", "pytest"]:
        return CommandRiskAnalysis(
            category="test",
            risk="low",
            decision="ask",
            reason="Command appears to run tests or validation.",
        )

    if "-c" in lowered_parts:
        inline_parts = lowered_parts[lowered_parts.index("-c") + 1 :]
        if _contains_dangerous_text(inline_parts):
            return CommandRiskAnalysis(
                category="delete_or_destructive",
                risk="high",
                decision="deny",
                reason="Python inline command is blocked because it may delete or alter files.",
            )

        return CommandRiskAnalysis(
            category="python_inline",
            risk="medium",
            decision="ask",
            reason="Python inline command requires confirmation.",
        )

    return CommandRiskAnalysis(
        category="python_execution",
        risk="medium",
        decision="ask",
        reason="Python command requires confirmation.",
    )


def _analyze_uv_command(lowered_parts: list[str]) -> CommandRiskAnalysis:
    if len(lowered_parts) > 1 and lowered_parts[1] in {"--version", "-v"}:
        return CommandRiskAnalysis(
            category="inspect",
            risk="low",
            decision="ask",
            reason="uv version command appears to inspect local state.",
        )

    if lowered_parts[1:3] == ["run", "pytest"]:
        return CommandRiskAnalysis(
            category="test",
            risk="low",
            decision="ask",
            reason="Command appears to run tests or validation.",
        )

    if len(lowered_parts) > 2 and lowered_parts[1] == "run":
        run_parts = lowered_parts[2:]
        run_executable = _normalized_executable(run_parts)
        if run_executable in PYTHON_COMMANDS:
            return _analyze_python_command(run_parts)
        if _contains_dangerous_text(run_parts):
            return _wrapped_dangerous_command("uv run")

        return CommandRiskAnalysis(
            category="uv_run",
            risk="medium",
            decision="ask",
            reason="uv run command requires confirmation.",
        )

    if _is_install_command("uv", lowered_parts):
        return CommandRiskAnalysis(
            category="install_dependency",
            risk="medium",
            decision="ask",
            reason="Command may install, add, or synchronize dependencies.",
        )

    return CommandRiskAnalysis(
        category="unknown",
        risk="medium",
        decision="ask",
        reason="uv command risk is unknown and requires confirmation.",
    )


def _is_python_version_command(lowered_parts: list[str]) -> bool:
    return len(lowered_parts) == 2 and lowered_parts[1] in {
        "--version",
        "-v",
        "-vv",
    }


def _parts_after_first_switch(
    lowered_parts: list[str],
    switches: set[str],
) -> list[str]:
    for index, part in enumerate(lowered_parts):
        if part in switches:
            return lowered_parts[index + 1 :]

    return []


def _contains_dangerous_text(parts: list[str]) -> bool:
    if not parts:
        return False

    text = " ".join(parts)
    normalized = f" {text.replace(';', ' ').replace('&', ' ')} "
    return any(marker in normalized for marker in DANGEROUS_TEXT_MARKERS)


def _wrapped_dangerous_command(wrapper: str) -> CommandRiskAnalysis:
    return CommandRiskAnalysis(
        category="delete_or_destructive",
        risk="high",
        decision="deny",
        reason=f"Dangerous command is blocked inside {wrapper}.",
    )


def _is_install_command(executable: str, lowered_parts: list[str]) -> bool:
    if executable in {"pip", "pip3"} and len(lowered_parts) > 1:
        return lowered_parts[1] in INSTALL_COMMANDS

    if executable == "uv" and len(lowered_parts) > 1:
        return lowered_parts[1] in {"add", "sync"} or lowered_parts[1:3] == [
            "pip",
            "install",
        ]

    if executable in {"npm", "pnpm", "yarn"} and len(lowered_parts) > 1:
        return lowered_parts[1] in INSTALL_COMMANDS

    if executable == "poetry" and len(lowered_parts) > 1:
        return lowered_parts[1] in {"add", "install", "sync"}

    return False


def _normalized_executable(command: list[str]) -> str:
    executable = Path(command[0]).name.lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    if executable.endswith(".cmd") or executable.endswith(".bat"):
        executable = executable[:-4]

    return executable
