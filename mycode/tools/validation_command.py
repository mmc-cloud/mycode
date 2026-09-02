from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PYTHON_COMMANDS = {"python", "python3", "py"}
TEST_COMMANDS = {"pytest", "unittest"}
LINT_COMMANDS = {"mypy", "pyright"}
COMPILE_MODULES = {"compileall", "py_compile"}
RUFF_MUTATING_OPTIONS = {"--fix", "--fix-only", "--unsafe-fixes"}
INFORMATIONAL_OPTIONS = {"-h", "--help", "--version"}
PYTEST_NON_EXECUTION_OPTIONS = {
    "--collect-only",
    "--collectonly",
    "--co",
    "--fixtures",
    "--fixtures-per-test",
    "--markers",
    "--setup-plan",
    "--trace-config",
}
TOX_NON_EXECUTION_OPTIONS = {
    "-a",
    "-l",
    "--listenvs",
    "--listenvs-all",
    "--notest",
    "--showconfig",
}
NOX_NON_EXECUTION_OPTIONS = {"-l", "--list", "--list-sessions"}
MAKE_NON_EXECUTION_OPTIONS = {
    "-n",
    "--dry-run",
    "--just-print",
    "--question",
    "--recon",
}
UV_RUN_FLAG_OPTIONS = {
    "--frozen",
    "--isolated",
    "--locked",
    "--no-project",
    "--no-sync",
    "--offline",
}
UV_RUN_VALUE_OPTIONS = {
    "-p",
    "--directory",
    "--project",
    "--python",
}
ORDINARY_COMMANDS = {
    "cat",
    "dir",
    "echo",
    "find",
    "findstr",
    "get-childitem",
    "grep",
    "head",
    "ls",
    "pwd",
    "rg",
    "sed",
    "tail",
    "type",
    "where",
    "which",
}

ValidationClassification = Literal["validation", "non_validation", "unknown"]


@dataclass(frozen=True)
class ValidationCommandAnalysis:
    allowed: bool
    classification: ValidationClassification
    category: str
    reason: str


def analyze_validation_command(command: list[str]) -> ValidationCommandAnalysis:
    if not command:
        return _unknown("Validation command must not be empty.")

    normalized = [part.casefold() for part in command]
    executable = _normalized_executable(normalized[0])

    if executable == "uv":
        if len(normalized) < 3 or normalized[1] != "run":
            return _non_validation("Command is not a uv validation run.")
        unwrapped = _unwrap_uv_run(normalized[2:])
        if isinstance(unwrapped, ValidationCommandAnalysis):
            return unwrapped
        return _analyze_direct(unwrapped)

    return _analyze_direct(normalized)


def _analyze_direct(parts: list[str]) -> ValidationCommandAnalysis:
    executable = _normalized_executable(parts[0])

    if executable == "pytest":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS | PYTEST_NON_EXECUTION_OPTIONS):
            return _non_validation(
                "pytest command only reports information or collects tests."
            )
        return _allowed("test", "Command runs pytest validation.")

    if executable == "unittest":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS):
            return _non_validation("unittest command only reports information.")
        return _allowed("test", "Command runs unittest validation.")

    if executable == "tox":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS | TOX_NON_EXECUTION_OPTIONS):
            return _non_validation("tox command does not execute validation.")
        return _allowed("test", "Command runs tox validation.")

    if executable == "nox":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS | NOX_NON_EXECUTION_OPTIONS):
            return _non_validation("nox command does not execute validation.")
        return _allowed("test", "Command runs nox validation.")

    if executable == "ctest":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS | {"-n", "--show-only"}):
            return _non_validation("ctest command does not execute tests.")
        return _allowed("test", "Command runs ctest validation.")

    if executable == "go":
        return _analyze_go(parts)

    if executable == "cargo":
        return _analyze_subcommand_validator(parts, command="cargo", subcommand="test")

    if executable in {"npm", "yarn", "pnpm"}:
        return _analyze_package_test(parts, executable=executable)

    if executable == "make":
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS | MAKE_NON_EXECUTION_OPTIONS):
            return _non_validation("make command does not execute the test target.")
        if len(parts) >= 2 and parts[1] == "test":
            return _allowed("test", "Command runs the make test target.")
        return _unknown("Make command is not the explicit test target.")

    if executable in LINT_COMMANDS:
        if _has_option(parts[1:], INFORMATIONAL_OPTIONS):
            return _non_validation(f"{executable} command only reports information.")
        return _allowed("lint", f"Command runs {executable} static validation.")

    if executable == "ruff":
        return _analyze_ruff(parts)

    if executable in PYTHON_COMMANDS:
        return _analyze_python(parts)

    if executable in {"pip", "pip3"}:
        if len(parts) >= 2 and parts[1] == "install":
            return _non_validation("Package installation is not validation.")
        return _unknown("pip command is not a recognized validator.")

    if executable == "git":
        if parts[1:] == ["diff", "--check"]:
            return _allowed("lint", "Command checks the Git diff for whitespace errors.")
        return _non_validation("Git command is not a recognized validation command.")

    if executable in ORDINARY_COMMANDS:
        return _non_validation("Command is a recognized non-validation utility.")

    return _unknown(
        "Command is not recognized confidently as validation or non-validation."
    )


def _analyze_python(parts: list[str]) -> ValidationCommandAnalysis:
    if len(parts) < 2:
        return _non_validation("Python interpreter was invoked without a validator.")
    if parts[1] != "-m":
        if _is_setup_script(parts[1]):
            return _analyze_setup_build(parts[2:])
        if _is_known_python_test_script(parts[1:]):
            if _has_option(parts[2:], INFORMATIONAL_OPTIONS):
                return _non_validation("Python test script only reports information.")
            return _allowed("test", "Command runs a recognized Python test script.")
        if parts[1] in {"-c", "-"}:
            return _unknown("Inline Python may or may not perform validation.")
        return _unknown("Python script is not a recognized validator.")
    if len(parts) < 3:
        return _unknown("Python -m did not name a module.")

    module = parts[2]
    module_parts = [module, *parts[3:]]
    if module in TEST_COMMANDS:
        return _analyze_direct(module_parts)
    if module in COMPILE_MODULES:
        if _has_option(parts[3:], INFORMATIONAL_OPTIONS):
            return _non_validation(f"python -m {module} only reports information.")
        return _allowed("compile", f"Command runs python -m {module} validation.")
    if module == "build":
        if _has_option(parts[3:], INFORMATIONAL_OPTIONS):
            return _non_validation("python -m build only reports information.")
        return _allowed("build", "Command runs python -m build validation.")
    if module in LINT_COMMANDS or module == "ruff":
        return _analyze_direct(module_parts)

    if module in {"pip", "venv"}:
        return _non_validation(f"Python module is not validation: {module}")
    return _unknown(f"Python module is not a recognized validator: {module}")


def _analyze_go(parts: list[str]) -> ValidationCommandAnalysis:
    if len(parts) < 2:
        return _unknown("Go command did not name a validation subcommand.")
    if parts[1] != "test":
        if parts[1] in {"env", "help", "version"}:
            return _non_validation("Go command only reports information.")
        return _unknown("Go command is not the test subcommand.")
    if _has_option(parts[2:], INFORMATIONAL_OPTIONS | {"-list"}):
        return _non_validation("go test command only lists tests or reports information.")
    return _allowed("test", "Command runs go test validation.")


def _analyze_subcommand_validator(
    parts: list[str], *, command: str, subcommand: str
) -> ValidationCommandAnalysis:
    if len(parts) >= 2 and parts[1] == subcommand:
        if _has_option(parts[2:], INFORMATIONAL_OPTIONS):
            return _non_validation(f"{command} {subcommand} only reports information.")
        return _allowed("test", f"Command runs {command} {subcommand} validation.")
    if _has_option(parts[1:], INFORMATIONAL_OPTIONS):
        return _non_validation(f"{command} command only reports information.")
    return _unknown(f"{command} command is not the {subcommand} subcommand.")


def _analyze_package_test(
    parts: list[str], *, executable: str
) -> ValidationCommandAnalysis:
    arguments = parts[1:]
    if _has_option(arguments, INFORMATIONAL_OPTIONS):
        return _non_validation(f"{executable} command only reports information.")
    is_test = bool(arguments) and (
        arguments[0] == "test"
        or (len(arguments) >= 2 and arguments[:2] == ["run", "test"])
    )
    if not is_test:
        return _unknown(f"{executable} command is not the explicit test script.")
    if _has_option(arguments, {"--dry-run", "--ignore-scripts"}):
        return _non_validation(f"{executable} test command does not execute tests.")
    if _has_option(arguments, {"--if-present"}):
        return _unknown(f"{executable} test script may not exist.")
    return _allowed("test", f"Command runs {executable} test validation.")


def _analyze_ruff(parts: list[str]) -> ValidationCommandAnalysis:
    if len(parts) < 2:
        return _unknown("ruff requires an explicit non-mutating subcommand.")
    if _has_option(parts[1:], INFORMATIONAL_OPTIONS):
        return _non_validation("ruff command only reports information.")
    if any(_is_mutating_ruff_option(part) for part in parts[1:]):
        return _non_validation("Mutating ruff fix options are not validation.")
    if parts[1] == "check":
        return _allowed("lint", "Command runs non-fixing ruff checks.")
    if parts[1] == "format" and "--check" in parts[2:]:
        return _allowed("lint", "Command checks ruff formatting without writing files.")
    return _unknown("Ruff command is not a recognized non-mutating validation.")


def _is_known_python_test_script(parts: list[str]) -> bool:
    if not parts:
        return False
    script = Path(parts[0].replace("\\", "/"))
    name = script.name.casefold()
    if name == "manage.py":
        return len(parts) >= 2 and parts[1] == "test"
    if name in {"runtest.py", "runtests.py"}:
        return True
    return name.endswith(".py") and (
        name.startswith("test") or name.endswith("_test.py")
    )


def _is_setup_script(value: str) -> bool:
    return Path(value.replace("\\", "/")).name.casefold() == "setup.py"


def _analyze_setup_build(parts: list[str]) -> ValidationCommandAnalysis:
    if not parts or parts[0] not in {"build", "build_ext"}:
        return _unknown("setup.py command is not a recognized build validator.")
    if _has_option(parts[1:], INFORMATIONAL_OPTIONS):
        return _non_validation("setup.py build command only reports information.")
    return _allowed("build", f"Command runs setup.py {parts[0]} validation.")


def _normalized_executable(value: str) -> str:
    executable = Path(value).name.casefold()
    for suffix in (".exe", ".cmd", ".bat"):
        if executable.endswith(suffix):
            return executable[: -len(suffix)]
    return executable


def _is_mutating_ruff_option(value: str) -> bool:
    return value in RUFF_MUTATING_OPTIONS or any(
        value.startswith(f"{option}=") for option in RUFF_MUTATING_OPTIONS
    )


def _allowed(category: str, reason: str) -> ValidationCommandAnalysis:
    return ValidationCommandAnalysis(
        allowed=True,
        classification="validation",
        category=category,
        reason=reason,
    )


def _has_option(parts: list[str], options: set[str]) -> bool:
    return any(
        part in options or any(part.startswith(f"{option}=") for option in options)
        for part in parts
    )


def _unwrap_uv_run(
    parts: list[str],
) -> list[str] | ValidationCommandAnalysis:
    index = 0
    python_module = False
    while index < len(parts):
        part = parts[index]
        if part == "--":
            index += 1
            break
        if part in INFORMATIONAL_OPTIONS:
            return _non_validation("uv run command only reports information.")
        if part in UV_RUN_FLAG_OPTIONS:
            index += 1
            continue
        if part in {"-m", "--module"}:
            python_module = True
            index += 1
            continue
        if part in UV_RUN_VALUE_OPTIONS:
            if index + 1 >= len(parts):
                return _unknown(f"uv run option requires a value: {part}")
            index += 2
            continue
        if any(part.startswith(f"{option}=") for option in UV_RUN_VALUE_OPTIONS):
            index += 1
            continue
        if part.startswith("-"):
            return _unknown(f"Cannot identify validator after uv run option: {part}")
        break

    if index >= len(parts):
        return _unknown("uv run did not name a validator.")
    if python_module:
        return ["python", "-m", *parts[index:]]
    return parts[index:]


def _non_validation(reason: str) -> ValidationCommandAnalysis:
    return ValidationCommandAnalysis(
        allowed=False,
        classification="non_validation",
        category="unsupported",
        reason=reason,
    )


def _unknown(reason: str) -> ValidationCommandAnalysis:
    return ValidationCommandAnalysis(
        allowed=False,
        classification="unknown",
        category="unsupported",
        reason=reason,
    )
