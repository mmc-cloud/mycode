from dataclasses import dataclass
from pathlib import Path


PYTHON_COMMANDS = {"python", "python3", "py"}
TEST_COMMANDS = {"pytest", "unittest"}
LINT_COMMANDS = {"mypy", "pyright"}
COMPILE_MODULES = {"compileall", "py_compile"}
RUFF_MUTATING_OPTIONS = {"--fix", "--fix-only", "--unsafe-fixes"}


@dataclass(frozen=True)
class ValidationCommandAnalysis:
    allowed: bool
    category: str
    reason: str


def analyze_validation_command(command: list[str]) -> ValidationCommandAnalysis:
    if not command:
        return _denied("Validation command must not be empty.")

    normalized = [part.casefold() for part in command]
    executable = _normalized_executable(normalized[0])

    if executable == "uv":
        if len(normalized) < 3 or normalized[1] != "run":
            return _denied("Only 'uv run <validator>' is allowed for validation.")
        if normalized[2].startswith("-"):
            return _denied("uv run options are not allowed before the validator.")
        return _analyze_direct(normalized[2:])

    return _analyze_direct(normalized)


def _analyze_direct(parts: list[str]) -> ValidationCommandAnalysis:
    executable = _normalized_executable(parts[0])

    if executable == "pytest":
        return _allowed("test", "Command runs pytest validation.")

    if executable in LINT_COMMANDS:
        return _allowed("lint", f"Command runs {executable} static validation.")

    if executable == "ruff":
        return _analyze_ruff(parts)

    if executable in PYTHON_COMMANDS:
        return _analyze_python(parts)

    return _denied(
        "Command is not an allowed test, compile, or lint validator."
    )


def _analyze_python(parts: list[str]) -> ValidationCommandAnalysis:
    if len(parts) < 3 or parts[1] != "-m":
        return _denied("Only allowlisted 'python -m <validator>' commands are allowed.")

    module = parts[2]
    module_parts = [module, *parts[3:]]
    if module in TEST_COMMANDS:
        return _allowed("test", f"Command runs python -m {module} validation.")
    if module in COMPILE_MODULES:
        return _allowed("compile", f"Command runs python -m {module} validation.")
    if module in LINT_COMMANDS or module == "ruff":
        return _analyze_direct(module_parts)

    return _denied(f"Python module is not an allowed validator: {module}")


def _analyze_ruff(parts: list[str]) -> ValidationCommandAnalysis:
    if len(parts) < 2:
        return _denied("ruff requires an explicit non-mutating subcommand.")
    if any(_is_mutating_ruff_option(part) for part in parts[1:]):
        return _denied("Mutating ruff fix options are not allowed.")
    if parts[1] == "check":
        return _allowed("lint", "Command runs non-fixing ruff checks.")
    if parts[1] == "format" and "--check" in parts[2:]:
        return _allowed("lint", "Command checks ruff formatting without writing files.")
    return _denied("Only 'ruff check' or 'ruff format --check' is allowed.")


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
    return ValidationCommandAnalysis(allowed=True, category=category, reason=reason)


def _denied(reason: str) -> ValidationCommandAnalysis:
    return ValidationCommandAnalysis(
        allowed=False,
        category="unsupported",
        reason=reason,
    )
