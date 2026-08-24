from pathlib import Path

from mycode.tools.ignore import (
    DEFAULT_SAFE_ENV_TEMPLATE_FILE_NAMES,
    DEFAULT_SENSITIVE_FILE_NAMES,
    DEFAULT_SENSITIVE_FILE_SUFFIXES,
)


GLOB_META_CHARS = frozenset("*?[")


def validate_relative_pattern(pattern: str, *, label: str) -> str | None:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        return f"{label} must be relative: {pattern}"

    if ".." in pattern_path.parts:
        return f"{label} must not contain '..': {pattern}"

    return None


def is_explicit_path_pattern(pattern: str) -> bool:
    return not any(char in pattern for char in GLOB_META_CHARS)


def is_sensitive_path_pattern(pattern: str) -> bool:
    name_pattern = Path(pattern).name.casefold()
    if not name_pattern:
        return False

    if name_pattern in DEFAULT_SAFE_ENV_TEMPLATE_FILE_NAMES:
        return False

    if name_pattern in DEFAULT_SENSITIVE_FILE_NAMES:
        return True

    if any(
        name_pattern.startswith(f"{sensitive_name}*")
        for sensitive_name in DEFAULT_SENSITIVE_FILE_NAMES
    ):
        return True

    if name_pattern.startswith(".env"):
        return True

    return any(name_pattern.endswith(f"*{suffix}") for suffix in DEFAULT_SENSITIVE_FILE_SUFFIXES)
