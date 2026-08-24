from pathlib import Path


DEFAULT_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

DEFAULT_LOW_RELEVANCE_DIR_NAMES = frozenset(
    {
        "__fixtures__",
        "__snapshots__",
        "archive",
        "archives",
        "demo",
        "demos",
        "example",
        "examples",
        "external",
        "fixture",
        "fixtures",
        "reference",
        "references",
        "sample",
        "samples",
        "snapshot",
        "snapshots",
        "third_party",
        "vendor",
    }
)

DEFAULT_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".npmrc",
        ".pypirc",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)

DEFAULT_SENSITIVE_FILE_SUFFIXES = (
    ".key",
    ".p12",
    ".pem",
    ".pfx",
)

DEFAULT_SAFE_ENV_TEMPLATE_FILE_NAMES = frozenset(
    {
        ".env.example",
        ".env.sample",
        ".env.template",
    }
)


def is_ignored_path(
    path: Path,
    root: Path,
    excluded_dir_names: frozenset[str] = DEFAULT_EXCLUDED_DIR_NAMES,
) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return True

    if any(part in excluded_dir_names for part in relative_path.parts[:-1]):
        return True

    return is_sensitive_path(path, root)


def is_low_relevance_path(
    path: Path,
    root: Path,
    low_relevance_dir_names: frozenset[str] = DEFAULT_LOW_RELEVANCE_DIR_NAMES,
) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return True

    return any(part.casefold() in low_relevance_dir_names for part in relative_path.parts[:-1])


def is_sensitive_path(
    path: Path,
    root: Path,
    sensitive_file_names: frozenset[str] = DEFAULT_SENSITIVE_FILE_NAMES,
    sensitive_file_suffixes: tuple[str, ...] = DEFAULT_SENSITIVE_FILE_SUFFIXES,
) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True

    name = path.name.casefold()
    if name in DEFAULT_SAFE_ENV_TEMPLATE_FILE_NAMES:
        return False

    if name in sensitive_file_names:
        return True

    if name.startswith(".env."):
        return True

    return any(name.endswith(suffix) for suffix in sensitive_file_suffixes)
