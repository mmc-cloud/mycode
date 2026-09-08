import ast
from pathlib import Path


APPLICATION_ROOT = Path(__file__).parents[1] / "mycode" / "application"
MCP_TRUST_PATH = Path(__file__).parents[1] / "mycode" / "mcp" / "trust.py"
CORE_ROOTS = tuple(
    Path(__file__).parents[1] / "mycode" / name
    for name in (
        "agent",
        "context",
        "persistence",
        "mcp",
        "tools",
        "subagents",
        "application",
    )
)


def test_application_modules_do_not_depend_on_presentation_or_terminal_io() -> None:
    forbidden_import_prefixes = (
        "mycode.presentation",
        "mycode.adapters",
    )
    forbidden_names = {
        "CliPresenter",
        "CliSubAgentObserver",
        "TerminalConfirmer",
    }
    violations: list[str] = []

    for path in APPLICATION_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name.startswith(prefix)
                        for prefix in forbidden_import_prefixes
                    ):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if any(
                    (node.module or "").startswith(prefix)
                    for prefix in forbidden_import_prefixes
                ):
                    violations.append(f"{path}: from {node.module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "input",
                    "print",
                }:
                    violations.append(f"{path}: {node.func.id}()")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}: {node.id}")

    assert violations == []


def test_mcp_trust_is_presentation_neutral() -> None:
    tree = ast.parse(MCP_TRUST_PATH.read_text(encoding="utf-8"), filename=str(MCP_TRUST_PATH))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.startswith("mycode.presentation")
                for alias in node.names
            ):
                violations.append(f"{MCP_TRUST_PATH}: presentation import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("mycode.presentation"):
                violations.append(f"{MCP_TRUST_PATH}: presentation import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"input", "print"}:
                violations.append(f"{MCP_TRUST_PATH}: {node.func.id}()")

    assert violations == []


def test_core_packages_do_not_import_presentation() -> None:
    violations: list[str] = []
    for root in CORE_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("mycode.presentation"):
                            violations.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").startswith("mycode.presentation"):
                        violations.append(f"{path}: from {node.module}")

    assert violations == []
