import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Literal

import yaml

from mycode.config import MYCODE_CONFIG_DIR_NAME

SkillSource = Literal["builtin", "user", "project"]
_SOURCE_PRIORITY: dict[SkillSource, int] = {
    "builtin": 0,
    "user": 1,
    "project": 2,
}
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class DuplicateSkillError(ValueError):
    pass


class SkillNotFoundError(KeyError):
    pass


class SkillPathError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root: Path
    skill_file: Path
    source: SkillSource
    frontmatter: Mapping[str, object]

    @property
    def allowed_tools(self) -> object | None:
        """Retain compatibility metadata without granting any permission."""
        return self.frontmatter.get("allowed-tools")


@dataclass(frozen=True)
class SkillDiscoveryWarning:
    source: SkillSource
    path: Path
    message: str

    @property
    def display(self) -> str:
        return f"{self.source}:{self.path.as_posix()}: {self.message}"


@dataclass
class SkillRegistry:
    _skills: dict[str, Skill] = field(default_factory=dict)
    warnings: list[SkillDiscoveryWarning] = field(default_factory=list)

    @classmethod
    def discover(
        cls,
        workspace_root: Path,
        *,
        builtin_root: Path | None = None,
        user_root: Path | None = None,
    ) -> "SkillRegistry":
        registry = cls()
        roots: tuple[tuple[SkillSource, Path], ...] = (
            (
                "builtin",
                Path(__file__).resolve().parent / "builtin"
                if builtin_root is None
                else builtin_root,
            ),
            (
                "user",
                Path.home() / MYCODE_CONFIG_DIR_NAME / "skills"
                if user_root is None
                else user_root,
            ),
            (
                "project",
                Path(workspace_root) / MYCODE_CONFIG_DIR_NAME / "skills",
            ),
        )
        for source, root in roots:
            registry._discover_root(root, source=source)
        return registry

    def _discover_root(self, root: Path, *, source: SkillSource) -> None:
        if not root.exists():
            return
        if not root.is_dir():
            self._warn(source, root, "Skill source is not a directory; skipped.")
            return
        try:
            candidates = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError as error:
            self._warn(source, root, f"Cannot enumerate Skill source: {error}")
            return

        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                skill = _load_skill(candidate, source=source)
                self.register(skill)
            except Exception as error:
                self._warn(source, candidate, str(error))

    def _warn(self, source: SkillSource, path: Path, message: str) -> None:
        self.warnings.append(
            SkillDiscoveryWarning(source=source, path=path, message=message)
        )

    def register(self, skill: Skill) -> None:
        existing = self._skills.get(skill.name)
        if existing is not None:
            existing_priority = _SOURCE_PRIORITY[existing.source]
            new_priority = _SOURCE_PRIORITY[skill.source]
            if new_priority == existing_priority:
                raise DuplicateSkillError(
                    f"Skill already registered from {skill.source}: {skill.name}"
                )
            if new_priority < existing_priority:
                return
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            raise SkillNotFoundError(f"Skill not found: {name}")
        return skill

    def list_skills(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def get_catalog(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (skill.name, skill.description) for skill in self.list_skills()
        )

    def load_instructions(self, name: str) -> str:
        skill = self.require(name)
        try:
            text = skill.skill_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("SKILL.md must be valid UTF-8.") from error
        except OSError as error:
            raise ValueError(f"Cannot read SKILL.md: {error}") from error
        frontmatter, body = _parse_frontmatter(text)
        loaded_name, loaded_description = _validate_metadata(
            frontmatter, directory_name=skill.root.name
        )
        if loaded_name != skill.name or loaded_description != skill.description:
            raise ValueError(
                "SKILL.md name or description changed after discovery; "
                "restart MyCode to rediscover Skills."
            )
        return body

    def resolve_resource(self, skill: Skill, relative_path: str) -> Path:
        return _resolve_bounded_path(skill.root, relative_path)

    def resolve_script(self, skill: Skill, relative_path: str) -> Path:
        requested = _validate_relative_path(relative_path)
        if requested.parts and requested.parts[0] == "scripts":
            requested = Path(*requested.parts[1:])
        if not requested.parts:
            raise SkillPathError("Script path must name a file inside scripts/.")
        return _resolve_bounded_path(skill.root / "scripts", requested)


def _load_skill(root: Path, *, source: SkillSource) -> Skill:
    skill_file = root / "SKILL.md"
    if not skill_file.exists():
        raise ValueError("Missing SKILL.md; skipped.")
    if not skill_file.is_file():
        raise ValueError("SKILL.md is not a regular file; skipped.")
    frontmatter = _read_frontmatter(skill_file)
    name, description = _validate_metadata(frontmatter, directory_name=root.name)
    resolved_root = root.resolve(strict=False)
    return Skill(
        name=name,
        description=description,
        root=resolved_root,
        skill_file=resolved_root / "SKILL.md",
        source=source,
        frontmatter=dict(frontmatter),
    )


def _validate_metadata(
    frontmatter: Mapping[str, object], *, directory_name: str
) -> tuple[str, str]:
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md frontmatter requires a non-empty string name.")
    if len(name) > 64:
        raise ValueError("Skill name must contain at most 64 characters.")
    if not _SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Skill name must contain lowercase letters, digits, and single hyphens."
        )
    if name != directory_name:
        raise ValueError(
            f"Skill name '{name}' does not match directory '{directory_name}'."
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "SKILL.md frontmatter requires a non-empty string description."
        )
    normalized_description = description.strip()
    if len(normalized_description) > 1024:
        raise ValueError("Skill description must contain at most 1024 characters.")
    return name, normalized_description


def _read_frontmatter(skill_file: Path) -> dict[str, object]:
    try:
        with skill_file.open("rb", buffering=0) as stream:
            first_line = stream.readline()
            if first_line.strip() != b"---":
                raise ValueError("SKILL.md must start with YAML frontmatter.")
            yaml_lines: list[str] = []
            for raw_line in stream:
                if raw_line.strip() == b"---":
                    return _load_frontmatter_yaml("".join(yaml_lines))
                try:
                    yaml_lines.append(raw_line.decode("utf-8"))
                except UnicodeDecodeError as error:
                    raise ValueError(
                        "SKILL.md frontmatter must be valid UTF-8; skipped."
                    ) from error
    except OSError as error:
        raise ValueError(f"Cannot read SKILL.md frontmatter: {error}") from error
    raise ValueError("SKILL.md YAML frontmatter is not closed.")


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter.")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("SKILL.md YAML frontmatter is not closed.")
    yaml_text = "".join(lines[1:closing_index])
    loaded = _load_frontmatter_yaml(yaml_text)
    return loaded, "".join(lines[closing_index + 1 :]).lstrip("\r\n")


def _load_frontmatter_yaml(yaml_text: str) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid SKILL.md YAML frontmatter: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("SKILL.md YAML frontmatter must be a mapping.")
    if any(not isinstance(key, str) for key in loaded):
        raise ValueError("SKILL.md YAML frontmatter keys must be strings.")
    return dict(loaded)


def _validate_relative_path(relative_path: str | PurePath) -> Path:
    if not isinstance(relative_path, (str, PurePath)):
        raise SkillPathError("Skill path must be relative.")
    text = str(relative_path)
    if not text or "\x00" in text:
        raise SkillPathError("Skill path must be a non-empty relative path.")
    requested = Path(text)
    if requested.is_absolute() or requested.drive or requested.root:
        raise SkillPathError("Absolute Skill paths are not allowed.")
    if any(part == ".." for part in requested.parts):
        raise SkillPathError("Parent traversal is not allowed in Skill paths.")
    return requested


def _resolve_bounded_path(root: Path, relative_path: str | PurePath) -> Path:
    requested = _validate_relative_path(relative_path)
    resolved_root = root.resolve(strict=False)
    resolved = (resolved_root / requested).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise SkillPathError("Skill path escapes the Skill root.")
    return resolved
