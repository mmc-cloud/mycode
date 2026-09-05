from pydantic import Field, field_validator

from mycode.skills import ActiveSkillState, SkillPathError, SkillRegistry
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult
from mycode.tools.bounds import clamp_positive_int_upper_bound


DEFAULT_MAX_SKILL_RESOURCE_CHARS = 4000
MAX_SKILL_RESOURCE_CHARS = 20000


class ReadSkillResourceArgs(ToolArgs):
    skill: str = Field(min_length=1)
    path: str = Field(min_length=1)
    offset_chars: int = Field(default=0, ge=0, strict=True)
    max_chars: int = Field(
        default=DEFAULT_MAX_SKILL_RESOURCE_CHARS,
        ge=1,
        le=MAX_SKILL_RESOURCE_CHARS,
        strict=True,
    )

    @field_validator("max_chars", mode="before")
    @classmethod
    def clamp_max_chars(cls, value: object) -> object:
        return clamp_positive_int_upper_bound(
            value, upper_bound=MAX_SKILL_RESOURCE_CHARS
        )


class ReadSkillResourceTool(PydanticTool[ReadSkillResourceArgs]):
    name = "read_skill_resource"
    description = "从已激活的 Skill 中按边界读取 UTF-8 文本资源。"
    args_model = ReadSkillResourceArgs
    capability = "read"
    risk = "low"
    concurrency_safe = True

    def __init__(self, registry: SkillRegistry, state: ActiveSkillState) -> None:
        self.registry = registry
        self.state = state

    def _run(self, args: ReadSkillResourceArgs) -> ToolResult:
        skill = self.registry.get(args.skill)
        metadata: dict[str, object] = {
            "skill_name": args.skill,
            "resource_path": args.path,
        }
        if skill is None or not self.state.is_active(args.skill):
            return ToolResult.failure(
                error=f"Skill is not active: {args.skill}", metadata=metadata
            )
        metadata["skill_source"] = skill.source
        try:
            path = self.registry.resolve_resource(skill, args.path)
        except SkillPathError as error:
            return ToolResult.failure(error=str(error), metadata=metadata)
        if not path.exists():
            return ToolResult.failure(
                error=f"Skill resource not found: {args.path}", metadata=metadata
            )
        if not path.is_file():
            return ToolResult.failure(
                error=f"Skill resource is not a file: {args.path}", metadata=metadata
            )
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                raise UnicodeDecodeError("utf-8", raw, 0, 1, "NUL byte")
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(
                error=f"Skill resource is not valid UTF-8 text: {args.path}",
                metadata={**metadata, "reason": "unsupported_text"},
            )
        except OSError as error:
            return ToolResult.failure(
                error=f"Cannot read Skill resource: {error}", metadata=metadata
            )
        content = text[args.offset_chars : args.offset_chars + args.max_chars]
        next_offset = args.offset_chars + len(content)
        has_more = next_offset < len(text)
        return ToolResult.success(
            content=content,
            metadata={
                **metadata,
                "offset_chars": args.offset_chars,
                "returned_chars": len(content),
                "total_chars": len(text),
                "has_more": has_more,
                "next_offset_chars": next_offset if has_more else None,
            },
        )
