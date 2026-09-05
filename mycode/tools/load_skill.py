from pydantic import Field

from mycode.skills import (
    MAX_ACTIVE_SKILLS,
    ActiveSkillLimitError,
    ActiveSkillState,
    SkillNotFoundError,
    SkillRegistry,
)
from mycode.tools.base import PydanticTool, ToolArgs, ToolResult


class LoadSkillArgs(ToolArgs):
    name: str = Field(min_length=1)


class LoadSkillTool(PydanticTool[LoadSkillArgs]):
    name = "load_skill"
    description = (
        "加载一个与当前任务相关的 Skill。加载后的完整 Skill 指导会从下一轮模型调用"
        "开始可见；如果后续操作依赖该 Skill，请先加载 Skill，再根据加载后的指导继续执行。"
    )
    args_model = LoadSkillArgs
    capability = "control"
    risk = "low"

    def __init__(self, registry: SkillRegistry, state: ActiveSkillState) -> None:
        self.registry = registry
        self.state = state

    def _run(self, args: LoadSkillArgs) -> ToolResult:
        try:
            skill = self.registry.require(args.name)
        except SkillNotFoundError:
            return ToolResult.failure(
                error=f"Skill not found: {args.name}",
                metadata={"skill_name": args.name, "already_active": False},
            )
        if self.state.is_active(skill.name):
            return ToolResult.success(
                content=f"Skill '{skill.name}' 已经激活。",
                metadata={
                    "skill_name": skill.name,
                    "skill_source": skill.source,
                    "already_active": True,
                },
            )
        if len(self.state.get_active()) >= MAX_ACTIVE_SKILLS:
            return ToolResult.failure(
                error=f"最多只能同时激活 {MAX_ACTIVE_SKILLS} 个 Skill。",
                metadata={
                    "skill_name": skill.name,
                    "skill_source": skill.source,
                    "already_active": False,
                    "active_skill_count": len(self.state.get_active()),
                    "max_active_skills": MAX_ACTIVE_SKILLS,
                },
            )
        try:
            instructions = self.registry.load_instructions(skill.name)
            self.state.activate(skill, instructions)
        except ActiveSkillLimitError as error:
            return ToolResult.failure(
                error=str(error),
                metadata={
                    "skill_name": skill.name,
                    "skill_source": skill.source,
                    "already_active": False,
                    "active_skill_count": len(self.state.get_active()),
                    "max_active_skills": MAX_ACTIVE_SKILLS,
                },
            )
        except ValueError as error:
            return ToolResult.failure(
                error=f"无法加载 Skill '{skill.name}'：{error}",
                metadata={
                    "skill_name": skill.name,
                    "skill_source": skill.source,
                    "already_active": False,
                },
            )
        metadata = {
            "skill_name": skill.name,
            "skill_source": skill.source,
            "already_active": False,
        }
        return ToolResult.success(
            content=(
                f"Skill '{skill.name}' 已为当前任务激活。完整指导将从下一轮模型调用开始可见。"
            ),
            metadata=metadata,
        )
