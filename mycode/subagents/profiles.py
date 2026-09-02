from dataclasses import dataclass
from collections.abc import Callable

from mycode.permissions import Confirmer
from mycode.subagents.contracts import (
    BoundedResultArgs,
    ExplorerResult,
    ReviewerResult,
    SubAgentRole,
    TesterReport,
)
from mycode.tools.glob import GlobTool
from mycode.tools.grep import GrepTool
from mycode.tools.inspect_changes import InspectChangesTool
from mycode.tools.read_file import ReadFileTool
from mycode.tools.registry import ToolRegistry
from mycode.tools.run_validation import RunValidationTool
from mycode.tools.submit_result import SubmitResultTool
from mycode.tools.workspace import Workspace


class ProfileToolConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class AgentProfile:
    role: SubAgentRole
    display_name: str
    purpose: str
    tool_names: tuple[str, ...]
    submission_model: type[BoundedResultArgs]
    near_limit_prompt: str
    role_rules: tuple[str, ...]
    success_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise ValueError("AgentProfile display_name must not be blank.")
        if not self.purpose.strip():
            raise ValueError("AgentProfile purpose must not be blank.")
        if not self.near_limit_prompt.strip():
            raise ValueError("AgentProfile near_limit_prompt must not be blank.")
        if not self.tool_names or self.tool_names[-1] != "submit_result":
            raise ValueError("AgentProfile tool_names must end with submit_result.")
        if len(self.tool_names) != len(set(self.tool_names)):
            raise ValueError("AgentProfile tool_names must be unique.")
        if not self.role_rules:
            raise ValueError("AgentProfile role_rules must not be empty.")
        if not self.success_criteria:
            raise ValueError("AgentProfile success_criteria must not be empty.")
        expected_model = ROLE_SUBMISSION_MODELS[self.role]
        if self.submission_model is not expected_model:
            raise ValueError(
                f"AgentProfile submission_model does not match role {self.role}."
            )


ROLE_SUBMISSION_MODELS: dict[SubAgentRole, type[BoundedResultArgs]] = {
    "explorer": ExplorerResult,
    "tester": TesterReport,
    "reviewer": ReviewerResult,
}


EXPLORER_PROFILE = AgentProfile(
    role="explorer",
    display_name="Explorer",
    purpose="调查代码、定位相关文件、解释调用链，并返回可核对的证据。",
    tool_names=("read_file", "glob", "grep", "submit_result"),
    submission_model=ExplorerResult,
    near_limit_prompt=(
        "当前接近运行轮次上限。请重新评估剩余工作，并优先使用剩余轮次完成最重要的"
        "必要步骤。如果无法在剩余轮次内完成任务，请尽量保留已有成果，并明确说明"
        "未完成部分或阻塞原因。"
    ),
    role_rules=(
        "只调查和读取，不执行命令，不修改文件。",
        "每条结论都要关联文件位置和实际读取到的证据。",
        "搜索范围不足时明确记录不确定项，不把推测写成事实。",
    ),
    success_criteria=(
        "找到相关实现时，至少提交一条带 path、claim 和 evidence 的 finding。",
        "没有找到时，使用 no_match 并提交非空 searched_scope。",
        "提交 summary、searched_scope、findings 和 uncertainties。",
    ),
)

TESTER_PROFILE = AgentProfile(
    role="tester",
    display_name="Tester",
    purpose="执行受限验证、分析真实退出状态，并给出精炼的失败或阻塞说明。",
    tool_names=("read_file", "glob", "grep", "run_validation", "submit_result"),
    submission_model=TesterReport,
    near_limit_prompt=(
        "当前接近运行轮次上限。请重新评估剩余工作，并优先使用剩余轮次完成最重要的"
        "必要步骤。如果无法在剩余轮次内完成任务，请尽量保留已有成果，并明确说明"
        "未完成部分或阻塞原因。"
    ),
    role_rules=(
        "只使用 run_validation 执行测试、编译或 lint，不请求任意命令执行。",
        "验证命令会执行项目代码，必须尊重权限确认和拒绝结果。",
        "不要把未运行、被拒绝或启动失败的验证描述为通过。",
    ),
    success_criteria=(
        "passed 只能在至少一个真实验证命令成功后提交，Runtime 会复核退出码。",
        "failed 必须提交 failure_summary，blocked 必须提交 blocked_reason。",
        "提交 status、summary、失败或阻塞信息以及 uncertainties。",
    ),
)

REVIEWER_PROFILE = AgentProfile(
    role="reviewer",
    display_name="Reviewer",
    purpose="独立审查代码、设计、安全边界和测试缺口，并给出分级问题。",
    tool_names=("read_file", "glob", "grep", "inspect_changes", "submit_result"),
    submission_model=ReviewerResult,
    near_limit_prompt=(
        "当前接近运行轮次上限。请重新评估剩余工作，并优先使用剩余轮次完成最重要的"
        "必要步骤。如果无法在剩余轮次内完成任务，请尽量保留已有成果，并明确说明"
        "未完成部分或阻塞原因。"
    ),
    role_rules=(
        "只读取代码和检查有界变更，不运行测试，不修改文件。",
        "问题必须包含 severity、path、problem、evidence 和 suggestion。",
        "需要额外验证时把它写成建议，由主 Agent 决定是否委派 Tester。",
    ),
    success_criteria=(
        "提交非空 reviewed_scope 和明确的合并建议。",
        "changes_requested 至少包含一条 finding。",
        "没有问题时提交空 findings，但仍说明实际审查范围和不确定项。",
    ),
)

BUILTIN_AGENT_PROFILES: dict[SubAgentRole, AgentProfile] = {
    profile.role: profile
    for profile in (EXPLORER_PROFILE, TESTER_PROFILE, REVIEWER_PROFILE)
}


def get_agent_profile(role: SubAgentRole) -> AgentProfile:
    try:
        return BUILTIN_AGENT_PROFILES[role]
    except KeyError as error:  # pragma: no cover - callers normally validate SubAgentTask.
        raise ValueError(f"Unsupported SubAgent role: {role}") from error


def create_subagent_tool_registry(
    profile: AgentProfile,
    workspace: Workspace,
    *,
    confirmer: Confirmer | None = None,
    result_validator: Callable[[BoundedResultArgs], str | None] | None = None,
) -> ToolRegistry:
    common_tools = [
        ReadFileTool(workspace),
        GlobTool(workspace),
        GrepTool(workspace),
    ]
    if profile.role == "explorer":
        role_tools = []
    elif profile.role == "tester":
        role_tools = [RunValidationTool(workspace)]
    elif profile.role == "reviewer":
        role_tools = [InspectChangesTool(workspace)]
    else:  # pragma: no cover - AgentProfile role is a closed Literal at runtime.
        raise ValueError(f"Unsupported SubAgent role: {profile.role}")

    submit_tool = SubmitResultTool(
        role=profile.role,
        result_model=profile.submission_model,
        acceptance_validator=result_validator,
    )
    registry = ToolRegistry.from_tools(
        [*common_tools, *role_tools, submit_tool],
        confirmer=confirmer,
    )
    actual_names = tuple(tool.name for tool in registry.list_tools())
    if actual_names != profile.tool_names:
        raise ProfileToolConfigurationError(
            f"Profile tool contract mismatch for {profile.role}: "
            f"declared={profile.tool_names}, actual={actual_names}"
        )
    return registry
