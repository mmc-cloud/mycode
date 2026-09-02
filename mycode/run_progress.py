from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
from typing import Iterable, Literal

from mycode.conversation import Conversation
from mycode.tools.patterns import is_explicit_path_pattern
from mycode.tools.validation_command import analyze_validation_command
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_MAX_TURNS = 50
MAIN_NEAR_LIMIT_REMAINING_TURNS = 5
DEFAULT_STAGNATION_REPEAT_LIMIT = 2
RUN_CHECKPOINT_HEADING = "## 续跑检查点"

MAX_TURNS_FINALIZATION_PROMPT = (
    "本轮可用的模型轮次已经用完。请停止继续探索，不要请求或假设任何新的工具结果。"
    "请仅根据当前对话中已有的工具结果，用中文生成可直接供下一次续跑使用的检查点。"
    f"必须以“{RUN_CHECKPOINT_HEADING}”开头，并依次包含：已确认事实、已读取文件及范围、"
    "已修改文件、测试状态、剩余动作、当前阻塞。没有内容的栏目也要写“无”。"
    "不要声称读取过实际未读取的文件，并检查数量、列表和文件名是否前后一致。"
    "内容要简洁、可执行，不要重新展开方案论证；运行状态将由 CLI 单独提示。"
)
MAIN_NEAR_LIMIT_PROMPT = (
    "当前距离运行轮次上限还有 5 轮。请重新评估剩余工作，并优先使用剩余轮次"
    "完成最重要的必要步骤。如果无法在剩余轮次内完成任务，请尽量保留已有成果，"
    "并明确说明未完成部分或阻塞原因。"
)
RESUME_PROMPT = (
    "这是达到上限后的续跑。请直接使用上一条“续跑检查点”衔接工作，"
    "不要重新读取已经确认的文件，不要重复输出原方案；先核对已有改动和测试状态，"
    "然后执行检查点中的剩余动作。"
)

ToolResultStatus = Literal["success", "failure"] | None
MutationStatus = Literal["yes", "no", "unknown"]
ValidationStatus = Literal["none", "pass", "fail", "unknown"]


class RuntimePolicy(str, Enum):
    NO_INTERVENTION = "NO_INTERVENTION"
    CONVERGENCE_GUIDANCE = "CONVERGENCE_GUIDANCE"


@dataclass(frozen=True)
class RuntimeObservation:
    """Facts observed at the model/tool boundary; contains no policy semantics."""

    tool_result: ToolResultStatus = None
    tool_signature: str | None = None
    resource: str | None = None
    mutation: MutationStatus = "no"
    validation: ValidationStatus = "none"
    result_signature: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    policy: RuntimePolicy
    guidance: tuple[str, ...] = ()
    notice: str = ""


@dataclass
class RuntimeState:
    stagnation_turns: int = 0
    same_tool_repeat: int = 0
    same_result_repeat: int = 0
    resource_repeat: int = 0
    convergence_guided: bool = False
    last_observation: RuntimeObservation | None = None
    last_tool_observation: RuntimeObservation | None = None
    last_reason: str = "run_started"
    _previous_tool_pattern: tuple[str, ...] | None = field(default=None, repr=False)
    _previous_result_pattern: tuple[str, ...] | None = field(default=None, repr=False)
    _previous_resource_pattern: tuple[str, ...] | None = field(default=None, repr=False)

    def observe_tool_turn(
        self,
        observations: tuple[RuntimeObservation, ...],
    ) -> None:
        if not observations:
            return
        had_stagnation = self.stagnation_turns > 0
        for observation in observations:
            self.last_observation = observation
            self.last_tool_observation = observation

        tool_pattern = _present_pattern(
            observation.tool_signature for observation in observations
        )
        result_pattern = _present_pattern(
            observation.result_signature for observation in observations
        )
        resource_pattern = _present_pattern(
            observation.resource for observation in observations
        )
        self.same_tool_repeat, self._previous_tool_pattern = _next_repeat(
            tool_pattern, self._previous_tool_pattern, self.same_tool_repeat
        )
        self.same_result_repeat, self._previous_result_pattern = _next_repeat(
            result_pattern, self._previous_result_pattern, self.same_result_repeat
        )
        self.resource_repeat, self._previous_resource_pattern = _next_repeat(
            resource_pattern, self._previous_resource_pattern, self.resource_repeat
        )
        repeated = max(
            self.same_tool_repeat,
            self.same_result_repeat,
            self.resource_repeat,
        ) >= DEFAULT_STAGNATION_REPEAT_LIMIT
        if repeated:
            self.stagnation_turns += 1
            self.last_reason = "repetition_observed"
            return
        self.stagnation_turns = max(0, self.stagnation_turns - 1)
        if self.stagnation_turns == 0:
            self.convergence_guided = False
            if had_stagnation:
                self.last_reason = "stagnation_episode_ended"
        elif had_stagnation:
            self.last_reason = "stagnation_decayed"

def observe_tool_result(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
    ok: bool,
    content: str,
    error: str | None,
    metadata: dict[str, object],
    workspace: Workspace | None = None,
) -> RuntimeObservation:
    """Build a factual observation from one completed tool call."""
    return RuntimeObservation(
        tool_result="success" if ok else "failure",
        tool_signature=tool_call_signature(tool_name, arguments),
        resource=classify_resource(
            tool_name,
            arguments,
            workspace,
            metadata=metadata,
        ),
        mutation=classify_mutation(
            tool_name=tool_name,
            capability=capability,
            arguments=arguments,
            ok=ok,
        ),
        validation=classify_validation(
            tool_name=tool_name,
            arguments=arguments,
            metadata=metadata,
        ),
        result_signature=tool_result_signature(
            tool_name=tool_name,
            ok=ok,
            content=content,
            error=error,
            metadata=metadata,
        ),
    )


def classify_mutation(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
    ok: bool,
) -> MutationStatus:
    if tool_name == "run_command":
        return "unknown"
    if tool_name == "run_validation":
        return "no"
    if tool_name in {"edit_file", "write_file"}:
        path = arguments.get("path")
        if isinstance(path, str) and _is_temporary_analysis_path(path):
            return "no"
        return "yes" if ok else "no"
    if capability in {"read", "delegate"} or tool_name == "read_artifact":
        return "no"
    return "unknown"


def classify_validation(
    *,
    tool_name: str,
    arguments: dict[str, object],
    metadata: dict[str, object],
) -> ValidationStatus:
    if tool_name not in {"run_command", "run_validation"}:
        return "none"
    command = arguments.get("command")
    if not isinstance(command, list) or any(
        not isinstance(part, str) for part in command
    ):
        return "unknown"
    analysis = analyze_validation_command(command)
    if analysis.classification == "non_validation":
        return "none"
    if analysis.classification == "unknown":
        return "unknown"
    exit_code = metadata.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return "pass" if exit_code == 0 else "fail"
    return "unknown"


def classify_resource(
    tool_name: str,
    arguments: dict[str, object],
    workspace: Workspace | None,
    *,
    metadata: dict[str, object] | None = None,
) -> str | None:
    if workspace is None:
        return None
    if tool_name == "read_file":
        metadata = {} if metadata is None else metadata
        path_value = metadata.get("path")
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or not isinstance(end_line, int)
            or isinstance(end_line, bool)
            or end_line < start_line
        ):
            return None
    elif tool_name in {"edit_file", "write_file"}:
        path_value = arguments.get("path")
    elif tool_name == "grep":
        path_value = arguments.get("path_pattern")
        if not isinstance(path_value, str) or not is_explicit_path_pattern(path_value):
            return None
    else:
        return None
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        resolved = workspace.resolve_path(path_value.replace("\\", "/"))
        relative = resolved.relative_to(workspace.root).as_posix()
    except (OSError, ValueError, WorkspacePathError):
        return None
    if tool_name == "read_file":
        return f"file:{relative}:{start_line}-{end_line}"
    return f"file:{relative}"


def decide_runtime_policy(state: RuntimeState) -> PolicyDecision:
    if (
        state.stagnation_turns >= 2
        and not state.convergence_guided
    ):
        state.convergence_guided = True
        evidence = (
            f"same_tool_repeat={state.same_tool_repeat}, "
            f"same_result_repeat={state.same_result_repeat}, "
            f"resource_repeat={state.resource_repeat}"
        )
        return PolicyDecision(
            policy=RuntimePolicy.CONVERGENCE_GUIDANCE,
            guidance=(
                "Runtime 观察到重复主导的停滞特征（"
                f"{evidence}）。请重新评估当前假设和路径；"
                "下一步行动由你决定。"
            ,),
            notice="检测到重复停滞特征，已提示 Agent 重新评估当前路径",
        )
    return PolicyDecision(policy=RuntimePolicy.NO_INTERVENTION)


def tool_call_signature(tool_name: str, arguments: dict[str, object]) -> str:
    normalized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{tool_name}:{normalized}"


def tool_result_signature(
    *,
    tool_name: str,
    ok: bool,
    content: str,
    error: str | None,
    metadata: dict[str, object],
) -> str:
    payload = json.dumps(
        {
            "tool": tool_name,
            "ok": ok,
            "content": content,
            "error": error,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _present_pattern(values: Iterable[str | None]) -> tuple[str, ...] | None:
    pattern = tuple(value for value in values if value is not None)
    return pattern or None


def _next_repeat(
    pattern: tuple[str, ...] | None,
    previous: tuple[str, ...] | None,
    current: int,
) -> tuple[int, tuple[str, ...] | None]:
    if pattern is None:
        return 0, None
    if pattern == previous:
        return current + 1, pattern
    return 1, pattern


def _is_temporary_analysis_path(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    parts = tuple(part.casefold() for part in path.parts if part not in {"", "/"})
    if not parts:
        return False
    ignored_directories = {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".temp",
        ".tmp",
        "__pycache__",
        "build",
        "dist",
        "temp",
        "tmp",
    }
    if any(part in ignored_directories for part in parts[:-1]):
        return True
    name = parts[-1]
    return name.endswith((".tmp", ".temp"))


def resume_guidance(conversation: Conversation, user_message: str) -> str | None:
    if not user_message.strip().startswith("继续"):
        return None
    for message in reversed(conversation.get_messages()):
        if message.role != "assistant":
            continue
        return RESUME_PROMPT if RUN_CHECKPOINT_HEADING in message.content else None
    return None


def normalize_run_checkpoint(content: str) -> str:
    if content.startswith(RUN_CHECKPOINT_HEADING):
        return content
    return f"{RUN_CHECKPOINT_HEADING}\n\n{content}"


def tool_result_evidence(
    *, tool_name: str, ok: bool, content: str, error: str | None
) -> str:
    return tool_result_signature(
        tool_name=tool_name,
        ok=ok,
        content=content,
        error=error,
        metadata={},
    )
