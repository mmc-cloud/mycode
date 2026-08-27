from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from mycode.conversation import Conversation
from mycode.tools.patterns import is_explicit_path_pattern
from mycode.tools.validation_command import analyze_validation_command
from mycode.tools.workspace import Workspace, WorkspacePathError


DEFAULT_MAX_TURNS = 50
DEFAULT_STAGNANT_TURN_LIMIT = 3
DEFAULT_READONLY_TURN_LIMIT = 8
DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT = 2
DEFAULT_READY_ACTION_TURN_LIMIT = 4
RESOURCE_STAGNATION_WINDOW = 7
RESOURCE_STAGNATION_REPEAT = 5
COMPLETION_CORRECTION_EXTRA_TURNS = 2
MAIN_CONVERGENCE_REMAINING_TURNS = 5
RUN_CHECKPOINT_HEADING = "## 续跑检查点"

MAX_TURNS_FINALIZATION_PROMPT = (
    "本轮可用的模型轮次已经用完。请停止继续探索，不要请求或假设任何新的工具结果。"
    "请仅根据当前对话中已有的工具结果，用中文生成可直接供下一次续跑使用的检查点。"
    f"必须以“{RUN_CHECKPOINT_HEADING}”开头，并依次包含：已确认事实、已读取文件及范围、"
    "已修改文件、测试状态、剩余动作、当前阻塞。没有内容的栏目也要写“无”。"
    "不要声称读取过实际未读取的文件，并检查数量、列表和文件名是否前后一致。"
    "内容要简洁、可执行，不要重新展开方案论证；运行状态将由 CLI 单独提示。"
)
MAIN_CONVERGENCE_PROMPT = (
    "本轮即将达到工具调用上限。立即收敛：先判断安全修改所需信息是否齐全；"
    "若齐全，停止扩展调查并实施最小修改和关键验证；若不足，只读取直接相关的关键文件；"
    "若无法安全完成，明确阻塞并保留当前进度。不要重复输出方案或重新读取已确认内容。"
)
REPLAN_PROMPT = (
    "检测到最近的工具调用没有带来有效推进，或正在重复原路径。"
    "请暂停当前路径并重新规划：如果仍缺少关键证据，只选择一个能够产生新信息的直接动作；"
    "如果已有足够依据，则进行必要修改、验证，或形成结论并结束。"
    "不要重复相同的失败动作或无信息增益的调查路径。"
)
COMPLETION_CORRECTION_PROMPT = (
    "当前 revision 发生了修改，但还没有通过 Runtime 认可的最终验证。"
    "如果当前任务可以执行验证，请优先使用 `run_validation` 运行与修改相关的最终验证；"
    "不要仅因为之前通过 `run_command` 做过检查或临时测试就直接结束。"
    "如果当前任务无需执行验证（例如分析或 Review），或客观上无法执行验证，"
    "可基于已有证据说明情况并完成任务。"
)
RESUME_PROMPT = (
    "这是达到上限后的续跑。请直接使用上一条“续跑检查点”衔接工作，"
    "不要重新读取已经确认的文件，不要重复输出原方案；先核对已有改动和测试状态，"
    "然后执行检查点中的剩余动作。"
)
READINESS_SOFT_CONVERGENCE_PROMPT = (
    "已经进行了较长调查，请整合现有证据并决定下一步。"
    "若依据充分，请修改、验证或形成结论并结束；"
    "只有存在具体信息缺口时才继续调查，不要扩大搜索范围。"
)
READINESS_ACTION_PROMPT = (
    "开放式调查暂时停止。请基于已有信息收敛：修改、验证、恢复既有 Artifact、"
    "形成结论并结束，或执行其他非开放式调查的必要动作。"
)
TASK_PHASE_PROMPTS = {
    "INVESTIGATE": (
        "只查直接证据；信息够则立即最小修改或回答，不等调查阈值。"
    ),
    "ACT": (
        "按已确认根因做最小正式修改；验证失败只修具体失败，不重新扩大调查。"
    ),
    "VERIFY": (
        "运行最相关验证；失败修正具体问题，成功后停止扩展工作。"
    ),
    "VALIDATED": (
        "当前 revision 已成功验证。如果没有明确未完成的用户需求、缺陷或验证缺口，"
        "应结束任务；只有已经识别出具体剩余问题时才继续调用工具。"
    ),
}

ReadinessMode = Literal[
    "open",
    "soft_convergence",
    "ready_action",
]
ToolEffect = Literal["investigate", "rehydrate", "mutate", "validate", "neutral"]
ToolBehavior = Literal["investigate", "rehydrate", "act", "verify", "other"]
TaskPhase = Literal["INVESTIGATE", "ACT", "VERIFY", "VALIDATED"]
DecisionTarget = Literal["INVESTIGATE", "ACT", "VERIFY", "VALIDATED", "CONVERGE"]
ToolPolicy = Literal["open", "block_investigate"]
CompletionDecision = Literal["accept", "correct"]

_TEMPORARY_ANALYSIS_DIRECTORIES = {
    ".temp",
    ".tmp",
    "scratch",
    "temp",
    "tmp",
}
_TEMPORARY_ANALYSIS_STEMS = {
    "debug",
    "repro",
    "reproduction",
    "scratch",
    "temp",
    "tmp",
}
_TEST_DIRECTORIES = {"test", "tests"}
_MUTATION_COMMANDS = frozenset(
    {
        "add-content",
        "copy",
        "copy-item",
        "cp",
        "del",
        "erase",
        "md",
        "mkdir",
        "move",
        "move-item",
        "mv",
        "new-item",
        "out-file",
        "rd",
        "remove-item",
        "ren",
        "rename",
        "rm",
        "rmdir",
        "set-content",
        "touch",
    }
)
_COMMAND_CHAIN_TOKENS = frozenset({"&", "&&", "|", "||", ";", ">", ">>", "<"})


@dataclass(frozen=True, init=False)
class ToolObservation:
    effect: ToolEffect
    succeeded: bool
    validation_passed: bool | None = None
    tool_name: str | None = None
    resource_key: str | None = None

    def __init__(
        self,
        effect: ToolEffect | None = None,
        succeeded: bool = False,
        validation_passed: bool | None = None,
        tool_name: str | None = None,
        resource_key: str | None = None,
        *,
        behavior: ToolBehavior | None = None,
    ) -> None:
        if effect is None:
            if behavior is None:
                raise TypeError("effect is required")
            effect = {
                "investigate": "investigate",
                "rehydrate": "rehydrate",
                "act": "mutate",
                "verify": "validate",
                "other": "neutral",
            }[behavior]
            if effect == "validate" and validation_passed is None:
                validation_passed = succeeded
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "succeeded", succeeded)
        object.__setattr__(self, "validation_passed", validation_passed)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "resource_key", resource_key)


ToolBehaviorObservation = ToolObservation


@dataclass(frozen=True)
class TurnDirective:
    target_phase: DecisionTarget
    reason: str
    guidance: str

    def render(self) -> str:
        return (
            "Current state:\n"
            f"target_phase: {self.target_phase}\n"
            f"reason: {self.reason}\n\n"
            "Guidance:\n"
            f"{self.guidance}"
        )


@dataclass(frozen=True)
class RuntimeDecision:
    guidance: tuple[str, ...]
    notice: str
    tool_policy: ToolPolicy


def observe_tool_result(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
    ok: bool,
    metadata: dict[str, object],
    workspace: Workspace | None = None,
) -> ToolObservation:
    """Describe one tool result without making a Runtime decision."""
    effect = classify_tool_effect(
        tool_name=tool_name,
        capability=capability,
        arguments=arguments,
    )
    return ToolObservation(
        effect=effect,
        succeeded=ok,
        validation_passed=(
            _validation_result(metadata) if effect == "validate" else None
        ),
        tool_name=tool_name,
        resource_key=classify_resource_key(
            tool_name=tool_name,
            arguments=arguments,
            workspace=workspace,
        ),
    )


def classify_resource_key(
    *,
    tool_name: str,
    arguments: dict[str, object],
    workspace: Workspace | None,
) -> str | None:
    """Return one high-confidence workspace resource identity, or None."""
    if workspace is None:
        return None

    path_value: object
    if tool_name == "read_file":
        path_value = arguments.get("path")
    elif tool_name == "grep":
        path_value = arguments.get("path_pattern")
        if not isinstance(path_value, str) or not is_explicit_path_pattern(
            path_value
        ):
            return None
    else:
        return None

    if not isinstance(path_value, str) or path_value.strip() == "":
        return None

    try:
        resolved_path = workspace.resolve_path(path_value.replace("\\", "/"))
    except (OSError, WorkspacePathError):
        return None
    if not resolved_path.is_file():
        return None

    relative_path = resolved_path.relative_to(workspace.root).as_posix()
    return f"file:{relative_path}"


def classify_tool_effect(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
) -> ToolEffect:
    """Classify high-confidence Runtime effects independently of permission risk."""
    if tool_name == "read_artifact":
        return "rehydrate"
    if capability == "read":
        return "investigate"

    if tool_name == "run_command":
        command = arguments.get("command")
        if not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ):
            return "neutral"
        return _classify_command_effect(command)

    if tool_name == "run_validation":
        return "validate"

    if tool_name in {"edit_file", "write_file"}:
        path = arguments.get("path")
        if isinstance(path, str) and _is_temporary_analysis_path(path):
            return "investigate"
        return "mutate"

    return "neutral"


def blocks_investigation_for_policy(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
) -> bool:
    """Identify open-ended investigation for ToolPolicy without changing State facts."""
    if tool_name == "delegate_task":
        return arguments.get("role") in {"explorer", "reviewer"}
    return classify_tool_effect(
        tool_name=tool_name,
        capability=capability,
        arguments=arguments,
    ) == "investigate"


def classify_tool_behavior(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
    ok: bool,
    metadata: dict[str, object],
) -> ToolBehavior:
    """Compatibility view for callers that still expose the older telemetry names."""
    del ok, metadata
    effect = classify_tool_effect(
        tool_name=tool_name,
        capability=capability,
        arguments=arguments,
    )
    return {
        "investigate": "investigate",
        "rehydrate": "rehydrate",
        "mutate": "act",
        "validate": "verify",
        "neutral": "other",
    }[effect]


def _validation_result(metadata: dict[str, object]) -> bool | None:
    exit_code = metadata.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code == 0
    return None


def _classify_command_effect(command: list[str]) -> ToolEffect:
    if not command:
        return "neutral"

    validation = analyze_validation_command(command)
    if validation.allowed:
        return "validate"

    lowered = [part.casefold() for part in command]
    executable = _command_name(lowered[0])
    unwrapped = _unwrap_known_mutation_command(lowered, executable=executable)
    if unwrapped is not None:
        lowered = unwrapped
        executable = _command_name(lowered[0])

    if executable == "sed":
        return (
            "mutate"
            if any(part == "-i" or part.startswith("--in-place") for part in lowered[1:])
            else "investigate"
        )
    if executable in {
        "cat",
        "diff",
        "dir",
        "file",
        "find",
        "findstr",
        "get-childitem",
        "grep",
        "head",
        "ls",
        "rg",
        "select-string",
        "stat",
        "strings",
        "tail",
        "type",
        "wc",
        "xxd",
    }:
        return "investigate"
    if executable == "git" and _is_readonly_git_command(lowered):
        return "investigate"
    if executable in _MUTATION_COMMANDS:
        return "mutate"
    return "neutral"


def _command_name(value: str) -> str:
    executable = PurePosixPath(value.replace("\\", "/")).name
    for suffix in (".exe", ".cmd", ".bat"):
        if executable.endswith(suffix):
            return executable[: -len(suffix)]
    return executable


def _unwrap_known_mutation_command(
    command: list[str],
    *,
    executable: str,
) -> list[str] | None:
    if executable in {"pwsh", "powershell"}:
        wrapper_flag = "-command"
    elif executable == "cmd":
        wrapper_flag = "/c"
    else:
        return None

    if len(command) < 3 or command[1] != wrapper_flag:
        return None
    inner = command[2:]
    if any(token in _COMMAND_CHAIN_TOKENS for token in inner):
        return None
    if any(character.isspace() for character in inner[0]):
        return None
    if _command_name(inner[0]) not in _MUTATION_COMMANDS:
        return None
    return inner


def _is_readonly_git_command(command: list[str]) -> bool:
    action = next((part for part in command[1:] if not part.startswith("-")), None)
    return action in {"branch", "diff", "log", "show", "status"}


def _is_temporary_analysis_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    lowered_parts = tuple(part.lower() for part in path.parts if part not in {"/", ""})
    if not lowered_parts:
        return False

    parent_parts = lowered_parts[:-1]
    if any(part in _TEMPORARY_ANALYSIS_DIRECTORIES for part in parent_parts):
        return True

    name = lowered_parts[-1]
    stem = PurePosixPath(name).stem
    if stem in _TEMPORARY_ANALYSIS_STEMS:
        return True
    if any(
        stem.startswith(f"{prefix}_")
        for prefix in {"repro", "reproduction", "scratch", "temp", "tmp"}
    ):
        return True
    if stem.startswith("test_repro") and not any(
        part in _TEST_DIRECTORIES for part in parent_parts
    ):
        return True
    return name.endswith(".tmp")


@dataclass
class RunProgress:
    stagnant_turn_limit: int = DEFAULT_STAGNANT_TURN_LIMIT
    readonly_turn_limit: int | None = DEFAULT_READONLY_TURN_LIMIT
    soft_convergence_turn_limit: int = DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT
    ready_action_turn_limit: int = DEFAULT_READY_ACTION_TURN_LIMIT
    seen_evidence: set[str] = field(default_factory=set)
    stagnant_turn_count: int = 0
    pending_replan_reason: str | None = None
    consecutive_readonly_turns: int = 0
    readiness_turn: int | None = None
    first_edit_turn: int | None = None
    first_key_test_turn: int | None = None
    readiness_mode: ReadinessMode = "open"
    soft_convergence_turn_count: int = 0
    ready_action_turn_count: int = 0
    ready_investigation_turn_count: int = 0
    investigation_turn_count: int = 0
    rehydration_turn_count: int = 0
    action_turn_count: int = 0
    verification_turn_count: int = 0
    other_turn_count: int = 0
    last_tool_effects: frozenset[ToolEffect] = field(default_factory=frozenset)
    task_phase: TaskPhase = "INVESTIGATE"
    task_phase_history: list[TaskPhase] = field(
        default_factory=lambda: ["INVESTIGATE"]
    )
    last_verification_succeeded: bool | None = None
    mutation_revision: int = 0
    validated_revision: int = 0
    completion_correction_revision: int | None = None
    post_validation_tool_turn_count: int = 0
    recent_investigation_resources: deque[str] = field(
        default_factory=lambda: deque(maxlen=RESOURCE_STAGNATION_WINDOW)
    )
    last_stagnant_resource: str | None = None
    last_progress_reason: str = "run_started"
    _pending_readiness_notice: str | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def readiness_active(self) -> bool:
        return self.readiness_mode != "open"

    @property
    def readiness_to_edit_gap(self) -> int | None:
        if self.readiness_turn is None or self.first_edit_turn is None:
            return None
        return self.first_edit_turn - self.readiness_turn

    @property
    def has_unvalidated_mutation(self) -> bool:
        return self.mutation_revision > self.validated_revision

    def observe_evidence(self, evidence: set[str]) -> None:
        has_new_evidence = any(item not in self.seen_evidence for item in evidence)
        self.seen_evidence.update(evidence)
        if has_new_evidence:
            self.stagnant_turn_count = 0
            return

        self.stagnant_turn_count += 1
        if self.stagnant_turn_count >= self.stagnant_turn_limit:
            self.request_replan("连续多轮没有新增工具证据")
            self.stagnant_turn_count = 0

    def request_replan(self, reason: str) -> None:
        if self.pending_replan_reason is None:
            self.pending_replan_reason = reason

    def take_replan_reason(self) -> str | None:
        reason = self.pending_replan_reason
        self.pending_replan_reason = None
        return reason

    def take_readiness_notice(self, turn_number: int) -> str | None:
        if self.readiness_mode == "open" or self.readonly_turn_limit is None:
            return None

        if self.readiness_turn is None:
            self.readiness_turn = turn_number

        notice = self._pending_readiness_notice
        self._pending_readiness_notice = None
        return notice

    def record_completion_correction(self) -> None:
        if not self.has_unvalidated_mutation:
            raise ValueError("Current revision has no unvalidated mutation.")
        self.completion_correction_revision = self.mutation_revision
        self.last_tool_effects = frozenset()
        self.last_progress_reason = "completion_correction_required"

    def observe_tool_turn(
        self,
        *,
        turn_number: int,
        observations: tuple[ToolObservation, ...],
    ) -> None:
        phase_before_turn = self.task_phase
        effects = frozenset(observation.effect for observation in observations)
        self.last_tool_effects = effects
        self.last_progress_reason = "tool_turn_observed"
        if "investigate" in effects:
            self.investigation_turn_count += 1
        if "rehydrate" in effects:
            self.rehydration_turn_count += 1
        if "mutate" in effects:
            self.action_turn_count += 1
        if "validate" in effects:
            self.verification_turn_count += 1
        if "neutral" in effects:
            self.other_turn_count += 1

        successful_edit = any(
            observation.effect == "mutate" and observation.succeeded
            for observation in observations
        )
        key_test_executed = "validate" in effects
        successful_progress = any(
            observation.succeeded
            and observation.effect in {"mutate", "validate"}
            for observation in observations
        )
        if successful_edit and self.first_edit_turn is None:
            self.first_edit_turn = turn_number
        if key_test_executed and self.first_key_test_turn is None:
            self.first_key_test_turn = turn_number

        if self.readiness_mode == "ready_action" and "investigate" in effects:
            self.ready_investigation_turn_count += 1
            self.last_progress_reason = "ready_investigation_observed"
        if self.task_phase == "VALIDATED":
            self.post_validation_tool_turn_count += 1
            self.last_progress_reason = "post_validation_tool_observed"
        self._observe_resource_activity(
            observations,
            phase_before_turn=phase_before_turn,
        )
        self._observe_task_phase_events(observations)

        if successful_progress:
            self._reset_readonly_cycle()
            return

        if self.task_phase != "INVESTIGATE":
            return

        if self.readiness_mode == "ready_action":
            if effects and effects <= {"rehydrate"}:
                return
            self._observe_ready_action_attempt()
            return

        if self.readiness_mode == "soft_convergence":
            if effects == frozenset({"investigate"}):
                self._observe_soft_convergence_attempt()
            return

        if effects == frozenset({"investigate"}):
            self.consecutive_readonly_turns += 1
            if (
                self.readonly_turn_limit is not None
                and self.consecutive_readonly_turns >= self.readonly_turn_limit
            ):
                self._transition_readiness(
                    "soft_convergence",
                    (
                        f"已连续只读 {self.readonly_turn_limit} 轮，"
                        "进入短暂收敛提醒，优先实施和验证"
                    ),
                )

    def observe_final_answer(self) -> None:
        self.last_tool_effects = frozenset()
        self.last_progress_reason = "final_answer"
        self._reset_readonly_cycle()

    def observe_turn_limit(self) -> None:
        self.last_tool_effects = frozenset()
        self.last_progress_reason = "max_turns"
        self._reset_readonly_cycle()

    def _reset_readonly_cycle(self) -> None:
        self.consecutive_readonly_turns = 0
        self.readiness_mode = "open"
        self.soft_convergence_turn_count = 0
        self.ready_action_turn_count = 0
        self._pending_readiness_notice = None

    def _observe_soft_convergence_attempt(self) -> None:
        self.soft_convergence_turn_count += 1
        self.last_progress_reason = "soft_convergence_observed"
        if self.soft_convergence_turn_count >= self.soft_convergence_turn_limit:
            self._transition_readiness(
                "ready_action",
                (
                    "调查预算已经用完，进入有限动作窗口；"
                    "请实施修改、运行必要命令和验证，或在窗口结束后重新规划"
                ),
            )

    def _observe_ready_action_attempt(
        self,
        *,
        retry_notice: str | None = None,
    ) -> None:
        self.ready_action_turn_count += 1
        if self.ready_action_turn_count >= self.ready_action_turn_limit:
            self._reset_readonly_cycle()
            self.request_replan(
                "有限收敛窗口没有产生明确进展或完成结果"
            )
            self.last_progress_reason = "readiness_action_window_exhausted"
            return
        if retry_notice is not None:
            self._pending_readiness_notice = retry_notice

    def _transition_readiness(
        self,
        mode: ReadinessMode,
        notice: str,
    ) -> None:
        if mode == "soft_convergence" and self.readiness_mode != "soft_convergence":
            self.soft_convergence_turn_count = 0
        if mode == "ready_action" and self.readiness_mode != "ready_action":
            self.ready_action_turn_count = 0
        self.readiness_mode = mode
        self._pending_readiness_notice = notice

    def _observe_task_phase_events(
        self,
        observations: tuple[ToolObservation, ...],
    ) -> None:
        for observation in observations:
            if observation.effect == "mutate" and observation.succeeded:
                self.mutation_revision += 1
                self.last_verification_succeeded = None
                self.last_progress_reason = "formal_action_succeeded"
                self._transition_task_phase("VERIFY")
                self._assert_progress_invariants()
                continue

            if observation.effect != "validate":
                continue

            self.last_verification_succeeded = observation.validation_passed
            if (
                observation.validation_passed is True
                and self.mutation_revision > 0
            ):
                self.validated_revision = self.mutation_revision
                self.last_progress_reason = "validation_succeeded"
                self._transition_task_phase("VALIDATED")
                self._assert_progress_invariants()
                continue
            if (
                observation.validation_passed is False
                and self.mutation_revision > 0
            ):
                if self.validated_revision == self.mutation_revision:
                    self.validated_revision = 0
                if self.completion_correction_revision == self.mutation_revision:
                    self.completion_correction_revision = None
                self.last_progress_reason = "validation_failed"
                self._transition_task_phase("ACT")
            self._assert_progress_invariants()

    def _observe_resource_activity(
        self,
        observations: tuple[ToolObservation, ...],
        *,
        phase_before_turn: TaskPhase,
    ) -> None:
        successful_mutation = any(
            observation.effect == "mutate" and observation.succeeded
            for observation in observations
        )
        effective_validation = any(
            observation.effect == "validate"
            and observation.validation_passed is not None
            for observation in observations
        )
        if successful_mutation or effective_validation:
            self.recent_investigation_resources.clear()
            return

        if (
            self.readonly_turn_limit is None
            or phase_before_turn != "INVESTIGATE"
            or self.readiness_mode == "ready_action"
        ):
            return

        for observation in observations:
            if (
                observation.effect != "investigate"
                or not observation.succeeded
                or observation.resource_key is None
            ):
                continue
            self.recent_investigation_resources.append(observation.resource_key)
            stagnant_resource = self._stagnant_resource()
            if stagnant_resource is None:
                continue
            self.last_stagnant_resource = stagnant_resource
            self.request_replan(
                "最近的调查反复集中在"
                f" `{stagnant_resource.removeprefix('file:')}`，"
                "期间没有出现实施或验证推进"
            )
            self.recent_investigation_resources.clear()
            return

    def _stagnant_resource(self) -> str | None:
        if len(self.recent_investigation_resources) < RESOURCE_STAGNATION_WINDOW:
            return None
        resource, count = Counter(
            self.recent_investigation_resources
        ).most_common(1)[0]
        return resource if count >= RESOURCE_STAGNATION_REPEAT else None

    def _assert_progress_invariants(self) -> None:
        assert self.validated_revision <= self.mutation_revision
        if self.task_phase == "VALIDATED":
            assert self.mutation_revision > 0
            assert self.validated_revision == self.mutation_revision
        if self.task_phase == "VERIFY":
            assert self.mutation_revision > self.validated_revision

    def _transition_task_phase(self, phase: TaskPhase) -> None:
        if phase == self.task_phase:
            return
        self.task_phase = phase
        self.task_phase_history.append(phase)
        if phase != "INVESTIGATE":
            self.recent_investigation_resources.clear()
            self._reset_readonly_cycle()


def decide_completion_request(progress: RunProgress) -> CompletionDecision:
    """Accept a terminal request unless this revision needs its one correction."""
    if not progress.has_unvalidated_mutation:
        return "accept"
    if progress.completion_correction_revision == progress.mutation_revision:
        return "accept"
    return "correct"


def resolve_runtime_decision(
    *,
    progress: RunProgress,
    turn_index: int,
    max_turns: int,
    convergence_remaining_turns: int | None,
    convergence_prompt: str | None,
    readiness_notice: str | None,
    resume_guidance: str | None,
    completion_correction_guidance: str | None,
) -> RuntimeDecision:
    task_phase_guidance = (
        TASK_PHASE_PROMPTS[progress.task_phase]
        if progress.readonly_turn_limit is not None
        and progress.task_phase != "INVESTIGATE"
        else None
    )
    readiness_guidance = None
    if progress.task_phase == "INVESTIGATE":
        if progress.readiness_mode == "soft_convergence":
            readiness_guidance = READINESS_SOFT_CONVERGENCE_PROMPT
        elif progress.readiness_mode == "ready_action":
            readiness_guidance = READINESS_ACTION_PROMPT

    guidance, notice = turn_guidance(
        turn_index=turn_index,
        max_turns=max_turns,
        convergence_remaining_turns=convergence_remaining_turns,
        convergence_prompt=convergence_prompt,
        task_phase=progress.task_phase,
        task_phase_guidance=task_phase_guidance,
        readiness_guidance=readiness_guidance,
        readiness_notice=readiness_notice,
        resume_guidance=resume_guidance,
        replan_reason=progress.pending_replan_reason,
        completion_correction_guidance=completion_correction_guidance,
    )
    tool_policy: ToolPolicy = (
        "block_investigate"
        if completion_correction_guidance is None
        and progress.task_phase == "INVESTIGATE"
        and progress.readiness_mode == "ready_action"
        else "open"
    )
    return RuntimeDecision(
        guidance=guidance,
        notice=notice,
        tool_policy=tool_policy,
    )


def turn_guidance(
    *,
    turn_index: int,
    max_turns: int,
    convergence_remaining_turns: int | None,
    convergence_prompt: str | None,
    task_phase: TaskPhase,
    task_phase_guidance: str | None,
    readiness_guidance: str | None,
    readiness_notice: str | None,
    resume_guidance: str | None,
    replan_reason: str | None,
    completion_correction_guidance: str | None,
) -> tuple[tuple[str, ...], str]:
    notices: list[str] = []
    remaining_turns = max_turns - turn_index
    convergence_active = (
        convergence_remaining_turns is not None
        and convergence_prompt is not None
        and remaining_turns <= convergence_remaining_turns
    )

    directive: TurnDirective | None = None
    if completion_correction_guidance is not None:
        directive = TurnDirective(
            target_phase="VERIFY",
            reason=_reason_with_replan(
                "completion_correction_required",
                replan_reason,
            ),
            guidance=_with_replan_correction(
                completion_correction_guidance,
                replan_reason=replan_reason,
                target_phase="VERIFY",
            ),
        )
        notices.append("上一次结束请求缺少当前修改的成功验证，已给予一次纠偏机会")
    elif task_phase_guidance is not None:
        directive = TurnDirective(
            target_phase=task_phase,
            reason=_reason_with_replan(
                f"task_phase_{task_phase.lower()}",
                replan_reason,
            ),
            guidance=_with_convergence_scope(
                _with_replan_correction(
                    task_phase_guidance,
                    replan_reason=replan_reason,
                    target_phase=task_phase,
                ),
                convergence_active=convergence_active,
            ),
        )
    elif readiness_guidance is not None:
        directive = TurnDirective(
            target_phase="CONVERGE",
            reason=_reason_with_replan(
                (
                    "investigation_budget_exhausted"
                    if readiness_guidance == READINESS_ACTION_PROMPT
                    else "investigation_converging"
                ),
                replan_reason,
            ),
            guidance=_with_convergence_scope(
                _with_replan_correction(
                    readiness_guidance,
                    replan_reason=replan_reason,
                    target_phase="CONVERGE",
                ),
                convergence_active=convergence_active,
            ),
        )
    elif replan_reason is not None:
        directive = TurnDirective(
            target_phase="INVESTIGATE",
            reason="stagnation_detected",
            guidance=f"{REPLAN_PROMPT}\n触发事实：{replan_reason}。",
        )
    elif turn_index == 0 and resume_guidance is not None:
        directive = TurnDirective(
            target_phase=task_phase,
            reason="session_resumed",
            guidance=resume_guidance,
        )
        notices.append("已载入上次检查点，将直接衔接剩余工作")
    elif convergence_active:
        assert convergence_prompt is not None
        directive = TurnDirective(
            target_phase=task_phase,
            reason="turn_budget_converging",
            guidance=convergence_prompt,
        )

    if replan_reason is not None:
        notices.append(f"{replan_reason}，已要求 Agent 调整方案")

    if readiness_notice is not None:
        notices.append(readiness_notice)

    if convergence_active:
        if remaining_turns == convergence_remaining_turns:
            notices.append(
                f"剩余约 {convergence_remaining_turns} 轮，Agent 将优先收敛"
            )

    return (
        () if directive is None else (directive.render(),),
        "；".join(notices),
    )


def _with_convergence_scope(
    guidance: str,
    *,
    convergence_active: bool,
) -> str:
    if not convergence_active:
        return guidance
    return (
        f"{guidance}\n"
        "剩余轮次有限；围绕上述唯一目标，只执行直接相关动作，不扩大范围。"
    )


def _reason_with_replan(reason: str, replan_reason: str | None) -> str:
    if replan_reason is None:
        return reason
    return f"{reason}_with_replan"


def _with_replan_correction(
    guidance: str,
    *,
    replan_reason: str | None,
    target_phase: DecisionTarget,
) -> str:
    if replan_reason is None:
        return guidance

    target = {
        "INVESTIGATE": "获取直接相关的新证据",
        "ACT": "完成必要的修改",
        "VERIFY": "完成对当前修改的验证",
        "VALIDATED": "基于现有结果收尾",
        "CONVERGE": "基于已有证据收敛",
    }[target_phase]
    return (
        f"{guidance}\n"
        f"纠偏：上一动作没有推进（{replan_reason}）。"
        f"不要重复相同的失败动作；调整方案后继续{target}。"
    )


def resume_guidance(
    conversation: Conversation,
    user_message: str,
) -> str | None:
    if not user_message.strip().startswith("继续"):
        return None
    for message in reversed(conversation.get_messages()):
        if message.role != "assistant":
            continue
        if RUN_CHECKPOINT_HEADING in message.content:
            return RESUME_PROMPT
        return None
    return None


def normalize_run_checkpoint(content: str) -> str:
    if content.startswith(RUN_CHECKPOINT_HEADING):
        return content
    return f"{RUN_CHECKPOINT_HEADING}\n\n{content}"


def tool_result_evidence(
    *,
    tool_name: str,
    ok: bool,
    content: str,
    error: str | None,
) -> str:
    payload = json.dumps(
        {
            "tool": tool_name,
            "ok": ok,
            "content": content,
            "error": error,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
