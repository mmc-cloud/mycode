from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from mycode.conversation import Conversation


DEFAULT_MAX_TURNS = 50
DEFAULT_STAGNANT_TURN_LIMIT = 3
DEFAULT_READONLY_TURN_LIMIT = 8
DEFAULT_SOFT_CONVERGENCE_TURN_LIMIT = 2
DEFAULT_READY_ACTION_TURN_LIMIT = 4
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
    "检测到最近的工具调用没有带来新增证据，或正在重复读取相同资源。"
    "请暂停原路径并重新规划：说明缺少的唯一关键信息，选择一个能产生新证据的动作；"
    "如果信息已经充分，立即进入修改和关键验证。不要再次执行相同工具调用。"
)
COMPLETION_CORRECTION_PROMPT = (
    "你在最后一次正式修改后还没有成功执行 run_validation。刚才的结束请求已被拦截一次。"
    "请现在做一次明确决策：如果仍能推进，使用 run_validation 验证当前修改，并根据失败继续修复；"
    "如果环境阻塞、测试无法运行、依赖无法获得或已经尽力仍失败，请明确说明未完成原因和现有验证状态后结束。"
    "不要把未运行、失败或不覆盖当前修改的验证声称为通过。此提醒只会强制一次，之后允许你诚实结束。"
)
RESUME_PROMPT = (
    "这是达到上限后的续跑。请直接使用上一条“续跑检查点”衔接工作，"
    "不要重新读取已经确认的文件，不要重复输出原方案；先核对已有改动和测试状态，"
    "然后执行检查点中的剩余动作。"
)
READINESS_SOFT_CONVERGENCE_PROMPT = (
    "已连续进行较长时间的调查。优先根据现有证据实施最小修改并验证；"
    "如果确实还缺关键信息，只补充直接决定实现的证据，不要扩大搜索范围。"
    "本阶段最多再给少量调查机会，之后将进入收敛动作窗口。"
)
READINESS_ACTION_PROMPT = (
    "调查预算已经用完。接下来的有限窗口用于实施修改、运行必要项目操作和验证。"
    "不要使用 run_command 代替 read_file/grep/glob 继续扩大调查；如果现有证据仍不足以安全完成，"
    "应在窗口结束后重新规划，而不是反复绕过限制。"
)
TASK_PHASE_PROMPTS = {
    "INVESTIGATE": (
        "阶段=INVESTIGATE：只查直接证据；信息够则立即最小修改或回答，不等调查阈值。"
    ),
    "ACT": (
        "阶段=ACT：按已确认根因做最小正式修改；验证失败只修具体失败，不重新扩大调查。"
    ),
    "VERIFY": (
        "阶段=VERIFY：运行最相关验证；失败修正具体问题，成功后停止扩展工作。"
    ),
    "DONE": (
        "阶段=DONE：当前修改已有成功验证结果。若没有新的明确失败证据，停止工具调用并总结；"
        "不要为了反向证明、额外探索或无明确依据的优化再次修改。"
        "若确有新证据要求继续修改，修改后旧验证将失效，并应重新运行最相关验证。"
    ),
}

ReadinessMode = Literal[
    "open",
    "soft_convergence",
    "ready_action",
]
ToolBehavior = Literal["investigate", "act", "verify", "other"]
TaskPhase = Literal["INVESTIGATE", "ACT", "VERIFY", "DONE"]

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


@dataclass(frozen=True)
class ToolBehaviorObservation:
    behavior: ToolBehavior
    succeeded: bool
    tool_name: str | None = None


def classify_tool_behavior(
    *,
    tool_name: str,
    capability: str | None,
    arguments: dict[str, object],
    ok: bool,
    metadata: dict[str, object],
) -> ToolBehavior:
    """Classify one observed tool result without guessing arbitrary command intent."""
    if capability == "read":
        return "investigate"

    process_started = (
        metadata.get("exit_code") is not None
        or metadata.get("timed_out") is True
    )
    if tool_name == "run_command":
        category = metadata.get("command_risk_category")
        if category == "inspect":
            return "investigate"
        if category == "test" and process_started:
            return "verify"
        return "other"

    if tool_name == "run_validation" and process_started:
        return "verify"

    if tool_name in {"edit_file", "write_file"} and ok:
        path = arguments.get("path")
        if isinstance(path, str) and _is_temporary_analysis_path(path):
            return "investigate"
        return "act"

    return "other"


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
    action_turn_count: int = 0
    verification_turn_count: int = 0
    other_turn_count: int = 0
    last_tool_behaviors: frozenset[ToolBehavior] = field(default_factory=frozenset)
    task_phase: TaskPhase = "INVESTIGATE"
    task_phase_history: list[TaskPhase] = field(
        default_factory=lambda: ["INVESTIGATE"]
    )
    last_verification_succeeded: bool | None = None
    mutation_revision: int = 0
    validated_revision: int = 0
    completion_correction_used: bool = False
    done_extra_tool_turn_count: int = 0
    last_progress_reason: str = "run_started"
    _pending_readiness_notice: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _completion_correction_pending: bool = field(
        default=False,
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
    def needs_completion_correction(self) -> bool:
        return (
            self.mutation_revision > 0
            and self.validated_revision != self.mutation_revision
            and not self.completion_correction_used
        )

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

    def readiness_guidance_for_turn(
        self,
        turn_number: int,
    ) -> tuple[str | None, str | None]:
        if self.readiness_mode == "open" or self.readonly_turn_limit is None:
            return None, None

        if self.readiness_turn is None:
            self.readiness_turn = turn_number

        notice = self._pending_readiness_notice
        self._pending_readiness_notice = None
        if self.readiness_mode == "soft_convergence":
            return READINESS_SOFT_CONVERGENCE_PROMPT, notice
        return READINESS_ACTION_PROMPT, notice

    def task_phase_guidance_for_turn(self) -> str | None:
        if (
            self.readonly_turn_limit is None
            or self.task_phase == "INVESTIGATE"
        ):
            return None
        return TASK_PHASE_PROMPTS[self.task_phase]

    def take_completion_correction_guidance(self) -> str | None:
        if not self._completion_correction_pending:
            return None
        self._completion_correction_pending = False
        return COMPLETION_CORRECTION_PROMPT

    def intercept_unverified_completion_once(self) -> bool:
        if not self.needs_completion_correction:
            return False
        self.completion_correction_used = True
        self._completion_correction_pending = True
        self.last_tool_behaviors = frozenset()
        self.last_progress_reason = "completion_correction_required"
        self._reset_readonly_cycle()
        return True

    def observe_tool_turn(
        self,
        *,
        turn_number: int,
        observations: tuple[ToolBehaviorObservation, ...],
    ) -> None:
        behaviors = frozenset(observation.behavior for observation in observations)
        self.last_tool_behaviors = behaviors
        self.last_progress_reason = "tool_turn_observed"
        if "investigate" in behaviors:
            self.investigation_turn_count += 1
        if "act" in behaviors:
            self.action_turn_count += 1
        if "verify" in behaviors:
            self.verification_turn_count += 1
        if "other" in behaviors:
            self.other_turn_count += 1

        successful_edit = any(
            observation.behavior == "act" and observation.succeeded
            for observation in observations
        )
        key_test_executed = "verify" in behaviors
        successful_progress = any(
            observation.succeeded
            and observation.behavior in {"act", "verify"}
            for observation in observations
        )
        if successful_edit and self.first_edit_turn is None:
            self.first_edit_turn = turn_number
        if key_test_executed and self.first_key_test_turn is None:
            self.first_key_test_turn = turn_number

        if self.readiness_mode == "ready_action" and "investigate" in behaviors:
            self.ready_investigation_turn_count += 1
            self.last_progress_reason = "ready_investigation_observed"
        if self.task_phase == "DONE":
            self.done_extra_tool_turn_count += 1
            self.last_progress_reason = "done_extra_tool_observed"
        self._observe_task_phase_events(observations)

        if successful_progress:
            self._reset_readonly_cycle()
            return

        if self.readiness_mode == "ready_action":
            self._observe_ready_action_attempt()
            return

        if self.readiness_mode == "soft_convergence":
            self._observe_soft_convergence_attempt()
            return

        if behaviors == frozenset({"investigate"}):
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

    def observe_restricted_tool_turn(
        self,
        *,
        attempted_investigation: bool = False,
    ) -> None:
        if self.readiness_mode != "ready_action":
            return
        self.last_tool_behaviors = (
            frozenset({"investigate"})
            if attempted_investigation
            else frozenset({"other"})
        )
        self.last_progress_reason = "readiness_tool_rejected"
        if attempted_investigation:
            self.ready_investigation_turn_count += 1
            self.last_progress_reason = "ready_investigation_observed"

        self._observe_ready_action_attempt(
            retry_notice=(
                "READY 动作窗口包含越界工具，未执行；READY 仍然有效，"
                "请只使用实施和验证工具"
            )
        )

    def observe_final_answer(self) -> None:
        self.last_tool_behaviors = frozenset()
        self.last_progress_reason = "final_answer"
        self._reset_readonly_cycle()
        self._transition_task_phase("DONE")

    def observe_turn_limit(self) -> None:
        self.last_tool_behaviors = frozenset()
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
                "有限动作窗口没有产生有效修改或验证"
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
        observations: tuple[ToolBehaviorObservation, ...],
    ) -> None:
        for observation in observations:
            if observation.behavior == "act" and observation.succeeded:
                self.mutation_revision += 1
                self.last_verification_succeeded = None
                self.last_progress_reason = "formal_action_succeeded"
                self._transition_task_phase("VERIFY")
                continue

            if observation.behavior != "verify":
                continue

            self.last_verification_succeeded = observation.succeeded
            if observation.succeeded and observation.tool_name == "run_validation":
                self.validated_revision = self.mutation_revision
            if self.task_phase not in {"VERIFY", "DONE"}:
                continue
            self.last_progress_reason = (
                "validation_succeeded"
                if observation.succeeded
                else "validation_failed"
            )
            self._transition_task_phase("DONE" if observation.succeeded else "ACT")

    def _transition_task_phase(self, phase: TaskPhase) -> None:
        if phase == self.task_phase:
            return
        self.task_phase = phase
        self.task_phase_history.append(phase)


def turn_guidance(
    *,
    turn_index: int,
    max_turns: int,
    convergence_remaining_turns: int | None,
    convergence_prompt: str | None,
    task_phase_guidance: str | None,
    readiness_guidance: str | None,
    readiness_notice: str | None,
    resume_guidance: str | None,
    replan_reason: str | None,
    completion_correction_guidance: str | None,
) -> tuple[tuple[str, ...], str]:
    guidance: list[str] = []
    notices: list[str] = []
    remaining_turns = max_turns - turn_index

    if turn_index == 0 and resume_guidance is not None:
        guidance.append(resume_guidance)
        notices.append("已载入上次检查点，将直接衔接剩余工作")

    if replan_reason is not None:
        guidance.append(REPLAN_PROMPT)
        notices.append(f"{replan_reason}，已要求 Agent 重新规划")

    if completion_correction_guidance is not None:
        guidance.append(completion_correction_guidance)
        notices.append("上一次结束请求缺少当前修改的成功验证，已给予一次纠偏机会")

    if task_phase_guidance is not None:
        guidance.append(task_phase_guidance)

    if readiness_guidance is not None:
        guidance.append(readiness_guidance)
    if readiness_notice is not None:
        notices.append(readiness_notice)

    if (
        convergence_remaining_turns is not None
        and convergence_prompt is not None
        and remaining_turns <= convergence_remaining_turns
    ):
        guidance.append(convergence_prompt)
        if remaining_turns == convergence_remaining_turns:
            notices.append(
                f"剩余约 {convergence_remaining_turns} 轮，Agent 将优先收敛"
            )

    return tuple(guidance), "；".join(notices)


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
