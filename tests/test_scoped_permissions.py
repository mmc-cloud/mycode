import asyncio
import sys

import pytest

from mycode.agent import AgentModelResponse, AgentToolCall
from mycode.confirmers import TerminalConfirmer
from mycode.llm import FakeLLMClient
from mycode.permissions import ConfirmationResult, PermissionDecision, ScopedApprovalState
from mycode.runner import AgentRunner
from mycode.tools import ToolRegistry, Workspace
from mycode.tools.write_file import WriteFileTool
from mycode.tools.run_command import RunCommandTool
from test_confirmers import _confirmation_request
from test_tools_registry import AuthorizedDecisionTool, RecordingConfirmer, RecordingPermissionChecker


@pytest.mark.parametrize('scope', ['once', 'task', 'session'])
def test_confirmation_scope(scope):
    assert ConfirmationResult.approved().scope == 'once'
    assert ConfirmationResult(status='approved').scope == 'once'
    assert ConfirmationResult.approved(scope=scope).scope == scope
    assert ConfirmationResult.rejected().scope is None
    with pytest.raises(ValueError):
        ConfirmationResult(status='rejected', scope=scope)
    with pytest.raises(ValueError):
        ConfirmationResult.approved(scope='all')


@pytest.mark.parametrize('answer,scope', [
    ('y', 'once'), ('yes', 'once'), (' t ', 'task'), ('TASK', 'task'),
    ('s', 'session'), ('session', 'session'), ('n', None), ('no', None),
    ('', None), ('unknown', None), ('allow all', None),
])
def test_terminal_scopes(answer, scope):
    result = TerminalConfirmer(lambda _: answer, lambda _: None).confirm(_confirmation_request())
    assert result.scope == scope
    assert result.status == ('approved' if scope else 'rejected')


@pytest.mark.parametrize('scope', ['once', 'task', 'session'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_registry_scope_and_hard_decisions(scope, asynchronous):
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope=scope))
    checker = RecordingPermissionChecker(PermissionDecision.ask())
    tool = AuthorizedDecisionTool('probe')
    registry = ToolRegistry.from_tools([tool], confirmer=confirmer, permission_checker=checker)
    registry.scoped_approvals.begin_task()

    def call():
        if not asynchronous:
            return registry.run_tool('probe', {'text': 'ok'})
        async def run():
            return await registry.run_tool_async('probe', {'text': 'ok'}, permission_lock=asyncio.Lock())
        return asyncio.run(run())

    first, second = call(), call()
    assert first.ok and second.ok
    assert first.metadata['confirmation_source'] == 'explicit'
    assert first.metadata['confirmation_scope'] == scope
    assert len(confirmer.requests) == (2 if scope == 'once' else 1)
    assert second.metadata['confirmation_source'] == ('explicit' if scope == 'once' else 'scoped_grant')
    if scope != 'once':
        assert second.metadata['grant_scope'] == scope
    for reason in ['outside_workspace', 'unsupported_operation', 'dangerous_command']:
        checker.decision = PermissionDecision.deny(reason)
        assert not call().ok
    assert tool.run_count == 2
    checker.decision = PermissionDecision.allow()
    assert call().ok
    assert tool.run_count == 3
    assert len(checker.requests) == 6


def test_reject_creates_no_grant():
    registry = ToolRegistry.from_tools(
        [AuthorizedDecisionTool('probe')],
        permission_checker=RecordingPermissionChecker(PermissionDecision.ask()),
    )
    assert not registry.run_tool('probe', {'text': 'ok'}).ok
    assert registry.scoped_approvals.allows_ask() is None


def test_async_concurrent_asks_share_grant():
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope='task'))
    registry = ToolRegistry.from_tools(
        [AuthorizedDecisionTool('probe')], confirmer=confirmer,
        permission_checker=RecordingPermissionChecker(PermissionDecision.ask()),
    )
    async def run():
        lock = asyncio.Lock()
        return await asyncio.gather(*[
            registry.run_tool_async('probe', {'text': str(i)}, permission_lock=lock)
            for i in range(8)
        ])
    assert all(result.ok for result in asyncio.run(run()))
    assert len(confirmer.requests) == 1


def make_runner(tmp_path, scope, ending='final'):
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope=scope))
    registry = ToolRegistry.from_tools([WriteFileTool(Workspace(tmp_path))], confirmer=confirmer)
    calls = AgentModelResponse(tool_calls=[
        AgentToolCall(id=f'call-{i}', name='write_file', arguments={'path': f'{i}.txt', 'content': str(i)})
        for i in range(2)
    ])
    responses = [calls]
    if ending == 'final':
        responses += [AgentModelResponse(content='done'), calls, AgentModelResponse(content='done')]
    runner = AgentRunner(
        llm_client=FakeLLMClient(responses=[], tool_responses=responses), tool_registry=registry,
        max_turns=1 if ending == 'max_turns' else 10,
        near_limit_remaining_turns=None, near_limit_prompt=None, finalize_on_max_turns=False,
    )
    return runner, confirmer


@pytest.mark.parametrize('scope,expected', [('once', 4), ('task', 2), ('session', 1)])
def test_runner_two_tasks_and_new_registry(tmp_path, scope, expected):
    runner, confirmer = make_runner(tmp_path, scope)
    for task in ['first', 'second']:
        events = list(runner.run(task))
        assert events[-1].stop_reason == 'final_answer'
        assert not runner.tool_registry.scoped_approvals.task_approved
    assert len(confirmer.requests) == expected
    assert (tmp_path / '0.txt').read_text() == '0'
    fresh, fresh_confirmer = make_runner(tmp_path, 'once')
    list(fresh.run('fresh'))
    assert len(fresh_confirmer.requests) == 2


@pytest.mark.parametrize('ending', ['final', 'error', 'max_turns', 'close', 'throw', 'overflow'])
def test_task_cleanup_all_exits(tmp_path, ending):
    runner, _ = make_runner(tmp_path, 'task', ending)
    stream = runner.run('task')
    saw_grant = False
    stops = []
    for event in stream:
        if event.type == 'stop':
            stops.append(event.stop_reason)
        if runner.tool_registry.scoped_approvals.task_approved:
            saw_grant = True
            if ending == 'close':
                stream.close()
                break
            if ending == 'throw':
                with pytest.raises(KeyboardInterrupt):
                    stream.throw(KeyboardInterrupt())
                break
            if ending == 'overflow':
                from mycode.context_budget import ContextBudget
                runner.context_budget = ContextBudget(context_window_tokens=1, reserved_output_tokens=0, safety_margin_tokens=0)
    assert saw_grant
    if ending in {'final', 'error', 'max_turns', 'overflow'}:
        assert stops == [{'final': 'final_answer', 'error': 'model_error',
                          'max_turns': 'max_turns', 'overflow': 'context_overflow'}[ending]]
    assert not runner.tool_registry.scoped_approvals.task_approved
    assert runner.tool_registry.scoped_approvals.allows_ask() is None


def test_state_resets_only_task():
    state = ScopedApprovalState()
    state.approve('task')
    state.begin_task()
    assert state.allows_ask() is None
    state.approve('session')
    state.approve('task')
    state.end_task()
    assert state.allows_ask() == 'session'


def test_real_tools_share_scope_and_deny_directory_write(tmp_path):
    workspace = Workspace(tmp_path)
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope='task'))
    registry = ToolRegistry.from_tools([WriteFileTool(workspace), RunCommandTool(workspace)], confirmer=confirmer)
    assert registry.run_tool('write_file', {'path': 'ok.txt', 'content': 'ok'}).ok
    result = registry.run_tool('run_command', {'command': [sys.executable, '-c', 'print(14)']})
    assert result.ok
    assert result.metadata['confirmation_source'] == 'scoped_grant'
    denied = registry.run_tool('write_file', {'path': '.', 'content': 'bad'})
    assert not denied.ok
    assert denied.metadata['permission_reason'] == 'unsupported_operation'
    assert len(confirmer.requests) == 1



def test_driver_detects_new_prompt_and_preserves_once_policy():
    from evals.live_cli_driver import CONFIRMATION_PROMPTS
    prompts = []
    result = TerminalConfirmer(lambda prompt: prompts.append(prompt) or 'y', lambda _: None).confirm(_confirmation_request())
    assert prompts[0] in CONFIRMATION_PROMPTS
    assert result.scope == 'once'


@pytest.mark.parametrize('failure', [False, True])
def test_mcp_native_async_inherits_grant_and_metadata(failure):
    from mcp.types import CallToolResult, TextContent
    from mycode.mcp.tool_adapter import MCPToolAdapter
    from test_mcp_tool_adapter import make_tool
    calls = []
    async def invoke(alias, name, arguments):
        calls.append(arguments)
        if failure:
            raise RuntimeError('synthetic failure')
        return CallToolResult(content=[TextContent(text='ok')])
    tool = MCPToolAdapter('local', make_tool(), invoke)
    confirmer = RecordingConfirmer(ConfirmationResult.approved(scope='session'))
    registry = ToolRegistry.from_tools([tool], confirmer=confirmer)
    async def run():
        lock = asyncio.Lock()
        return [await registry.run_tool_async(tool.name, {'value': 'ok'}, permission_lock=lock) for _ in range(2)]
    first, second = asyncio.run(run())
    assert first.ok is not failure
    assert second.ok is not failure
    assert first.metadata['confirmation_source'] == 'explicit'
    assert second.metadata['confirmation_source'] == 'scoped_grant'
    assert second.metadata['grant_scope'] == 'session'
    assert len(confirmer.requests) == 1
    assert len(calls) == 2
