from pathlib import Path

import pytest

from mycode.config import (
    DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS,
    DEFAULT_LLM_CONTEXT_WINDOW_TOKENS,
    DEFAULT_LLM_MEMORY_CONTEXT_TOKENS,
    DEFAULT_LLM_RESERVED_OUTPUT_TOKENS,
    LLMConfig,
    MissingLLMConfigError,
    _RedactedConfigValues,
    load_layered_environment,
    load_llm_config,
)

CONFIG_ENVIRONMENT_NAMES = (
    "MYCODE_API_KEY",
    "MYCODE_BASE_URL",
    "MYCODE_MODEL",
    "MYCODE_COMPACT_MODEL",
    "MYCODE_SUBAGENT_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "LLM_CONTEXT_WINDOW_TOKENS",
    "LLM_RESERVED_OUTPUT_TOKENS",
    "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
    "LLM_MEMORY_CONTEXT_TOKENS",
    "LLM_STREAM_INCLUDE_USAGE",
    "LLM_THINKING_ENABLED",
    "LLM_REASONING_EFFORT",
    "LLM_MAX_OUTPUT_TOKENS",
)


@pytest.fixture(autouse=True)
def clear_llm_environment(monkeypatch) -> None:
    for name in CONFIG_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def write_env(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_valid_env(path: Path, *lines: str) -> None:
    write_env(
        path,
        "MYCODE_API_KEY=test-key",
        "MYCODE_BASE_URL=https://example.com/v1",
        "MYCODE_MODEL=test-model",
        *lines,
    )


def test_config_value_mapping_repr_is_redacted() -> None:
    values = _RedactedConfigValues({"MYCODE_API_KEY": "synthetic-secret"})

    assert repr(values) == "<redacted config values>"


def test_layered_environment_preserves_priority_and_empty_fallback(tmp_path) -> None:
    user_env = tmp_path / "user.env"
    write_env(
        user_env,
        "PROCESS_WINS=user",
        "PROJECT_WINS=user",
        "USER_FALLBACK=user",
    )
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    write_env(
        project_env,
        "PROCESS_WINS=project",
        "PROJECT_WINS=project",
        "USER_FALLBACK=",
    )

    values = load_layered_environment(
        user_env,
        workspace_root=workspace,
        environ={
            "PROCESS_WINS": "process",
            "PROJECT_WINS": "   ",
            "USER_FALLBACK": "",
        },
    )

    assert values == {
        "PROCESS_WINS": "process",
        "PROJECT_WINS": "project",
        "USER_FALLBACK": "user",
    }


def test_llm_config_repr_does_not_include_api_key() -> None:
    config = LLMConfig(
        api_key="synthetic-secret",
        base_url="https://example.com/v1",
        model="test-model",
    )

    assert "synthetic-secret" not in repr(config)


def test_load_llm_config_reads_user_env_file(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "MYCODE_API_KEY=test-key",
        "MYCODE_BASE_URL=https://example.com/v1",
        "MYCODE_MODEL=test-model",
    )

    config = load_llm_config(env_file)

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.com/v1"
    assert config.model == "test-model"
    assert config.compact_model == "test-model"
    assert config.subagent_model == "test-model"


def test_load_llm_config_reads_project_env_without_user_env(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    write_valid_env(project_env)

    config = load_llm_config(
        tmp_path / "missing-user.env",
        workspace_root=workspace,
    )

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.com/v1"
    assert config.model == "test-model"


def test_project_env_overrides_user_and_falls_back_for_missing_or_empty_values(
    tmp_path,
) -> None:
    user_env = tmp_path / "user.env"
    write_env(
        user_env,
        "MYCODE_API_KEY=user-key",
        "MYCODE_BASE_URL=https://user.example.com/v1",
        "MYCODE_MODEL=user-model",
        "MYCODE_COMPACT_MODEL=user-compact",
        "MYCODE_SUBAGENT_MODEL=user-subagent",
        "LLM_CONTEXT_WINDOW_TOKENS=128000",
        "LLM_RESERVED_OUTPUT_TOKENS=8192",
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS=4096",
        "LLM_MEMORY_CONTEXT_TOKENS=2048",
        "LLM_STREAM_INCLUDE_USAGE=true",
        "LLM_THINKING_ENABLED=false",
    )
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    write_env(
        project_env,
        "MYCODE_API_KEY=",
        "MYCODE_MODEL=project-model",
        "MYCODE_COMPACT_MODEL=project-compact",
        "LLM_CONTEXT_WINDOW_TOKENS=256000",
        "LLM_RESERVED_OUTPUT_TOKENS=16384",
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS=8192",
        "LLM_MEMORY_CONTEXT_TOKENS=1024",
        "LLM_STREAM_INCLUDE_USAGE=false",
        "LLM_THINKING_ENABLED=true",
        "LLM_REASONING_EFFORT=max",
        "LLM_MAX_OUTPUT_TOKENS=4096",
    )

    config = load_llm_config(user_env, workspace_root=workspace)

    assert config.api_key == "user-key"
    assert config.base_url == "https://user.example.com/v1"
    assert config.model == "project-model"
    assert config.compact_model == "project-compact"
    assert config.subagent_model == "user-subagent"
    assert config.context_window_tokens == 256000
    assert config.reserved_output_tokens == 16384
    assert config.context_safety_margin_tokens == 8192
    assert config.memory_context_tokens == 1024
    assert config.stream_include_usage is False
    assert config.thinking_enabled is True
    assert config.reasoning_effort == "max"
    assert config.max_output_tokens == 4096


def test_load_llm_config_uses_defaults_for_optional_values(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(env_file)

    config = load_llm_config(env_file)

    assert config.base_url == "https://example.com/v1"
    assert config.model == "test-model"
    assert config.compact_model == "test-model"
    assert config.subagent_model == "test-model"
    assert config.context_window_tokens == DEFAULT_LLM_CONTEXT_WINDOW_TOKENS
    assert config.reserved_output_tokens == DEFAULT_LLM_RESERVED_OUTPUT_TOKENS
    assert (
        config.context_safety_margin_tokens
        == DEFAULT_LLM_CONTEXT_SAFETY_MARGIN_TOKENS
    )
    assert config.memory_context_tokens == DEFAULT_LLM_MEMORY_CONTEXT_TOKENS
    assert config.stream_include_usage is True
    assert config.thinking_enabled is None
    assert config.reasoning_effort is None
    assert config.max_output_tokens is None


def test_load_llm_config_reads_deepseek_reasoning_values(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_THINKING_ENABLED=true",
        "LLM_REASONING_EFFORT=max",
        "LLM_MAX_OUTPUT_TOKENS=4096",
    )

    config = load_llm_config(env_file)

    assert config.thinking_enabled is True
    assert config.reasoning_effort == "max"
    assert config.max_output_tokens == 4096


def test_llm_config_defaults_enabled_thinking_effort_to_high() -> None:
    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="deepseek-v4-flash",
        thinking_enabled=True,
    )

    assert config.reasoning_effort == "high"


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh", "invalid"])
def test_load_llm_config_rejects_unsupported_reasoning_effort(
    tmp_path,
    effort: str,
) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_THINKING_ENABLED=true",
        f"LLM_REASONING_EFFORT={effort}",
    )

    with pytest.raises(ValueError, match="high, max"):
        load_llm_config(env_file)


def test_load_llm_config_rejects_reasoning_effort_without_thinking(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_REASONING_EFFORT=max",
    )

    with pytest.raises(ValueError, match="requires LLM_THINKING_ENABLED=true"):
        load_llm_config(env_file)


def test_load_llm_config_rejects_output_limit_above_reserved_budget(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_RESERVED_OUTPUT_TOKENS=1024",
        "LLM_MAX_OUTPUT_TOKENS=2048",
    )

    with pytest.raises(ValueError, match="must not exceed"):
        load_llm_config(env_file)


def test_load_llm_config_reads_explicit_role_models(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "MYCODE_API_KEY=test-key",
        "MYCODE_BASE_URL=https://example.com/v1",
        "MYCODE_MODEL=main-model",
        "MYCODE_COMPACT_MODEL=compact-model",
        "MYCODE_SUBAGENT_MODEL=subagent-model",
    )

    config = load_llm_config(env_file)

    assert config.model == "main-model"
    assert config.compact_model == "compact-model"
    assert config.subagent_model == "subagent-model"


def test_load_llm_config_ignores_empty_role_models(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "MYCODE_API_KEY=test-key",
        "MYCODE_BASE_URL=https://example.com/v1",
        "MYCODE_MODEL=main-model",
        "MYCODE_COMPACT_MODEL=",
        "MYCODE_SUBAGENT_MODEL=   ",
    )

    config = load_llm_config(env_file)

    assert config.compact_model == "main-model"
    assert config.subagent_model == "main-model"


def test_load_llm_config_reads_unchanged_context_budget_values(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_CONTEXT_WINDOW_TOKENS=200000",
        "LLM_RESERVED_OUTPUT_TOKENS=16000",
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS=8000",
        "LLM_MEMORY_CONTEXT_TOKENS=1024",
        "LLM_STREAM_INCLUDE_USAGE=false",
    )

    config = load_llm_config(env_file)

    assert config.context_window_tokens == 200000
    assert config.reserved_output_tokens == 16000
    assert config.context_safety_margin_tokens == 8000
    assert config.memory_context_tokens == 1024
    assert config.stream_include_usage is False


def test_load_llm_config_rejects_context_reserves_that_consume_window(
    tmp_path,
) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_CONTEXT_WINDOW_TOKENS=100",
        "LLM_RESERVED_OUTPUT_TOKENS=60",
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS=40",
    )

    with pytest.raises(ValueError, match="leave at least 1 input token"):
        load_llm_config(env_file)


def test_load_llm_config_allows_disabling_memory_context(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_MEMORY_CONTEXT_TOKENS=0",
    )

    config = load_llm_config(env_file)

    assert config.memory_context_tokens == 0


def test_load_llm_config_rejects_negative_memory_context_budget(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_valid_env(
        env_file,
        "LLM_MEMORY_CONTEXT_TOKENS=-1",
    )

    with pytest.raises(ValueError, match="LLM_MEMORY_CONTEXT_TOKENS"):
        load_llm_config(env_file)


def test_load_llm_config_uses_default_user_file_and_ignores_cwd_env(
    tmp_path,
    monkeypatch,
) -> None:
    user_env_file = tmp_path / "user.env"
    write_env(
        user_env_file,
        "MYCODE_API_KEY=user-key",
        "MYCODE_BASE_URL=https://user.example.com/v1",
        "MYCODE_MODEL=user-model",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_env(
        workspace / ".env",
        "MYCODE_API_KEY=workspace-key",
        "MYCODE_BASE_URL=https://workspace.example.com/v1",
        "MYCODE_MODEL=workspace-model",
    )
    monkeypatch.setattr(
        "mycode.config.default_user_env_file",
        lambda: user_env_file,
    )

    config = load_llm_config(workspace_root=workspace)

    assert config.api_key == "user-key"
    assert config.base_url == "https://user.example.com/v1"
    assert config.model == "user-model"


def test_load_llm_config_prefers_process_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MYCODE_API_KEY", "process-key")
    monkeypatch.setenv("MYCODE_BASE_URL", "https://process.example.com/v1")
    monkeypatch.setenv("MYCODE_MODEL", "process-model")
    monkeypatch.setenv("MYCODE_COMPACT_MODEL", "process-compact")
    monkeypatch.setenv("MYCODE_SUBAGENT_MODEL", "process-subagent")
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "MYCODE_API_KEY=user-key",
        "MYCODE_BASE_URL=https://user.example.com/v1",
        "MYCODE_MODEL=user-model",
        "MYCODE_COMPACT_MODEL=user-compact",
        "MYCODE_SUBAGENT_MODEL=user-subagent",
    )
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    write_env(
        project_env,
        "MYCODE_API_KEY=project-key",
        "MYCODE_BASE_URL=https://project.example.com/v1",
        "MYCODE_MODEL=project-model",
        "MYCODE_COMPACT_MODEL=project-compact",
        "MYCODE_SUBAGENT_MODEL=project-subagent",
    )

    config = load_llm_config(env_file, workspace_root=workspace)

    assert config.api_key == "process-key"
    assert config.base_url == "https://process.example.com/v1"
    assert config.model == "process-model"
    assert config.compact_model == "process-compact"
    assert config.subagent_model == "process-subagent"


def test_load_llm_config_ignores_empty_process_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MYCODE_API_KEY", "")
    monkeypatch.setenv("MYCODE_BASE_URL", "   ")
    monkeypatch.setenv("MYCODE_MODEL", "")
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "MYCODE_API_KEY=user-key",
        "MYCODE_BASE_URL=https://user.example.com/v1",
        "MYCODE_MODEL=user-model",
    )

    config = load_llm_config(env_file)

    assert config.api_key == "user-key"
    assert config.base_url == "https://user.example.com/v1"
    assert config.model == "user-model"


def test_load_llm_config_does_not_accept_legacy_openai_names(tmp_path) -> None:
    env_file = tmp_path / "user.env"
    write_env(
        env_file,
        "OPENAI_API_KEY=legacy-key",
        "OPENAI_BASE_URL=https://legacy.example.com/v1",
        "OPENAI_MODEL=legacy-model",
    )

    with pytest.raises(MissingLLMConfigError, match="MYCODE_API_KEY"):
        load_llm_config(env_file)


@pytest.mark.parametrize(
    ("missing_name", "lines"),
    [
        (
            "MYCODE_API_KEY",
            (
                "MYCODE_API_KEY=",
                "MYCODE_BASE_URL=https://example.com/v1",
                "MYCODE_MODEL=test-model",
            ),
        ),
        (
            "MYCODE_BASE_URL",
            (
                "MYCODE_API_KEY=test-key",
                "MYCODE_BASE_URL=",
                "MYCODE_MODEL=test-model",
            ),
        ),
        (
            "MYCODE_MODEL",
            (
                "MYCODE_API_KEY=test-key",
                "MYCODE_BASE_URL=https://example.com/v1",
                "MYCODE_MODEL=",
            ),
        ),
    ],
)
def test_load_llm_config_requires_connection_values(
    tmp_path,
    missing_name: str,
    lines: tuple[str, ...],
) -> None:
    env_file = tmp_path / "user.env"
    write_env(env_file, *lines)

    with pytest.raises(MissingLLMConfigError, match=missing_name):
        load_llm_config(env_file)
