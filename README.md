# MyCode

MyCode 是一个使用 Python 实现、面向个人开发者的可扩展终端 Coding Agent。

An extensible terminal coding agent built with Python.

它以启动命令时的当前目录作为工作区，通过 OpenAI-compatible Chat Completions 模型完成代码理解、文件修改、命令执行、会话管理和受控的 SubAgent 协作。这是一个个人项目，按“能力依赖 + 风险边界”逐步实现，不建议未经审查直接用于生产环境。

## Features / 主要能力

- Agent Loop 与结构化 Tool Calling
- 工作区内的文件查找、读取、写入和编辑
- 受路径边界和风险分级约束的命令执行
- Permission 与人工确认
- 上下文预算、历史压缩、会话持久化与跨会话记忆
- SubAgent 委派：explorer / tester / reviewer
- 通过 MCP 接入外部工具
- Skill：可复用的任务流程包
- CLI 与 Textual 终端界面

## Installation / 安装

要求：

- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 一个支持 OpenAI-compatible Chat Completions 的模型服务及其 API Key

使用 uv 安装：

```powershell
uv tool install mycode-coding-agent
```

安装后 `mycode` 可以在任意目录使用。从源码安装见 [Development](#development--开发与测试)。

## Quick Start / 快速开始

MyCode 以启动命令时的当前目录作为 Agent 工作区，所以先进入你想让 Agent 操作的项目：

```powershell
cd D:\path\to\your-project
```

配置模型连接，这三项必填，取值见 [Configuration](#configuration--配置)：

```dotenv
MYCODE_API_KEY=your-api-key
MYCODE_BASE_URL=https://your-provider.example/v1
MYCODE_MODEL=your-chat-completions-model
```

启动：

```powershell
mycode agent
```

当前 CLI 没有 `--workspace PATH` 参数，切换工作区需要先进入目标目录再启动。建议只在受信任、并且已经用 Git 管理的项目中运行。

## Usage / 常用命令

| 命令                       | 说明                                                                     |
| -------------------------- | ------------------------------------------------------------------------ |
| `mycode agent`           | 在当前目录启动 coding agent                                              |
| `mycode tui`             | 启动 Textual 终端界面                                                    |
| `mycode runtime --jsonl` | 启动机器可消费的 JSONL runtime；stdout 只输出 JSONL，诊断信息写到 stderr |
| `mycode --help`          | 查看命令列表；`mycode <子命令> --help` 查看子命令帮助                  |

`mycode agent` 的会话选项（三者互斥）：

- `--new`：跳过菜单，直接创建新会话。
- `--continue`：跳过菜单，续接当前项目最近使用的会话；没有历史会话时新建。
- `--resume SESSION_ID`：跳过菜单，续接指定的未删除会话。已永久删除的会话无法恢复。

输出选项同样互斥：`--verbose` 显示更完整的运行信息，`--debug` 显示调试级运行信息。

不指定会话选项时，如果当前项目已有历史会话，CLI 会显示交互式菜单，可以选择历史会话、新建会话、永久删除会话或退出；没有历史会话时直接新建，不显示空菜单。

在 `agent` 和 `tui` 的交互会话里可以使用 slash command：`/help`、`/new`、`/sessions`、`/resume <session_id>`、`/context`、`/compact`、`/exit`（别名 `/quit`）。

会话和记忆保存在用户目录下的 `.mycode`，并按工作区路径区分项目，同名目录不会互相覆盖：

```text
%USERPROFILE%\.mycode\projects\<workspace-basename>-<12-char-hash>\sessions\<session-id>\
%USERPROFILE%\.mycode\projects\<workspace-basename>-<12-char-hash>\MEMORY.md
%USERPROFILE%\.mycode\MEMORY.md
```

## Configuration / 配置

模型连接写在 `.env` 文件中，支持用户级和项目级两级：

```text
用户级（Windows）：%USERPROFILE%\.mycode\.env
项目级：<workspace>\.mycode\.env
```

逐字段优先级为 `process environment > project > user > defaults`，空字符串不构成有效覆盖。项目级只需要写要覆盖的字段，其余回退到用户级配置；临时覆盖用进程环境变量。项目根目录的 `<workspace>\.env` 不会被读取。

Windows PowerShell 首次配置：

```powershell
$mycodeConfigDir = Join-Path $env:USERPROFILE ".mycode"
$mycodeConfigFile = Join-Path $mycodeConfigDir ".env"
New-Item -ItemType Directory -Path $mycodeConfigDir -Force | Out-Null
if (-not (Test-Path $mycodeConfigFile)) {
    Copy-Item .env.example $mycodeConfigFile
}
```

然后编辑该文件。必填三项：`MYCODE_API_KEY`、`MYCODE_BASE_URL`、`MYCODE_MODEL`。缺少任意一项时 MyCode 会拒绝启动并指出缺少的配置。

常用可选项：

- `MYCODE_COMPACT_MODEL`、`MYCODE_SUBAGENT_MODEL`：Compact 和 SubAgent 使用的模型，留空时继承 `MYCODE_MODEL`。
- `LLM_CONTEXT_WINDOW_TOKENS`、`LLM_RESERVED_OUTPUT_TOKENS`、`LLM_CONTEXT_SAFETY_MARGIN_TOKENS`、`LLM_MEMORY_CONTEXT_TOKENS`：上下文预算，需要按模型实际的上下文窗口和输出上限设置。
- `LLM_STREAM_INCLUDE_USAGE`：要求兼容的流式模型服务返回 token 用量。
- `LLM_THINKING_ENABLED`、`LLM_REASONING_EFFORT`、`LLM_MAX_OUTPUT_TOKENS`：可选的推理配置，模型服务不支持时保持留空。

完整字段和注释见 [`.env.example`](.env.example)。

项目级 secret 放在 `<workspace>\.mycode\.env`，该文件已被 `.env` 的 Git ignore 规则排除；不要提交真实 API Key 或 token。

## MCP / Skills

### MCP

MCP 用于把外部工具接入同一个 Agent Loop，支持 `stdio` 和 `streamable_http` 两种 transport，配置分用户级和项目级：

```text
用户级：%USERPROFILE%\.mycode\mcp.json
项目级：<workspace>\.mycode\mcp.json
```

```json
{
  "mcpServers": {
    "local": {
      "transport": "stdio",
      "command": "python",
      "args": ["D:\\path\\to\\server.py"],
      "env": {"TOKEN": "${MCP_LOCAL_TOKEN}"}
    },
    "remote": {
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${MCP_REMOTE_TOKEN}"}
    }
  }
}
```

secret 不写进 `mcp.json`，而是放在 `.mycode\.env` 里用 `${ENV_VAR}` 引用。用户级配置视为你本人的主动配置；项目级 MCP 会在启动时要求一次信任确认，确认前不会建立连接。MCP 工具注册后仍然走 MyCode 的 JSON Schema 校验、Permission 和确认链，与内置工具一致。

### Skills

Skill 是可复用的任务流程包，由 `SKILL.md` 和可选的参考文件、脚本组成，按 `builtin -> user -> project` 顺序发现，同名时项目级覆盖用户级、用户级覆盖内置。内置 Skill 目前是 `database-recovery`。

```text
用户级：%USERPROFILE%\.mycode\skills\
项目级：<workspace>\.mycode\skills\
```

## Safety / 安全说明

MyCode 能够修改文件和执行命令，使用前请注意：

- 先提交或备份重要修改，并确认当前目录就是目标工作区。
- 路径和命令策略会允许、拒绝或要求人工确认工具调用，但不能替代系统级沙箱。
- 命令执行使用结构化参数、不经过 shell；需要人工确认的操作会明确提示。
- Agent 完成后用 `git status` 和 `git diff` 检查实际改动。
- MCP Server、文件和网页内容都按不可信外部输入处理，其中的指令不能覆盖 System Prompt、Permission 规则和运行时控制。
- 不要在源码、测试、提示词或 Git 历史中写入真实密钥。

## Development / 开发与测试

从源码运行：

```powershell
git clone https://github.com/mmc-cloud/mycode.git
cd mycode
uv sync
uv run mycode agent
```

把本地 checkout 安装成全局命令：

```powershell
uv tool install .
```

运行测试：

```powershell
uv run pytest
```

项目结构：

```text
mycode/          核心 Agent、CLI/TUI、工具、会话、上下文、MCP、Skill 和 SubAgent 实现
tests/           核心测试
.env.example     用户级或项目级模型配置模板
pyproject.toml   包信息、依赖、CLI 入口和测试配置
uv.lock          锁定的依赖版本
```

## Current Limitations / 当前限制

- 主要验证环境是 Windows + PowerShell；macOS/Linux 尚未充分实际验证，目前不作为受支持平台。
- 需要 Python 3.11 或更高版本。
- 不同 OpenAI-compatible Provider 对流式 usage、thinking 和 reasoning 字段的支持可能存在差异，协议行为不完全一致。
- Agent 可以修改文件和执行命令，建议在 Git 管理的项目中运行并检查 diff；输出和工具决策仍需人工审查。
- MCP 第一版不支持 OAuth、Resources、Prompts、Sampling、热加载或原生 Image/Audio ToolResult。
- 当前以终端交互为主，不包含 IDE 插件或图形界面。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
