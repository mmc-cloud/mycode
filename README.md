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

在 `agent` 和 `tui` 的交互会话里可以使用 slash command：`/help`、`/new`、`/sessions`、`/resume <session_id>`、`/context`、`/compact`、`/connect`、`/model`、`/exit`（别名 `/quit`）。

`/context` 展示当前 Main 请求可见投影的预计输入量，以 `~` 标记估算；分别显示“预计输入 / 总 Context Window”的占用率，以及“预计输入 / 可用输入预算”的安全预算比例，后者已经扣除输出预留和安全余量。同时显示窗口来源、估算来源和最近一次 API 输入 Usage；最近一次 API 数字仅供参考，不是当前上下文占用或累计消耗。

TUI 在原 StatusBar 的运行状态旁常驻显示如 `~53K / 1M (5.3%)` 的摘要，使用总窗口作分母。启动/Resume、模型或窗口切换、任务完成及 Compact 后刷新；后台 worker 计算，界面不逐 Token 重算，权限等待和错误提示仍保留。Compact 后显示压缩投影；Resume 和切换模型重新估算，不使用上一会话的屏幕缓存。

估算器仍在内存中利用有效 Provider 输入 Usage 校准；缺失、布尔值或非整数等异常 Usage 不参与校准。模型/连接的有效请求配置改变时沿用已有隔离/重置逻辑，窗口调整本身不换用其他模型的样本。Resume 后恢复默认估算来源，数值可能与关闭前的校准估算不同；没有新增校准数据库、累计 Token、费用或账单统计。

会话和记忆保存在用户目录下的 `.mycode`，并按工作区路径区分项目，同名目录不会互相覆盖：

```text
%USERPROFILE%\.mycode\projects\<workspace-basename>-<12-char-hash>\sessions\<session-id>\
%USERPROFILE%\.mycode\projects\<workspace-basename>-<12-char-hash>\MEMORY.md
%USERPROFILE%\.mycode\MEMORY.md
```

## Configuration / 配置

首次启动可以沿用 `.env`。连接管理和角色模型配置也可以在会话中完成：

| 命令 | 行为 |
| --- | --- |
| `/connect` 或 `/connect list` | 列出已保存连接，隐藏凭据；`env` 是旧 `.env` 的只读连接引用 |
| `/connect add ID`、`/connect edit ID` | CLI 使用隐藏输入采集 Key/Headers；TUI 打开密码输入表单，空字段保留原值，Headers 填 `{}` 清空 |
| `/connect delete ID` | 删除连接；活动 Session、默认配置仍引用它时拒绝删除 |
| `/connect models ID` | 尝试获取模型列表；失败则明确提示可手动输入未验证的模型 ID |
| `/connect use ID MODEL` | 将当前 Main 切换到指定完整连接与模型 |
| `/connect default ID` | 保存新 Session 的默认连接；`env` 清除此项。没有完整默认角色配置时，模型仍来自旧配置 |
| `/model`、`/model edit` | 查看三角色配置；`edit` 在 CLI 提问、在 TUI 打开角色/模型表单 |
| `/model ROLE MODEL [选项]` | 修改 Main/SubAgent/Compact；ROLE 为 `main`、`subagent`、`compact` |
| `/model subagent follow`、`/model compact follow` | 恢复跟随 Main 的连接与模型；Compact 仍默认关闭 Thinking |
| `/model default`、`/model default clear` | 保存/清除完整默认角色配置，供新 Session 使用；不更改其他活动 Session |

例如，在 MyCode 的交互会话中逐条输入：

```text
/connect add deepseek
/connect models deepseek
/model main deepseek-flash connection=deepseek thinking=on effort=high
/model subagent deepseek-flash connection=follow thinking=auto window=32000 reserve=4096 margin=2048
/model compact deepseek-flash connection=deepseek thinking=off
/model default
```

不要在 slash command 参数中输入 API Key；添加/编辑连接时使用专门的隐藏输入。`connection=follow` 只让子角色的连接跟随，模型仍固定；`connection=ID` 固定引用完整连接。不同连接的 Key、URL、Headers 不会拼接继承。`.env` 与 `/model` 统一采用当前规则：`thinking=auto` 不发送 Thinking/effort 参数，`on` 不强制补 `high`，可以显式提供 `effort=low|high|max`；`off` 关闭 Thinking。需要 high 时明确配置，开发阶段的旧默认行为不再作为兼容模式保留。

`/connect models` 是辅助查询，不是模型白名单。即使列表未包含某个 Model ID，也允许手动选择；该模型缺少明确 API 限制时仍显示未验证，不借用列表中其他模型的窗口。

模型参数支持 `window=N reserve=N margin=N output=N`。`window=N` 按当前角色实际使用的 **API Connection + Model ID** 保存到现有 `connections.json`；相同组合的角色共享此窗口设置，输出预留和安全余量仍按角色独立。修改完整连接的 URL、Key 或 Headers 后，不会借用原手动窗口。切换到其他组合时重新解析窗口，切换回来复用已保存值。`window=inherit` 清除此组合的手动窗口，恢复自动选择，不是继承另一模型的窗口。

窗口只采用三个来源：当前连接 API 明确返回的可信限制、此连接/模型的用户设置、默认 **128000 tokens（128K，未验证）**。有 API 限制时，未设置手动值就采用 API 值，手动值可以更小但不能更大；没有 API 限制时，手动值同样标记为未验证。`/model` 显示三个角色的有效窗口、来源和 API 限制/未验证状态，不表示远端调用一定成功。

在现有 `/connect models ID` 中查询窗口；仅接受响应中 `context_window` / `context_window_tokens` 的正整数声明，布尔值、字符串和非正数不作可信限制。查询会在空闲时更新当前角色预算；若新限制比过去的手动设置小，实际窗口受 API 限制约束。显式设置超过已知限制的窗口会被拒绝。API 声明只保留在当前运行实例，不做元数据落盘、第三方目录或长 Prompt 探测；恢复 Session 后，未重新查询的窗口显示未验证。

切换前仍检查当前可见历史；小窗口无法容纳时要求先 `/compact` 或新建 Session，不静默裁剪 Conversation。若查询得知更小限制却无法更新预算，保留当前配置和历史，受影响角色暂停模型请求，直到修正预算或新建 Session；其他连接的摘要角色仍可用于 Compact。请求中的窗口/预留/余量继续复用原 ContextBudget，不改变 Compact 算法。

旧 `.env` 和显式注入配置中的已有窗口保留为当前组合的用户意图；不会带到新连接/新模型。编辑保存连接会清除此连接的手动窗口，避免新端点沿用旧设置；删除连接一并清理窗口。`env` 与临时注入引用按实际连接摘要隔离设置，不保存额外凭据。

连接、模型参数和预算只在 Main/SubAgent/Compact 都空闲时切换。存储失败、参数错误、客户端构造失败或预算超限时保留原配置。切换的静态校验不等于远端认证/模型能力已通过；只有模型列表或真实调用能提供相应证据。被活动 Session 引用的连接暂不允许编辑，需先切换离开或关闭相关 Session；删除前还需清除默认连接/角色引用。

用户连接与完整默认配置保存在 `%USERPROFILE%\.mycode\connections.json`，凭据仅在此处以本地文件形式保存（不是加密保险库）。写入前限制权限：Windows 使用当前用户/SYSTEM ACL，POSIX 使用 `0600`；原子替换失败不覆盖旧数据。Session 的 `model-selection.json` 只保存连接引用与角色参数。Resume 优先恢复该 Session 的选择，引用失效时明确失败，不静默替换连接。

新 Session 依次使用完整默认角色配置、默认连接加旧模型配置、旧 `.env`。保存的完整默认配置全部引用已保存连接时，即使没有 `.env` 也可启动；仍显式引用 `env` 的配置需要旧配置存在。

缺失旧配置时，从保存连接生成的启动快照不代表 `env`。Resume/Continue 中任何角色仍引用 `env` 都会明确失败，不会将其重新绑定到新 Session 的默认连接；`/connect models env` 同样拒绝查询。

TUI 的 `/model edit` 会加载当前角色的连接、模型、Thinking、Reasoning 与预算；切换角色保留各自草稿，“Save selected role”只保存当前角色。无改动保存不重建客户端；只改预算时保留已有 Thinking/effort，切换模型时不会提交未编辑的旧窗口。窗口填 `inherit` 恢复上述自动选择，其他字段的 `inherit` 保留跟随意图；`effort=auto` / `output=auto` 表示明确清除可选值。

Python 调用方的配置链路为 `ModelSelection/RoleSelection（角色意图）→ ModelConfiguration（绑定完整连接和窗口）→ 三角色 LLMConfig（有效请求参数）`。不再经过 ModelParameters、RoleModelConfig 或双向配置转换。RoleSelection 的缺省/None 表示跟随或采用当前默认，`effort="auto"`、`max_output_tokens="auto"` 明确不发送可选控制；同连接同模型可继承 Main，换模型/连接不会继承不兼容的 Thinking、Reasoning、输出限制和手动窗口。Compact 未指定 Thinking 时关闭。

显式注入完整 `ModelConfiguration(selection=..., connections=...)` 时，它优先于保存的默认配置和 Session 快照。注入连接使用保留的临时引用，不自动保存凭据；恢复这类 Session 需再次注入，或在关闭前改为已保存连接。`.env` 中未填写的子模型保持“跟随”意图，不在 LLMConfig 中重复保存展开后的名称与原始名称。

连接文件及角色选择快照只使用当前 Schema，不包含配置版本字段、版本分支或迁移机制。配置校验失败会报错并保留原文件。Context 读取暂时失败时，TUI 保留上一次成功摘要；新 Session 使用自己的摘要，不沿用其他 Session 的值。

### JSONL 模型与连接管理

`mycode runtime --jsonl` 在原有 `turn`、`context_status`、`compact`、`close` 之外支持结构化 `model` / `connect` 输入；协议仍为 version 1。每行一个 JSON 对象，新管理请求必须带非空字符串 `request_id`，结果以同一个 ID 返回 `model_result` / `connect_result`。参数直接用 JSON 字段，不传 slash command 或 `key=value` 字符串。

| 输入 type | action | 参数与行为 |
| --- | --- | --- |
| `model` | `status`（默认） | 查询三个角色的配置意图及有效模型、连接、Thinking、effort、窗口和预算 |
| `model` | `set` | 必填 `role`、`model`；可选 `connection`、`thinking`、`effort`、`window`、`reserve`、`margin`、`output` |
| `model` | `default` / `clear_default` | 保存当前三个角色为新 Session 默认配置，或清除此默认选择 |
| `connect` | `list`（默认） | 查询连接 ID、Base URL、只读状态及默认连接，不返回认证字段 |
| `connect` | `add` | 必填 `id`、`base_url`、`api_key`；可选 `headers` 对象 |
| `connect` | `edit` | 必填 `id` 和至少一个待更新字段；省略字段保持原值，`headers:{}` 清空 Headers |
| `connect` | `delete` / `default` / `models` | 必填 `id`，分别删除连接、设置新 Session 默认连接、查询当前 API 的模型列表；`default` 的 `id:"env"` 清除保存连接默认值 |
| `connect` | `use` | 必填 `id`、`model`，将 Main 切换至该完整连接和模型；更多角色参数使用 `model set` |

`role` 为 `main` / `subagent` / `compact`。SubAgent/Compact 的 `model:null` 表示跟随 Main；不带其他选项时重置整个角色为跟随。`connection:null` 表示连接跟随 Main，Main 必须保持具体连接。未提供的选项按现有模型管理规则保留，切换模型/连接时仍会清除不兼容参数。

`thinking` 接受 `auto` / `on` / `off`，`effort` 接受 `auto` / `low` / `high` / `max`，两者的 `null` 表示继承/缺省。`window`、`reserve`、`margin`、`output` 使用整数（不接受布尔值或数字字符串）；`null` 清除对应覆盖。`window:null` 清除当前连接/模型的共享手动窗口；`output:"auto"` 明确省略输出限制。窗口上限、历史预算检查和跟随关系与 CLI/TUI 共用规则。

```jsonl
{"version":1,"type":"model","request_id":"status-1"}
{"version":1,"type":"connect","request_id":"connections-1","action":"list"}
{"version":1,"type":"connect","request_id":"add-1","action":"add","id":"work","base_url":"https://api.example.com/v1","api_key":"YOUR_LOCAL_KEY","headers":{}}
{"version":1,"type":"model","request_id":"main-1","action":"set","role":"main","model":"your-model-id","connection":"work","thinking":"on","effort":"low","window":64000,"reserve":8192,"margin":4096}
{"version":1,"type":"model","request_id":"worker-1","action":"set","role":"subagent","model":null}
```

成功结果包含 `ok:true` 和操作数据。`model_result.roles` 按角色给出 `selection` 原始意图，以及有效 `connection`、`model`、`thinking`、`effort`、`window`、`window_source`、`server_limit`、`reserve`、`margin`、`output`。连接结果只返回允许展示的字段；`default_connection` 表示新 Session 实际默认使用的 Main 连接，优先取 `model default` 保存的 Main 连接，其次取 `connect default`，最后回退 `env`，不表示当前活动 Session 的连接。模型列表只返回 `id` 与可信 `context_window`。

失败结果使用 `ok:false` 和 `error:{code,message}`。错误码包括 `invalid_parameters`、`configuration_rejected`、`operation_failed`、`runtime_busy`；缺失或错误类型的请求 ID 返回 `request_id:null`。校验失败后继续接收下一条输入，不回显输入凭据或底层 SDK 异常。模型列表请求不可用时按现有业务返回 `ok:true,available:false,models:[],warning:"model_list_unavailable"`，仍可手动指定 Model ID；空列表与失败不会清除之前的可信限制。

启动仍需有效的 `.env` 或已保存的完整默认角色配置。JSONL 顺序处理输入，活动 turn 期间通常会在 turn 完成后读取下一条管理请求；权限/MCP 确认等待中读到管理请求时返回 `runtime_busy`，不执行或排队。实际修改仍受共享忙闲门禁、存储锁、活动引用及失败回退保护。切换不重建 Session/Conversation/Tool/MCP/Compact；保存的默认配置只影响新 Session，Resume 恢复当前 Session 的选择。API Key、Headers 值不进入管理结果或诊断输出。

旧 `.env` 支持用户级和项目级两级：

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
