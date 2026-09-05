# MyCode

MyCode 是一个使用 Python 实现的轻量级 coding agent。它以当前目录为工作区，通过 OpenAI-compatible 模型完成代码理解、文件修改、命令执行、会话管理和受控的 SubAgent 协作。

这是一个个人学习和工程实践项目，不建议未经审查直接用于生产环境。

## 主要能力

- OpenAI-compatible Chat Completions 流式调用和结构化 Tool Calling
- 工作区内的文件读取、搜索、写入和编辑
- 带路径边界、风险分级和确认机制的命令执行
- 按项目保存、选择、恢复和删除会话
- 上下文管理、历史压缩和持久记忆
- 受限的 SubAgent 探索、验证与审查
- 通过 MCP stdio 或 Streamable HTTP 接入外部工具
- 多级 CLI 运行信息

## 环境要求

- 当前已验证的运行环境：Windows + PowerShell
- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 一个支持 Chat Completions 的 OpenAI-compatible 模型服务及其 API Key

## 安装

下载或克隆仓库后，在 MyCode 仓库根目录执行：

```powershell
uv tool install .
```

安装成功后，`mycode` 命令可以在其他目录中使用：

```powershell
mycode
```

## 配置模型

MyCode 支持用户级和项目级模型配置：

```text
用户级：%USERPROFILE%\.mycode\.env
项目级：<workspace>\.mycode\.env
```

`mycode agent` 按 `process environment > project > user > defaults` 逐字段取值；空字符串不构成有效覆盖。项目级路径严格使用 Agent 已确定的 workspace root，不向父目录搜索。项目根目录的 `<workspace>\.env` 仍不会读取，只有 `<workspace>\.mycode\.env` 是项目配置入口。`mycode chat` 暂无 workspace 语义，因此只使用进程环境、用户级配置和默认值。

Windows PowerShell 首次配置：

```powershell
$mycodeConfigDir = Join-Path $env:USERPROFILE ".mycode"
$mycodeConfigFile = Join-Path $mycodeConfigDir ".env"
New-Item -ItemType Directory -Path $mycodeConfigDir -Force | Out-Null
if (-not (Test-Path $mycodeConfigFile)) {
    Copy-Item .env.example $mycodeConfigFile
}
```

然后编辑该文件。以下三项均为必填，必须将占位值替换为支持 OpenAI-compatible Chat Completions 的模型服务地址、模型名称和凭证：

```dotenv
MYCODE_API_KEY=your-api-key
MYCODE_BASE_URL=https://your-provider.example/v1
MYCODE_MODEL=your-chat-completions-model
```

缺少其中任意一项时，MyCode 都会拒绝启动并指出缺少的配置。完整配置项及说明参见 [`.env.example`](.env.example)。项目可以只覆盖少数字段，其余字段回退到用户级配置。项目 secret 应放在 `<workspace>\.mycode\.env`，该文件已由现有 `.env` Git ignore 规则排除；不要提交真实 API Key 或 token。

## 配置 MCP 工具

可选 MCP 配置位于用户级 `%USERPROFILE%\.mycode\mcp.json` 和项目级 `<workspace>\.mycode\mcp.json`。两份文件按 Server alias 合并：不同 alias 全部保留，同名 alias 由项目级覆盖；任一实际存在的文件无效都会报告配置错误。两份文件都缺少时 MCP 保持关闭。每个 transport 必须显式声明；secret 放在用户级或项目级 `.mycode\.env`，配置中只引用 `${ENV_VAR}`：

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

MCP secret 使用与 Agent 模型配置一致的 `process environment > project .mycode/.env > user .mycode/.env` 优先级，空值不会清空较低层的有效值。`mcp.json` 可以纳入 Git，`.mycode/skills/` 也仍是项目资产；不要忽略整个 `.mycode/` 目录。

用户级 `mcp.json` 视为用户主动配置，stdio 与 Streamable HTTP 都默认可信。自动从项目 `mcp.json` 发现的所有 Server 都必须先通过独立的 `MCP trust>` 整体确认；这同时保护 stdio 本地进程和 HTTP 网络连接、header secret 发送。确认默认拒绝且只有 `y/yes` 才批准。拒绝会过滤整个 project MCP 层，只保留 user MCP；若项目 Server 覆盖了同名 user alias，该 alias 会回退到用户配置。显式通过 Python API 注入的 `mcp_config=` 视为调用方已批准，不触发项目级确认。

批准状态原子写入用户目录 `%USERPROFILE%\.mycode\mcp-trust.json`，格式为 `{"version": 1, "projects": {"<ProjectIdentity.key>": "<SHA-256>"}}`。canonical JSON 指纹覆盖整份项目 MCP：stdio 包含 alias、transport、解析前后 command/args、未解析 env 定义与 connect/tool timeout；HTTP 包含 alias、transport、解析前后 URL、未解析 headers 定义与两个 timeout。集合或任一执行字段变化都会重新确认。文件只保存 digest，不保存 URL、command、参数、header、env 或 secret；旧的 stdio-only digest 无法授权后来加入的 HTTP。信任文件缺失表示未信任；JSON/schema 损坏会 fail closed、警告并重新确认；用户已批准但写入失败时，本次仍启用并警告下次会再次询问。

统一确认会用安全 repr 展示 stdio 的未解析 command/args 和 env key。HTTP 展示未解析 URL 模板的安全形式、从解析后 URL 提取的 `scheme://host[:port]` destination，以及 header key；不会显示 URL userinfo、query、fragment、resolved header value 或 token。确认发生在构造实际 effective config 与 `MCPManager.start()` 之前，未信任的项目 HTTP 不会进入 client、DNS、初始化或 discovery 路径。

启动 `mycode agent` 时会并行连接各 Server、一次性分页发现工具并显示各自状态；单个 Server 失败不会阻止 Agent 启动。失败状态会显示经过安全归类的简短摘要，例如连接超时、命令不存在或 HTTP 401/403，不显示 header、token、query 或响应正文。本 Runtime 内工具快照固定，修改配置后需退出并重启。模型侧工具名使用 `mcp__<server_alias>__<safe_remote_name>`，不合法字符或超长名称会附加稳定短 hash，协议调用仍使用 Server 的原始工具名。

MCP 工具仍经过 MyCode 的 JSON Schema、Permission 与确认链。`destructiveHint=true` 映射为 write/high；明确 `readOnlyHint=true` 且非 destructive 时映射为 read/low，`openWorldHint` 不会把明确只读工具改写为 write；缺失或未知信息保守走既有确认。所有 annotations 都只是风险提示，不是可信授权。退出时 Manager 会限时等待连接清理；Python 线程不能被强杀，超时会保留线程引用并记录结构化失败状态，而不会假称关闭成功。

## 在目标项目中运行

MyCode 将启动命令时的**当前目录**作为 Agent 工作区。安装命令后，先进入想让 Agent 操作的项目：

```powershell
cd D:\path\to\your-project
mycode agent
```

当前 CLI 没有 `--workspace PATH` 参数。切换工作区时，需要先进入目标目录再启动；建议只在受信任且已经使用 Git 的项目中运行。

### 查看命令帮助

```powershell
mycode --help
mycode agent --help
```

帮助命令也可以简写为 `-h`。`mycode agent` 启动 coding agent，`mycode chat` 启动不带 coding tools 的普通模型对话。会话默认保存在 `%USERPROFILE%\.mycode\state.sqlite3`，并按工作区路径区分项目。

### Agent 会话启动方式

直接运行 `mycode agent` 时，如果当前项目存在历史会话，CLI 会显示交互式菜单，供你选择历史会话、创建新会话、永久删除会话或退出；如果当前项目没有历史会话，则自动创建新会话，不显示空菜单。

也可以使用会话选项跳过菜单：

- `mycode agent --new`：直接创建新会话。
- `mycode agent --continue`：续接当前项目最近使用的会话；没有历史会话时创建新会话。
- `mycode agent --resume SESSION_ID`：续接指定的未删除会话。

`--resume` 只用于续接仍然存在的会话，不支持恢复已经永久删除的会话。上述三个会话选项互斥，不能同时使用。

## 安全边界

MyCode 能够修改文件和执行命令。使用前请注意：

- 先提交或备份重要修改，并确认当前目录就是目标工作区。
- 路径和命令策略会允许、拒绝或要求人工确认工具调用，但不能替代系统级沙箱。
- Agent 完成后使用 `git status` 和 `git diff` 检查实际改动。
- 不要在源码、测试、提示词或 Git 历史中写入真实密钥。

## 运行测试

在 MyCode 仓库根目录执行：

```powershell
uv run pytest
```

当前公开版本已在 Windows、Python 3.11 环境中验证。

## 项目结构

```text
mycode/          核心 Agent、CLI、工具、会话、上下文和 SubAgent 实现
tests/           可公开的核心测试
.env.example     用户级或项目级模型配置模板
pyproject.toml   包信息、依赖、CLI 入口和测试配置
uv.lock          锁定的依赖版本
```

## 当前限制

- 当前以 CLI 交互为主，不包含 IDE 插件或图形界面。
- 源码包含部分 POSIX 兼容实现，但 macOS/Linux 尚未经过实际验证，目前不作为受支持平台。
- 不同 OpenAI-compatible 服务对流式 usage、thinking 和 reasoning 字段的支持可能不同。
- Agent 的输出和工具决策仍需要人工审查，不能替代代码评审和测试。
- MCP 第一版不支持 OAuth、Resources、Prompts、Sampling、热加载、复杂管理命令或原生 Image/Audio ToolResult。
