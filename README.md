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

MyCode 默认从用户配置目录读取模型配置，而不是读取项目根目录的 `.env`：

```text
%USERPROFILE%\.mycode\.env
```

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

缺少其中任意一项时，MyCode 都会拒绝启动并指出缺少的配置。完整配置项及说明参见 [`.env.example`](.env.example)。也可以直接设置优先级更高的进程环境变量。不要把真实 API Key 写入项目或提交到 Git。

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
.env.example     用户级模型配置模板
pyproject.toml   包信息、依赖、CLI 入口和测试配置
uv.lock          锁定的依赖版本
```

## 当前限制

- 当前以 CLI 交互为主，不包含 IDE 插件或图形界面。
- 源码包含部分 POSIX 兼容实现，但 macOS/Linux 尚未经过实际验证，目前不作为受支持平台。
- 不同 OpenAI-compatible 服务对流式 usage、thinking 和 reasoning 字段的支持可能不同。
- Agent 的输出和工具决策仍需要人工审查，不能替代代码评审和测试。
