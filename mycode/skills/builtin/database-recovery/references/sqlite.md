# SQLite Recovery Reference

本参考适用于 SQLite 数据库损坏、WAL/journal 恢复、异常中断、文件截断和数据 salvage。

## 1. 首先保护 SQLite evidence set

在使用 `sqlite3`、Python `sqlite3` 或其他 SQLite API 打开原始数据库之前，先检查同目录下与数据库相关的文件。

重点包括：

```text
<database>
<database>-wal
<database>-journal
<database>-shm
```

文件扩展名不一定是 `.db`，不要根据扩展名假定数据库身份。

### WAL

`<database>-wal` 可能包含已经提交、但尚未 checkpoint 回主数据库的事务。

WAL 是 SQLite 数据库持久状态的一部分。

如果 WAL 存在：

* 将主数据库和对应 WAL 视为同一数据库状态的一部分；
* 不要只复制主数据库而遗漏 WAL；
* 不要删除、重命名或把 WAL 与其他数据库错配。

主数据库与 WAL 分离可能导致已提交事务丢失，甚至造成数据库损坏。

### Rollback journal

`<database>-journal` 可能是 hot journal。

如果异常中断发生在事务过程中，SQLite 可能依靠 hot journal 把数据库恢复到事务开始前的一致状态。

因此：

* journal 存在时与主数据库一起保存；
* 不要在保护证据前删除、重命名或替换 journal。

### SHM

`<database>-shm` 是 WAL-index/shared-memory 辅助文件。

它本身不保存数据库持久数据，SQLite 可以根据 WAL 重建 WAL-index，因此 crash recovery 不依赖原始 SHM 内容。

如果现场存在，可以随 evidence set 一起保存以保留完整现场，但不要把它当作主要恢复数据来源。

## 2. 不要先打开原始数据库“看看”

SQLite 数据库访问并不总是无副作用调查。

Rollback-journal 模式下，在读取数据库之前，SQLite 会检查是否存在 hot journal；如果存在，会先自动执行 rollback，使数据库恢复到一致状态。

因此，在 original evidence 尚未保护之前，避免直接对原始数据库执行：

```text
sqlite3 <database>
Python sqlite3.connect(...)
PRAGMA integrity_check
PRAGMA quick_check
.dump
.recover
VACUUM
REINDEX
journal_mode 修改
checkpoint
```

即使某个操作看起来主要用于读取或检查，也应优先在 working copy 上执行。

原始现场调查优先使用普通文件系统能力，例如：

* 列出目录；
* 查看文件名；
* 查看文件大小和时间；
* 检查相关 sidecar 是否存在；
* 必要时检查文件 header 或原始字节。

## 3. 建立分层副本

建议保持：

```text
original evidence
      ↓
preserved evidence
      ↓
working copy
      ↓
recovered output
```

其中：

* `original evidence`：不修改；
* `preserved evidence`：作为重新尝试恢复的干净起点；
* `working copy`：允许 SQLite 自动 recovery、rollback、checkpoint 或其他状态变化；
* `recovered output`：最终交付结果。

如果存在 WAL 或 journal，创建 preserved evidence 和 working copy 时保持它们与数据库文件的原有 basename 和目录关系。

## 4. 优先尝试 SQLite 正常恢复

完成 evidence preservation 后，再在 working copy 上使用 SQLite。

### WAL 场景

如果：

```text
database
database-wal
```

同时存在，应保持二者正确配对，然后让 SQLite 在 working copy 上正常打开数据库。

打开后检查：

* schema 是否可读；
* 用户需要的数据是否可见；
* WAL 中已经提交的数据是否反映在查询结果中。

不要为了“清理文件”先手工删除 WAL。

如果正常打开已经获得完整、正确的数据，应优先把当前逻辑数据库状态导出或备份为新的 recovered output，而不是继续修改 preserved evidence。

### Rollback-journal 场景

如果：

```text
database
database-journal
```

同时存在，应保持二者正确配对。

在 working copy 上正常打开数据库时，SQLite 可能自动识别 hot journal 并进行 rollback。

这是正常的 crash-recovery 行为。

自动恢复后再检查数据库，不要先手工删除 journal。

## 5. 检查数据库状态

在 working copy 或 recovered output 上，可以根据情况使用：

```sql
PRAGMA quick_check;
PRAGMA integrity_check;
```

并检查：

```text
.schema
.tables
```

以及任务真正关心的数据。

`integrity_check` 返回正常，只能说明 SQLite 结构层面没有发现相应问题，不能证明用户要求的数据已经正确恢复。

因此还必须进行 task-specific semantic verification。

## 6. 结构损坏时再考虑 `.recover`

如果正常 SQLite 访问失败、数据库存在结构损坏，或者普通 `.dump` 无法完整读取，可以考虑 SQLite CLI 的：

```text
.recover
```

典型流程：

```text
working corrupt database
        ↓
.recover
        ↓
SQL output
        ↓
新的 recovered database
```

`.recover` 是 salvage 机制，而不是精确恢复保证。

损坏数据库有时可以完全恢复，但并不能假定一定能够完整恢复。

`.recover` 可能产生无法归属到原表的数据，例如 `lost_and_found` 中的 orphaned rows。

因此 `.recover` 输出必须重新执行：

```text
integrity / quick check
schema validation
task-specific data validation
```

不要因为 `.recover` 命令成功执行就直接宣布任务完成。

## 7. 原始 evidence 上避免的操作

在 original evidence 上避免：

```text
删除 -wal
删除 -journal
只复制 DB 而遗漏存在的 WAL/journal
替换或错配 WAL/journal
VACUUM
REINDEX
修改 journal_mode
主动 checkpoint
覆盖原数据库进行恢复
```

不要因为某个 sidecar “看起来没用”就在理解其角色之前删除它。

## 8. 恢复成功判定

只有至少满足以下条件时，才将任务视为成功：

1. 原始 evidence 已得到保护；
2. 恢复过程没有覆盖原始输入；
3. recovered output 可以正常读取；
4. SQLite 结构检查合理；
5. 用户要求恢复的数据经过明确查询或其他方式得到验证。

如果只能 salvage 部分数据，应明确报告部分恢复，而不是描述为完整恢复。
