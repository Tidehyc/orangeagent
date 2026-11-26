# OrangeAgent

[English](#english) | [中文](#中文)

---

## English

A pure-Rust, single-binary Coding Agent & Harness combining minimalist model-surface ergonomics (`Pi`) with production-grade runtime concurrency, prompt-cache discipline, native OS sandboxing, and recoverable execution (`CodeFlow`, `Codex`, `OpenCode`, `Claude Code`).

### Architecture Overview

OrangeAgent is organized as a modular Cargo workspace with strict one-way dependencies:

- `orangeagent-ai` *(M1 ready)* — Multi-provider streaming wire protocols (OpenAI Chat Completions compatible; Anthropic / Responses / Gemini planned), thinking-block sanitization, structured error verdicts, and deterministic VCR record/replay.
- `orangeagent-core` *(M1 ready)* — Minimal event-driven ReAct loop (`AgentLoop`), `AgentEvent` / `EventSink`, `ToolRegistry` (`ReadOnlyParallel` vs `ExclusiveSequential`), bounded malformed-JSON/empty recovery, mid-turn `SteerQueue`, 6-phase `AgentHook` chain, and hierarchical `CancellationToken` trees.
- `orangeagent-harness` *(M1 ready — core 6 tools)* — `WorkspaceContext` containment, `ReadFileTracker` anti-blind-edit guard, `read`, `write` (atomic `rename`), `edit` (multi-hunk + fuzzy indent + unified diff), `bash` (timeout + PGID cleanup + head/tail truncation), and in-process `grep` / `glob`; dual-mode `subagent` + `worktree`, `ask_user`, Shadow-Git checkpoints, LSP diagnostics, `AGENTS.md`/`SKILL.md`, and lazy-disclosure MCP planned in M2/M3.
- `orangeagent-spine` *(M1 ready — tree JSONL store)* — Tree-structured append-only JSONL session store (`id` / `parent_id`, `/branch` time travel, torn-tail recovery); `Lane` serialization, `OriginPools`, `BusyPolicy`, SQLite WAL/FTS5 derived index, and Daemon/ACP transports scheduled for M2/M4.
- `orangeagent-cli` *(M1 ready — `run` mode)* — Single-binary CLI entrypoint (`orangeagent run` with streaming terminal UI, headless `--json` JSONL events, `--session`, and `--vcr-replay`/`--vcr-record`); `ratatui` inline TUI, Daemon, and ACP scheduled for M3/M4.
- `orangeagent-context` *(Planned — Phase M2)* — Prompt-cache-safe context assembly, 30KB spill-to-disk pointers, and 75% high-watermark atomic-closure compaction.
- `orangeagent-sandbox` *(Planned — Phase M2)* — Zero-dependency native OS sandboxing (Linux Landlock/Seccomp, macOS Seatbelt), fail-closed startup probes, and `ApprovalBroker`.

### Status

**Phase M1 (Steps 1–9) complete** — minimal end-to-end pure-Rust coding loop operational across 5 implemented crates (`orangeagent-ai`, `orangeagent-core`, `orangeagent-harness`, `orangeagent-spine`, `orangeagent-cli`), verified by **43 offline unit and E2E tests** (`cargo test --workspace`).

`orangeagent-context` and `orangeagent-sandbox` are scheduled for **Phase M2**.

### Quick Start

```bash
# Run formatting, strict clippy lints, and all 43 offline tests
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Replay a recorded VCR cassette offline (zero API cost)
cargo run -p orangeagent-cli -- run \
  --workspace . \
  --vcr-replay ./tape.jsonl \
  --json \
  "Inspect the repo and summarize key modules"

# Run against a live OpenAI Chat Completions compatible endpoint with session persistence
ORANGEAGENT_BASE_URL="https://api.deepseek.com/v1" \
ORANGEAGENT_API_KEY="sk-..." \
ORANGEAGENT_MODEL="deepseek-v4-pro" \
cargo run -p orangeagent-cli -- run \
  --workspace . \
  --session ./.orangeagent/session.jsonl \
  "Read README.md and check cargo test"
```

---

## 中文

基于纯 Rust 单二进制打造的 Coding Agent 与 Harness 平台：兼具极简透明的模型交互表面（参考 `Pi`），以及生产级的并发调度、Prompt Cache 前缀保护、原生 OS 沙箱与可恢复执行底盘（融合 `CodeFlow`、`Codex`、`OpenCode`、`Claude Code` 工程实践）。

### 架构总览

OrangeAgent 采用模块化 Cargo Workspace 划分，保持严格单向无环依赖：

- `orangeagent-ai` *（M1 已完成）* — 多协议流式引擎（已实现 OpenAI Chat Completions 兼容协议；后续扩展 Anthropic / Responses / Gemini），支持跨厂商 Thinking 签名净化、结构化错误判决与零成本 VCR 录制/回放。
- `orangeagent-core` *（M1 已完成）* — 极简事件驱动 ReAct 主循环（`AgentLoop`）、`AgentEvent` / `EventSink`、工具并发分类注册表（`ReadOnlyParallel` vs `ExclusiveSequential`）、空响应/畸形 JSON 受限自愈、回合内插话队列 `SteerQueue`、6 阶段 `AgentHook` 拦截链与层级 `CancellationToken` 取消树。
- `orangeagent-harness` *（M1 已完成基础 6 工具）* — 统一工作区边界校验 `WorkspaceContext`、防盲改哈希追踪 `ReadFileTracker`、`read`、`write`（同目录临时文件原子 `rename`）、`edit`（多 Hunk 原子替换 + 缩进模糊匹配 + Unified Diff）、`bash`（超时 + Unix PGID 整组无孤儿清理 + Head/Tail 输出截断），以及进程内 `grep` / `glob`；M2/M3 将补齐双模 `subagent` + `worktree`、`ask_user`、Shadow-Git 快照、LSP 诊断、`AGENTS.md`/`SKILL.md` 与 MCP 懒披露。
- `orangeagent-spine` *（M1 已完成树状 JSONL 存储）* — 树状 Append-only JSONL 会话主存储（`id` / `parent_id`、非破坏 `/branch` 分叉、崩溃尾行截断自愈）；M2/M4 将补齐会话级 `Lane`、`OriginPools` 双池隔离、`BusyPolicy`、SQLite WAL/FTS5 派生索引与 Daemon/ACP 传输层。
- `orangeagent-cli` *（M1 已完成 `run` 闭环）* — 单二进制统一入口（`orangeagent run` 流式终端输出、`--json` 无头评测事件流、`--session` 续聊、`--vcr-replay`/`--vcr-record`）；M3/M4 将补齐 `ratatui` 内联 TUI、Daemon 与 ACP 适配。
- `orangeagent-context` *（规划中 — Phase M2）* — Prompt Cache 优先上下文装配、30KB 大输出落盘指针与 75% 高水位原子闭包压缩。
- `orangeagent-sandbox` *（规划中 — Phase M2）* — 零外部依赖原生 OS 沙箱（Linux Landlock/Seccomp、macOS Seatbelt）、Fail-closed 启动探针与三态审批流 `ApprovalBroker`。

### 当前进度

**Phase M1（Step 1–9）已完成** —— 已打通跨越 5 个 Crate（`orangeagent-ai`、`orangeagent-core`、`orangeagent-harness`、`orangeagent-spine`、`orangeagent-cli`）的端到端纯 Rust 编码闭环，全 Workspace **43 条离线单测与全栈 E2E 测试全部通过**（`cargo test --workspace`）。

`orangeagent-context` 与 `orangeagent-sandbox` 将在 **Phase M2** 推进实现。

### 快速开始

```bash
# 运行格式检查、严格 Clippy 校验与全部 43 条离线测试
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# 离线回放 VCR 录制磁带（0 Token 费用）
cargo run -p orangeagent-cli -- run \
  --workspace . \
  --vcr-replay ./tape.jsonl \
  --json \
  "Inspect the repo and summarize key modules"

# 接入兼容 OpenAI Chat 的真实模型 API 并开启 JSONL 会话持久化
ORANGEAGENT_BASE_URL="https://api.deepseek.com/v1" \
ORANGEAGENT_API_KEY="sk-..." \
ORANGEAGENT_MODEL="deepseek-v4-pro" \
cargo run -p orangeagent-cli -- run \
  --workspace . \
  --session ./.orangeagent/session.jsonl \
  "Read README.md and check cargo test"
```
