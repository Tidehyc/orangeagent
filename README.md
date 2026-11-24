# OrangeAgent

A pure-Rust, single-binary Coding Agent & Harness combining minimalist model-surface ergonomics (`Pi`) with production-grade runtime concurrency, prompt-cache discipline, native OS sandboxing, and recoverable execution (`CodeFlow`, `Codex`, `OpenCode`, `Claude Code`).

## Architecture Overview

OrangeAgent is organized as a modular Cargo workspace with strict one-way dependencies:

- `orangeagent-ai` — Multi-provider streaming wire protocols (OpenAI Compatible / Responses, Anthropic, Gemini), prompt-cache breakpoints, thinking-block preservation, structured error verdicts, and VCR deterministic replay.
- `orangeagent-core` — Minimal event-driven ReAct loop, `Submission` / `AgentEvent` channels, mid-turn `Steer` queue, lifecycle hook chain, and hierarchical `CancellationToken` trees.
- `orangeagent-sandbox` — Zero-dependency native OS sandboxing (Linux Landlock/Seccomp/bwrap, macOS Seatbelt), process-group lifecycle management, fail-closed startup probes, and capability approval broker.
- `orangeagent-context` — Prompt-cache-safe context assembly, large tool output spill-to-disk pointers, and high-watermark atomic-closure compaction (`tool_call` / `tool_result` pair preservation).
- `orangeagent-spine` — Multi-session `Lane` serialization, `OriginPools` concurrency isolation (`User` vs `System`), `BusyPolicy` (`Append` / `Inject` / `Interrupt`), tree-structured append-only JSONL sessions + derived SQLite FTS5 index, and local daemon / ACP transport.
- `orangeagent-harness` — `6+2` core coding tools (`read`, `write`, `edit`, `bash`, in-process `grep`/`glob`, dual-mode `subagent` + `git worktree` isolation, `ask_user`), shadow-git checkpoints, optional post-edit LSP diagnostics, progressive-disclosure `AGENTS.md`/`SKILL.md`, and lazy-disclosure MCP (`tool_search` / `tool_call`).
- `orangeagent-cli` — Single-binary entrypoint (`ratatui` inline streaming TUI, headless `--json` mode, daemon server, ACP adapter).

## Status

Repository initialized. Ready for Phase M1 implementation.
