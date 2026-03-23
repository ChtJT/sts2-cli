# STS2 Agent Runtime

这个目录现在提供了一层可直接跑在 `sts2-cli` 之上的 agent scaffold，目标不是先把 OpenAI/Codex 真接上，而是先把自动运行、分层 memory、RAG、策略循环和 provider 抽象搭出来。

版本说明：
- 当前底层 C# 项目 target framework 是 `net10.0`。
- 不是 `net11.0`。
- 如果本机装的是更新的 `.NET 11` SDK/runtime，通常也能启动，但推荐环境仍然是 `.NET 10`。

## 入口

```bash
python3 agent/run_agent.py --provider openai --character Ironclad --max-steps 200
```

如果游戏不在默认 Steam 路径，显式传入：

```bash
python3 agent/run_agent.py \
  --provider openai \
  --game-dir "/Volumes/T9/SteamLibrary/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64"
```

## 当前结构

- `agent/runtime.py`
  负责 `dotnet` 和游戏目录探测，启动 `Sts2Headless` 子进程，并通过 JSON stdin/stdout 通信。
- `agent/providers.py`
  Provider 抽象层。当前支持：
  - `openai`：真实 OpenAI Responses API provider
  - `codex`：预留接口，暂未接线
- `agent/memory.py`
  分层 memory 落盘：
  - `working_memory.json`：工作记忆，保存最近状态和事实
  - `episodes.jsonl`：逐步 episode 记录
  - `reflections.jsonl`：较高层的反思和异常总结
- `agent/retrieval.py`
  纯标准库的本地检索层，默认会索引：
  - 根目录 `README.md`
  - `agent/bug.md`
  - `agent/knowledge/`
- `agent/runner.py`
  主循环：读状态、检索、取 provider 决策、做动作校验、写 memory、继续下一步。

## 现在已经具备的机制

- 自动运行 STS2 JSON 协议，不需要人工输入
- provider 抽象，可替换为真实 API
- command validation，避免 provider 输出非法动作
- fallback policy，provider 决策无效时回退到安全策略
- working / episodic / reflective 三层 memory
- 本地 RAG 检索
- 每一步落盘日志，便于 replay 和后续分析

## OpenAI Provider

先设置环境变量：

```bash
export OPENAI_API_KEY="你的 key"
export OPENAI_MODEL="gpt-5-mini"
```

可选项：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_TIMEOUT_SECS="60"
export OPENAI_MAX_OUTPUT_TOKENS="400"
export OPENAI_REASONING_EFFORT="low"
```

然后运行：

```bash
python3 agent/run_agent.py --provider openai --character Ironclad --max-steps 200
```

当前 `openai` 走真实 API，`codex` 仍然只是预留接口。你后面如果要继续扩展 provider，建议只改 `agent/providers.py`：

1. 保持 `ProviderDecision` 结构不变
2. 把 `payload` 直接转成模型 prompt / JSON schema 输入
3. 输出仍然收敛成：
   - `command`
   - `rationale`
   - `memory_note`
4. 保留 `runner.py` 里的 validation 和 fallback，不要把安全性交给模型本身

## 建议下一步

先用 `openai` provider 跑通几局，确认：

- game dir 自动探测没有问题
- `agent/state/` 正常落盘
- reward / combat / rest / event / shop 都能闭环

接着再做：

1. 把 `codex` provider 接成真实 API
2. 在 retrieval 里引入向量检索或 embeddings
3. 给不同角色单独加 playbook 和 memory schema
4. 增加 battle planner / evaluator / reflection agent 这种多角色机制
