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
- `agent/world_model.py`
  全局路线规划层。进入 `map_select` 时会读取整张地图，生成当前路线模式、节点偏好和 immediate choice 打分。
- `agent/combat_log.py`
  战斗记录器。每场战斗会单独输出一个终端视图日志文件，方便逐场回看。
- `agent/retrieval.py`
  纯标准库的本地检索层，默认会索引：
  - 根目录 `README.md`
  - `agent/bug.md`
  - `agent/knowledge/`
- `agent/tracing.py`
  节点级 trace 记录器。每个 run 会在 `agent/state/runs/<run_id>/trace.jsonl` 写入类似 LangSmith 的节点事件。
- `agent/safety.py`
  Agent 层 safety policy。负责生成风险提示、硬性 guardrail，并在执行前拦截少量明显高风险动作。
- `agent/skills.py`
  显式 skill registry。把不同 decision type 的策略拆成可读的 skill，而不是全压在一次模型调用里。
- `agent/episodic.py`
  Episodic retrieval。会从历史 `episodes.jsonl` 中召回相似局面，给当前决策提供过去经验参考。
- `agent/prompt_context.py`
  Prompt 摘要层。把 `memory / world_model / skills / retrieval / episodic / safety` 压成一份紧凑上下文，再送给 provider。
- `agent/rl/`
  离线强化学习骨架：
  - `schema.py`：transition / reward schema
  - `reward.py`：连续 reward shaping
  - `dataset.py`：candidate action 枚举、特征摘要、训练集构建
  - `train_bc.py`：offline behavior cloning 入口
  - `train_iql.py`：offline IQL 风格训练入口
- `agent/runner.py`
  主循环：读状态、检索、取 provider 决策、做动作校验、写 memory、继续下一步。

## 现在已经具备的机制

- 自动运行 STS2 JSON 协议，不需要人工输入
- provider 抽象，可替换为真实 API
- command validation，避免 provider 输出非法动作
- fallback policy，provider 决策无效时回退到安全策略
- working / episodic / reflective 三层 memory
- map_select 时的 world-model 路线规划
- 本地 RAG 检索
- 每一步落盘日志，便于 replay 和后续分析
- 每场战斗单独落盘完整终端画面，路径在 `agent/state/combat_logs/`
- 节点级 trace，路径在 `agent/state/runs/<run_id>/trace.jsonl`
- provider 现在只接收一份 `prompt_context` 摘要，而不是整包原始 `memory / retrieval / episodic / world_model`
- 显式 skill 选择：
  - `combat_*`
  - `route_*`
  - `rest_*`
  - `shop_*`
  - `reward_*`
  - `event_*`
- episodic retrieval：
  - 从历史 runs 中找相似 `decision`
  - 结合 `floor / hp_ratio / enemy names / card names` 做局面相似度
  - 返回可供模型参考的 past action / rationale / memory_note
- safety guardrails：
  - 关键状态风险提示会进 prompt
  - 少量明显错误或高风险动作会在执行前被拦截
  - 连续相同状态会触发 stuck-loop 保护并提前停止
- step 级可观测性：
  - `steps.jsonl` / `episodes.jsonl` 现在会带上 `agent_context`
  - 其中包含 `skills / safety / episodic / world_model / memory` 的摘要
- RL dataset logging：
  - `agent/state/rl_transitions.jsonl`
  - `agent/state/runs/<run_id>/rl_transitions.jsonl`
  - 每条 transition 会记录：
    - `state / next_state`
    - `action_key / available_actions / action_hints`
    - `state_features / next_state_features`
    - `reward / reward_breakdown / terminal_type`
    - `agent_context`

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

## Offline RL

当前已经有一层离线 RL 骨架，默认读取：

```bash
agent/state/rl_transitions.jsonl
```

先训练 BC：

```bash
python3 agent/rl/train_bc.py
```

再训练 IQL 风格 scorer：

```bash
python3 agent/rl/train_iql.py
```

这两个脚本默认针对高层 decision：
- `map_select`
- `rest_site`
- `shop`
- `card_reward`
- `event_choice`
- `bundle_select`
- `card_select`

默认不包含 `combat_play`。如果后面要把战斗也纳入离线训练，可以加：

```bash
--include-combat
```

说明：
- 训练脚本需要 `torch`
- reward 不是二元胜负，而是连续分解：
  - `terminal`
  - `room_progress`
  - `combat_delta`
  - `hp_delta`
  - `deck_delta`
  - `economy_delta`
  - `action_penalty`
  - `shaping`
