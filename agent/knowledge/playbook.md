# Agent Playbook

## Baseline Heuristics

- `combat_play`
  优先打出可出的牌；若需要目标，默认打血量最低的敌人。
- `map_select`
  低血量优先篝火；高血量时可以接受精英；金币多时可考虑商店。
- `card_reward`
  默认跳过 `Status` / `Curse`，优先普通可用牌。
- `rest_site`
  血量 `<= 45%` 优先 `HEAL`；若接近 Boss 且血量 `<= 60%` 也优先 `HEAL`。其余情况优先 `SMITH`，并升级高影响非 starter 牌。
- `shop`
  先检查是否该删牌：如果删牌价格可负担，且牌组里还有很多 `Strike/Defend`，删牌优先级很高。
  再看折扣牌和高分 relic；药水只在有空槽且短期保命有帮助时买。
  若没有删牌或明显高价值购买，就离开，避免把钱花在平庸项上。
- `event_choice`
  当前默认选第一个可选项，后续再把事件知识做细。

## Memory Usage

- `deck_profile`
  跟踪攻击 / 技能 / 能力牌数量、starter 数量、抽牌 / 格挡 / 易伤 / 消耗协同，以及最佳升级候选。
- `run_plan`
  每步都要有 2-4 条当前 run 优先级，例如“先保命”、“优先删 Strike”、“下个篝火升级 Pommel Strike”。
- `decision_context`
  当前决策点要有专门建议：
  - `rest_site`：推荐 `HEAL` 还是 `SMITH`，以及前三个升级候选
  - `shop`：删牌是否值得、最优卡牌 / relic / 药水候选、药水槽是否已满

## Safety Rules

- provider 输出必须经过 action validation
- 非法动作要自动回退到安全策略
- 引擎返回 `error` 时先尝试 `proceed`
- 所有关键步骤写入 episodic memory，方便 replay 和 bug 复盘

## Retrieval Sources

- `README.md`
- `agent/bug.md`
- 角色专属策略文档
- 历史反思和回放摘要
