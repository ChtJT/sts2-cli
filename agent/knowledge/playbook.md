# Agent Playbook

## Baseline Heuristics

- `combat_play`
  优先打出可出的牌；若需要目标，默认打血量最低的敌人。
- `map_select`
  低血量优先篝火；高血量时可以接受精英；金币多时可考虑商店。
- `card_reward`
  默认跳过 `Status` / `Curse`，优先普通可用牌。
- `rest_site`
  血量偏低先 `HEAL`，否则优先 `SMITH`。
- `shop`
  当前默认离开，后续再加完整购物策略。
- `event_choice`
  当前默认选第一个可选项，后续再把事件知识做细。

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
