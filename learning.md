# STS2 CLI Learning Notes

## Latest Run: 10 games, avg floor 8.1, max floor 14, 0 stuck

## Observations from 10 Games

### Death Patterns
- **Fogmog (74hp)** — massive single enemy, eats lots of HP (Game 4: 72→31 HP)
- **Cubex Construct (65hp)** — tanky, hard to kill before taking too much damage
- **Vine Shambler (61hp)** — same problem, takes 5+ rounds
- **Inklet x3** — small but annoying in groups (3x 12-17hp each)
- **Flyconid + Slime combo** — flyconid has 47-49hp AND slimes have 28-35hp
- **Byrdonis (91-94hp)** — Elite, massive HP pool, almost guaranteed death without upgrades
- **Phrog Parasite (62-63hp)** — Elite, dangerous at low HP

### Key Issues
1. **Rest site heal doesn't show in log** — F5 heal at HP=35, F6 still shows HP=35 (Game 8). Bug?
2. **Gold loss at rest** — F5 heal, G=180→180 ok, but G6 Game 6: G=112 after heal at G=132. -20 gold?
3. **No potion usage** — potions are never used in combat
4. **Elites kill us** — Byrdonis 91hp and Phrog 62hp are death sentences at < 40 HP
5. **Too many fights** — agent takes every Monster, gets worn down by floor 6-8

### Strategy Improvements Needed
- **Skip some fights** — prefer Unknown/Treasure when HP < 60%
- **Use potions** — need to implement `use_potion` action
- **Card synergy** — picking random cards doesn't build a deck. Need:
  - Strength scaling (Inflame, Demon Form)
  - AOE for multi-enemy (Whirlwind, Thunderclap)
  - Block scaling (Barricade, Body Slam)
- **Don't fight Elites** at low HP (< 50%)
- **Heal threshold** — heal at 60% not 65% (preserve smith opportunities)
- **Buy from shop** — cheap relics/potions can make difference

### Card Tier List (from what I've seen)
**S-tier (take always):**
- Inflame (Power: +2 strength permanently)
- Demon Form (Power: +2 strength per turn)
- Barricade (Power: block doesn't decay)
- Impervious (Rare: 30 block for 2 energy)

**A-tier (usually take):**
- Battle Trance (draw 3 cards)
- Whirlwind (AOE damage, scales with energy)
- Feel No Pain (Power: gain block when exhausting)
- Hemokinesis (self-damage but high output)

**B-tier (situational):**
- Thunderclap (AOE + vulnerable)
- Pommel Strike (attack + draw)
- Burning Pact (exhaust for draw)

## Game Mechanics
- Starting: 80 HP, 3 energy, 99 gold, 10 cards (5 Strike/4 Defend/1 Bash)
- Strike: 6 dmg, cost 1 | Defend: 5 blk, cost 1 | Bash: 8 dmg + 2 vuln, cost 2
- Burning Blood: heal 6 HP after combat
- Rest: 30% max HP heal
- Act 1 (Overgrowth/密林): 15 floors + Boss, enemies 7-94 HP
- Vulnerable: take 50% more damage for N turns

## TODO
- [ ] Re-enable Neow (load loc data into LocManager tables)
- [ ] Implement potion usage in combat
- [ ] Track powers/buffs/debuffs in combat output
- [ ] Better card pick strategy (synergy-based)
- [ ] Shop buying strategy (cheap relics/potions)
- [ ] Try to beat Act 1 Boss

## Game claude_v2 Analysis (Floor 7 defeat)
- Intents work perfectly — can see exact damage values
- Fairy in a Bottle relic saved from death once (HP 5→24)
- Had 3 potions (Vulnerable, Orobic Acid, Fairy) but never used any!
- Byrdonis Elite 92hp with 17-24 dmg/turn killed us
- Gold 180 never spent on shop
- Key lesson: USE POTIONS in tough fights, especially Vulnerable Potion (+50% dmg)
- Key lesson: buy from shop when gold > 100
- Key lesson: Bash → Vulnerable → big attacks is the winning pattern

## Bug Found
- Potion usage code exists but agent never triggers it
- Need to add potion usage to combat strategy

## Game w26 Analysis (Floor 11 defeat, Demon Form seed)
- Got Demon Form at F3 (score 100 — best possible card!)
- Got Battle Trance (draw 3), Feel No Pain, Pommel Strike
- Used Strength Potion (+2 STR) and Flex Potion (+5 STR) ✅
- Bought Toolbox + Planisphere relics from shop
- Powers display working: Strength=2, Vulnerable=2, Shrink=-1, FLEX=5
- Died to Raiders at F11 (3 enemies, 16+30 incoming damage)
- Key issue: Demon Form costs 3 energy, uses entire turn to play
- Key issue: 15-card deck means Demon Form drawn every ~3 turns
- Need: play Demon Form ASAP, then survive while strength builds

## Neow Blessings Working!
- Golden Pearl (+150 gold) was selected and applied
- 3 options shown correctly from Neow event
- Avg floor improved from 8.4 to 9.2 with Neow
