# CLI reference

[← Back to the README](../README.md)

All commands are run as `python osuai/cli.py <command> [options]`.

| Command                                                         | What it does                                          |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| [`import-lazer`](#import-lazer--extract-beatmaps-from-osulazer) | Extract beatmaps from your osu!lazer install          |
| [`generate`](#generate--make-synthetic-maps)                    | Make synthetic maps at any star rating                |
| [`inspect`](#inspect--inspect-a-beatmap)                        | Print a beatmap's settings, star rating and PP        |
| [`judge`](#judge--score-a-replay)                               | Score a `.osr` replay the way osu!lazer does          |
| [`bench`](#bench--accuracy-across-a-map-pool)                   | Accuracy, timing and PP of an agent across a map pool |
| [`train`](#train--train-the-agent)                              | Train a round                                         |
| [`rollout`](#rollout--play-maps-into-osr-replays)               | Play maps into osu!lazer replays                      |
| [`render`](#render--preview-what-the-agent-sees)                | Save what the agent sees as a PNG sheet or GIF        |

## Map pools

Maps live in **pools**: folders inside `maps/`, each holding `.osu` files and a `manifest.json` that lists them with their star ratings.

| Pool             | Made by                  | Notes                                                                                                                     |
| ---------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `maps/imported/` | `import-lazer`           | Real maps from osu!lazer. The default pool of `train` and `bench`.                                                        |
| `maps/<name>/`   | `generate --name <name>` | Synthetic maps. Without `--name`: `maps/synthetic/`, or `maps/circles/`, `maps/sliders/`, `maps/spinners/` with `--only`. |

`train --maps` and `bench` take **one or more pools**. Name a pool by its folder (`maps/synthetic`) or its manifest (`maps/synthetic/manifest.json`); both mean the same thing. A map listed in several pools is used once. `inspect`, `judge`, `rollout` and `render` work on a single `.osu` file instead (e.g. `maps/imported/<map_md5>.osu`).

```powershell
# train on real maps plus two synthetic pools
python osuai/cli.py train --maps maps/imported maps/synthetic maps/sliders --max-stars 6

# benchmark only the synthetic pools
python osuai/cli.py bench maps/synthetic maps/sliders
```

**Validation maps and `--maps-dir`.** A checkpoint remembers its validation maps by MD5 hash only, not by path. `bench --val-only` and `rollout --val-only` find each `<md5>.osu` by searching `--maps-dir` (default `maps`) and all its sub-folders, so they find maps in every pool. Moving or renaming pools inside `maps/` never breaks a checkpoint. Change `--maps-dir` only if you keep maps outside `./maps`. Without `--val-only` it is not used.

## `import-lazer` — Extract beatmaps from osu!lazer

Scans the local osu!lazer file store (`%APPDATA%\osu\files`), extracts standard `.osu` beatmaps into `maps/imported/`, and writes `maps/imported/manifest.json`, sorted from the highest star rating down. That manifest is the default map pool of `train` and `bench`. Re-importing rewrites it.

```powershell
python osuai/cli.py import-lazer --limit 40
# A beginner-friendly training pool
python osuai/cli.py import-lazer --max-stars 3 --limit 0
```

<details>
<summary>Options</summary>

| Option                | Default         | Description                                                         |
| --------------------- | --------------- | ------------------------------------------------------------------- |
| `--lazer-dir <path>`  | `%APPDATA%\osu` | Path to the osu!lazer data directory containing `files/`.           |
| `--out-maps <dir>`    | `maps/imported` | Output directory for extracted `.osu` beatmaps and `manifest.json`. |
| `--limit <int>`       | `40`            | Maximum number of beatmaps to import (hardest first; `0` = all).    |
| `--min-stars <float>` | `0.0`           | Minimum star rating.                                                |
| `--max-stars <float>` | `0.0`           | Maximum star rating (`0.0` = no cap).                               |

</details>

## `generate` — Make synthetic maps

Generates playable maps at a target star rating into a pool `maps/<name>/`, with a `manifest.json` that `train` and `bench` read like any other pool (see [map pools](#map-pools)). Useful to fill star ranges your imported maps don't cover, or to drill one skill (only circles, only sliders, only spinners).

Each map picks its BPM, rhythm (1/1, 1/2 and 1/4 streams, denser at higher ratings), AR, OD and slider velocity from the target rating, then tunes its jump spacing until rosu-pp's star rating is within `--tolerance` of the target. Running the command again with the same `--name` adds to that folder and its manifest.

```powershell
# 20 maps spread evenly from 3 to 6 stars, default mix (60% circles, 35% sliders, 5% spinners)
python osuai/cli.py generate --stars 3-6 --count 20

# Slider practice at 4 stars; --only puts them in maps/sliders/ unless --name says otherwise
python osuai/cli.py generate --stars 4 --count 10 --only sliders

# A custom mix of circles/sliders/spinners
python osuai/cli.py generate --stars 7 --count 5 --ratio 60/30/10 --name hard

# Train on imported and synthetic maps together
python osuai/cli.py train --maps maps/imported maps/synthetic --max-stars 6 --resume checkpoints/round3_0-5star
```

<details>
<summary>Options</summary>

| Option                | Default                           | Description                                                                                                                                 |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--stars <x\|lo-hi>`  | _(required)_                      | Target star rating, or a range spread evenly over the maps.                                                                                 |
| `--count <int>`       | `10`                              | Number of maps.                                                                                                                             |
| `--name <folder>`     | `synthetic`, or the `--only` type | Folder inside `maps/` (not `imported`, which belongs to `import-lazer`). With `--only circles` and no `--name`, maps go to `maps/circles/`. |
| `--only <type>`       | _(none)_                          | `circles`, `sliders` or `spinners`: maps of a single object type.                                                                           |
| `--ratio <c/s/sp>`    | `60/35/5`                         | Share of circles/sliders/spinners by object count (any scale, e.g. `6/3/1`).                                                                |
| `--length <seconds>`  | `90`                              | Map length.                                                                                                                                 |
| `--tolerance <float>` | `0.05`                            | Allowed distance from the target star rating.                                                                                               |
| `--seed <int>`        | random                            | Map _i_ uses seed + _i_, so a seed reproduces the same maps.                                                                                |

</details>

- **Spinners have no star rating**, so `--only spinners` maps rate about 0★ whatever `--stars` says. Keep `--min-stars` at 0 when training on them.
- **Limits of a mix:** a mix with few circles has a ceiling (e.g. sliders-only maps top out around 7★). A map that can't reach its target is still written, and the command says so.
- **No audio:** synthetic maps are for training, `bench` and `render`, not for playing in osu!lazer.

## `inspect` — Inspect a beatmap

Prints beatmap metadata, difficulty settings (`CS`, `AR`, `OD`, `HP`), BPM range, object counts, max combo, star rating, and SS PP.

```powershell
python osuai/cli.py inspect maps/imported/<map_md5>.osu --mods HR,DT
```

<details>
<summary>Options</summary>

| Argument / Option    | Default      | Description                                                                                    |
| -------------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| `map` _(positional)_ | _(required)_ | Path to the `.osu` beatmap file.                                                               |
| `--mods <str>`       | `NM`         | Comma-separated mod string (`NM`, `HR`, `EZ`, `DT`, `HT`, `NC`, `HD`, `FL`, `NF`, `SD`, `PF`). |

</details>

## `judge` — Score a replay

Judges a `.osr` replay against its `.osu` beatmap the way osu!lazer does. It prints what lazer's results screen shows: accuracy, score, max combo, 300/100/50/miss, slider ticks, slider ends, spinner spins and bonus, plus star rating and PP.

```powershell
python osuai/cli.py judge maps/imported/<map_md5>.osu out/<map_md5>/dqn.osr
```

<details>
<summary>How the judge scores</summary>

<br>

`judge.py` follows osu!lazer's rules:

| Rule                     | How it works                                                                                                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hit windows              | lazer's 300 / 100 / 50 windows; pressing earlier than the 50 window (within 400 ms) is a miss                                                                               |
| Note lock                | a press is blocked only before the previous circle's time; hitting a circle misses earlier circles still waiting                                                            |
| Slider head              | timed like a circle (300/100/50/miss)                                                                                                                                       |
| Slider ticks and repeats | 30 points; a miss breaks combo                                                                                                                                              |
| Slider end               | 150 points; a miss costs accuracy but not combo                                                                                                                             |
| Slider tracking          | starts inside the ball and holds within $`2.4\times`$ its radius, with the head's key rules; ends are decided within the slider's last 36 ms                                |
| Spinners                 | rotation is a signed sum, so back-and-forth movement cancels out; each move between replay frames counts under the keys held where it starts, as in lazer's replay playback |
| Accuracy and score       | accuracy is floored to 2 decimals; the score uses lazer's standardised formula                                                                                              |

The results closely match lazer's but not exactly: lazer judges on rendered frames, so even the same replay can score slightly differently from one playback to the next. **PP** comes from rosu-pp's osu!stable formula and only approximates the PP lazer shows.

</details>

<details>
<summary>Options</summary>

| Argument / Option       | Default         | Description                                                              |
| ----------------------- | --------------- | ------------------------------------------------------------------------ |
| `map` _(positional)_    | _(required)_    | Path to the `.osu` beatmap file.                                         |
| `replay` _(positional)_ | _(required)_    | Path to the `.osr` replay file.                                          |
| `--mods <str>`          | _(from `.osr`)_ | Optional mod override; defaults to the mods stored in the `.osr` header. |

</details>

## `bench` — Accuracy across a map pool

Lets a trained agent play every map in the pools you pass (default: `maps/imported`), or only its held-out validation maps with `--val-only`, and prints per-map accuracy, score, combo, 100s, 50s, misses, unstable rate and PP, plus a summary. Each map is played in full and scored by the replay judge.

The summary ends with a **timing line**: the share of 300s, 100s, 50s and misses, the mean hit error (press time minus object time; negative = early) with the share of early hits, and the unstable rate ($`\mathrm{UR} = 10\,\sigma_{\Delta t}`$, ten times the standard deviation of hit errors, as in osu!). Use it to tell timing problems (many 100s, a large UR or a strong early/late bias) from aim problems (misses).

```powershell
# The newest round, on its validation maps (found anywhere under maps/)
python osuai/cli.py bench --val-only

# A specific round, on the 10 hardest imported maps
python osuai/cli.py bench --checkpoint checkpoints/round1_0-3star --limit 10

# Several pools: imported, then two synthetic ones (folders or manifest files)
python osuai/cli.py bench maps/imported maps/synthetic maps/sliders/manifest.json
```

Maps are played pool by pool in the order given, each pool hardest first, so `--limit N` takes the first N of that order. `bench` has no star filter; to benchmark one star band, generate it into its own pool (e.g. `generate --stars 5 --name bench5`) and pass that pool.

<details>
<summary>Options</summary>

| Argument / Option         | Default           | Description                                                                                               |
| ------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------- |
| `manifest` _(positional)_ | `maps/imported`   | One or more pools: map folders or their `manifest.json` (ignored with `--val-only`).                      |
| `--checkpoint <path>`     | newest round      | Round folder (uses `best.pt`) or `.pt` file.                                                              |
| `--val-only`              | off               | Only the validation maps stored in the checkpoint.                                                        |
| `--maps-dir <dir>`        | `maps`            | With `--val-only`: the folder searched (with all sub-folders) for the validation maps' `<md5>.osu` files. |
| `--mods <str>`            | _(from manifest)_ | Optional mod override applied to all maps.                                                                |
| `--limit <int>`           | `0`               | Only the first N maps (`0` = all).                                                                        |

</details>

## `train` — Train the agent

Trains on simulated frames from the maps in one or more pools (by default `maps/imported`). See the [training guide](training.md) for rounds, evaluation and the curriculum.

```powershell
# First round on the easiest maps (3M decisions, 32 parallel games, about 15 hours on an A10)
python osuai/cli.py train --max-stars 3

# Pure game-score reward, no aim shaping
python osuai/cli.py train --max-stars 3 --aim-shaping 0

# Real and synthetic maps together; the star filter applies to all pools
python osuai/cli.py train --maps maps/imported maps/synthetic --max-stars 3

# Only synthetic pools (e.g. made with generate --only circles and generate --only sliders)
python osuai/cli.py train --maps maps/circles maps/sliders --max-stars 4
```

Rewards (`--reward-*`, `--spin-reward`, `--spin-bonus`, `--timing-bonus`) are signed: hits are 0 or more and misses 0 or less. Penalties and costs (`--press-penalty`, `--empty-press-cost`, `--hold-cost`, `--slider-break-penalty`, `--slider-offset-cost`) are positive amounts that are subtracted. `train` refuses a value with the wrong sign.

<details>
<summary>Options: maps and rounds</summary>

| Option                                        | Default                        | Description                                                                                                                              |
| --------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--maps <pool> [<pool> …]`                    | `maps/imported`                | One or more pools: map folders or their `manifest.json`, e.g. `--maps maps/imported maps/synthetic`. A map listed twice is used once.    |
| `--out <folder>`                              | `checkpoints/round<N>_<stars>` | Folder for this round's files.                                                                                                           |
| `--force`                                     | off                            | Overwrite `--out` if it already holds a round.                                                                                           |
| `--min-stars <float>` / `--max-stars <float>` | `0` / `0` (no cap)             | Star-rating filter for the training pool (both ends included).                                                                           |
| `--limit-maps <int>`                          | `0`                            | Use only the first N matching maps.                                                                                                      |
| `--val-ratio <float>`                         | `0.1`                          | Roughly this fraction of new maps is held out for validation.                                                                            |
| `--resume <path>`                             | _(none)_                       | Continue from a previous round: its folder (uses `last.pt`) or a `.pt` file such as `<folder>/best.pt`.                                  |
| `--steps <int>`                               | `3000000`                      | Agent decisions in this round, summed over the parallel games.                                                                           |
| `--envs <int>`                                | `32`                           | Parallel simulated games. More games give less correlated replay data; the simulation is cheap next to the updates.                      |
| `--episode-seconds <float>`                   | `30`                           | Length of each random training window.                                                                                                   |
| `--eval-interval <int>`                       | `100000`                       | Decisions between evaluations.                                                                                                           |
| `--eval-maps <int>`                           | `12`                           | Validation maps per evaluation, spread over the round's star range. The previous round's eval maps are played as well, as `eval_prev_*`. |
| `--log-interval <int>`                        | `10000`                        | Decisions between log lines.                                                                                                             |
| `--seed <int>`                                | `0`                            | Random seed.                                                                                                                             |

</details>

<details>
<summary>Options: game, frame and network (round 1 only)</summary>

| Option                                   | Default  | Description                                                                                                                                                                                                                                                                           |
| ---------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--hz <float>`                           | `60`     | Simulation ticks per second of map time. Round 1 only; kept when resuming.                                                                                                                                                                                                            |
| `--action-repeat <int>`                  | `2`      | Ticks per agent decision; 2 = 30 decisions/s. Round 1 only; kept when resuming.                                                                                                                                                                                                       |
| `--width <int>`                          | `96`     | Playfield width in pixels, height follows 4:3 plus a 4-row HUD. Round 1 only; kept when resuming.                                                                                                                                                                                     |
| `--grid <cols>x<rows>`                   | `24x18`  | Aim grid over the playfield, e.g. `16x12` (coarser) or `32x24` (finer); one number sets the columns, rows follow 4:3. At most about $`W/4`$ by $`3W/16`$ for $`W`$ = `--width` (see [aim grid](training.md#aim-grid---grid---width)). Round 1 only; kept when resuming.                                 |
| `--late-press` / `--no-late-press`       | on       | K1 late and K2 late, two key states that press at the decision's middle tick: presses land on a 17 ms grid while the agent still decides 30 times/s. 5 key states (2,160 actions); `--no-late-press` keeps 3 (1,296). Off with `--action-repeat 1`. Round 1 only; kept when resuming. |
| `--max-speed <float>`                    | `8000`   | Cursor speed cap in osu!px per second.                                                                                                                                                                                                                                                |
| `--frame-stack <int>`                    | `4`      | Frames of history the network sees, newest last. 4 at 30 decisions/s covers the last 133 ms. Round 1 only; kept when resuming.                                                                                                                                                        |
| `--channels <int>`                       | `128`    | Network width: the default BTR network has about 2.1M parameters, 64 channels about a quarter as many. Round 1 only; kept when resuming.                                                                                                                                              |
| `--depth <int>`                          | `4`      | Layers after the two downsampling ones, each reaching twice as far: with 4 a cell sees about 128 frame pixels around it, the whole playfield width, with 3 about 64. Round 1 only; kept when resuming.                                                                                |
| `--encoder <impala\|conv>`               | `impala` | `impala`: BTR's residual encoder (two IMPALA stages, then `--depth` residual units). `conv`: the original plain network, for comparison. Round 1 only; kept when resuming.                                                                                                            |
| `--spectral-norm` / `--no-spectral-norm` | on       | Spectral normalization in the residual units. Round 1 only; kept when resuming.                                                                                                                                                                                                       |
| `--quantiles <int>`                      | `8`      | IQN quantile levels sampled per state (`0` = one value per action and a Huber loss). Round 1 only; kept when resuming.                                                                                                                                                                |
| `--dueling` / `--no-dueling`             | on       | Dueling value/advantage split of the Q-map. Round 1 only; kept when resuming.                                                                                                                                                                                                         |
| `--noisy` / `--no-noisy`                 | on       | Noisy output layers explore instead of random actions. Round 1 only; kept when resuming.                                                                                                                                                                                              |
| `--noisy-sigma <float>`                  | `0.5`    | Starting noise of the noisy layers. Round 1 only.                                                                                                                                                                                                                                     |

</details>

<details>
<summary>Options: reward</summary>

| Option                                         | Default   | Description                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--aim-shaping <float>`                        | `0.5`     | Potential-based aim shaping weight (`0` = pure game reward).                                                                                                                                                                                                        |
| `--reward-300 <float>`                         | `1.0`     | Reward for a 300 (circle, slider head or spinner).                                                                                                                                                                                                                  |
| `--reward-100 <float>`                         | `0.333`   | Reward for a 100. The default is $`\tfrac{1}{3}`$ of a 300, lazer's accuracy weight. Lower makes timing matter more.                                                                                                                                                |
| `--reward-50 <float>`                          | `0.167`   | Reward for a 50. Hit rewards must keep `--reward-50` $`\le`$ `--reward-100` $`\le`$ `--reward-300`.                                                                                                                                                                 |
| `--reward-miss <float>`                        | `-0.1`    | Reward for a missed circle, slider head or spinner (0 or less). Kept small so hits stand out early in training.                                                                                                                                                     |
| `--timing-bonus <float>`                       | `0.2`     | Extra reward for a 300 on a circle or slider head: the full amount for a press exactly on time, falling linearly to 0 at the edge of the 300 window, so presses aim for the window's centre (`0` = off).                                                            |
| `--pay-hits-on-time` / `--no-pay-hits-on-time` | on        | Pay an early hit when its object's time comes instead of on the press, so pressing early gains nothing from the discount.                                                                                                                                           |
| `--reward-tick <float>`                        | `0.1`     | Reward for a slider tick or repeat.                                                                                                                                                                                                                                 |
| `--reward-tick-miss <float>`                   | `-0.02`   | Reward for a missed slider tick or repeat (0 or less).                                                                                                                                                                                                              |
| `--reward-tail <float>`                        | `0.5`     | Reward for a slider end.                                                                                                                                                                                                                                            |
| `--slider-break-penalty <float>`               | `0.3`     | Taken off the reward when a slider is lost before its end, including a hit head let go before the slider was ever tracked (`0` = off).                                                                                                                              |
| `--slider-offset-cost <float>`                 | `1.0`     | Taken off the reward per second of a slider in progress with the cursor at the follow circle's edge, linear in distance from the ball (`0` = off).                                                                                                                  |
| `--spin-reward <float>`                        | `0.5`     | Reward per spinner rotation toward the spins its 300 needs, paid tick by tick as the net rotation grows.                                                                                                                                                            |
| `--spin-bonus <float>`                         | `0.1`     | Reward per spinner rotation beyond those.                                                                                                                                                                                                                           |
| `--spin-max-deg <float>`                       | `0` (off) | Forward rotation the spin reward credits per tick, in degrees (`0` = no cap). Off by default: a cap makes random spinning lose reward, so an untrained agent learns not to spin at all.                                                                             |
| `--press-penalty <float>`                      | `0.03`    | Taken off the reward per key press, including switching K1 ↔ K2 while holding.                                                                                                                                                                                      |
| `--empty-press-cost <float>`                   | `0.1`     | Taken off the reward, on top of `--press-penalty`, for a press that hits nothing: no circle or slider head, no slider in progress, no spinner running or about to start. A K1 ↔ K2 switch while a key is held costs it even during a slider or spinner (`0` = off). |
| `--hold-cost <float>`                          | `0.2`     | Taken off the reward per second a key is held while nothing needs one: no slider in progress, no spinner running (`0` = off).                                                                                                                                       |

</details>

<details>
<summary>Options: learning</summary>

| Option                                    | Default                          | Description                                                                                                                                                                                           |
| ----------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--buffer <int>`                          | `1000000`                        | Replay capacity in transitions (~7.3 GB of frames at default size).                                                                                                                                   |
| `--batch-size <int>`                      | `256`                            | Mini-batch size.                                                                                                                                                                                      |
| `--lr <float>`                            | `1e-4`                           | Adam learning rate.                                                                                                                                                                                   |
| `--adam-eps <float>`                      | `0.005 / batch size`             | Adam epsilon, as in BTR.                                                                                                                                                                              |
| `--gamma <float>`                         | `0.9`                            | Discount per decision; $`\gamma = 0.9`$ at 30 decisions/s looks about $`\tfrac{1}{3}`$ s ahead.                                                                                                       |
| `--n-step <int>`                          | `3`                              | n-step return length.                                                                                                                                                                                 |
| `--learning-starts <int>`                 | `20000`                          | Decisions collected before learning starts.                                                                                                                                                           |
| `--replay-ratio <float>`                  | `1/8`                            | Gradient updates per decision: one batch of 256 every 8 decisions (32 replayed samples per decision; BTR uses 4 over a much longer run). Higher learns more per decision but each round takes longer. |
| `--target-update <int>`                   | `500`                            | Gradient updates between target-network syncs.                                                                                                                                                        |
| `--munchausen` / `--no-munchausen`        | on                               | Munchausen targets: a small bonus for actions the agent already rates highly, and a soft next-state value instead of a max.                                                                           |
| `--double` / `--no-double`                | off                              | Double DQN targets; only used with `--no-munchausen`, which replaces them.                                                                                                                            |
| `--per-alpha <float>`                     | `0.2`                            | Prioritized replay: how strongly a transition's error decides how often it is replayed (`0` = uniform).                                                                                               |
| `--per-beta <float>`                      | `0.4`                            | Prioritized replay: starting strength of the importance-weight correction, rising to 1 over the round.                                                                                                |
| `--eps-start <float>`                     | `1.0` (`0` resuming)             | Initial share of random actions; 0.3 when resuming with `--no-noisy`.                                                                                                                                 |
| `--eps-decay <int>` / `--eps-end <float>` | `500000` / `0.02` (`0` resuming) | Linear $`\varepsilon`$ schedule from `--eps-start`. Round 1's small floor keeps some hits in the replay buffer; resumed rounds use no random actions (0.05 with `--no-noisy`).                        |
| `--key-explore <float>`                   | `0.2`                            | How often a random action presses a key (K1, K2 or a late key).                                                                                                                                       |

</details>

## `rollout` — Play maps into `.osr` replays

Lets a trained agent play a whole map from pixels and writes `out/<map_folder>/dqn.osr`, ready to import into osu!lazer. The printed result is what lazer will show for that replay, judged on exactly the frames written to the file. The `.osr` is in lazer's format: besides the 300/100/50/miss counts, combo and score, it stores the full statistics (slider ticks, slider ends, spinner spins and bonus) and the mod list.

```powershell
# Without --checkpoint, the newest round folder in checkpoints/ is used
python osuai/cli.py rollout maps/imported/<map_md5>.osu

# A synthetic map
python osuai/cli.py rollout maps/synthetic/<map_md5>.osu

# Every validation map of a round, found anywhere under maps/ (or under --maps-dir)
python osuai/cli.py rollout --val-only --checkpoint checkpoints/round2_0-5star
```

`rollout` plays one `.osu` file, or with `--val-only` all of a checkpoint's validation maps. To get replays for a whole pool, use `--val-only` or call it once per map.

<details>
<summary>Options</summary>

| Argument / Option     | Default             | Description                                                                                               |
| --------------------- | ------------------- | --------------------------------------------------------------------------------------------------------- |
| `map` _(positional)_  | _(none)_            | Beatmap to play (optional with `--val-only`).                                                             |
| `--val-only`          | off                 | Play every validation map stored in the checkpoint.                                                       |
| `--maps-dir <dir>`    | `maps`              | With `--val-only`: the folder searched (with all sub-folders) for the validation maps' `<md5>.osu` files. |
| `--checkpoint <path>` | newest round        | Round folder (uses `best.pt`) or `.pt` file.                                                              |
| `--mods <str>`        | `NM`                | Comma-separated mods.                                                                                     |
| `--out <path>`        | `out/<map>/dqn.osr` | Base output directory or explicit `.osr` path.                                                            |

</details>

## `render` — Preview what the agent sees

Saves the simulated grayscale frames the agent sees as a PNG contact sheet, or with `--gif` as an animated GIF. With `--checkpoint`, a trained agent plays and its cursor and keys are shown. Without it, only the map is shown: objects appearing, approaching and disappearing, with no cursor. Previews run in real map time, breaks included. A GIF lasts `--seconds` from `--start-ms` and is capped at the end of the map; `--seconds full` renders the whole song.

```powershell
# Contact sheet of 16 frames of the map
python osuai/cli.py render maps/imported/<map_md5>.osu --count 16 --every 2

# 20-second GIF of a trained agent playing, starting 30 s into the map
python osuai/cli.py render maps/imported/<map_md5>.osu --gif --start-ms 30000 --seconds 20 --checkpoint checkpoints/round1_0-3star

# The whole song (a 6-minute map is ~44 MB at the defaults; use --upscale 1 or --fps 15 for smaller files)
python osuai/cli.py render maps/imported/<map_md5>.osu --gif --seconds full --checkpoint checkpoints/round1_0-3star
```

<details>
<summary>Options</summary>

| Argument / Option                 | Default                                       | Description                                                                                    |
| --------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `map` _(positional)_              | _(required)_                                  | Path to the `.osu` beatmap file (from any pool, e.g. `maps/imported/` or `maps/synthetic/`).   |
| `--checkpoint <path>`             | _(none)_                                      | Let this trained agent (round folder or `.pt`) play; without it only the map is shown.         |
| `--mods <str>`                    | `NM`                                          | Comma-separated mods (e.g. `HR`, `DT`).                                                        |
| `--start-ms <float>`              | first object − 300 ms (PNG) / map start (GIF) | Map time of the first captured frame.                                                          |
| `--gif`                           | off                                           | Write an animated GIF instead of a PNG contact sheet.                                          |
| `--seconds <s\|full>`             | `10`                                          | GIF length in seconds, capped at the end of the map; `full` = the whole song.                  |
| `--fps <float>`                   | `30`                                          | GIF frame rate (max 50). The GIF always plays in real time.                                    |
| `--count <int>` / `--every <int>` | `16` / `2`                                    | PNG only: frames to capture / capture every N-th step.                                         |
| `--cols <int>`                    | `4`                                           | PNG only: columns in the contact sheet.                                                        |
| `--upscale <int>`                 | `4` (PNG) / `3` (GIF)                         | Nearest-neighbour upscale factor.                                                              |
| `--width <int>` / `--hz <float>`  | `96` / `60`                                   | Frame size and rate of map-only previews (with `--checkpoint`, the round's settings are used). |
| `--out <path>`                    | `out/<map>/frames.png` or `frames.gif`        | Output file.                                                                                   |

</details>
