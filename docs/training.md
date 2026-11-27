# Training guide

[← Back to the README](../README.md)

How rounds, checkpoints and evaluations work, how to build a curriculum, and the settings fixed in round 1. Every `train` option is listed in the [CLI reference](cli.md#train--train-the-agent).

## Rounds and their files

Each `train` run (a **round**) writes into its own folder, by default `checkpoints/round<N>_<stars>` (e.g. `round1_0-3star`, `round2_0-4star`, `round1_allstars`):

| File         | Contents                                                                                                                                                               |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `best.pt`    | Best checkpoint of the round by validation reward. Passing a round folder to `--checkpoint` (`rollout`, `bench`, `render`) uses this one.                              |
| `last.pt`    | Latest checkpoint, saved at every evaluation. Passing a round folder to `--resume` continues from this one.                                                            |
| `log.jsonl`  | One JSON line per progress log (steps/s, $`\varepsilon`$, episode reward / accuracy / hit rate, loss, mean Q) and per evaluation.                                      |
| `round.json` | Readable summary: status, step counts, best and latest evaluation, the round history, settings and this round's training/validation maps. Updated while training runs. |

`train` refuses to write into a folder that already holds a round. Pass `--force` to replace one on purpose. Anywhere a checkpoint is expected (`--resume`, `--checkpoint`), you can pass a round folder or a specific `.pt` file.

## Evaluation

Every `--eval-interval` decisions, and at the end of the round, the agent plays the first 60 s of `--eval-maps` validation maps (12 by default) without exploration. The results are logged as `eval_*`. They decide `best.pt` (by `eval_reward`) and are when `last.pt` is saved. They don't affect learning.

- **Which maps:** the round's eval maps are spread evenly over the star range of its validation pool, from the easiest to the hardest. Each round's set is saved in its checkpoint and listed in `round.json` (`eval_maps`).
- **Comparing rounds:** a later round's eval maps are harder, so its `eval_accuracy` is lower even when the agent improved. Each evaluation therefore also replays the previous round's eval maps, logged as `eval_prev_*` (listed as `prev_eval_maps` in `round.json`). Compare a round's `eval_prev_accuracy` with the previous round's final `eval_accuracy`: that's the same maps, before and after this round's training.
- **Cost:** a few seconds per map, so one evaluation with both sets takes about a minute, a few percent of the training time.

## Curriculum: continue training from a checkpoint

`--resume <checkpoint>` continues from a checkpoint's network weights and Adam optimizer state on whatever maps you select. That lets you train in rounds of increasing difficulty:

```powershell
python osuai/cli.py train --max-stars 3                                      # -> checkpoints/round1_0-3star/
python osuai/cli.py train --max-stars 5 --resume checkpoints/round1_0-3star  # -> checkpoints/round2_0-5star/
python osuai/cli.py train --max-stars 7 --resume checkpoints/round2_0-5star  # -> checkpoints/round3_0-7star/
```

- **When to move on:** once accuracy stops rising: `ep_accuracy` in the training log lines and `eval_accuracy` in the evaluations (`log.jsonl` / `round.json`). Hit rate isn't a good signal: it reaches about 0.99 early and says nothing about timing. A round that is going nowhere can be stopped with Ctrl+C; it keeps its checkpoints from the last evaluation.
- **Which checkpoint to resume:** a round folder resumes from its `last.pt`, the weights with the most training. `best.pt` is picked by a short evaluation and can be a lucky spike. If a round clearly got worse towards its end (training `ep_accuracy` falling for several log lines), resume from `checkpoints/<round>/best.pt` instead. When unsure, compare both with `bench --val-only`.
- **Resume from the latest round:** each checkpoint carries the training/validation roles of all rounds before it. Resume each round from the one before it, and older round folders can be deleted once the new round has saved its first checkpoint.
- **Round numbers:** the round number comes from the checkpoint's history, so the next folder is named automatically. Use `--out <folder>` to choose your own name.
- **Keep easier maps in later rounds:** a range like `--max-stars 5` rather than `--min-stars 4 --max-stars 5` keeps them in the mix, so the agent doesn't forget them while learning harder maps.
- **What carries over:** the replay buffer starts empty each round. `--steps` counts the new round only, while the step count stored in the checkpoint is cumulative.
- **Exploration:** noisy layers do most of the exploring. Round 1 starts with a random phase ($`\varepsilon`$ from 1 to 0.02 over `--eps-decay` decisions, pressing keys only `--key-explore` of the time), and 2% of actions stay random after it. Resumed rounds play with no random actions ($`\varepsilon = 0`$), as BTR ends: the agent already hits with its own policy, and random actions only add noise to the data. Without the round 1 floor the agent learned that presses cost reward before it learned that hits pay, stopped pressing once $`\varepsilon`$ reached 0, and the noisy layers never got it pressing again. With `--no-noisy`, $`\varepsilon`$-greedy as before ($`\varepsilon`$ restarts at 0.3 when resuming).

## Round 1 settings

The frame size (`--width`), tick rate (`--hz`), decision rate (`--action-repeat`), aim grid (`--grid`), key states (`--no-late-press`), frame history (`--frame-stack`) and network (`--channels`, `--depth`, `--encoder`, `--spectral-norm`, `--quantiles`, `--noisy`, `--dueling`) are **fixed after round 1** and come from the checkpoint, because they define the network's input and output. Reward and optimisation settings (`--lr`, `--aim-shaping`, `--n-step`, …) can change per round.

### Aim grid (`--grid`, `--width`)
Each cell is a spot the agent can aim at; the cursor glides to its centre. Finer cells aim more precisely (the worst-case error is half a cell diagonal, $`\tfrac{1}{2}\sqrt{w^2 + h^2}`$ for a $`w \times h`$ cell) but give more actions to learn: the network outputs one value per cell and key state (5 by default).

| `--grid` | Cell size | Worst-case aim error | Actions | Needs `--width` |
| -------- | --------- | -------------------- | ------- | --------------- |
| `16x12`  | 32 osu!px | ~23 px               | 960     | 64 or more      |
| `24x18`  | 21 osu!px | ~15 px               | 2,160   | 96 or more      |
| `32x24`  | 16 osu!px | ~11 px               | 3,840   | 128 or more     |
| `48x36`  | 11 osu!px | ~8 px                | 8,640   | 192 or more     |

The network reads every pixel of the frame, then halves the resolution twice, packing the detail into more channels. Its output map, one value per cell and key state, is therefore a quarter of the frame's width: about $`4 \times 4`$ frame pixels per cell. A grid finer than about `--width`/4 columns would only repeat values across neighbouring cells, so `train` refuses it and names the width that fits. This limits where the agent can aim, not what it can see. A wider frame costs more compute per step and more replay-buffer RAM (about 7.3 GB at width 96, 13 GB at 128 with the default 1,000,000-transition buffer).

```powershell
# Finer aim for small circles (high CS): 32x24 cells on a 128-pixel-wide frame
python osuai/cli.py train --max-stars 3 --grid 32x24 --width 128
```


### Press timing (`--late-press`, `--action-repeat`)
At 30 decisions/s a key could only go down every 33 ms, while the 300 window at OD 9 is $`\pm 25.5`$ ms, so a press one decision off would usually be a 100. So by default there are two more key states, "K1 late" and "K2 late": the key stays up for the first tick of the decision and goes down on the second, 17 ms later. A late key also re-taps a key that was already held. `--no-late-press` drops them (3 key states); with `--action-repeat 1` there is no middle tick, so they are off. Deciding more often instead (`--action-repeat 1`) trains much worse: each decision then changes the outcome so little that the network's errors drown out the differences between actions.

```powershell
# Only none / K1 / K2 (presses on the 33 ms decision grid)
python osuai/cli.py train --max-stars 3 --no-late-press
```


### Frame history (`--frame-stack`)
A single frame shows where things are, and the difference between stacked frames shows motion: the cursor's velocity, slider balls and how fast approach circles are closing. The history spans `--frame-stack` decisions, so 4 frames cover 133 ms at 30 decisions/s. More frames give a longer view of motion at a small cost in network compute. The replay-buffer RAM does not change, since the buffer stores single frames and rebuilds stacks when sampling.

```powershell
# 8 frames: the last 267 ms
python osuai/cli.py train --max-stars 3 --frame-stack 8
```


### Network size (`--channels`, `--depth`)
The default BTR network (`--channels 128 --depth 4`) has about 2.1M parameters, most of them in its residual encoder (BTR's Atari network has 2.9M). It is sized for the hardest maps, since size can't change after round 1: 128 channels keep more detail per spot and a richer summary of the whole screen, and depth 4 lets each aim cell see the whole playfield width. `--channels 64 --depth 3` (about 465k) trains about twice as fast per update and is enough for easy maps. The residual encoder dominates training time: updates run in bf16 mixed precision on a GPU, and `--replay-ratio` sets how many of them each round does.

```powershell
# The smaller network, for a quick experiment
python osuai/cli.py train --max-stars 3 --channels 64 --depth 3
```
