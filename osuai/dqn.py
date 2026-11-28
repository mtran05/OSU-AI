"""Beyond The Rainbow (BTR, Clark et al. 2024) agent that learns osu! from simulated grayscale frames.

From DeepMind's DQN (Mnih et al. 2015): the observation is the last 4 rendered frames, experience replay, a
periodically synced target network and action repeat (the env decides every few ticks). From Rainbow (Hessel et al.
2018): n-step returns, a dueling value/advantage split, prioritized replay (Schaul et al. 2016) and noisy layers for
exploration (Fortunato et al. 2018). BTR's additions: an IMPALA residual encoder with spectral normalization, an
implicit quantile head (IQN, Dabney et al. 2018) in place of C51, and Munchausen targets (Vieillard et al. 2020) in
place of Double DQN. Each part can be switched off for comparison; with all of them off it is a plain DQN with a
Huber loss and epsilon-greedy exploration, which is how checkpoints from before BTR are loaded.

The action space is *spatial*, like DeepMind's StarCraft II agents: the network is fully convolutional and
outputs one Q-value per (screen cell, key state) pair, i.e. "aim here, holding no key / K1 / K2". Two keys let
the agent tap on every decision by alternating them, as players do in streams. Choosing a screen cell suits a
convolutional network, which is good at finding where something is in an image.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from osuai.beatmap import Beatmap, Frame
from osuai.env import EnvConfig, MapData, OsuEnv, PlayStats

CHECKPOINT_VERSION = "dqn-v4"  # checkpoint format identifier, stored in every checkpoint

# Every training round writes into its own folder:
#   <round dir>/best.pt     best checkpoint of the round (by validation reward)
#   <round dir>/last.pt     latest checkpoint (resume from this one to continue exactly where the round stopped)
#   <round dir>/log.jsonl   progress + evaluation log
#   <round dir>/round.json  human-readable summary: round history, settings, maps, results
BEST_CKPT = "best.pt"
LAST_CKPT = "last.pt"
LOG_FILE = "log.jsonl"
ROUND_FILE = "round.json"


# ---------------------------------------------------------------------------------------------- network
class NoisyLayer(nn.Module):
    """A fully connected layer with learned, factorised Gaussian noise on its weights (Fortunato et al. 2018).

    Applied to (B, in) it is a linear layer; applied to (B, in, H, W) it is a 1x1 convolution, the same layer at every
    cell. While training, the weights are mu + sigma * noise, with the noise redrawn by reset_noise(); in eval mode
    only mu is used, so greedy play is deterministic. The agent explores by learning how much noise each weight keeps.
    """

    def __init__(self, n_in: int, n_out: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        bound = 1.0 / math.sqrt(n_in)
        self.weight_mu = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((n_out, n_in), sigma0 / math.sqrt(n_in)))
        self.bias_mu = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(torch.full((n_out,), sigma0 / math.sqrt(n_in)))
        self.register_buffer("eps_in", torch.zeros(n_in), persistent=False)
        self.register_buffer("eps_out", torch.zeros(n_out), persistent=False)
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        self.eps_in.copy_(self._f(torch.randn(self.n_in, device=self.eps_in.device)))
        self.eps_out.copy_(self._f(torch.randn(self.n_out, device=self.eps_out.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * torch.outer(self.eps_out, self.eps_in)
            b = self.bias_mu + self.bias_sigma * self.eps_out
        else:
            w, b = self.weight_mu, self.bias_mu
        if x.dim() == 4:
            return F.conv2d(x, w[:, :, None, None], b)
        return F.linear(x, w, b)


def _head_precision(device: torch.device):
    """Turn mixed precision off for the network's head. bf16 keeps about 3 significant digits (steps of about 0.008
    near 1.0), so action values computed in it would round close calls such as pressing now or a tick later, and the
    greedy choice between them; the head is a small share of the work, so float32 there costs almost nothing."""
    return torch.autocast(device_type=device.type, enabled=False)


class ResBlock(nn.Module):
    """IMPALA residual unit, x + conv(relu(conv(relu(x)))), optionally dilated and with spectral normalization on both
    convolutions (as BTR does), which keeps each layer from amplifying its input and steadies learning."""

    def __init__(self, c: int, dilation: int = 1, spectral: bool = True) -> None:
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if spectral else (lambda m: m)
        self.c1 = sn(nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation))
        self.c2 = sn(nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.relu(self.c1(F.relu(x))))


def _impala_stage(c_in: int, c_out: int, spectral: bool) -> nn.Sequential:
    """conv 3x3, max-pool 3x3 stride 2 (half the size), two residual units."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.MaxPool2d(3, stride=2, padding=1),
        ResBlock(c_out, 1, spectral), ResBlock(c_out, 1, spectral),
    )


class QNetwork(nn.Module):
    """Fully convolutional spatial Q-network.

    Input: (B, frames, H, W) uint8 frames (playfield rows on top, HUD rows below). Output of forward() / q_values():
    (B, grid_h*grid_w*n_keys) expected returns, index = (row * grid_w + col) * n_keys + key, matching the env's action
    table. Two constant coordinate channels (CoordConv) are appended so cells near the edges can be told apart.

    Encoder: "impala" (BTR): two IMPALA stages (conv, max-pool to half size, two residual units; channels//2 then
    channels), then `depth` residual units whose spacing doubles (1, 2, 4, ...) so each cell sees further, with
    spectral normalization in every residual unit. "conv": the original plain network (two strided convolutions and
    `depth` dilated convolutions), kept so older checkpoints load. Unlike BTR, the features are not max-pooled to 6x6
    for a fully connected head: every aim cell keeps its own output, which is what makes the action space spatial.

    Head: a globally pooled summary of the screen is broadcast onto every cell. With quantiles > 0 the head is IQN
    (Dabney et al. 2018, as in BTR): it outputs return quantiles for sampled levels tau in (0, 1), each tau's cosine
    embedding scaling the head's hidden features. Dueling adds a value stream from the pooled summary to the cells'
    advantages minus their mean. Noisy layers replace the head's layers when noisy is set.
    """

    def __init__(
        self,
        in_frames: int,
        obs_hw: tuple[int, int],
        field_h: int,
        grid_hw: tuple[int, int],
        n_keys: int = 3,
        dueling: bool = False,
        channels: int = 64,
        depth: int = 3,
        quantiles: int = 0,
        noisy: bool = False,
        noisy_sigma: float = 0.5,
        encoder: str = "conv",
        spectral_norm: bool = False,
        n_cos: int = 64,
    ) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.obs_hw = tuple(obs_hw)
        self.field_h = int(field_h)
        self.grid_hw = tuple(grid_hw)
        self.n_keys = n_keys
        self.dueling = dueling
        self.channels, self.depth = int(channels), int(depth)
        self.quantiles, self.noisy, self.encoder, self.n_cos = int(quantiles), bool(noisy), encoder, int(n_cos)
        h, w = obs_hw
        ys = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
        self.register_buffer("coords", torch.cat([xs, ys], dim=1).contiguous(), persistent=False)
        c, c0 = self.channels, max(1, self.channels // 2)
        if encoder == "impala":
            self.torso = nn.Sequential(
                _impala_stage(in_frames + 2, c0, spectral_norm), _impala_stage(c0, c, spectral_norm),
                *[ResBlock(c, 2**k, spectral_norm) for k in range(self.depth)], nn.ReLU(),
            )
        elif encoder == "conv":
            # two downsampling layers, then `depth` layers at the same size whose spacing doubles (1, 2, 4, 8, ...)
            layers: list[nn.Module] = [
                nn.Conv2d(in_frames + 2, c0, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(c0, c, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            ]
            for k in range(self.depth):
                d = 2**k
                layers += [nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            self.torso = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown encoder {encoder!r} (use 'impala' or 'conv')")
        self.context = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True))

        def layer(n_in: int, n_out: int, conv: bool) -> nn.Module:
            if self.noisy:
                return NoisyLayer(n_in, n_out, noisy_sigma)
            return nn.Conv2d(n_in, n_out, 1) if conv else nn.Linear(n_in, n_out)

        # head: hidden layer, then the output layer (the names are those of the original network for old checkpoints)
        self.head = nn.Sequential(layer(2 * c, c, True), nn.ReLU(inplace=True), layer(c, n_keys, True))
        if dueling:
            self.value = nn.Sequential(layer(c, c, False), nn.ReLU(inplace=True), layer(c, 1, False))
        if self.quantiles > 0:
            # tau -> cos(pi * i * tau), i = 0..n_cos-1 -> one scale per hidden feature, for the cells and for the value
            self.register_buffer("cos_i", torch.arange(self.n_cos, dtype=torch.float32) * math.pi, persistent=False)
            self.tau_embed = nn.Sequential(nn.Linear(self.n_cos, 2 * c), nn.ReLU(inplace=True))

    @property
    def n_actions(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1] * self.n_keys

    @staticmethod
    def max_grid(obs_hw: tuple[int, int], field_h: int) -> tuple[int, int]:
        """The finest aim grid (rows, cols) this network resolves: its playfield feature map (about 1/4 of the frame).

        A finer grid would only repeat feature cells, so neighbouring cells would share the same Q-values.
        """

        def torso(n: int) -> int:  # two halvings (strided conv or max-pool, 3x3 stride 2 padding 1 sizes)
            n = (n + 2 - 3) // 2 + 1
            return (n + 2 - 3) // 2 + 1

        fh, fw = torso(obs_hw[0]), torso(obs_hw[1])
        return (max(1, round(field_h * fh / obs_hw[0])), fw)

    def _features(self, obs_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-cell hidden features (B, c, grid_h, grid_w) and the pooled screen summary (B, c)."""
        x = obs_u8.float() * (1.0 / 255.0)
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        f = self.torso(x)  # under mixed precision: bf16, where nearly all the computation is
        # everything after the encoder stays float32 (see _head_precision)
        with _head_precision(f.device):
            f = f.float()
            pooled = f.mean(dim=(2, 3))
            ctx = self.context(pooled)[:, :, None, None].expand_as(f)
            hid = F.relu(self.head[0](torch.cat([f, ctx], dim=1)))
            # keep only the feature rows that cover the playfield (drop the HUD strip), then fit them to the grid
            rows = max(1, round(self.field_h * f.shape[2] / self.obs_hw[0]))
            return F.adaptive_avg_pool2d(hid[:, :, :rows], self.grid_hw), pooled

    def sample_taus(self, batch: int, n: int, device) -> torch.Tensor:
        """n quantile levels per state: random in training, evenly spaced in eval mode (so greedy play is fixed)."""
        if self.training:
            return torch.rand(batch, n, device=device)
        return ((torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n).expand(batch, n)

    def quantile_values(self, obs_u8: torch.Tensor, taus: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, n_actions) returns at the quantile levels `taus` (B, N); N = 1 and taus unused without IQN."""
        hid, pooled = self._features(obs_u8)
        with _head_precision(hid.device):
            return self._head(hid, pooled, taus)

    def _head(self, hid: torch.Tensor, pooled: torch.Tensor, taus: torch.Tensor | None) -> torch.Tensor:
        b, c, gh, gw = hid.shape
        if self.quantiles > 0:
            if taus is None:
                taus = self.sample_taus(b, self.quantiles, hid.device)
            emb = self.tau_embed(torch.cos(taus[..., None] * self.cos_i))  # (B, N, 2c)
            e_cell, e_val = emb[..., :c], emb[..., c:]
            n = taus.shape[1]
            hid = (hid[:, None] * e_cell[..., None, None]).reshape(b * n, c, gh, gw)
        else:
            n = 1
        adv = self.head[2](hid)  # (B*N, n_keys, gh, gw)
        q = adv.view(b, n, self.n_keys, gh, gw).permute(0, 1, 3, 4, 2).reshape(b, n, gh * gw * self.n_keys)
        if self.dueling:
            v = F.relu(self.value[0](pooled))  # (B, c)
            v = v[:, None] * e_val if self.quantiles > 0 else v[:, None]
            q = self.value[2](v) + q - q.mean(dim=2, keepdim=True)  # (B, N, 1) + (B, N, A)
        return q

    def q_values(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """(B, n_actions) expected returns: the mean over quantile levels (the single value without IQN)."""
        return self.quantile_values(obs_u8).mean(dim=1)

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs_u8)

    def greedy(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs_u8).argmax(dim=1)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLayer):
                m.reset_noise()


# ---------------------------------------------------------------------------------------------- replay
class SumTree:
    """Leaf values in a binary tree of sums: set values and draw indices in proportion to them in O(log n)."""

    def __init__(self, n: int) -> None:
        self.P = 1 << max(1, (n - 1).bit_length())
        self.tree = np.zeros(2 * self.P, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[self.P + np.asarray(idx)]

    def set(self, idx, values) -> None:
        i = self.P + np.atleast_1d(np.asarray(idx, dtype=np.int64))
        self.tree[i] = np.atleast_1d(np.asarray(values, dtype=np.float64))
        i = np.unique(i // 2)
        while i[0] >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i = np.unique(i // 2)
            if i[0] == 0:
                break

    def find(self, values: np.ndarray) -> np.ndarray:
        """Leaf index for each value in [0, total): leaves are laid end to end by size."""
        v = np.asarray(values, dtype=np.float64).copy()
        idx = np.ones(len(v), dtype=np.int64)
        while idx[0] < self.P:
            left = self.tree[2 * idx]
            right = v >= left
            v = np.where(right, v - left, v)
            idx = 2 * idx + right
        return idx - self.P


class ReplayBuffer:
    """Frame-efficient replay for several parallel envs.

    Each env owns a ring of single frames; stacked observations and n-step returns are rebuilt at sample time,
    so a transition costs one frame of memory instead of eight. Frame k of an env's ring is the observation on
    which action[k] was taken; `has_action[k]` is False for the last frame of an episode (no action from it).

    With alpha > 0 the replay is prioritized (Schaul et al. 2016): a transition is drawn in proportion to
    priority ** alpha, new ones at the highest priority seen so far, and sample() returns importance weights
    ((N * P) ** -beta, scaled to at most 1) that undo the bias of not sampling uniformly. update_priorities() sets a
    sampled transition's priority to its latest error. alpha = 0 is uniform replay with weights of 1.
    """

    def __init__(
        self, n_envs: int, capacity: int, obs_hw: tuple[int, int], stack: int, n_step: int, gamma: float, alpha: float = 0.0,
    ) -> None:
        self.E = n_envs
        self.C = max(capacity // n_envs, stack + n_step + 2)
        self.stack = stack
        self.n_step = n_step
        self.gamma = gamma
        h, w = obs_hw
        self.frames = np.zeros((n_envs, self.C, h, w), dtype=np.uint8)
        self.action = np.zeros((n_envs, self.C), dtype=np.int64)
        self.reward = np.zeros((n_envs, self.C), dtype=np.float32)
        self.done = np.zeros((n_envs, self.C), dtype=bool)
        self.has_action = np.zeros((n_envs, self.C), dtype=bool)
        self.ep_start = np.zeros((n_envs, self.C), dtype=bool)
        self.written = np.zeros(n_envs, dtype=np.int64)  # absolute number of frames written per env
        self.alpha = float(alpha)
        self.tree = SumTree(n_envs * self.C) if self.alpha > 0 else None
        self.max_priority = 1.0

    def __len__(self) -> int:
        return int(np.minimum(self.written, self.C).sum())

    def _put_frame(self, e: int, frame: np.ndarray, start: bool) -> None:
        k = int(self.written[e] % self.C)
        self.frames[e, k] = frame
        if self.tree is not None and self.has_action[e, k]:
            self.tree.set(e * self.C + k, 0.0)  # the transition stored in this slot is overwritten
        self.has_action[e, k] = False
        self.done[e, k] = False
        self.reward[e, k] = 0.0
        self.ep_start[e, k] = start
        self.written[e] += 1

    def begin_episode(self, e: int, frame: np.ndarray) -> None:
        self._put_frame(e, frame, start=True)

    def add_step(self, e: int, action: int, reward: float, terminal: bool, next_frame: np.ndarray) -> None:
        k = int((self.written[e] - 1) % self.C)
        self.action[e, k] = action
        self.reward[e, k] = reward
        self.done[e, k] = terminal
        self.has_action[e, k] = True
        if self.tree is not None:
            self.tree.set(e * self.C + k, self.max_priority**self.alpha)  # new transitions are seen at least once
        self._put_frame(e, next_frame, start=False)

    def update_priorities(self, index: np.ndarray, errors: np.ndarray, eps: float = 1e-6) -> None:
        if self.tree is None:
            return
        p = np.abs(np.asarray(errors, dtype=np.float64)) + eps
        self.max_priority = max(self.max_priority, float(p.max()))
        self.tree.set(index, p**self.alpha)

    def _stack(self, e: np.ndarray, idx: np.ndarray) -> np.ndarray:
        cols = [idx]
        cur = idx
        for _ in range(self.stack - 1):
            start = self.ep_start[e, cur % self.C]
            cur = np.where(start, cur, cur - 1)
            cols.append(cur)
        order = np.stack(cols[::-1], axis=1) % self.C  # (B, stack) oldest -> newest
        return self.frames[e[:, None], order]

    def _sample_prioritized(self, batch: int, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, beta: float):
        """Draw transitions in proportion to their priority (one per equal slice of the total, as in the paper)."""
        es: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        need = batch
        total = self.tree.total
        for _ in range(64):
            u = (np.arange(need) + rng.random(need)) * (total / need)
            g = self.tree.find(np.minimum(u, np.nextafter(total, 0)))
            g = g[g < self.E * self.C]  # the tree is padded to a power of two; padding leaves are empty
            e, k = g // self.C, g % self.C
            w = self.written[e] - 1  # newest absolute frame of that env; slot k holds absolute index a
            a = w - ((w - k) % self.C)
            keep = self.has_action[e, k] & (a >= lo[e]) & (a <= hi[e]) & (self.tree.get(g) > 0)
            es.append(e[keep])
            ids.append(a[keep])
            gs.append(g[keep])
            need -= int(keep.sum())
            if need <= 0:
                break
        e, a, g = np.concatenate(es)[:batch], np.concatenate(ids)[:batch], np.concatenate(gs)[:batch]
        n_valid = max(int(np.count_nonzero(self.has_action)), 1)
        probs = self.tree.get(g) / max(total, 1e-12)
        weights = (n_valid * probs) ** (-beta)
        return e, a, g, (weights / weights.max()).astype(np.float32)

    def sample(self, batch: int, rng: np.random.Generator, beta: float = 0.4) -> dict[str, np.ndarray]:
        # Once a ring has wrapped, the oldest stack-1 frames may be missing their history
        lo = np.where(self.written > self.C, self.written - self.C + self.stack - 1, 0)
        hi = self.written - 2  # newest frame with a successor frame
        ok_env = np.nonzero(hi >= lo)[0]
        if len(ok_env) == 0:
            raise RuntimeError("Replay buffer does not have enough data to sample")
        if self.tree is not None and self.tree.total > 0:
            e, a, g, weights = self._sample_prioritized(batch, rng, lo, hi, beta)
            return self._assemble(e, a, g, weights)
        es: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        need = batch
        for _ in range(64):
            e = rng.choice(ok_env, size=need * 2)
            a = lo[e] + (rng.random(need * 2) * (hi[e] - lo[e] + 1)).astype(np.int64)
            keep = self.has_action[e, a % self.C]
            es.append(e[keep])
            ids.append(a[keep])
            need -= int(keep.sum())
            if need <= 0:
                break
        e = np.concatenate(es)[:batch]
        a = np.concatenate(ids)[:batch]
        return self._assemble(e, a, e * self.C + a % self.C, np.ones(len(a), dtype=np.float32))

    def _assemble(self, e: np.ndarray, a: np.ndarray, g: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
        ret = np.zeros(len(a), dtype=np.float32)
        disc = np.ones(len(a), dtype=np.float32)
        alive = np.ones(len(a), dtype=bool)
        term = np.zeros(len(a), dtype=bool)
        boot = a + 1
        for m in range(self.n_step):
            idx = a + m
            valid = alive & (idx < self.written[e] - 1) & self.has_action[e, idx % self.C]
            r = self.reward[e, idx % self.C]
            ret += np.where(valid, disc * r, 0.0).astype(np.float32)
            d = valid & self.done[e, idx % self.C]
            term |= d
            boot = np.where(valid, idx + 1, boot)
            disc = np.where(valid, disc * self.gamma, disc).astype(np.float32)
            alive = valid & ~d
        return {
            "obs": self._stack(e, a),
            "action": self.action[e, a % self.C],
            "ret": ret,
            "disc": np.where(term, 0.0, disc).astype(np.float32),
            "next_obs": self._stack(e, boot),
            "index": g,  # for update_priorities
            "weight": weights,
        }


# ---------------------------------------------------------------------------------------------- config
@dataclass(slots=True)
class DQNConfig:
    # Defaults favour what the agent learns over training speed: they are sized for a dedicated GPU
    # (an NVIDIA A10 runs a round 1 in about 15 hours)
    total_steps: int = 3_000_000  # environment transitions (summed over parallel envs)
    n_envs: int = 32  # more games at once: less correlated data, and the simulation is cheap next to the updates
    frame_stack: int = 4
    buffer_size: int = 1_000_000  # as in BTR; about 7.3 GB of frames
    batch_size: int = 256
    lr: float = 1e-4
    # with a value, the learning rate follows a cosine from lr down to lr_end over the round (None: constant). For a
    # polishing round: smaller steps late in the round make the value estimates less noisy, and the agent's fine
    # choices (which tick to press on, when to leave a slider) depend on small differences between them
    lr_end: float | None = None
    adam_eps: float = 0.005 / 256  # BTR: 0.005 / batch size
    gamma: float = 0.9  # short horizon: at 30 decisions/s this looks ~1/3 s ahead, enough to aim and press
    n_step: int = 3
    double: bool = False  # Double DQN targets; BTR drops them, Munchausen targets replace them
    dueling: bool = True  # value + advantage streams (round 1 only)
    # 128 channels and depth 4: about 2.1M parameters (BTR's Atari network has 2.9M), and every aim cell sees the
    # whole playfield width; size is fixed after round 1, so it is chosen for the hardest maps, not the first ones
    channels: int = 128  # network width (round 1 only; the first stage has half as many)
    depth: int = 4  # units after the two downsampling stages, each reaching twice as far (round 1 only)
    encoder: str = "impala"  # "impala" (BTR's residual encoder) or "conv" (the original plain one); round 1 only
    spectral_norm: bool = True  # in the residual units (round 1 only)
    quantiles: int = 8  # IQN quantile levels sampled per state (0: one value per action); round 1 only
    n_cos: int = 64  # cosine features of a quantile level
    kappa: float = 1.0  # quantile Huber loss threshold
    munchausen: bool = True  # Munchausen targets (alpha * clipped scaled log-policy bonus, soft next-state value)
    m_alpha: float = 0.9
    m_tau: float = 0.03  # softmax temperature of the implied policy
    m_l0: float = -1.0  # lower clip of the log-policy bonus
    noisy: bool = True  # round 1 only
    noisy_sigma: float = 0.5
    per_alpha: float = 0.2  # prioritized replay strength (0: uniform)
    per_beta: float = 0.4  # importance-weight correction, rising to 1 over each round
    learning_starts: int = 20_000
    # gradient updates per environment transition: one batch of 256 every 8 decisions, 32 replayed samples per decision,
    # as many as the plain DQN that reached 98.5% on 0-3 star maps (BTR: 4 per decision, but over 50M decisions; at
    # 1/32 a 2M-decision round learned too little before the random phase ended and the agent stopped pressing)
    replay_ratio: float = 1 / 8
    target_update: int = 500  # gradient updates between target-network syncs
    eps_start: float = 1.0  # with noisy layers, a random phase in round 1 only (it presses keys rarely, see key_explore_prob)
    # a small floor: with ε = 0 the agent learned that presses cost reward before it learned that hits pay, stopped
    # pressing, and the noisy layers never made it press again; 2% random actions keep hits in the replay buffer
    eps_end: float = 0.02  # 0.05 without noisy layers; resumed rounds play with none (0, set by the CLI)
    # at replay ratio 1/8 the agent learns fast: with a 1M-step decay it was at 57% eval accuracy by 200k steps while
    # still acting at random 80% of the time, and only took off once ε neared its floor; 500k halves that wait
    eps_decay_steps: int = 500_000
    key_explore_prob: float = 0.2  # a random key action holds the key down this often (osu! punishes spam)
    grad_clip: float = 10.0
    eval_interval: int = 100_000
    eval_maps: int = 12  # validation maps per evaluation, spread over the round's star range (more: a steadier best.pt)
    eval_seconds: float = 60.0
    log_interval: int = 10_000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DQNConfig:
        known = {f.name for f in fields(cls)}
        # checkpoints from before BTR have none of its settings: they are plain DQN with the original encoder
        legacy = {"encoder": "conv", "spectral_norm": False, "quantiles": 0, "munchausen": False, "noisy": False,
                  "per_alpha": 0.0} if "encoder" not in d else {}
        # checkpoints from before the size flags existed used 64 channels and depth 3 (the defaults then)
        size = {k: v for k, v in (("channels", 64), ("depth", 3)) if k not in d}
        return cls(**{**legacy, **size, **{k: v for k, v in d.items() if k in known}})


def td_loss(
    online: QNetwork, target: QNetwork, obs: torch.Tensor, act: torch.Tensor, ret: torch.Tensor, disc: torch.Tensor,
    nxt: torch.Tensor, cfg: DQNConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample loss (with gradient), per-sample priority and the taken actions' current values for one batch.

    ret is the n-step discounted reward and disc = gamma**n (0 after a terminal). The target network is in eval mode
    (no noise). Munchausen (Vieillard et al. 2020), as in BTR: the target adds alpha * clip(tau * log pi(a_t|s_t), l0, 0)
    with pi = softmax(q_target / tau), and bootstraps the soft value sum_a pi(a|s')(z(s', a) - tau log pi(a|s')) instead
    of a max. Without it, a max (or the Double DQN choice) as usual. With IQN the online and target quantiles are
    compared with the quantile Huber loss (summed over online levels, averaged over target levels).
    """
    b = act.shape[0]
    rows = torch.arange(b, device=act.device)
    n = max(cfg.quantiles, 1)
    iqn = online.quantiles > 0
    if iqn:
        taus = torch.rand(b, n, device=obs.device)
        z = online.quantile_values(obs, taus).float()  # (B, N, A)
        z_sel = z.gather(2, act[:, None, None].expand(b, n, 1)).squeeze(2)  # (B, N)
    else:
        z_sel = online.quantile_values(obs).float().gather(2, act[:, None, None].expand(b, 1, 1)).squeeze(2)  # (B, 1)
    with torch.no_grad():
        z_next = target.quantile_values(nxt, torch.rand(b, n, device=obs.device) if iqn else None).float()  # (B, N', A)
        q_next = z_next.mean(dim=1)  # (B, A)
        if cfg.munchausen:
            t, alpha = cfg.m_tau, cfg.m_alpha
            logpi_next = F.log_softmax(q_next / t, dim=1)
            soft_next = (logpi_next.exp()[:, None, :] * (z_next - t * logpi_next[:, None, :])).sum(dim=2)  # (B, N')
            logpi_cur = F.log_softmax(target.q_values(obs).float() / t, dim=1)[rows, act]
            bonus = alpha * (t * logpi_cur).clamp(cfg.m_l0, 0.0)
            y = ret[:, None] + bonus[:, None] + disc[:, None] * soft_next  # (B, N')
        else:
            a_star = (online.q_values(nxt).float() if cfg.double else q_next).argmax(dim=1)
            y = ret[:, None] + disc[:, None] * z_next.gather(2, a_star[:, None, None].expand(b, z_next.shape[1], 1)).squeeze(2)
    delta = y[:, None, :] - z_sel[:, :, None]  # (B, N online, N' target)
    huber = F.smooth_l1_loss(z_sel[:, :, None].expand_as(delta), y[:, None, :].expand_as(delta), reduction="none",
                             beta=cfg.kappa)  # Huber(delta) / kappa
    if iqn:
        weight = (taus[:, :, None] - (delta.detach() < 0).float()).abs()
        per_sample = (weight * huber).sum(dim=1).mean(dim=1)
    else:
        per_sample = huber.mean(dim=(1, 2))
    priority = delta.detach().abs().mean(dim=2).sum(dim=1)
    return per_sample, priority, z_sel.detach().mean(dim=1)


def epsilon_at(cfg: DQNConfig, step: int) -> float:
    frac = min(max(step / max(cfg.eps_decay_steps, 1), 0.0), 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def lr_at(cfg: DQNConfig, step: int) -> float:
    """Learning rate at a step of the round: constant, or with lr_end set, a cosine from lr down to lr_end."""
    if cfg.lr_end is None:
        return cfg.lr
    frac = min(max(step / max(cfg.total_steps, 1), 0.0), 1.0)
    return cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1.0 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------------------------- checkpoint
def save_checkpoint(
    path: str | Path,
    net: QNetwork,
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_maps: list[str],
    val_maps: list[str],
    step: int,
    metrics: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict] | None = None,
    all_train_maps: list[str] | None = None,
    all_val_maps: list[str] | None = None,
    eval_maps: list[str] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": net.state_dict(),
            "grid_hw": list(net.grid_hw),
            "field_h": net.field_h,
            "n_keys": net.n_keys,
            "env_cfg": env_cfg.to_dict(),
            "dqn_cfg": dqn_cfg.to_dict(),
            "train_maps": train_maps,
            "val_maps": val_maps,
            "step": step,
            "metrics": metrics or {},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "history": history or [],
            # roles over every round so far, so later rounds never validate on a map that was trained on
            "all_train_maps": sorted(set(all_train_maps if all_train_maps is not None else train_maps)),
            "all_val_maps": sorted(set(all_val_maps if all_val_maps is not None else val_maps)),
            # the maps this round evaluated on; the next round evaluates them again to compare rounds like for like
            "eval_maps": list(eval_maps if eval_maps is not None else val_maps[: dqn_cfg.eval_maps]),
        },
        p,
    )


@dataclass(slots=True)
class LoadedAgent:
    net: QNetwork
    env_cfg: EnvConfig
    dqn_cfg: DQNConfig
    train_maps: list[str]
    val_maps: list[str]
    step: int
    metrics: dict
    opt_state: dict | None = None
    history: list[dict] = field(default_factory=list)
    all_train_maps: list[str] = field(default_factory=list)  # every map trained on in any round so far
    all_val_maps: list[str] = field(default_factory=list)  # every map held out for validation in any round so far
    eval_maps: list[str] = field(default_factory=list)  # the maps this checkpoint's round evaluated on


def load_checkpoint(path: str | Path, device: str | torch.device | None = None) -> LoadedAgent:
    path = resolve_checkpoint(path, prefer=BEST_CKPT)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a DQN checkpoint (version={ck.get('version')!r})")
    env_cfg = EnvConfig.from_dict(ck["env_cfg"])
    dqn_cfg = DQNConfig.from_dict(ck["dqn_cfg"])
    net = QNetwork(
        dqn_cfg.frame_stack,
        env_cfg.obs_shape,
        int(ck["field_h"]),
        tuple(ck["grid_hw"]),
        int(ck["n_keys"]),
        dueling=dqn_cfg.dueling,
        channels=dqn_cfg.channels,
        depth=dqn_cfg.depth,
        quantiles=dqn_cfg.quantiles,
        noisy=dqn_cfg.noisy,
        noisy_sigma=dqn_cfg.noisy_sigma,
        encoder=dqn_cfg.encoder,
        spectral_norm=dqn_cfg.spectral_norm,
        n_cos=dqn_cfg.n_cos,
    ).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return LoadedAgent(
        net,
        env_cfg,
        dqn_cfg,
        ck["train_maps"],
        ck["val_maps"],
        int(ck["step"]),
        ck.get("metrics", {}),
        opt_state=ck.get("optimizer"),
        history=list(ck.get("history", [])),
        all_train_maps=list(ck.get("all_train_maps", ck["train_maps"])),
        all_val_maps=list(ck.get("all_val_maps", ck["val_maps"])),
        # older checkpoints evaluated on the first maps of their validation list
        eval_maps=list(ck.get("eval_maps", ck["val_maps"][: dqn_cfg.eval_maps])),
    )


def pick_eval_maps(ids: list[str], stars: dict[str, float], k: int) -> list[str]:
    """k maps spread evenly over the star range of `ids` (easiest, hardest and evenly in between), deterministic."""
    ranked = sorted(ids, key=lambda m: (stars.get(m, 0.0), m))
    if k <= 0:
        return []
    if len(ranked) <= k:
        return ranked
    if k == 1:
        return [ranked[len(ranked) // 2]]
    return [ranked[round(i * (len(ranked) - 1) / (k - 1))] for i in range(k)]


def collect_map_roles(agent: LoadedAgent) -> tuple[set[str], set[str]]:
    """(maps ever trained on, maps ever held out) over every round that led to this checkpoint.

    Checkpoints store the roles of all rounds so far, but ones written before that existed only know their own round,
    so this also reads the checkpoints named as `resumed_from` in the round history when they are still on disk.
    """
    train, val = set(agent.all_train_maps), set(agent.all_val_maps)
    for h in agent.history:
        src = h.get("resumed_from")
        if not src or not Path(src).exists():
            continue
        ck = torch.load(Path(src), map_location="cpu", weights_only=False)
        train |= set(ck.get("all_train_maps", ck.get("train_maps", [])))
        val |= set(ck.get("all_val_maps", ck.get("val_maps", [])))
    return (train, val - train)


def resolve_checkpoint(path: str | Path, prefer: str = BEST_CKPT) -> Path:
    """Accept a checkpoint file or a round folder (then `prefer` inside it, falling back to the other file)."""
    p = Path(path)
    if p.is_dir():
        for name in (prefer, LAST_CKPT if prefer == BEST_CKPT else BEST_CKPT):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"No {BEST_CKPT} or {LAST_CKPT} in round folder {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def latest_round_dir(root: str | Path = "checkpoints") -> Path:
    """The round folder under `root` whose checkpoint was written most recently."""
    r = Path(root)
    cands = [d for d in r.iterdir() if d.is_dir() and ((d / BEST_CKPT).exists() or (d / LAST_CKPT).exists())] if r.is_dir() else []
    if not cands:
        raise FileNotFoundError(f"No trained round folders in {r}/ (run `train` first or pass --checkpoint)")
    return max(cands, key=lambda d: max((d / n).stat().st_mtime for n in (BEST_CKPT, LAST_CKPT) if (d / n).exists()))


# ---------------------------------------------------------------------------------------------- acting
@torch.no_grad()
def _autocast(device: torch.device | str):
    """bf16 mixed precision on a GPU (about twice as fast for the residual encoder); a no-op on the CPU."""
    dev = torch.device(device)
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()


def greedy_actions(net: QNetwork, stacks: np.ndarray, device: torch.device) -> np.ndarray:
    with _autocast(device):
        return net.q_values(torch.from_numpy(stacks).to(device, non_blocking=True)).float().argmax(dim=1).cpu().numpy()


@torch.no_grad()
def play_map(
    net: QNetwork,
    env_cfg: EnvConfig,
    beatmap: Beatmap | MapData,
    frame_stack: int,
    max_seconds: float | None = None,
    epsilon: float = 0.0,
    seed: int = 0,
    skip_breaks: bool = False,
) -> tuple[list[Frame], PlayStats]:
    """Play a whole map (or its first max_seconds) with the greedy policy; returns replay frames and stats.

    Breaks are played through in real time by default (the agent keeps acting while nothing is on screen), so the
    replay covers every moment of the song. Training fast-forwards through breaks instead, to save compute.
    """
    device = next(net.parameters()).device
    env_cfg = EnvConfig.from_dict({**env_cfg.to_dict(), "skip_breaks": skip_breaks})
    md = beatmap if isinstance(beatmap, MapData) else MapData(beatmap, env_cfg)
    env = OsuEnv(md, env_cfg, seed=seed, record=True)
    frame = env.reset(mode="full")
    stack = np.repeat(frame[None], frame_stack, axis=0)
    rng = np.random.default_rng(seed)
    t_stop = env.t + max_seconds * 1000.0 if max_seconds else float("inf")
    net.eval()
    while True:
        if epsilon > 0.0 and rng.random() < epsilon:
            a = int(rng.integers(env.n_actions))
        else:
            a = int(greedy_actions(net, stack[None], device)[0])
        frame, _, terminal, _ = env.step(a)
        stack = np.concatenate([stack[1:], frame[None]], axis=0)
        if terminal or env.t >= t_stop:
            break
    return (env.frames, env.stats)


# ---------------------------------------------------------------------------------------------- training
def train_dqn(
    train_maps: list[MapData],
    val_maps: list[MapData],
    out_dir: str | Path,
    env_cfg: EnvConfig,
    cfg: DQNConfig,
    train_names: list[str] | None = None,
    val_names: list[str] | None = None,
    log=print,
    resume: LoadedAgent | None = None,
    round_info: dict | None = None,
    eval_maps: list[MapData] | None = None,
    prev_eval_maps: list[MapData] | None = None,
) -> dict:
    """Train a DQN agent on random windows of the training maps; keeps the best checkpoint by validation reward.

    Everything of this round goes into `out_dir` (best.pt, last.pt, log.jsonl, round.json). With `resume`, training continues from that checkpoint's network weights and optimizer state (e.g. for a
    star-rating curriculum: <4* maps first, then 4-5*, ...). The replay buffer always starts empty, cfg.total_steps
    counts this round only, and the step stored in checkpoints is cumulative across rounds.

    Evaluations play `eval_maps` (default: the first cfg.eval_maps validation maps); best.pt is chosen on them. With
    `prev_eval_maps` (the maps the previous round evaluated on), every evaluation also plays those and logs them as
    eval_prev_*, so a round can be compared with how the one before it ended on the same maps.
    """
    run_dir = Path(out_dir)
    if run_dir.suffix == ".pt":
        raise ValueError(f"out_dir must be a folder, got a .pt path: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / BEST_CKPT
    last_path = run_dir / LAST_CKPT
    log_path = run_dir / LOG_FILE
    summary_path = run_dir / ROUND_FILE
    train_names = train_names or [m.bm.md5 for m in train_maps]
    val_names = val_names or [m.bm.md5 for m in val_maps]
    eval_set = list(eval_maps) if eval_maps is not None else list(val_maps[: cfg.eval_maps])
    prev_set = list(prev_eval_maps or [])
    eval_names = [m.bm.md5 for m in eval_set]

    prev_history = list(resume.history) if resume is not None else []
    round_no = len(prev_history) + 1
    base_step = resume.step if resume is not None else 0
    seed = cfg.seed + 7919 * (round_no - 1)  # fresh randomness for every round
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    env_cfg = EnvConfig.from_dict({**env_cfg.to_dict(), "gamma": cfg.gamma})  # shaping uses the agent's gamma

    envs = [OsuEnv(train_maps, env_cfg, seed=seed * 1000 + i) for i in range(cfg.n_envs)]
    n_actions = envs[0].n_actions
    n_cells, n_keys = env_cfg.n_cells, env_cfg.n_keys
    obs_hw = env_cfg.obs_shape
    grid_hw = (env_cfg.grid_h, env_cfg.grid_w)
    field_h = env_cfg.render.field_height
    def make_net() -> QNetwork:
        return QNetwork(cfg.frame_stack, obs_hw, field_h, grid_hw, n_keys, dueling=cfg.dueling,
                        channels=cfg.channels, depth=cfg.depth, quantiles=cfg.quantiles, noisy=cfg.noisy,
                        noisy_sigma=cfg.noisy_sigma, encoder=cfg.encoder, spectral_norm=cfg.spectral_norm,
                        n_cos=cfg.n_cos).to(device)

    online, target = make_net(), make_net()
    if device.type == "cuda":
        online, target = online.to(memory_format=torch.channels_last), target.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    if resume is not None:
        r = resume.net
        mine = (n_actions, cfg.frame_stack, tuple(obs_hw), cfg.dueling, cfg.channels, cfg.depth, cfg.quantiles,
                cfg.noisy, cfg.encoder)
        theirs = (r.n_actions, r.in_frames, tuple(r.obs_hw), r.dueling, r.channels, r.depth, r.quantiles, r.noisy, r.encoder)
        if mine != theirs:
            raise ValueError(
                "Cannot resume: the checkpoint was trained with a different observation/action setup or network "
                f"(actions, frames, obs, dueling, channels, depth, quantiles, noisy, encoder = {theirs}, this round: {mine})"
            )
        online.load_state_dict(r.state_dict())
        if resume.opt_state:
            opt.load_state_dict(resume.opt_state)
            for g in opt.param_groups:
                g["lr"] = cfg.lr
        log(f"[dqn] resumed from round {round_no - 1} at {base_step:,} total steps -> starting round {round_no}")
    target.load_state_dict(online.state_dict())
    target.eval()
    buf = ReplayBuffer(cfg.n_envs, cfg.buffer_size, obs_hw, cfg.frame_stack, cfg.n_step, cfg.gamma, alpha=cfg.per_alpha)

    stacks = np.zeros((cfg.n_envs, cfg.frame_stack, *obs_hw), dtype=np.uint8)
    for i, env in enumerate(envs):
        f = env.reset()
        buf.begin_episode(i, f)
        stacks[i] = f[None]

    log(
        f"[dqn] device={device} grid={grid_hw[1]}x{grid_hw[0]} keys={n_keys} decision_hz={env_cfg.decision_hz:g} "
        f"obs={cfg.frame_stack}x{obs_hw[0]}x{obs_hw[1]} "
        f"net={cfg.channels}ch/depth {cfg.depth} ({sum(p.numel() for p in online.parameters()):,} parameters) "
        f"btr: encoder={cfg.encoder}{'+sn' if cfg.spectral_norm else ''} quantiles={cfg.quantiles} "
        f"munchausen={cfg.munchausen} noisy={cfg.noisy} dueling={cfg.dueling} per_alpha={cfg.per_alpha:g} "
        f"train_maps={len(train_maps)} val_maps={len(val_maps)} eval_maps={len(eval_set)}"
        + (f" (+{len(prev_set)} from the previous round)" if prev_set else "")
        + f" buffer={buf.C * cfg.n_envs:,} "
        f"({buf.frames.nbytes / 1e9:.2f} GB)"
    )

    recent: deque[dict] = deque(maxlen=50)
    losses: deque[float] = deque(maxlen=200)
    qs: deque[float] = deque(maxlen=200)
    step = 0
    updates = 0
    update_credit = 0.0
    best_score = -float("inf")
    saved_best = False
    next_log = cfg.log_interval
    next_eval = cfg.eval_interval
    t_start = time.perf_counter()
    last_metrics: dict = {}

    def history_now() -> list[dict]:
        entry = {
            "round": round_no,
            "start_step": base_step,
            "steps": step,
            "train_maps": len(train_names),
            "val_maps": len(val_names),
            "best_eval_reward": None if math.isinf(best_score) else round(best_score, 3),
            **(round_info or {}),
        }
        return prev_history + [entry]

    all_train = sorted(set(resume.all_train_maps if resume is not None else []) | set(train_names))
    all_val = sorted((set(resume.all_val_maps if resume is not None else []) | set(val_names)) - set(all_train))

    def save(path: Path) -> None:
        save_checkpoint(
            path, online, env_cfg, cfg, train_names, val_names, base_step + step, last_metrics, opt, history_now(),
            all_train_maps=all_train, all_val_maps=all_val, eval_maps=eval_names,
        )

    def write_summary(done: bool) -> None:
        summary = {
            "round": round_no,
            "status": "finished" if done else "running",
            "folder": str(run_dir).replace("\\", "/"),
            "round_steps": step,
            "target_round_steps": cfg.total_steps,
            "total_steps": base_step + step,
            "best_eval_reward": None if math.isinf(best_score) else round(best_score, 3),
            "last_eval": last_metrics,
            "history": history_now(),
            "env_cfg": env_cfg.to_dict(),
            "dqn_cfg": cfg.to_dict(),
            "train_maps": train_names,
            "val_maps": val_names,
            "eval_maps": eval_names,
            "prev_eval_maps": [m.bm.md5 for m in prev_set],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def play_set(maps: list[MapData], prefix: str, played: dict[str, PlayStats]) -> dict:
        # greedy play is deterministic, so a map in both sets is played once per evaluation and its result reused
        for md in maps:
            if md.bm.md5 not in played:
                played[md.bm.md5] = play_map(online, env_cfg, md, cfg.frame_stack, max_seconds=cfg.eval_seconds)[1]
        results = [played[md.bm.md5] for md in maps]
        return {
            f"{prefix}_reward": float(np.mean([s.reward for s in results])),
            f"{prefix}_accuracy": float(np.mean([s.accuracy for s in results])),
            f"{prefix}_hit_rate": float(np.mean([s.hit_rate for s in results])),
            f"{prefix}_max_combo": float(np.mean([s.max_combo for s in results])),
        }

    def evaluate() -> dict:
        if not eval_set:
            return {}
        played: dict[str, PlayStats] = {}
        ev = play_set(eval_set, "eval", played)
        if prev_set:
            ev.update(play_set(prev_set, "eval_prev", played))
        online.train()
        return ev

    online.train()
    while step < cfg.total_steps:
        eps = epsilon_at(cfg, step)
        if cfg.noisy:
            online.reset_noise()  # fresh exploration noise for every decision
        actions = greedy_actions(online, stacks, device)
        # epsilon-greedy: a random action aims at a uniformly random cell and holds a key (K1 or K2, 50/50)
        # only key_explore_prob of the time (osu! punishes spam clicking, so frequent random presses mostly miss)
        rand = rng.random(cfg.n_envs) < eps
        if rand.any():
            k = int(rand.sum())
            cells = rng.integers(n_cells, size=k)
            keys = np.where(rng.random(k) < cfg.key_explore_prob, rng.integers(1, n_keys, size=k), 0)
            actions[rand] = cells * n_keys + keys

        for i, env in enumerate(envs):
            a = int(actions[i])
            f, r, terminal, truncated = env.step(a)
            buf.add_step(i, a, r, terminal, f)
            if terminal or truncated:
                st = env.stats
                recent.append({"reward": st.reward, "accuracy": st.accuracy, "hit_rate": st.hit_rate, "judged": st.judged})
                f = env.reset()
                buf.begin_episode(i, f)
                stacks[i] = f[None]
            else:
                stacks[i, :-1] = stacks[i, 1:]
                stacks[i, -1] = f
        step += cfg.n_envs

        if step >= cfg.learning_starts:
            update_credit += cfg.n_envs * cfg.replay_ratio
            while update_credit >= 1.0:
                update_credit -= 1.0
                beta = cfg.per_beta + (1.0 - cfg.per_beta) * min(step / max(cfg.total_steps, 1), 1.0)
                b = buf.sample(cfg.batch_size, rng, beta=beta)
                obs = torch.from_numpy(b["obs"]).to(device, non_blocking=True)
                nxt = torch.from_numpy(b["next_obs"]).to(device, non_blocking=True)
                act = torch.from_numpy(b["action"]).to(device)
                ret = torch.from_numpy(b["ret"]).to(device)
                disc = torch.from_numpy(b["disc"]).to(device)
                weight = torch.from_numpy(b["weight"]).to(device)
                if cfg.noisy:
                    online.reset_noise()
                with _autocast(device):
                    per_sample, priority, q_sel = td_loss(online, target, obs, act, ret, disc, nxt, cfg)
                loss = (weight * per_sample).mean()
                if cfg.lr_end is not None:
                    for g in opt.param_groups:
                        g["lr"] = lr_at(cfg, step)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                opt.step()
                buf.update_priorities(b["index"], priority.cpu().numpy())
                updates += 1
                losses.append(float(loss.item()))
                qs.append(float(q_sel.mean().item()))
                if updates % cfg.target_update == 0:
                    target.load_state_dict(online.state_dict())

        if step >= next_log:
            next_log += cfg.log_interval
            elapsed = time.perf_counter() - t_start
            rec = {
                "round": round_no,
                "step": step,
                "total_step": base_step + step,
                "updates": updates,
                "eps": round(eps, 4),
                "lr": float(f"{opt.param_groups[0]['lr']:.3g}"),
                "sps": round(step / max(elapsed, 1e-9), 1),
                "episodes": len(recent),
                "ep_reward": round(float(np.mean([r["reward"] for r in recent])), 3) if recent else None,
                "ep_accuracy": round(float(np.mean([r["accuracy"] for r in recent])), 2) if recent else None,
                "ep_hit_rate": round(float(np.mean([r["hit_rate"] for r in recent])), 4) if recent else None,
                "loss": round(float(np.mean(losses)), 5) if losses else None,
                "q_mean": round(float(np.mean(qs)), 4) if qs else None,
            }
            log("[dqn] " + " ".join(f"{k}={v}" for k, v in rec.items()))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        if step >= next_eval or step >= cfg.total_steps:
            next_eval += cfg.eval_interval
            ev = evaluate()
            last_metrics = {"step": base_step + step, "round": round_no, **ev}
            if ev:
                log("[eval] " + " ".join(f"{k}={round(v, 3)}" for k, v in ev.items()))
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"eval": True, **last_metrics}) + "\n")
            score = ev.get("eval_reward", -float("inf")) if ev else float(np.mean([r["reward"] for r in recent] or [0.0]))
            if score >= best_score:
                best_score = score
                save(out)
                saved_best = True
                log(f"[dqn] new best checkpoint -> {out} (score={score:.3f})")
            save(last_path)
            write_summary(done=False)

    if not saved_best:
        save(out)
    write_summary(done=True)
    return {
        "round_dir": str(run_dir).replace("\\", "/"),
        "best_checkpoint": str(out).replace("\\", "/"),
        "last_checkpoint": str(last_path).replace("\\", "/"),
        "round": round_no,
        "steps": step,
        "total_steps": base_step + step,
        "updates": updates,
        "best_eval_reward": None if math.isinf(best_score) else round(best_score, 3),
        "final": last_metrics,
        "minutes": round((time.perf_counter() - t_start) / 60.0, 2),
    }


# ---------------------------------------------------------------------------------------------- map pool
def _val_bucket(md5: str, seed: int) -> float:
    """A fixed number in [0, 1) per map: a map is validation when it falls below val_ratio."""
    import hashlib

    return int(hashlib.sha1(f"{seed}:{md5}".encode()).hexdigest()[:12], 16) / float(16**12)


def split_maps(
    md5s: list[str],
    val_ratio: float = 0.1,
    seed: int = 42,
    forced_train: set[str] | list[str] = (),
    forced_val: set[str] | list[str] = (),
) -> tuple[list[str], list[str]]:
    """Split maps into training and validation by MD5, so no map is in both.

    Each map's role depends only on its own MD5 (a hash), not on which other maps are in the pool, so changing the
    star filter between rounds never turns a training map into a validation map or the other way round.
    forced_train / forced_val pin the roles a map already had in earlier rounds (they win over the hash). A map that
    was ever trained on is always a training map, even if it was a validation map before, because the agent has seen it.
    """
    ids = sorted(set(md5s))
    ft, fv = set(forced_train), set(forced_val)
    if len(ids) <= 1 or val_ratio <= 0.0:
        held = [i for i in ids if i in fv and i not in ft]
        return ([i for i in ids if i not in held], held)
    val = [i for i in ids if i not in ft and (i in fv or _val_bucket(i, seed) < val_ratio)]
    if not val:  # tiny pools: still hold out one map that has never been trained on
        fresh = [i for i in ids if i not in ft]
        if fresh:
            val = [min(fresh, key=lambda i: _val_bucket(i, seed))]
    vs = set(val)
    return ([i for i in ids if i not in vs], val)


def resolve_manifest(path: str | Path) -> Path:
    """A manifest.json path, or a map folder holding one (e.g. maps/imported)."""
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"No map manifest at {p}")
    return p


def load_manifest_entries(
    manifest_paths: str | Path | Sequence[str | Path],
    min_stars: float = 0.0,
    max_stars: float = 0.0,
) -> list[dict]:
    """Read the entries of one or more manifests (or map folders), resolving paths and filtering by star rating
    (max_stars=0 means no cap). A map listed in several manifests is kept once."""
    if isinstance(manifest_paths, (str, Path)):
        manifest_paths = [manifest_paths]
    out = []
    seen: set[str] = set()
    for mp in manifest_paths:
        m_path = resolve_manifest(mp)
        for e in json.loads(m_path.read_text(encoding="utf-8")).get("maps", []):
            stars = float(e.get("star_rating", 0.0))
            if stars < min_stars or (max_stars > 0.0 and stars > max_stars):
                continue
            p = Path(e.get("path") or e.get("map_path") or "")
            if not p.is_absolute() and not p.exists():
                cand = m_path.parent / p
                p = cand if cand.exists() else (m_path.parent / p.name)
            if not p.exists():
                continue
            key = e.get("md5") or p.stem
            if key in seen:
                continue
            seen.add(key)
            out.append({**e, "path": str(p)})
    return out
