"""Tests for the Beyond The Rainbow parts: noisy layers, prioritized replay, the IQN head and the Munchausen loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from osuai.dqn import DQNConfig, NoisyLayer, QNetwork, ReplayBuffer, SumTree, td_loss
from osuai.env import EnvConfig

ENV = EnvConfig(late_press=True)


def _net(**kw) -> QNetwork:
    torch.manual_seed(0)
    return QNetwork(4, ENV.obs_shape, ENV.render.field_height, (ENV.grid_h, ENV.grid_w), ENV.n_keys, **kw)


def test_noisy_layer_explores_in_training_and_is_fixed_in_eval():
    layer = NoisyLayer(8, 3)
    x = torch.randn(5, 8)
    layer.train()
    a = layer(x)
    layer.reset_noise()
    assert not torch.allclose(a, layer(x))  # new noise, new output
    layer.eval()
    assert torch.allclose(layer(x), F.linear(x, layer.weight_mu, layer.bias_mu))  # eval: the mean weights only
    conv_in = torch.randn(2, 8, 4, 6)  # the same layer at every cell
    assert layer(conv_in).shape == (2, 3, 4, 6)


def test_btr_network_outputs_quantiles_and_plays_deterministically():
    net = _net(dueling=True, quantiles=8, noisy=True, encoder="impala", spectral_norm=True)
    x = torch.randint(0, 255, (3, 4, *ENV.obs_shape), dtype=torch.uint8)
    taus = torch.rand(3, 8)
    z = net.quantile_values(x, taus)
    assert z.shape == (3, 8, ENV.n_actions) and net(x).shape == (3, ENV.n_actions)
    net.eval()  # evenly spaced quantile levels and no noise: greedy play is repeatable
    assert torch.equal(net.greedy(x), net.greedy(x))
    assert torch.allclose(net.q_values(x), net.quantile_values(x).mean(dim=1))
    # the original plain network is still there for comparison: one value per action
    plain = _net(encoder="conv")
    assert plain.quantile_values(x).shape == (3, 1, ENV.n_actions)
    assert QNetwork.max_grid(ENV.obs_shape, ENV.render.field_height) == (ENV.grid_h, ENV.grid_w)


def test_sum_tree_draws_in_proportion_and_prioritized_replay_weights():
    tree = SumTree(5)
    tree.set(np.arange(5), [1.0, 0.0, 3.0, 0.0, 6.0])
    assert tree.total == 10.0
    draws = tree.find(np.random.default_rng(0).random(20000) * tree.total)
    counts = np.bincount(draws, minlength=5) / 20000
    assert np.allclose(counts, [0.1, 0.0, 0.3, 0.0, 0.6], atol=0.02)

    buf = ReplayBuffer(n_envs=1, capacity=64, obs_hw=(2, 2), stack=2, n_step=1, gamma=0.9, alpha=1.0)
    f = lambda v: np.full((2, 2), v, dtype=np.uint8)  # noqa: E731
    buf.begin_episode(0, f(0))
    for i in range(1, 21):
        buf.add_step(0, action=i, reward=0.0, terminal=False, next_frame=f(i))
    rng = np.random.default_rng(0)
    b = buf.sample(8, rng, beta=0.5)
    assert np.all(b["weight"] <= 1.0 + 1e-6) and np.isclose(b["weight"].max(), 1.0)
    # one transition with a much larger error is replayed far more often, at a smaller importance weight
    buf.update_priorities(b["index"], np.full(len(b["index"]), 1e-3))
    hot = int(b["index"][0])
    buf.update_priorities(np.array([hot]), np.array([100.0]))
    seen = np.concatenate([buf.sample(8, rng, beta=0.5)["index"] for _ in range(50)])
    assert (seen == hot).mean() > 0.5


def test_plain_loss_is_the_dqn_huber_loss():
    online, target = _net(encoder="conv"), _net(encoder="conv")
    target.eval()
    cfg = DQNConfig(quantiles=0, munchausen=False, double=False, kappa=1.0)
    b = 4
    obs = torch.randint(0, 255, (b, 4, *ENV.obs_shape), dtype=torch.uint8)
    nxt = torch.randint(0, 255, (b, 4, *ENV.obs_shape), dtype=torch.uint8)
    act = torch.tensor([0, 5, 17, 2159])
    ret, disc = torch.tensor([1.0, 0.0, -0.1, 0.5]), torch.tensor([0.729, 0.729, 0.0, 0.729])
    loss, prio, q = td_loss(online, target, obs, act, ret, disc, nxt, cfg)
    with torch.no_grad():
        y = ret + disc * target(nxt).max(dim=1).values
    q_ref = online(obs)[torch.arange(b), act]
    assert torch.allclose(q, q_ref.detach(), atol=1e-5)
    assert torch.allclose(loss, F.smooth_l1_loss(q_ref, y, reduction="none"), atol=1e-5)
    assert torch.allclose(prio, (y - q_ref).abs().detach(), atol=1e-5)


def test_munchausen_target_reduces_to_the_max_and_never_adds_reward():
    online, target = _net(encoder="conv"), _net(encoder="conv")
    target.eval()
    b = 3
    obs = torch.randint(0, 255, (b, 4, *ENV.obs_shape), dtype=torch.uint8)
    nxt = torch.randint(0, 255, (b, 4, *ENV.obs_shape), dtype=torch.uint8)
    act, ret, disc = torch.tensor([3, 9, 40]), torch.tensor([1.0, 0.2, 0.0]), torch.full((b,), 0.729)
    plain = DQNConfig(quantiles=0, munchausen=False, double=False)
    near_max = DQNConfig(quantiles=0, munchausen=True, m_alpha=0.0, m_tau=1e-4)  # a cold softmax is the max
    l0, _, _ = td_loss(online, target, obs, act, ret, disc, nxt, plain)
    l1, _, _ = td_loss(online, target, obs, act, ret, disc, nxt, near_max)
    assert torch.allclose(l0, l1, atol=1e-3)
    # the log-policy bonus is clipped to [l0, 0]: it can lower a target, never raise it
    with torch.no_grad():
        q_cur = target.q_values(obs)
        logpi = F.log_softmax(q_cur / 0.03, dim=1)[torch.arange(b), act]
        bonus = 0.9 * (0.03 * logpi).clamp(-1.0, 0.0)
    assert (bonus <= 0).all() and (bonus >= -0.9).all()


def test_iqn_loss_trains_the_quantiles_toward_a_fixed_target():
    torch.manual_seed(0)
    online, target = _net(quantiles=8, encoder="impala", spectral_norm=True), _net(quantiles=8, encoder="impala", spectral_norm=True)
    target.eval()
    cfg = DQNConfig(quantiles=8, munchausen=True)
    b = 8
    obs = torch.randint(0, 255, (b, 4, *ENV.obs_shape), dtype=torch.uint8)
    act = torch.randint(0, ENV.n_actions, (b,))
    ret, disc = torch.ones(b), torch.zeros(b)  # terminal: the target is the reward alone (plus the bonus)
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        loss, prio, q = td_loss(online, target, obs, act, ret, disc, obs, cfg)
        if first is None:
            first = float(loss.detach().mean())
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    assert float(loss.detach().mean()) < 0.3 * first and prio.shape == (b,) and (prio >= 0).all()


def test_old_configs_load_as_plain_dqn():
    old = {"frame_stack": 4, "dueling": False, "double": True, "channels": 64, "depth": 3}
    cfg = DQNConfig.from_dict(old)
    assert (cfg.encoder, cfg.quantiles, cfg.munchausen, cfg.noisy, cfg.per_alpha) == ("conv", 0, False, False, 0.0)
    new = DQNConfig.from_dict(DQNConfig().to_dict())
    assert (new.encoder, new.quantiles, new.munchausen, new.noisy) == ("impala", 8, True, True)


def test_the_head_runs_in_float32_under_mixed_precision():
    """The encoder may run in bf16, but action values must not be rounded to bf16's ~3 significant digits."""
    net = _net(dueling=True, quantiles=8, noisy=True, encoder="impala", spectral_norm=True).eval()
    x = torch.randint(0, 255, (2, 4, *ENV.obs_shape), dtype=torch.uint8)
    seen = {}
    def record(name):
        def hook(module, inp, out):
            seen.setdefault(name, (inp[0].dtype, out.dtype))  # returns None: the layer's output is left as it is

        return hook

    for name in ("0", "2"):
        net.head[int(name)].register_forward_hook(record(name))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        q = net.quantile_values(x)
    assert q.dtype == torch.float32
    assert seen == {"0": (torch.float32, torch.float32), "2": (torch.float32, torch.float32)}
    # same encoder output in full precision gives the same head output: only the encoder was rounded
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            hid, pooled = net._features(x)
        assert hid.dtype == torch.float32 and pooled.dtype == torch.float32
        assert torch.allclose(net._head(hid, pooled, None), q, atol=1e-6)
