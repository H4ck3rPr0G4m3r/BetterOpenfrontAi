"""Training stack tests against the in-process fake worker (no Node needed)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from openfront import layout as L
from openfront import protocol as P
from openfront.agents import RandomAgent, ScriptedAgent
from openfront.config import load_config, stage_env
from openfront.model import OpenFrontPolicy, load_checkpoint, save_checkpoint
from openfront.ppo import Rollout, Trainer
from openfront.selfplay import LEARNER, OpponentPool, assign_seats
from openfront.vec import VecGames
from openfront.worker import WorkerConfig
from tests.fake_worker import fake_factory


def make_vec(servers, games=2, seats=2, workers=2, **kwargs):
    cfg = WorkerConfig(games=games, seats=seats)
    factory = fake_factory(servers, **kwargs)
    return VecGames(cfg, workers, seed="t", worker_factory=lambda i: factory(i, cfg))


def random_masks(n: int, rng) -> np.ndarray:
    m = np.zeros((n, L.MASK_SIZE), bool)
    m[:, L.MASK_INTENT + L.NOOP] = True
    m[:, L.MASK_INTENT + L.ATTACK] = True
    m[:, L.MASK_TARGET + L.ATTACK * L.NUM_SLOTS : L.MASK_TARGET + (L.ATTACK + 1) * L.NUM_SLOTS] = rng.random((n, L.NUM_SLOTS)) < 0.3
    m[:, L.MASK_TARGET + L.ATTACK * L.NUM_SLOTS] = True
    m[:, L.MASK_INTENT + L.BOAT] = True
    m[:, L.MASK_SPATIAL : L.MASK_SPATIAL + L.RASTER_CELLS] = rng.random((n, L.RASTER_CELLS)) < 0.01
    m[:, L.MASK_SPATIAL] = True
    m[:, L.MASK_RATIO : L.MASK_SPATIAL] = True
    return m


# ----- layout / protocol ------------------------------------------------------

def test_layout_refuses_drift():
    remote = L.layout_dict() | {"maskSize": 1}
    with pytest.raises(RuntimeError, match="maskSize"):
        L.check_layout(remote)


def test_step_encoding_roundtrip():
    actions = np.array([[1, 2, 3, 8191, 31], [0, 0, 0, 0, 0]])
    body = P.encode_step(actions, ["a", ""], [1])
    (n,) = np.frombuffer(body[:4], "<u4")
    assert n == 2
    assert np.frombuffer(body[4:24], "<u2").reshape(2, 5).tolist() == actions.tolist()


# ----- vec ---------------------------------------------------------------------

def test_vec_reset_step_and_game_reset():
    servers = []
    vec = make_vec(servers, game_length=3)
    batch = vec.reset()
    assert vec.num_rows == 8
    assert np.all(batch["flags"] == L.FLAG_NEXT | L.FLAG_FRESH)
    for _ in range(2):
        batch = vec.step(np.zeros((8, 5)))
        assert np.all(batch["flags"] == L.FLAG_VALID | L.FLAG_NEXT)
    batch = vec.step(np.zeros((8, 5)))
    assert np.all(batch["flags"] == L.FLAG_VALID | L.FLAG_DONE | L.FLAG_NEXT | L.FLAG_FRESH)
    assert np.all(batch["terminal"] == 3)
    vec.close()


def test_vec_routes_aborts_to_the_right_worker():
    servers = []
    vec = make_vec(servers, game_length=50)
    vec.reset()
    batch = vec.step(np.zeros((8, 5)), abort_games={3})
    assert servers[1].aborts == [1] and servers[0].aborts == []
    assert batch["terminal"][6:8].tolist() == [5, 5]
    assert batch["terminal"][:6].tolist() == [0] * 6
    vec.close()


def test_vec_respawns_a_dead_worker():
    servers = []
    vec = make_vec(servers, game_length=50)
    vec.reset()
    vec.clients[0].sock.close()
    with pytest.warns(RuntimeWarning, match="respawning"):
        batch = vec.step(np.zeros((8, 5)))
    assert batch["terminal"][:4].tolist() == [L.WORKER_DEATH] * 4
    assert np.all(batch["flags"][:4] & L.FLAG_FRESH)
    assert np.all(batch["terminal"][4:] == 0)
    vec.close()


# ----- agents and model ---------------------------------------------------------

def assert_legal(actions: np.ndarray, masks: np.ndarray):
    heads = L.split_masks(masks)
    for i, (intent, target, ratio, cell, site) in enumerate(actions):
        assert heads["intent"][i, intent]
        if L.USES_TARGET[intent]:
            assert heads["target"][i, intent, target]
        if L.SPATIAL_KIND[intent] >= 0:
            assert heads["spatial"][i, L.SPATIAL_KIND[intent], cell]


def test_agents_only_take_legal_actions():
    rng = np.random.default_rng(0)
    masks = random_masks(32, rng)
    features = rng.random((32, L.FEATURE_SIZE)).astype(np.float32)
    raster = np.zeros((32, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH), np.uint8)
    for agent in (RandomAgent(1), ScriptedAgent()):
        assert_legal(agent.act(raster, features, masks), masks)


def test_policy_samples_legally_and_rescores_exactly(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    model = OpenFrontPolicy()
    masks = torch.as_tensor(random_masks(16, rng))
    raster = torch.randint(0, 255, (16, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH), dtype=torch.uint8)
    feats = torch.rand(16, L.FEATURE_SIZE)
    out = model.act(raster, feats, masks)
    actions = out["actions"].numpy()
    assert_legal(actions, masks.numpy())
    # Unused heads are zeroed so stored actions are canonical.
    assert np.all(actions[actions[:, 0] == L.NOOP][:, 1:] == 0)
    logp, ent, value = model.evaluate(raster, feats, out["rows"], out["actions"])
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert set(ent) == {"intent", "target", "ratio", "spatial", "site"}
    # Noop rows carry no conditional-head entropy.
    noop = out["actions"][:, 0] == L.NOOP
    assert torch.all(ent["target"][noop] == 0)

    path = tmp_path / "p.pt"
    save_checkpoint(path, model, meta={"step": 5})
    loaded, payload = load_checkpoint(path)
    assert payload["meta"]["step"] == 5
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="v1"):
        load_checkpoint(tmp_path / "old.zip")


# ----- GAE ---------------------------------------------------------------------

def test_gae_skips_steps_a_row_did_not_act_in():
    ro = Rollout(4, 1)
    ro.stored[:, 0] = [True, True, False, False]  # acted twice, then dead/inactive
    ro.reward[:, 0] = [1.0, -2.0, 0.0, 0.0]
    ro.value[:, 0] = [0.5, 0.25, 9.0, 9.0]
    ro.done[:, 0] = [False, True, False, False]
    adv, ret = ro.gae(np.array([100.0]), gamma=0.9, lam=1.0)
    # Step 1 is terminal: no bootstrap. Step 0 bootstraps from step 1 only.
    assert adv[1, 0] == pytest.approx(-2.0 - 0.25)
    assert ret[0, 0] == pytest.approx(1.0 + 0.9 * -2.0)
    assert adv[2, 0] == 0 and adv[3, 0] == 0


def test_gae_bootstraps_a_continuing_row():
    ro = Rollout(2, 1)
    ro.stored[:, 0] = True
    ro.reward[:, 0] = [0.0, 0.0]
    ro.value[:, 0] = [0.0, 0.0]
    adv, ret = ro.gae(np.array([1.0]), gamma=0.5, lam=1.0)
    assert ret[1, 0] == pytest.approx(0.5)
    assert ret[0, 0] == pytest.approx(0.25)


# ----- self-play pool and config -----------------------------------------------

def test_seat_assignment_keeps_seat_zero_learner():
    rng = random.Random(0)
    for _ in range(50):
        seats = assign_seats(4, 3, 0.5, rng)
        assert seats[0] == LEARNER
        assert all(s == LEARNER or 0 <= s < 3 for s in seats)
    assert assign_seats(4, 0, 0.0, rng) == [LEARNER] * 4


def test_pool_prunes_and_records(tmp_path):
    pool = OpponentPool(tmp_path, max_size=4)
    model = OpenFrontPolicy()
    for step in range(6):
        pool.add(model, step * 100)
    assert len(pool.snapshots) == 4
    assert pool.snapshots[-1].step == 500
    assert len(list(tmp_path.glob("snapshot_*.pt"))) == 4
    picks = pool.sample(10, 1.0, random.Random(0))
    assert {p.step for p in picks} <= {s.step for s in pool.snapshots[-3:]}
    pool.record(pool.snapshots[0].path, True)
    pool.save()
    reloaded = OpponentPool(tmp_path, max_size=4)
    assert reloaded.snapshots[0].games == 1 and reloaded.snapshots[0].learner_wins == 1


def test_config_rejects_typos_and_applies_curriculum(tmp_path):
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(overrides={"ppo": {"lerning_rate": 1}})
    cfg = load_config(overrides={"curriculum": [{"bots": 3, "nations": "disabled", "promote": {"survival": 0.5}}, {"bots": 20}]})
    assert stage_env(cfg, 0)["bots"] == 3 and stage_env(cfg, 0)["nations"] == "disabled"
    assert stage_env(cfg, 1)["bots"] == 20


# ----- trainer end to end --------------------------------------------------------

def tiny_config(tmp_path, **run):
    return load_config(overrides={
        "run": {"name": "t", "total_steps": 96, "device": "cpu", "checkpoint_dir": str(tmp_path / "ck"),
                "log_dir": str(tmp_path / "runs"), "checkpoint_every": 1, **run},
        "env": {"workers": 2, "games_per_worker": 1, "seats": 2},
        "ppo": {"n_steps": 12, "epochs": 1, "minibatch": 16, "amp": False},
        "selfplay": {"snapshot_every": 1, "active_opponents": 1, "learner_seat_prob": 0.0},
        "model": {"trunk": 32, "slot_dim": 8, "site_dim": 8, "context": 16},
    })


def test_trainer_runs_self_play_end_to_end(tmp_path):
    servers = []
    factory = fake_factory(servers, game_length=5, death_at={1: 2})
    trainer = Trainer(tiny_config(tmp_path), worker_factory=factory, log=lambda m: None)
    result = trainer.train()
    assert result["global_step"] >= 96
    ck = tmp_path / "ck" / "t"
    assert (ck / "t_final.pt").is_file() and (ck / "t_latest.pt").is_file()
    assert len(trainer.pool.snapshots) >= 1
    # Once the pool exists, seat 1 is a frozen opponent (learner_seat_prob 0)
    # and its games feed the head-to-head record.
    assert sum(s.games for s in trainer.pool.snapshots) > 0
    lines = (tmp_path / "runs" / "t" / "metrics.jsonl").read_text().splitlines()
    assert lines and "rollout/survival_rate" in lines[-1]

    resumed = Trainer(tiny_config(tmp_path, total_steps=150), resume=str(ck / "t_latest.pt"),
                      worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    assert resumed.global_step == result["global_step"]
    assert resumed.train()["global_step"] >= 150


def test_trainer_stops_on_request(tmp_path):
    cfg = tiny_config(tmp_path, total_steps=10_000)
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    original = trainer._update

    def update_then_stop(*args, **kwargs):
        (tmp_path / "ck" / "t" / "STOP").touch()
        return original(*args, **kwargs)

    trainer._update = update_then_stop
    result = trainer.train()
    assert result["updates"] == 1
    assert (tmp_path / "ck" / "t" / "t_final.pt").is_file()


def test_evaluate_reports_every_predictor():
    from evaluate import evaluate

    result = evaluate(["scripted", "noop"], games=3, workers=2, log=lambda m: None,
                      worker_factory=fake_factory([], game_length=4, death_at={1: 2}))
    preds = result["predictors"]
    assert preds["scripted"]["episodes"] + preds["noop"]["episodes"] == 6
    assert set(result["head_to_head"]) <= {"scripted > noop", "noop > scripted"}
