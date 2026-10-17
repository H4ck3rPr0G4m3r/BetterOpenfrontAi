"""Training stack tests against the in-process fake worker (no Node needed)."""

from __future__ import annotations

import json
import random
from collections import Counter

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

    # A line from updates the checkpoint never saw (a killed run).
    metrics = tmp_path / "runs" / "t" / "metrics.jsonl"
    with open(metrics, "a") as handle:
        handle.write(json.dumps({"step": result["global_step"] + 999, "update": 99}) + "\n")
    resumed = Trainer(tiny_config(tmp_path, total_steps=150), resume=str(ck / "t_latest.pt"),
                      worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    assert resumed.global_step == result["global_step"]
    # Recent episodes resume too, so the first stats are not deaths only.
    assert list(resumed.episodes) == list(trainer.episodes) and resumed.episodes
    assert list(resumed.head_to_head) == list(trainer.head_to_head)
    assert resumed.train()["global_step"] >= 150
    steps = [json.loads(line)["step"] for line in metrics.read_text().splitlines()]
    assert steps == sorted(steps)
    assert "update\": 99" in (metrics.parent / "metrics.rewound.jsonl").read_text()


def test_latest_checkpoint_is_saved_every_update(tmp_path):
    cfg = tiny_config(tmp_path, total_steps=10_000)
    cfg["run"]["checkpoint_every"] = 1000
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    original = trainer._update
    seen = []

    def update_then_stop(*args, **kwargs):
        if trainer.update == 1:
            latest = tmp_path / "ck" / "t" / "t_latest.pt"
            seen.append(torch.load(latest, weights_only=False)["meta"]["update"])
            (tmp_path / "ck" / "t" / "STOP").touch()
        return original(*args, **kwargs)

    trainer._update = update_then_stop
    trainer.train()
    assert seen == [1]


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


# ----- mixed scenarios -----------------------------------------------------------

def test_scenarios_split_workers_by_weight():
    from openfront.config import scenario_workers

    cfg = load_config(overrides={
        "env": {"workers": 10},
        "scenarios": [{"name": "solo", "weight": 3, "seats": 1}, {"name": "ffa", "weight": 4, "seats": 4},
                      {"name": "duel", "weight": 2, "seats": 2, "close_spawn": 1.0}, {"name": "crowd", "weight": 1, "seats": 6}],
    })
    plan = scenario_workers(cfg)
    counts = {name: sum(1 for n, _ in plan if n == name) for name in ("solo", "ffa", "duel", "crowd")}
    assert sum(counts.values()) == 10 and min(counts.values()) >= 1
    assert counts["ffa"] >= counts["solo"] >= counts["duel"] >= counts["crowd"]
    duel = next(env for n, env in plan if n == "duel")
    assert duel["seats"] == 2 and duel["close_spawn"] == 1.0
    with pytest.raises(ValueError):
        load_config(overrides={"scenarios": [{"name": "x", "flavor": 1}]})
    with pytest.raises(ValueError):
        load_config(overrides={"scenarios": [{"name": "x"}], "curriculum": [{"weights": {"nope": 1}}]})
    with pytest.raises(ValueError):
        load_config(overrides={"env": {"difficulty": "Brutal"}})


def test_curriculum_stages_reweight_scenarios_and_set_difficulty():
    from openfront.config import scenario_workers

    cfg = load_config("configs/main.yaml")
    early, late = Counter(n for n, _ in scenario_workers(cfg, 0)), Counter(n for n, _ in scenario_workers(cfg, 3))
    assert early["easy"] > late["easy"] and early["close_rivals"] <= late["close_rivals"]
    assert all(env["difficulty"] == "Easy" for _, env in scenario_workers(cfg, 0))
    # Later stages mix difficulties over the workers with nations, Medium growing.
    medium = [sum(env["difficulty"] == "Medium" for _, env in scenario_workers(cfg, s) if env["nations"] != "disabled")
              for s in (1, 2, 3)]
    assert 0 < medium[0] < medium[1] <= medium[2]
    assert any(env["difficulty"] == "Easy" for _, env in scenario_workers(cfg, 3) if env["nations"] != "disabled")
    assert all(env["nations"] == "disabled" for n, env in scenario_workers(cfg, 0) if n == "easy")
    assert sum(1 for _ in scenario_workers(cfg, 0)) == cfg["env"]["workers"]


def test_difficulty_mix_splits_workers_and_interleaves():
    from openfront.config import scenario_workers

    cfg = load_config(overrides={
        "env": {"workers": 8},
        "scenarios": [{"name": "a", "seats": 1}, {"name": "b", "seats": 1},
                      {"name": "own", "seats": 1, "difficulty": "Hard"}, {"name": "calm", "seats": 1, "nations": "disabled"}],
        "curriculum": [{"difficulty": {"Easy": 1, "Medium": 1}, "weights": {"a": 2, "b": 2, "own": 1, "calm": 1}}],
    })
    plan = scenario_workers(cfg)
    assert {env["difficulty"] for n, env in plan if n == "own"} == {"Hard"}  # a scenario's own difficulty wins
    mixed = [env["difficulty"] for n, env in plan if n in ("a", "b")]
    assert Counter(mixed) == {"Easy": len(mixed) // 2, "Medium": len(mixed) - len(mixed) // 2}
    for name in ("a", "b"):  # each multi-worker scenario sees both
        assert {env["difficulty"] for n, env in plan if n == name} == {"Easy", "Medium"}
    with pytest.raises(ValueError):
        load_config(overrides={"curriculum": [{"difficulty": {"Brutal": 1}}]})
    with pytest.raises(ValueError):
        load_config(overrides={"curriculum": [{"difficulty": {"Easy": 0}}]})


def test_promotion_can_be_judged_on_the_hard_games_only(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["curriculum"] = [{"difficulty": {"Easy": 1, "Medium": 1},
                          "promote": {"survival": 0.5, "min_episodes": 4, "difficulty": "Medium"}}, {}]
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    ep = lambda term, level: {"terminal": term, "scenario": "default", "difficulty": level}
    trainer.stage_episodes.extend([ep("tick_cap", "Easy")] * 10 + [ep("death", "Medium")] * 4)
    assert not trainer._maybe_promote()  # Easy wins cannot carry it
    trainer.stage_episodes.extend([ep("tick_cap", "Medium")] * 4)
    assert trainer._maybe_promote() and trainer.stage == 1


def test_promotion_waits_for_min_steps_in_the_stage(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["curriculum"] = [{"promote": {"survival": 0.5, "min_episodes": 4, "min_steps": 1000}},
                         {"promote": {"survival": 0.5, "min_episodes": 4, "min_steps": 1000}}, {}]
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    won = [{"terminal": "tick_cap", "scenario": "default"}] * 4
    trainer.stage_episodes.extend(won)
    trainer.global_step = 999
    assert not trainer._maybe_promote()  # surviving, but too early
    trainer.global_step = 1000
    assert trainer._maybe_promote() and trainer.stage == 1 and trainer.stage_start_step == 1000
    trainer.stage_episodes.extend(won)
    trainer.global_step = 1500
    assert not trainer._maybe_promote()  # counted from the stage's own start
    trainer.global_step = 2000
    assert trainer._maybe_promote() and trainer.stage == 2


def test_promotion_is_judged_on_the_listed_scenarios(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["scenarios"] = [{"name": "easy", "seats": 1}, {"name": "hard", "seats": 1}]
    cfg["curriculum"] = [{"promote": {"survival": 0.5, "min_episodes": 4, "scenarios": ["hard"]}}, {}]
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    alive = lambda sc: {"terminal": "tick_cap", "scenario": sc}
    dead = lambda sc: {"terminal": "death", "scenario": sc}
    trainer.stage_episodes.extend([alive("easy")] * 10 + [dead("hard")] * 4)
    assert not trainer._maybe_promote()  # easy games cannot carry it
    trainer.stage_episodes.extend([alive("hard")] * 4)
    assert trainer._maybe_promote() and trainer.stage == 1


def test_vec_games_with_mixed_seat_counts():
    servers = []
    configs = [WorkerConfig(games=1, seats=1), WorkerConfig(games=2, seats=3), WorkerConfig(games=1, seats=2)]
    vec = VecGames(configs, 3, worker_factory=lambda i: fake_factory(servers, game_length=4)(i, configs[i]))
    try:
        assert vec.num_rows == 1 + 6 + 2 and vec.num_games == 4 and vec.seats is None and vec.max_seats == 3
        assert [list(vec.rows_of_game(g)) for g in range(4)] == [[0], [1, 2, 3], [4, 5, 6], [7, 8]]
        assert [vec.worker_of_game(g) for g in range(4)] == [0, 1, 1, 2]
        batch = vec.reset()
        for _ in range(6):
            batch = vec.step(np.zeros((vec.num_rows, 5), np.int64), abort_games={2})
        assert batch["flags"].shape == (9,)
    finally:
        vec.close()


def test_trainer_runs_mixed_scenarios(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["env"]["workers"] = 3
    cfg["scenarios"] = [{"name": "solo", "seats": 1}, {"name": "duel", "seats": 2, "close_spawn": 1.0},
                        {"name": "ffa", "seats": 3}]
    trainer = Trainer(cfg, worker_factory=fake_factory([], game_length=5), log=lambda m: None)
    trainer.train()
    last = json.loads((tmp_path / "runs" / "t" / "metrics.jsonl").read_text().splitlines()[-1])
    assert {"scenario/solo/survival_rate", "scenario/duel/survival_rate", "scenario/ffa/survival_rate"} <= set(last)


def test_v3_checkpoints_migrate_to_v4_and_act_as_before(tmp_path):
    from openfront.model import V3, load_checkpoint, save_checkpoint

    fresh = OpenFrontPolicy()
    # A v3 checkpoint is the v4 network without the nuke parts.
    cut = {"stem.0.weight": (slice(None), slice(0, V3["planes"])), "global_mlp.0.weight": (slice(None), slice(0, V3["globals"])),
           "intent_head.weight": (slice(0, V3["intents"]),), "intent_head.bias": (slice(0, V3["intents"]),),
           "intent_embed.weight": (slice(0, V3["intents"]),), "spatial_out.weight": (slice(0, V3["spatial_kinds"]),),
           "spatial_out.bias": (slice(0, V3["spatial_kinds"]),)}
    old = {k: (v[cut[k]] if k in cut else v).clone() for k, v in fresh.state_dict().items()}
    path = tmp_path / "v3.pt"
    save_checkpoint(path, fresh, meta={"global_step": 7})
    payload = torch.load(path, weights_only=False)
    payload["state_dict"] = old
    payload["layout"] = {**payload["layout"], "version": 3}
    payload["optimizer"] = {"state": {}}
    torch.save(payload, path)

    model, loaded = load_checkpoint(path)
    assert loaded["migrated_from"] == 3 and "optimizer" not in loaded and loaded["meta"]["global_step"] == 7
    assert torch.equal(model.stem[0].weight[:, :V3["planes"]], old["stem.0.weight"])
    # Whatever the new planes and globals hold, the old network's outputs do not move.
    raster = torch.randint(0, 255, (2, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH), dtype=torch.uint8)
    features = torch.rand(2, L.FEATURE_SIZE)
    blank_r, blank_f = raster.clone(), features.clone()
    blank_r[:, V3["planes"]:] = 0
    blank_f[:, L.SLOT_TABLE + L.SITE_TABLE + V3["globals"]:] = 0
    with torch.no_grad():
        a, b = model.encode(raster, features), model.encode(blank_r, blank_f)
        assert torch.allclose(model.intent_head(a["h"]), model.intent_head(b["h"]))
        assert torch.allclose(model.value(a), model.value(b))
