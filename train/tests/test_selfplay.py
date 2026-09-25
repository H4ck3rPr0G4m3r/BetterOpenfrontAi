"""Tests for the self-play pool and curriculum (stdlib only, no worker)."""

import random

import pytest

from selfplay.curriculum import STAGES, next_stage, should_promote, stage_by_name
from selfplay.pool import OpponentPool, selfplay_ready


def test_pool_sample_strategies_and_persistence(tmp_path):
    pool = OpponentPool(tmp_path / "pool")
    assert pool.sample() is None
    assert pool.best() is None
    # Fake checkpoints: real zip bytes are unnecessary, copyfile only moves bytes.
    sources = []
    for step in (100, 200, 300):
        src = tmp_path / f"ckpt{step}.zip"
        src.write_bytes(b"fake-checkpoint-%d" % step)
        sources.append((src, step))
    snaps = [pool.add_snapshot(src, step) for src, step in sources]
    assert [s.step for s in snaps] == [100, 200, 300]
    # Idempotent per step.
    assert pool.add_snapshot(sources[0][0], 100).step == 100
    assert len(pool.snapshots) == 3

    rng = random.Random(0)
    assert pool.sample(strategy="latest", rng=rng).step == 300
    assert pool.sample(strategy="last_k", last_k=1, rng=rng).step == 300
    with pytest.raises(ValueError):
        pool.sample(strategy="nonsense", rng=rng)

    pool.record_result(snaps[0], True)
    pool.record_result(snaps[0], False)
    assert pool.snapshots[0].games == 2
    assert pool.snapshots[0].wins == 1

    reloaded = OpponentPool.load(tmp_path / "pool")
    assert [s.step for s in reloaded.snapshots] == [100, 200, 300]
    assert reloaded.snapshots[0].games == 2
    # Best prefers played snapshots by win rate.
    assert reloaded.best().step == 100


def test_pool_rejects_missing_source(tmp_path):
    pool = OpponentPool(tmp_path / "pool")
    with pytest.raises(FileNotFoundError):
        pool.add_snapshot(tmp_path / "nope.zip", 1)


def test_selfplay_gate():
    assert not selfplay_ready(0.49)
    assert selfplay_ready(0.5)
    assert selfplay_ready(0.9)
    assert not selfplay_ready(0.9, threshold=0.95)


def test_curriculum_stages_and_promotion():
    assert [s.name for s in STAGES] == ["stage1", "stage2", "stage3"]
    stage1 = stage_by_name("stage1")
    assert (stage1.bots, stage1.nations) == (5, "disabled")
    assert stage_by_name("stage3").bots == 20
    with pytest.raises(ValueError):
        stage_by_name("stage99")

    assert not should_promote(stage1, 0.9, 5)  # too few episodes
    assert not should_promote(stage1, 0.1, 10)  # bots not beaten
    assert should_promote(stage1, 0.8, 10)  # pool dormant: bots suffice
    assert not should_promote(stage1, 0.8, 10, pool_win_rate=0.0)
    assert should_promote(stage1, 0.8, 10, pool_win_rate=0.9)
    # Final stage never promotes.
    assert next_stage("stage1") == "stage2"
    assert next_stage("stage2") == "stage3"
    assert next_stage("stage3") is None
