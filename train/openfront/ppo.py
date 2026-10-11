"""PPO with real self-play over multi-seat games.

Rollout: every step, rows that act next are split by controller — the
live learner (batched through the current policy, stored for training) or
a frozen pool snapshot (batched per snapshot, never stored). Games where
no learner seat is still playing are aborted so compute goes to learner
data. GAE runs per row and simply skips steps a row did not act in (dead
seats wait for their game to end; their rows carry nothing).

Time limits: the tick cap is a true terminal of the time-aware MDP (the
observation carries the time fraction, the reward pays placement there),
so there is no truncation bootstrap. A rollout boundary bootstraps from
V(s_T) as usual.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import layout as L
from .config import scenario_workers, worker_config
from .model import ModelConfig, OpenFrontPolicy, load_checkpoint, parameter_count, save_checkpoint
from .selfplay import LEARNER, OpponentPool, assign_seats
from .vec import VecGames

ROW_KEYS = ("intent", "target", "ratio", "spatial", "site")
ROW_SIZES = {"intent": L.NUM_INTENTS, "target": L.NUM_SLOTS, "ratio": L.NUM_RATIOS, "spatial": L.RASTER_CELLS, "site": L.MAX_BUILD_SITES}


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class Rollout:
    """Fixed [T, R] storage; ``stored`` marks learner transitions."""

    def __init__(self, steps: int, rows: int):
        self.T, self.R = steps, rows
        self.raster = np.zeros((steps, rows, L.RASTER_PLANES, L.RASTER_HEIGHT, L.RASTER_WIDTH), np.uint8)
        self.features = np.zeros((steps, rows, L.FEATURE_SIZE), np.float32)
        self.rows = {k: np.zeros((steps, rows, n), bool) for k, n in ROW_SIZES.items()}
        self.actions = np.zeros((steps, rows, 5), np.int64)
        self.logp = np.zeros((steps, rows), np.float32)
        self.value = np.zeros((steps, rows), np.float32)
        self.reward = np.zeros((steps, rows), np.float32)
        self.done = np.zeros((steps, rows), bool)
        self.stored = np.zeros((steps, rows), bool)

    def clear(self) -> None:
        self.stored[:] = False
        self.done[:] = False
        self.reward[:] = 0.0

    def gae(self, last_value: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        adv = np.zeros((self.T, self.R), np.float32)
        next_value = last_value.astype(np.float32).copy()
        last_adv = np.zeros(self.R, np.float32)
        for t in reversed(range(self.T)):
            s = self.stored[t]
            nonterminal = 1.0 - self.done[t].astype(np.float32)
            delta = self.reward[t] + gamma * next_value * nonterminal - self.value[t]
            a = delta + gamma * lam * nonterminal * last_adv
            adv[t] = np.where(s, a, 0.0)
            next_value = np.where(s, self.value[t], next_value)
            last_adv = np.where(s, a, last_adv)
        return adv, adv + self.value


class Trainer:
    def __init__(self, cfg: dict, resume: str | None = None, worker_factory=None, log=print):
        self.cfg = cfg
        self.log = log
        run = cfg["run"]
        self.name = str(run["name"])
        self.device = resolve_device(str(run["device"]))
        self.rng = random.Random(int(run["seed"]))
        torch.manual_seed(int(run["seed"]))
        self.ckpt_dir = Path(run["checkpoint_dir"]) / self.name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(run["log_dir"]) / self.name
        self.worker_factory = worker_factory
        self.global_step = 0
        self.update = 0
        self.stage = 0
        if resume:
            self.model, payload = load_checkpoint(resume, self.device)
            meta = payload.get("meta", {})
            self.global_step = int(meta.get("global_step", 0))
            self.update = int(meta.get("update", 0))
            self.stage = int(meta.get("stage", 0))
            self._resumed_meta = meta
        else:
            self.model = OpenFrontPolicy(ModelConfig(**cfg["model"])).to(self.device)
            payload = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg["ppo"]["lr"]), eps=1e-5)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        sp = cfg["selfplay"]
        self.pool = OpponentPool(Path(sp["pool_dir"] or self.ckpt_dir / "pool"), int(sp["pool_size"]))
        self.opponents: list[tuple[str, OpenFrontPolicy]] = []
        self.episodes: deque = deque(maxlen=400)
        self.row_scenario: list[str] = []
        self.row_difficulty: list[str] = []
        self.stage_episodes: deque = deque(maxlen=1000)
        self.head_to_head: deque = deque(maxlen=400)
        self.amp = bool(cfg["ppo"]["amp"]) and self.device == "cuda"
        # Recent episodes come back with the weights: the first rollouts
        # after a restart only see games that end early (deaths), so stats
        # and curriculum judged on those alone read as a collapse.
        meta = getattr(self, "_resumed_meta", {})
        self.episodes.extend(meta.get("episodes", []))
        self.stage_episodes.extend(meta.get("stage_episodes", []))
        self.head_to_head.extend(meta.get("head_to_head", []))

    # ----- environment ---------------------------------------------------

    def _apply_replay_env(self) -> None:
        replay = self.cfg["run"].get("replay")
        if replay:
            os.environ["OPENFRONT_REPLAY_DIR"] = str(Path(replay["dir"]).resolve())
            os.environ["OPENFRONT_REPLAY_EVERY"] = str(int(replay.get("every", 200)))
        else:
            os.environ.pop("OPENFRONT_REPLAY_DIR", None)
            os.environ.pop("OPENFRONT_REPLAY_EVERY", None)

    def build_vec(self) -> VecGames:
        self._apply_replay_env()
        seed = f"{self.name}-s{self.stage}-u{self.update}-{self.cfg['run']['seed']}"
        plan = scenario_workers(self.cfg, self.stage)
        wcs = [worker_config(env, self.cfg["reward"]) for _, env in plan]
        names = [name for name, _ in plan]
        # Nation difficulty per worker ("" without nations), for per-difficulty stats.
        levels = [env["difficulty"] if env["nations"] != "disabled" else "" for _, env in plan]
        factory = None if self.worker_factory is None else (lambda i: self.worker_factory(i, wcs[i]))
        vec = VecGames(wcs, len(wcs), seed=seed, worker_factory=factory)
        # Scenario name per row, for per-scenario episode stats.
        self.row_scenario = [names[vec.worker_of_game(vec.game_of_row(r))] for r in range(vec.num_rows)]
        self.row_difficulty = [levels[vec.worker_of_game(vec.game_of_row(r))] for r in range(vec.num_rows)]
        return vec

    def refresh_opponents(self) -> None:
        sp = self.cfg["selfplay"]
        picks = self.pool.sample(int(sp["active_opponents"]), float(sp["latest_bias"]), self.rng)
        loaded = []
        for snap in picks:
            model, _ = load_checkpoint(snap.path, self.device)
            model.eval()
            loaded.append((snap.path, model))
        self.opponents = loaded

    # ----- main loop -------------------------------------------------------

    def train(self) -> dict:
        cfg, ppo = self.cfg, self.cfg["ppo"]
        total = int(cfg["run"]["total_steps"])
        writer = SummaryWriter(str(self.log_dir))
        metrics_path = self.log_dir / "metrics.jsonl"
        self._rewind_metrics(metrics_path)
        self.log(f"[{self.name}] device {self.device}, {parameter_count(self.model):,} params, "
                 f"stage {self.stage}, resuming at step {self.global_step:,}")
        self.refresh_opponents()
        vec = self.build_vec()
        seats = vec.max_seats
        batch = vec.reset()
        ctrl = np.full(vec.num_rows, LEARNER, np.int64)
        opp_path = [""] * vec.num_rows
        self._assign(vec, range(vec.num_games), ctrl, opp_path)
        ep_ret = np.zeros(vec.num_rows, np.float64)
        ep_len = np.zeros(vec.num_rows, np.int64)
        rollout = Rollout(int(ppo["n_steps"]), vec.num_rows)
        stop_file = self.ckpt_dir / "STOP"
        stop_file.unlink(missing_ok=True)
        stopped = False
        try:
            while self.global_step < total and not stopped:
                t0 = time.monotonic()
                rollout.clear()
                intents = Counter()
                rows_stepped = 0
                for t in range(rollout.T):
                    acting = (batch["flags"] & L.FLAG_NEXT) != 0
                    actions = np.zeros((vec.num_rows, 5), np.int64)
                    learner = np.flatnonzero(acting & (ctrl == LEARNER))
                    if len(learner):
                        out = self._act(self.model, batch, learner)
                        actions[learner] = out["actions"]
                        rollout.raster[t, learner] = batch["raster"][learner]
                        rollout.features[t, learner] = batch["features"][learner]
                        for k in ROW_KEYS:
                            rollout.rows[k][t, learner] = out["rows"][k]
                        rollout.actions[t, learner] = out["actions"]
                        rollout.logp[t, learner] = out["logp"]
                        rollout.value[t, learner] = out["value"]
                        rollout.stored[t, learner] = True
                        intents.update(out["actions"][:, 0].tolist())
                    for k, (_, opp) in enumerate(self.opponents):
                        idx = np.flatnonzero(acting & (ctrl == k))
                        if len(idx):
                            actions[idx] = self._act(opp, batch, idx)["actions"]
                    # Spend no compute on games with no learner left in them.
                    abort = {g for g in range(vec.num_games)
                             if acting[vec.rows_of_game(g)].any() and not (acting & (ctrl == LEARNER))[vec.rows_of_game(g)].any()}
                    batch = vec.step(actions, abort)
                    rows_stepped += int(acting.sum())
                    flags = batch["flags"]
                    valid = (flags & L.FLAG_VALID) != 0
                    done = (flags & L.FLAG_DONE) != 0
                    stored = rollout.stored[t]
                    rollout.reward[t] = np.where(stored, batch["reward"], 0.0)
                    rollout.done[t] = stored & done
                    ep_ret += np.where(valid, batch["reward"], 0.0)
                    ep_len += valid
                    for row in np.flatnonzero(done & valid & stored):
                        self._finish_episode(row, ep_ret[row], ep_len[row], batch)
                    fresh_games = sorted({vec.game_of_row(r) for r in np.flatnonzero(flags & L.FLAG_FRESH)})
                    if fresh_games:
                        self._head_to_head(vec, fresh_games, ctrl, opp_path, ep_ret)
                        for g in fresh_games:
                            rs = list(vec.rows_of_game(g))
                            ep_ret[rs] = 0.0
                            ep_len[rs] = 0
                        self._assign(vec, fresh_games, ctrl, opp_path)
                    self.global_step += len(learner)

                # Bootstrap rows still mid-episode at the rollout boundary.
                last_value = np.zeros(vec.num_rows, np.float32)
                acting = (batch["flags"] & L.FLAG_NEXT) != 0
                cont = np.flatnonzero(acting & (ctrl == LEARNER) & ((batch["flags"] & L.FLAG_FRESH) == 0))
                if len(cont):
                    last_value[cont] = self._act(self.model, batch, cont)["value"]
                adv, ret = rollout.gae(last_value, float(ppo["gamma"]), float(ppo["gae_lambda"]))
                rollout_time = time.monotonic() - t0
                stats = self._update(rollout, adv, ret, total)
                self.update += 1
                elapsed = time.monotonic() - t0
                stats.update(self._episode_stats())
                stats["time/steps_per_second"] = int(rollout.stored.sum()) / max(elapsed, 1e-6)
                stats["time/rows_per_second"] = rows_stepped / max(rollout_time, 1e-6)
                stats["time/rollout_seconds"] = rollout_time
                stats["time/update_seconds"] = elapsed - rollout_time
                n_int = max(1, sum(intents.values()))
                for i, name in enumerate(L.INTENT_NAMES):
                    stats[f"intents/{name}"] = intents.get(i, 0) / n_int
                stats["selfplay/pool_size"] = len(self.pool.snapshots)
                stats["curriculum/stage"] = self.stage
                for key, value in stats.items():
                    writer.add_scalar(key, value, self.global_step)
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": self.global_step, "update": self.update, **stats}) + "\n")
                self.log(self._summary(stats))

                if self.update % int(cfg["selfplay"]["snapshot_every"]) == 0 and seats > 1:
                    self.pool.add(self.model, self.global_step)
                    self.refresh_opponents()
                if self.update % int(cfg["run"]["checkpoint_every"]) == 0:
                    self.save(f"{self.name}_{self.global_step}.pt")
                else:
                    # Latest after every update, so a crash or kill loses one.
                    self.save(None)
                if self._maybe_promote():
                    vec.close()
                    vec = self.build_vec()
                    seats = vec.max_seats
                    batch = vec.reset()
                    ctrl = np.full(vec.num_rows, LEARNER, np.int64)
                    opp_path = [""] * vec.num_rows
                    self._assign(vec, range(vec.num_games), ctrl, opp_path)
                    ep_ret = np.zeros(vec.num_rows, np.float64)
                    ep_len = np.zeros(vec.num_rows, np.int64)
                    rollout = Rollout(int(ppo["n_steps"]), vec.num_rows)
                if stop_file.exists():
                    self.log(f"[{self.name}] stop requested")
                    stopped = True
        finally:
            vec.close()
            final = self.save(f"{self.name}_final.pt")
            writer.close()
        return {"checkpoint": str(final), "global_step": self.global_step, "updates": self.update, **self._episode_stats()}

    # ----- pieces ------------------------------------------------------------

    def _act(self, model, batch, idx, deterministic=False) -> dict:
        d = self.device
        raster = torch.as_tensor(batch["raster"][idx]).to(d, non_blocking=True)
        feats = torch.as_tensor(batch["features"][idx]).to(d, non_blocking=True)
        masks = torch.as_tensor(batch["masks"][idx]).to(d, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = model.act(raster, feats, masks, deterministic=deterministic)
        return {
            "actions": out["actions"].cpu().numpy(),
            "logp": out["logp"].float().cpu().numpy(),
            "value": out["value"].float().cpu().numpy(),
            "rows": {k: v.cpu().numpy() for k, v in out["rows"].items()},
        }

    def _assign(self, vec, games, ctrl, opp_path) -> None:
        prob = float(self.cfg["selfplay"]["learner_seat_prob"])
        for g in games:
            rows = vec.rows_of_game(g)
            for row, who in zip(rows, assign_seats(len(rows), len(self.opponents), prob, self.rng)):
                ctrl[row] = who
                opp_path[row] = self.opponents[who][0] if who != LEARNER else ""

    def _finish_episode(self, row, ret, length, batch) -> None:
        stats = batch["stats"][row]
        record = {
            "return": float(ret),
            "length": int(length),
            "terminal": L.TERMINALS.get(int(batch["terminal"][row]), "unknown"),
            "tiles": float(stats[0]),
            "kills": float(stats[2]),
            "placement": float(stats[3]),
            "scenario": self.row_scenario[row] if row < len(self.row_scenario) else "default",
            "difficulty": self.row_difficulty[row] if row < len(self.row_difficulty) else "",
        }
        self.episodes.append(record)
        self.stage_episodes.append(record)

    def _head_to_head(self, vec, games, ctrl, opp_path, ep_ret) -> None:
        """Score finished mixed games: did the learner seats out-earn each snapshot seat?"""
        for g in games:
            rows = list(vec.rows_of_game(g))
            learner = [r for r in rows if ctrl[r] == LEARNER]
            if not learner:
                continue
            best_learner = max(ep_ret[r] for r in learner)
            for r in rows:
                if ctrl[r] != LEARNER and opp_path[r]:
                    won = best_learner > ep_ret[r]
                    self.pool.record(opp_path[r], won)
                    self.head_to_head.append(won)
        self.pool.save()

    def _episode_stats(self) -> dict:
        if not self.episodes:
            return {}
        eps = list(self.episodes)
        terms = Counter(e["terminal"] for e in eps)
        n = len(eps)
        out = {
            "rollout/ep_rew_mean": float(np.mean([e["return"] for e in eps])),
            "rollout/ep_len_mean": float(np.mean([e["length"] for e in eps])),
            "rollout/survival_rate": (terms["tick_cap"] + terms["win"]) / n,
            "rollout/final_tiles": float(np.mean([e["tiles"] for e in eps])),
            "rollout/kills": float(np.mean([e["kills"] for e in eps])),
            "rollout/placement": float(np.mean([e["placement"] for e in eps])),
        }
        scenarios = sorted({e.get("scenario", "default") for e in eps})
        if len(scenarios) > 1:
            for sc in scenarios:
                mine = [e for e in eps if e.get("scenario") == sc]
                out[f"scenario/{sc}/survival_rate"] = sum(e["terminal"] in ("tick_cap", "win") for e in mine) / len(mine)
                out[f"scenario/{sc}/ep_rew_mean"] = float(np.mean([e["return"] for e in mine]))
                out[f"scenario/{sc}/final_tiles"] = float(np.mean([e["tiles"] for e in mine]))
        levels = sorted({e.get("difficulty") for e in eps if e.get("difficulty")})
        if len(levels) > 1:
            for level in levels:
                mine = [e for e in eps if e.get("difficulty") == level]
                out[f"difficulty/{level}/survival_rate"] = sum(e["terminal"] in ("tick_cap", "win") for e in mine) / len(mine)
        for name in L.TERMINAL_NAMES:
            out[f"terminals/{name}"] = terms[name] / n
        if self.head_to_head:
            out["selfplay/learner_beats_pool"] = float(np.mean(self.head_to_head))
        return out

    def _maybe_promote(self) -> bool:
        stages = self.cfg["curriculum"]
        if self.stage >= len(stages) - 1:
            return False
        rule = stages[self.stage].get("promote") or {}
        need = int(rule.get("min_episodes", 200))
        if len(self.stage_episodes) < need:
            return False
        # Judged only on the listed scenarios and nation difficulties when
        # given (e.g. the Medium games of the scenarios with nations), so
        # easy games cannot carry a promotion.
        judged = set(rule.get("scenarios") or [])
        levels = rule.get("difficulty") or []
        levels = {levels} if isinstance(levels, str) else set(levels)
        pool = [e for e in self.stage_episodes if (not judged or e.get("scenario") in judged)
                and (not levels or e.get("difficulty") in levels)]
        if len(pool) < need:
            return False
        recent = pool[-need:]
        survival = sum(e["terminal"] in ("tick_cap", "win") for e in recent) / need
        if survival < float(rule.get("survival", 0.6)):
            return False
        self.stage += 1
        self.stage_episodes.clear()
        where = f" in {sorted(judged)}" if judged else ""
        where += f" at {'/'.join(sorted(levels))}" if levels else ""
        self.log(f"[{self.name}] curriculum: survival {survival:.2f}{where} over {need} episodes, promoted to stage {self.stage}")
        return True

    def _update(self, rollout: Rollout, adv: np.ndarray, ret: np.ndarray, total: int) -> dict:
        ppo = self.cfg["ppo"]
        t_idx, r_idx = np.nonzero(rollout.stored)
        n = len(t_idx)
        if n == 0:
            return {}
        progress = min(1.0, self.global_step / max(1, total))
        lr = float(ppo["lr"]) * (1.0 - (1.0 - float(ppo["lr_final_fraction"])) * progress)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        weights = ppo["ent_weights"]
        mb = int(ppo["minibatch"])
        clip = float(ppo["clip"])
        d = self.device
        sums: Counter = Counter()
        count = 0
        skipped = 0
        kl_stop = False
        # Advantages normalized once over the whole batch: a short last
        # minibatch (even one sample) would otherwise give a NaN std.
        stored = adv[t_idx, r_idx]
        adv = (adv - stored.mean()) / (stored.std() + 1e-8)
        self.model.train()
        for epoch in range(int(ppo["epochs"])):
            perm = np.random.permutation(n)
            epoch_kl = []
            for start in range(0, n, mb):
                sel = perm[start : start + mb]
                ti, ri = t_idx[sel], r_idx[sel]
                raster = torch.as_tensor(rollout.raster[ti, ri]).to(d, non_blocking=True)
                feats = torch.as_tensor(rollout.features[ti, ri]).to(d, non_blocking=True)
                rows = {k: torch.as_tensor(rollout.rows[k][ti, ri]).to(d) for k in ROW_KEYS}
                actions = torch.as_tensor(rollout.actions[ti, ri]).to(d)
                old_logp = torch.as_tensor(rollout.logp[ti, ri]).to(d)
                a = torch.as_tensor(adv[ti, ri]).to(d)
                target = torch.as_tensor(ret[ti, ri]).to(d)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    logp, ent, value = self.model.evaluate(raster, feats, rows, actions)
                logp, value = logp.float(), value.float()
                # Clamped so one stale or corrupt sample cannot overflow exp().
                log_ratio = (logp - old_logp).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                v_loss = 0.5 * (value - target).pow(2).mean()
                ent_terms = {k: e.float().mean() for k, e in ent.items()}
                entropy = sum(float(weights.get(k, 1.0)) * e for k, e in ent_terms.items())
                loss = pg_loss + float(ppo["vf_coef"]) * v_loss - float(ppo["ent_coef"]) * entropy
                self.optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(ppo["max_grad_norm"]))
                if not torch.isfinite(grad_norm):
                    # A non-finite step would poison every weight; drop it.
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > clip).float().mean().item()
                epoch_kl.append(approx_kl)
                sums.update({
                    "train/policy_gradient_loss": pg_loss.item(),
                    "train/value_loss": v_loss.item(),
                    "train/entropy_loss": -entropy.item(),
                    "train/approx_kl": approx_kl,
                    "train/clip_fraction": clipfrac,
                    "train/grad_norm": float(grad_norm),
                    **{f"entropy/{k}": v.item() for k, v in ent_terms.items()},
                })
                count += 1
            if float(ppo["target_kl"]) > 0 and epoch_kl and np.mean(epoch_kl) > float(ppo["target_kl"]):
                kl_stop = True
                break
        self.model.eval()
        if skipped:
            self.log(f"[{self.name}] skipped {skipped} non-finite minibatches")
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("policy weights went non-finite; refusing to continue (last good checkpoint is intact)")
        stats = {k: v / max(count, 1) for k, v in sums.items()}
        stats["train/skipped_minibatches"] = skipped
        values, returns = rollout.value[t_idx, r_idx], ret[t_idx, r_idx]
        var = np.var(returns)
        stats["train/explained_variance"] = float(1 - np.var(returns - values) / var) if var > 1e-8 else 0.0
        stats["train/learning_rate"] = lr
        stats["train/kl_early_stop"] = float(kl_stop)
        stats["train/samples"] = n
        return stats

    def _summary(self, s: dict) -> str:
        def g(key, fmt="{:.2f}"):
            return fmt.format(s[key]) if key in s else "-"
        return (f"[{self.name}] upd {self.update} step {self.global_step:,} | "
                f"return {g('rollout/ep_rew_mean')} survive {g('rollout/survival_rate')} "
                f"tiles {g('rollout/final_tiles', '{:.0f}')} len {g('rollout/ep_len_mean', '{:.0f}')} | "
                f"vs pool {g('selfplay/learner_beats_pool')} | kl {g('train/approx_kl', '{:.3f}')} "
                f"ev {g('train/explained_variance')} | {g('time/steps_per_second', '{:.0f}')} sps")

    def _rewind_metrics(self, path: Path) -> None:
        """Metrics past the resume point belong to updates that were lost
        (the run was killed before saving them); they move aside to
        metrics.rewound.jsonl so the charts do not run backwards."""
        if not path.exists():
            return
        keep, lost = [], []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            try:
                late = int(json.loads(line).get("step", 0)) > self.global_step
            except ValueError:
                late = False
            (lost if late else keep).append(line)
        if lost:
            with open(path.with_name("metrics.rewound.jsonl"), "a", encoding="utf-8") as handle:
                handle.writelines(lost)
            path.write_text("".join(keep), encoding="utf-8")
            self.log(f"[{self.name}] {len(lost)} metric lines past step {self.global_step:,} moved to metrics.rewound.jsonl")

    def save(self, filename: str | None) -> Path:
        """Write ``filename`` (when given) and ``<name>_latest.pt``."""
        meta = {"global_step": self.global_step, "update": self.update, "stage": self.stage,
                "config": self.cfg, "name": self.name, "episodes": list(self.episodes),
                "stage_episodes": list(self.stage_episodes), "head_to_head": list(self.head_to_head)}
        latest = self.ckpt_dir / f"{self.name}_latest.pt"
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            self.log(f"[{self.name}] not saving {filename or latest.name}: weights are non-finite")
            return latest
        if filename:
            save_checkpoint(self.ckpt_dir / filename, self.model, meta=meta, optimizer=self.optimizer)
        save_checkpoint(latest, self.model, meta=meta, optimizer=self.optimizer)
        return self.ckpt_dir / filename if filename else latest
