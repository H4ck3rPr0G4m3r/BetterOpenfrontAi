"""SB3 VecEnv around OpenFrontVecEnv.

SB3 speaks its own vectorized API: ``reset()`` returns obs only,
``step_async`` plus ``step_wait`` return ``(obs, rewards, dones, infos)``
with one info dict per env. Our gym wrapper speaks gymnasium with batched
info dicts and separate terminated and truncated arrays. This module
translates between the two so MaskablePPO can train without the gym side
learning anything about SB3.

Three semantics matter and are easy to get wrong:

* Node auto-resets finished episodes, so a done row already carries the
  fresh next observation. SB3 expects exactly this from an autoresetting
  VecEnv, with the terminal observation in the info dict for timeout
  bootstrapping.
* Tick cap is a truncation, not a true terminal. The bridge does not
  preserve the terminal tick count, so the bootstrap value reads the
  fresh observation. That approximation is documented in the phase 1
  spec and repeated here so nobody mistakes it for exactness.
* Win, death, sim error, and worker death get no bootstrap. Their info
  dicts carry no terminal observation, so SB3 ends the advantage chain.
  Every done row carries ``terminal_reason`` and an ``episode`` summary
  with return and length, which is what SB3 logs as ep_rew_mean.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

TRUNCATED_REASONS = {"tick_cap"}


class SB3OpenFrontVec(VecEnv):
    """Synchronous SB3 VecEnv delegating to one OpenFrontVecEnv."""

    def __init__(self, inner):
        self.inner = inner
        self._pending_actions = None
        self._ep_rewards = np.zeros(inner.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(inner.num_envs, dtype=np.int64)
        self.terminal_counts: Counter[str] = Counter()
        super().__init__(
            inner.num_envs,
            inner.single_observation_space,
            inner.single_action_space,
        )

    def reset(self):
        obs, _ = self.inner.reset(seed=self._seeds_to_inner())
        self._pending_actions = None
        self._ep_rewards[:] = 0.0
        self._ep_lengths[:] = 0
        return obs

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._seeds = [seed + index for index in range(self.num_envs)]
        return self._seeds

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait called without step_async")
        actions, self._pending_actions = self._pending_actions, None
        obs, rewards, terminated, truncated, infos = self.inner.step(actions)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        dones = terminated | truncated
        reasons = list(infos.get("terminal_reason", [None] * self.num_envs))
        ticks = list(infos.get("ticks", [0] * self.num_envs))

        self._ep_rewards += np.asarray(rewards, dtype=np.float64)
        self._ep_lengths += 1

        out_infos: list[dict] = []
        for index in range(self.num_envs):
            reason = reasons[index]
            info = {
                "terminal_reason": reason,
                "ticks": int(ticks[index]),
                "TimeLimit.truncated": bool(truncated[index]),
            }
            if dones[index] and truncated[index] and reason in TRUNCATED_REASONS:
                # Fresh-state bootstrap, per the module docstring. The
                # terminal observation is gone by the time Node replies,
                # so the value target reads the reset state.
                info["terminal_observation"] = np.asarray(obs[index]).copy()
            if dones[index]:
                name = reason if reason is not None else "continuing"
                self.terminal_counts[name] += 1
                info["episode"] = {
                    "r": float(self._ep_rewards[index]),
                    "l": int(self._ep_lengths[index]),
                    "t": 0.0,
                }
                self._ep_rewards[index] = 0.0
                self._ep_lengths[index] = 0
            out_infos.append(info)
        return obs, np.asarray(rewards, dtype=np.float64), dones, out_infos

    def action_masks(self) -> np.ndarray:
        return self.inner.action_masks()

    def get_attr(self, attr_name: str, indices=None):
        if attr_name == "render_mode":
            return [None] * self.num_envs
        if attr_name == "action_masks":
            return [row.copy() for row in self.inner.action_masks()]
        if attr_name == "num_envs":
            return [self.num_envs] * self.num_envs
        raise AttributeError(f"unknown attr {attr_name}")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError(f"cannot set attr {attr_name} on SB3OpenFrontVec")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            return [row.copy() for row in self.inner.action_masks()]
        raise AttributeError(f"unknown env method {method_name}")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def close(self) -> None:
        self.inner.close()

    def _seeds_to_inner(self):
        seeds = getattr(self, "_seeds", [None] * self.num_envs)
        if seeds is None or all(seed is None for seed in seeds):
            return None
        return [str(seed) for seed in seeds]
