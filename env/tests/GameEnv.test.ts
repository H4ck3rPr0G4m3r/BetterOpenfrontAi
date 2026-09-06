import path from "path";
import { fileURLToPath } from "url";
import { describe, expect, it } from "vitest";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import { GameEnv } from "../src/GameEnv";
import type { EnvConfig } from "../src/types";
import { NOOP_ACTION, NoopActionCodec, StubObservationEncoder } from "./helpers/stubs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MAPS_DIR = path.join(__dirname, "../../OpenFrontIO/resources/maps");

function findAgentSpawnTile(env: GameEnv): number | undefined {
  return env.game.player("agent0").spawnTile();
}

function makeConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    map: GameMapType.World,
    mapsDir: MAPS_DIR,
    bots: 20,
    ticksPerStep: 10,
    tickCap: 6000,
    ...overrides,
  };
}

describe("GameEnv", () => {
  it("reset() spawns the agent and returns a correctly-sized observation", async () => {
    const encoder = new StubObservationEncoder();
    const env = new GameEnv(makeConfig(), encoder, new NoopActionCodec());

    const obs = await env.reset("gameenv-reset");

    expect(obs.length).toBe(encoder.size);
    expect(env.game.player("agent0").isAlive()).toBe(true);
    expect(env.game.player("agent0").numTilesOwned()).toBeGreaterThan(0);
    expect(env.game.inSpawnPhase()).toBe(false);
  });

  it("step() advances exactly ticksPerStep ticks and isn't done early", async () => {
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10 }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await env.reset("gameenv-step");
    const before = env.game.ticks();

    const result = env.step(NOOP_ACTION);

    expect(env.game.ticks()).toBe(before + 10);
    expect(result.done).toBe(false);
    expect(result.info.terminalReason).toBeNull();
    expect(result.obs.length).toBe(4);
  });

  it("terminates with tick_cap once the post-spawn tick budget is exhausted", async () => {
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10, tickCap: 25 }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await env.reset("gameenv-tickcap");

    let result = env.step(NOOP_ACTION); // ticks 1-10, not done
    expect(result.done).toBe(false);
    result = env.step(NOOP_ACTION); // ticks 11-20, not done
    expect(result.done).toBe(false);
    result = env.step(NOOP_ACTION); // ticks 21-30, past the cap of 25
    expect(result.done).toBe(true);
    expect(result.info.terminalReason).toBe("tick_cap");
  });

  it("does not auto-reset after a terminal episode — that's the caller's job", async () => {
    const env = new GameEnv(
      makeConfig({ ticksPerStep: 10, tickCap: 5 }),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await env.reset("gameenv-poststep");
    const result = env.step(NOOP_ACTION);
    expect(result.done).toBe(true);

    // GameEnv only throws on a caught simulation exception (fatalError);
    // a plain tick-cap/win/death terminal doesn't latch that flag, so
    // calling step() again is the caller's bug to avoid, not GameEnv's to
    // reject — matching the auto-reset contract the bridge (§5.6) owns.
    expect(() => env.step(NOOP_ACTION)).not.toThrow();
  });

  it("reset() rebuilds the game rather than reusing the previous episode", async () => {
    const env = new GameEnv(
      makeConfig(),
      new StubObservationEncoder(),
      new NoopActionCodec(),
    );
    await env.reset("gameenv-fresh-1");
    const firstEpisodeTicks = env.game.ticks();
    const firstEpisodeSpawnTile = findAgentSpawnTile(env);
    env.step(NOOP_ACTION); // advance episode 1 so a stale reset would be visible
    env.step(NOOP_ACTION);
    const ticksAfterStepping = env.game.ticks();
    expect(ticksAfterStepping).toBeGreaterThan(firstEpisodeTicks);

    const obs = await env.reset("gameenv-fresh-2");

    expect(obs.length).toBe(4);
    expect(env.game.player("agent0").isAlive()).toBe(true);
    expect(env.game.player("agent0").numTilesOwned()).toBeGreaterThan(0);
    // A stale/reused game would still be sitting at ticksAfterStepping (or
    // beyond); a genuinely fresh one is back down near the post-spawn tick
    // count of a new episode.
    expect(env.game.ticks()).toBeLessThan(ticksAfterStepping);
    expect(Math.abs(env.game.ticks() - firstEpisodeTicks)).toBeLessThanOrEqual(5);
    // Different seeds feed different PRNG streams into spawn-tile
    // selection (§ pickSpawnTile) — a stale game would trivially "pass" a
    // tile-count check since it's the same object.
    expect(findAgentSpawnTile(env)).not.toBe(firstEpisodeSpawnTile);
  });
});
