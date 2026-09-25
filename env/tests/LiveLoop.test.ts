import { describe, expect, it, vi } from "vitest";
import { AgentLoop, type LoopHooks, type Predictor } from "../src/live/agent_loop";
import { checkVersionParity, RateLimiter } from "../src/live/rate_limiter";

describe("RateLimiter", () => {
  it("holds 8 intents per second and recovers the next second", () => {
    const limiter = new RateLimiter(8, 120);
    for (let i = 0; i < 8; i++) {
      expect(limiter.tryAcquire(i * 100, 100)).toBe(0);
    }
    expect(limiter.tryAcquire(800, 100)).toBeGreaterThan(0);
    expect(limiter.tryAcquire(1100, 100)).toBe(0);
  });

  it("holds 120 intents per minute", () => {
    const limiter = new RateLimiter(1000, 3);
    expect(limiter.tryAcquire(0, 10)).toBe(0);
    expect(limiter.tryAcquire(1000, 10)).toBe(0);
    expect(limiter.tryAcquire(2000, 10)).toBe(0);
    expect(limiter.tryAcquire(3000, 10)).toBeGreaterThan(0);
    expect(limiter.tryAcquire(61_000, 10)).toBe(0);
  });

  it("never admits oversize or over-budget payloads", () => {
    const limiter = new RateLimiter();
    expect(limiter.tryAcquire(0, 2001)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("checkVersionParity", () => {
  const pinned = "f51f165b947a92cc683ade3a72c3800300e86e61";
  it("accepts the pinned commit and dev servers", () => {
    expect(checkVersionParity(pinned, pinned).ok).toBe(true);
    expect(checkVersionParity("DEV", pinned).ok).toBe(true);
  });

  it("refuses mismatches and missing commits", () => {
    expect(checkVersionParity("deadbeef", pinned).ok).toBe(false);
    expect(checkVersionParity("", pinned).ok).toBe(false);
  });
});

describe("AgentLoop", () => {
  const hooks: LoopHooks = {
    encode: (_game, _id, out) => out.fill(1),
    masks: (_game, _id, _out) => {},
    decode: (_action, _game, _id) => [{ type: "noop" }] as never,
    createMaskBuffers: () => ({}) as never,
    obsSize: () => 8,
  };
  const predictor: Predictor = {
    predict: () => ({ intentType: 0, target: 0, ratioBucket: 0, spatialCell: 0, buildSite: 0 }),
  };
  const loop = () =>
    new AgentLoop({ turnBudgetMs: 100, maxIntentsPerMinute: 120, ticksPerStep: 10 }, hooks, predictor);

  function stubShadow(feedOk = true) {
    return {
      game: {},
      feed: vi.fn(() => feedOk),
    } as never;
  }

  it("emits intents on a healthy turn", () => {
    const result = loop().onTurn(stubShadow(), { turnNumber: 1, intents: [] } as never, "agent0", null, () => true);
    expect(result.dropped).toBeNull();
    expect(result.intents.length).toBe(1);
  });

  it("drops stale turns, shadow errors, and hash mismatches without acting", () => {
    const agent = loop();
    agent.onTurn(stubShadow(), { turnNumber: 5, intents: [] } as never, "agent0", null, () => true);
    expect(
      agent.onTurn(stubShadow(), { turnNumber: 5, intents: [] } as never, "agent0", null, () => true).dropped,
    ).toBe("stale-turn");
    expect(
      agent.onTurn(stubShadow(false), { turnNumber: 6, intents: [] } as never, "agent0", null, () => true).dropped,
    ).toBe("shadow-error");
    expect(
      agent.onTurn(stubShadow(), { turnNumber: 7, intents: [] } as never, "agent0", { tick: 9, hash: 1 }, () => false).dropped,
    ).toBe("hash-mismatch");
  });

  it("drops overruns instead of landing late actions", () => {
    let now = 0;
    const result = loop().onTurn(
      stubShadow(),
      { turnNumber: 1, intents: [] } as never,
      "agent0",
      null,
      () => true,
      () => (now += 500),
    );
    expect(result.dropped).toBe("overrun");
    expect(result.intents).toEqual([]);
  });
});
