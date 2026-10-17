import { describe, expect, it } from "vitest";
import { shouldRecordReplay } from "../src/Replay";

describe("replay schedule", () => {
  it("records game 0 and every Nth game for a single game", () => {
    const due = [...Array(25).keys()].filter((i) => shouldRecordReplay(i, 10));
    expect(due).toEqual([0, 10, 20]);
  });

  it("spreads about one in `every` games over all of a run's games", () => {
    const stride = 10;
    const every = 20;
    const perGame = 80; // a 1M-step run: each game slot plays ~80 games
    const recorded: [number, number][] = [];
    for (let slot = 0; slot < stride; slot++) {
      for (let i = 0; i < perGame; i++) if (shouldRecordReplay(i, every, slot, stride)) recorded.push([slot, i]);
    }
    expect(recorded).toHaveLength((stride * perGame) / every);
    // Every slot (scenario) records, and only one recording happens at the very start.
    expect(new Set(recorded.map(([s]) => s)).size).toBe(stride);
    expect(recorded.filter(([, i]) => i === 0)).toHaveLength(1);
    // Recordings keep arriving through the run, not just at the start.
    expect(Math.max(...recorded.map(([, i]) => i))).toBeGreaterThan(perGame * 0.75);
  });
});
