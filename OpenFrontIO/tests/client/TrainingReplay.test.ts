import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchReplayList,
  parseTrainingReplay,
} from "../../src/client/GameModeSelector";

/** Minimal record shaped like the training pipeline's output. */
function validRecordText(): string {
  return JSON.stringify({
    info: {
      gameID: "ABCD1234",
      lobbyCreatedAt: 1700000000000,
      config: {
        gameMap: "World",
        difficulty: "Medium",
        donateGold: false,
        donateTroops: false,
        gameType: "Public",
        gameMode: "Free For All",
        gameMapSize: "Normal",
        nations: "default",
        bots: 20,
        infiniteGold: false,
        infiniteTroops: false,
        instantBuild: false,
        randomSpawn: false,
      },
      players: [
        {
          clientID: "RLCLIENT",
          username: "RLAgent",
          clanTag: null,
          persistentID: null,
        },
      ],
      start: 1700000000000,
      end: 1700000060000,
      duration: 60,
      num_turns: 2,
      lobbyFillTime: 0,
    },
    version: "v0.0.2",
    gitCommit: "DEV",
    turns: [
      { turnNumber: 0, intents: [] },
      {
        turnNumber: 1,
        intents: [{ clientID: "RLCLIENT", type: "spawn", tile: 100 }],
      },
    ],
  });
}

describe("parseTrainingReplay", () => {
  it("accepts a valid training record", () => {
    const record = parseTrainingReplay(validRecordText());
    expect(record.info.gameID).toBe("ABCD1234");
    expect(record.turns.length).toBe(2);
  });

  it("rejects non-JSON and schema-violating files", () => {
    expect(() => parseTrainingReplay("not json")).toThrow();
    expect(() => parseTrainingReplay("{}")).toThrow();
    expect(() =>
      parseTrainingReplay(JSON.stringify({ info: {}, turns: [] })),
    ).toThrow();
  });
});

describe("fetchReplayList", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns entries from a good index", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({ replays: [{ name: "a.json", size: 10 }] }),
      ),
    );
    await expect(fetchReplayList("http://x")).resolves.toEqual([
      { name: "a.json", size: 10 },
    ]);
  });

  it("throws on HTTP errors and bad shapes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 500 })),
    );
    await expect(fetchReplayList("http://x")).rejects.toThrow();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ replays: [{ nope: 1 }] })),
    );
    await expect(fetchReplayList("http://x")).rejects.toThrow();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("down");
      }),
    );
    await expect(fetchReplayList("http://x")).rejects.toThrow();
  });
});
