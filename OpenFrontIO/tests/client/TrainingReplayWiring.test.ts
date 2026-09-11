import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ClientEnv } from "../../src/client/ClientEnv";

// Same socket stand-in as GameModeSelectorGatingWiring: the component opens
// a public-lobby socket on connect, which jsdom cannot serve.
vi.mock("../../src/client/LobbySocket", () => ({
  PublicLobbySocket: class {
    constructor(_onUpdate: unknown) {}
    start(): void {}
    stop(): void {}
  },
}));

// Registers <game-mode-selector> as a side effect.
import "../../src/client/GameModeSelector";

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

let selector: HTMLElement & { updateComplete: Promise<unknown> };
let pickers: HTMLInputElement[];

function watchReplayButton(): HTMLButtonElement | null {
  // translateText is untranslated under test (raw keys render), so match
  // the key. Production en.json carries "Watch Replay" for the same key.
  const buttons = Array.from(selector.querySelectorAll("button"));
  return buttons.find((b) => b.textContent?.includes("watch_replay")) ?? null;
}

function buttonByKey(key: string): HTMLButtonElement | null {
  const buttons = Array.from(selector.querySelectorAll("button"));
  return buttons.find((b) => b.textContent?.includes(key)) ?? null;
}

async function flush(): Promise<void> {
  await selector.updateComplete;
  await new Promise((r) => setTimeout(r, 0));
  await selector.updateComplete;
}

/** The file picker the disk flow creates, with a chosen file. */
async function chooseFile(name: string, text: string): Promise<void> {
  expect(pickers.length).toBe(1);
  const picker = pickers[0];
  const file = new File([text], name, { type: "application/json" });
  Object.defineProperty(picker, "files", { value: [file] });
  picker.onchange?.call(picker, new Event("change"));
  await flush();
}

function stubServer(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (String(url).endsWith("/api/replays")) {
        return new Response(
          JSON.stringify({ replays: [{ name: "ep0.json", size: 4321 }] }),
          { status: 200 },
        );
      }
      if (String(url).endsWith("/replays/ep0.json")) {
        return new Response(validRecordText(), { status: 200 });
      }
      return new Response("nope", { status: 404 });
    }),
  );
}

beforeEach(async () => {
  window.BOOTSTRAP_CONFIG = {
    gameEnv: "dev",
    numWorkers: 1,
    turnstileSiteKey: "",
    jwtAudience: "test",
    instanceId: "test",
    gitCommit: "test",
  } as never;
  ClientEnv.reset();
  pickers = [];
  const realCreateElement = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation(((tag: string) => {
    const el = realCreateElement(tag);
    if (tag === "input") pickers.push(el as HTMLInputElement);
    return el;
  }) as never);
  selector = document.createElement("game-mode-selector") as typeof selector;
  document.body.appendChild(selector);
  await selector.updateComplete;
  window.dispatchEvent(
    new CustomEvent("username-validity-change", { detail: { isValid: true } }),
  );
  await selector.updateComplete;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
  window.BOOTSTRAP_CONFIG = undefined;
  ClientEnv.reset();
});

describe("Watch Replay browser", () => {
  it("lists server replays and loads one on click", async () => {
    stubServer();
    const seen: CustomEvent[] = [];
    document.addEventListener("join-lobby", (e) => seen.push(e as CustomEvent));
    watchReplayButton()!.click();
    await flush();
    const entry = buttonByKey("ep0.json");
    expect(entry).not.toBeNull();
    entry!.click();
    await flush();
    expect(seen.length).toBe(1);
    expect(seen[0].detail.gameRecord.info.gameID).toBe("ABCD1234");
    expect(seen[0].detail.gameID).toBe("ABCD1234");
  });

  it("falls back gracefully when the server is down, disk still works", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("connection refused");
      }),
    );
    const seen: CustomEvent[] = [];
    document.addEventListener("join-lobby", (e) => seen.push(e as CustomEvent));
    watchReplayButton()!.click();
    await flush();
    expect(selector.textContent).toContain("replay_server_missing");
    expect(seen.length).toBe(0);
    // The disk path stays available inside the open panel.
    buttonByKey("replay_from_disk")!.click();
    await chooseFile("ep0-seed-death.json", validRecordText());
    expect(seen.length).toBe(1);
    expect(seen[0].detail.gameRecord.info.gameID).toBe("ABCD1234");
  });

  it("shows an inline error for a bad disk file and dispatches nothing", async () => {
    stubServer();
    const seen: CustomEvent[] = [];
    document.addEventListener("join-lobby", (e) => seen.push(e as CustomEvent));
    watchReplayButton()!.click();
    await flush();
    buttonByKey("replay_from_disk")!.click();
    await chooseFile("nope.json", "definitely not a record");
    expect(seen.length).toBe(0);
    expect(selector.textContent).toContain("replay_invalid");
  });
});
