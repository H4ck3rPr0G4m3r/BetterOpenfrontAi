import { describe, expect, it } from "vitest";
import { QueryCache, ViewGame, waterComponents } from "../src/live/GameViewAdapter";

// 5x3 map: two lakes split by a land column.
//   ~ ~ # ~ ~
//   ~ ~ # ~ ~
//   # # # # #
const W = 5;
const H = 3;
const WATER = new Set([0, 1, 5, 6, 3, 4, 8, 9]);

function fakeView(ids: string[] = []): any {
  return {
    width: () => W,
    height: () => H,
    isWater: (t: number) => WATER.has(t),
    hasPlayer: () => false, // the real GameView stub
    players: () => ids.map((id) => ({ id: () => id })),
  };
}

const game = (view: any) => new ViewGame(view, new QueryCache(), { tileOf: new Map(), myTiles: [] }, new Set());

describe("ViewGame", () => {
  it("finds players from the view's player list, not GameView.hasPlayer", () => {
    const g = game(fakeView(["a", "b"]));
    expect(g.hasPlayer("a")).toBe(true);
    expect(g.hasPlayer("c")).toBe(false);
  });

  it("labels separate water bodies apart and land as no component", () => {
    const g = game(fakeView());
    const west = g.getWaterComponent(0);
    const east = g.getWaterComponent(4);
    expect(west).not.toBeNull();
    expect(east).not.toBeNull();
    expect(west).not.toBe(east);
    expect(g.getWaterComponent(2)).toBeNull();
    expect(g.hasWaterComponent(6, west!)).toBe(true);
    expect(g.hasWaterComponent(8, west!)).toBe(false);
  });

  it("labels each view once", () => {
    const view = fakeView();
    expect(waterComponents(view)).toBe(waterComponents(view));
  });
});
