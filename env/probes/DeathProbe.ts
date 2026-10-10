/**
 * Diagnostic: how do scripted behaviours fare, and what does the reward pay
 * for them? Runs real games and logs each seat's economy (tiles, troops vs
 * max, attacks in flight both ways), its running return under the current
 * reward, and the per-step wall time.
 *
 * This is the probe behind the 2026-10 diagnosis: under the v1 weights a
 * passive agent that died at step 171 scored +16.6, because the kill term
 * paid for every elimination on the map. Use it to sanity-check any reward
 * change: survival and land should be what pays.
 *
 * Usage (from env/):
 *   TSX_TSCONFIG_PATH=tsconfig.probes.json npx tsx probes/DeathProbe.ts [policy] [games] [bots] [seats]
 *   (optional 5th/6th arguments: close-spawn probability, tick cap)
 * policy: noop | random | spam50 | paced | builder   (every seat plays the same policy)
 */
import path from "path";
import { fileURLToPath } from "url";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import { GameEnv, SeatStep } from "../src/GameEnv";
import { buildCandidates } from "../src/BuildSites";
import { HEAD_USAGE, IntentType, Masks, siteRow, spatialRow, targetRow } from "../src/Layout";
import { NOOP_ACTION, type Action } from "../src/types";

const MAPS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../OpenFrontIO/resources/maps");

let rng = 12345;
const rand = (): number => {
  rng = (rng * 1103515245 + 12345) & 0x7fffffff;
  return rng / 0x7fffffff;
};
const pick = (m: Uint8Array): number => {
  const open: number[] = [];
  m.forEach((v, i) => v && open.push(i));
  return open.length ? open[Math.floor(rand() * open.length)] : 0;
};

type Policy = (m: Masks, env: GameEnv, seat: number) => Action;
const expand = (m: Masks, ratioBucket: number): Action =>
  m.intent[IntentType.Attack] && targetRow(m, IntentType.Attack)[0]
    ? { ...NOOP_ACTION, intentType: IntentType.Attack, target: 0, ratioBucket }
    : NOOP_ACTION;

const POLICIES: Record<string, Policy> = {
  noop: () => NOOP_ACTION,
  // Conditional masked-uniform: what an untrained policy does.
  random: (m) => {
    const intent = pick(m.intent);
    const use = HEAD_USAGE[intent];
    return {
      intentType: intent,
      target: use.target ? pick(targetRow(m, intent)) : 0,
      ratioBucket: use.ratio ? pick(m.ratio) : 0,
      spatialCell: use.spatial >= 0 ? pick(spatialRow(m, use.spatial)) : 0,
      buildSite: use.site >= 0 ? pick(siteRow(m, use.site)) : 0,
    };
  },
  // v1's scripted baseline: expand at 50% every single step.
  spam50: (m) => expand(m, 4),
  // A human's rhythm: expand at 20% only while the army is above 45% of max.
  paced: (m, env, seat) => {
    const p = env.game.player(env.seatIDs[seat]);
    return p.troops() / env.game.config().maxTroops(p) > 0.45 ? expand(m, 2) : NOOP_ACTION;
  },
  // paced, plus a city (else an upgrade, else a port or factory) whenever one is affordable.
  builder: (m, env, seat) => {
    const p = env.game.player(env.seatIDs[seat]);
    const sites = buildCandidates(env.game, p);
    for (const want of ["City", "Port", "Factory"]) {
      const i = sites.findIndex((c, k) => c.type === want && siteRow(m, 0)[k]);
      if (i >= 0) return { ...NOOP_ACTION, intentType: IntentType.BuildUnit, buildSite: i };
    }
    const up = sites.findIndex((c, k) => c.type === "City" && siteRow(m, 1)[k]);
    if (up >= 0) return { ...NOOP_ACTION, intentType: IntentType.UpgradeStructure, buildSite: up };
    return p.troops() / env.game.config().maxTroops(p) > 0.45 ? expand(m, 2) : NOOP_ACTION;
  },
};

async function main(): Promise<void> {
  const [policyName = "random", gamesRaw = "3", botsRaw = "20", seatsRaw = "1", closeRaw = "0", capRaw = "6000"] = process.argv.slice(2);
  const policy = POLICIES[policyName];
  if (!policy) throw new Error(`unknown policy ${policyName}`);
  const seats = Number(seatsRaw);
  const env = new GameEnv({ map: GameMapType.World, mapsDir: MAPS, bots: Number(botsRaw), ticksPerStep: 10, tickCap: Number(capRaw), seats, closeSpawn: Number(closeRaw) });
  const results: string[] = [];
  let stepMs = 0;
  let steps = 0;
  for (let g = 0; g < Number(gamesRaw); g++) {
    let rows: SeatStep[] = await env.reset(`probe-${g}`);
    const returns = new Array(seats).fill(0);
    const ends: string[] = new Array(seats).fill("");
    for (let step = 0; !env.gameOver; step++) {
      const actions = rows.map((r, s) => (r.active ? policy(r.masks, env, s) : NOOP_ACTION));
      const t0 = performance.now();
      rows = env.step(actions);
      stepMs += performance.now() - t0;
      steps++;
      rows.forEach((r, s) => {
        returns[s] += r.reward;
        if (r.done) ends[s] = `${r.terminal}@${step}`;
      });
      if (step % 100 === 0 || env.gameOver) {
        for (let s = 0; s < seats; s++) {
          const p = env.game.player(env.seatIDs[s]);
          const max = p.isAlive() ? env.game.config().maxTroops(p) : 1;
          const out = p.outgoingAttacks().reduce((a, x) => a + x.troops(), 0);
          const inc = p.incomingAttacks().reduce((a, x) => a + x.troops(), 0);
          console.log(
            `game${g} seat${s} step ${String(step).padStart(4)} tiles ${String(p.numTilesOwned()).padStart(6)} ` +
              `troops ${Math.round(p.troops()).toString().padStart(8)} (${((p.troops() / max) * 100).toFixed(0)}% max) ` +
              `out ${Math.round(out)} in ${Math.round(inc)} return ${returns[s].toFixed(2)}`,
          );
        }
      }
    }
    // Per-term episode sums: which reward terms actually moved the return.
    const seatState = (env as unknown as { seats: { reward: { terms: Record<string, number> } }[] }).seats;
    results.push(
      ...returns.map((r, s) => {
        const terms = Object.entries(seatState[s].reward.terms)
          .map(([k, v]) => `${k} ${v.toFixed(2)}`)
          .join(", ");
        return `game${g} seat${s}: ${ends[s]} return ${r.toFixed(2)}  [${terms}]`;
      }),
    );
  }
  console.log(results.join("\n"));
  console.log(`${policyName}: mean step ${(stepMs / steps).toFixed(1)} ms over ${steps} steps (${seats} seats)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
