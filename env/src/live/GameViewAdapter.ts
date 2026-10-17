/**
 * Makes the live client's GameView look like the engine's Game to the
 * training encoder and codec, so a real openfront.io game is observed and
 * acted on by exactly the code the policy was trained with.
 *
 * Why not a shadow simulation: openfront.io deploys newer engine versions
 * than the one vendored here, and a re-simulation on our engine diverges
 * from the real game as soon as any game rule differs. The GameView is the
 * page's own state, produced by the page's own engine — the same version
 * as the server by construction.
 *
 * The view answers engine queries (buildables, attackability, boat spawns)
 * asynchronously through the client's worker, while the encoder and codec
 * are synchronous. QueryCache bridges that: a pass that misses records the
 * query and returns a conservative default; the caller awaits all recorded
 * queries and runs the pass again with the answers cached.
 *
 * View objects are typed `any` deliberately: this file adapts whatever
 * client build is live, not a compile-time API.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

export class QueryCache {
  private readonly answers = new Map<string, unknown>();
  private readonly pending = new Map<string, () => Promise<unknown>>();

  get<T>(key: string, fetch: () => Promise<T>, fallback: T): T {
    if (this.answers.has(key)) return this.answers.get(key) as T;
    if (!this.pending.has(key)) this.pending.set(key, fetch);
    return fallback;
  }

  get missing(): number {
    return this.pending.size;
  }

  /** Resolve every recorded query (in parallel). Failed queries cache their fallback. */
  async resolve(): Promise<void> {
    const work = [...this.pending.entries()];
    this.pending.clear();
    await Promise.all(
      work.map(async ([key, fetch]) => {
        try {
          this.answers.set(key, await fetch());
        } catch {
          this.answers.set(key, undefined);
        }
      }),
    );
  }
}

const TERRA_NULLIUS = { isPlayer: () => false, smallID: () => 0, id: () => null };

/** Per-decision ownership scan: one tile per owner, and our own tiles. */
export interface OwnershipScan {
  tileOf: Map<number, number>;
  myTiles: number[];
}

export function scanOwnership(view: any, mySmallID: number): OwnershipScan {
  const state: Uint16Array = view.tileStateBuffer();
  const tileOf = new Map<number, number>();
  const myTiles: number[] = [];
  for (let t = 0; t < state.length; t++) {
    const owner = state[t] & 0xfff;
    if (owner === 0) continue;
    if (!tileOf.has(owner)) tileOf.set(owner, t);
    if (owner === mySmallID) myTiles.push(t);
  }
  return { tileOf, myTiles };
}

const componentsByView = new WeakMap<object, Int32Array>();

/**
 * 4-connected water components of the view's map, labelled from 1 (0 is
 * land). Computed once per game view; ViewGame is rebuilt every decision.
 */
export function waterComponents(view: any): Int32Array {
  const cached = componentsByView.get(view);
  if (cached !== undefined) return cached;
  const w: number = view.width();
  const h: number = view.height();
  const label = new Int32Array(w * h);
  const queue = new Int32Array(w * h);
  let next = 0;
  for (let start = 0; start < label.length; start++) {
    if (label[start] !== 0 || !view.isWater(start)) continue;
    const id = ++next;
    label[start] = id;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    while (head < tail) {
      const t = queue[head++];
      const x = t % w;
      const visit = (n: number) => {
        if (label[n] === 0 && view.isWater(n)) {
          label[n] = id;
          queue[tail++] = n;
        }
      };
      if (x > 0) visit(t - 1);
      if (x < w - 1) visit(t + 1);
      if (t >= w) visit(t - w);
      if (t < w * (h - 1)) visit(t + w);
    }
  }
  componentsByView.set(view, label);
  return label;
}

export class ViewGame {
  private readonly wrappers = new Map<number, ViewPlayer>();

  constructor(
    readonly view: any,
    readonly cache: QueryCache,
    readonly scan: OwnershipScan,
    readonly myBorder: ReadonlySet<number>,
  ) {}

  wrap(p: any): ViewPlayer {
    const sid = p.smallID();
    let w = this.wrappers.get(sid);
    if (w === undefined || w.view !== p) {
      w = new ViewPlayer(this, p);
      this.wrappers.set(sid, w);
    }
    return w;
  }

  bySmallID(sid: number): any {
    if (sid === 0) return TERRA_NULLIUS;
    const p = this.view.playerBySmallID(sid);
    return p && typeof p.isPlayer === "function" && p.isPlayer() ? this.wrap(p) : TERRA_NULLIUS;
  }

  me(): ViewPlayer {
    return this.wrap(this.view.myPlayer());
  }

  // ---- Game surface used by the encoder / codec --------------------------
  map(): any {
    return this.view;
  }
  ticks(): number {
    return this.view.ticks();
  }
  inSpawnPhase(): boolean {
    return this.view.inSpawnPhase();
  }
  config(): any {
    return this.view.config();
  }
  players(): ViewPlayer[] {
    return this.view.players().filter((p: any) => p.isAlive()).map((p: any) => this.wrap(p));
  }
  allPlayers(): ViewPlayer[] {
    return this.view.players().map((p: any) => this.wrap(p));
  }
  player(id: string): ViewPlayer {
    return this.wrap(this.view.player(id));
  }
  // GameView.hasPlayer() is a stub that always answers false; ask the
  // player list instead.
  hasPlayer(id: string): boolean {
    return this.view.players().some((p: any) => p.id() === id);
  }
  units(types: any, ...rest: any[]): any[] {
    const list = Array.isArray(types) ? types : [types, ...rest];
    return this.view.units(...list).map((u: any) => this.wrapUnit(u));
  }
  unitInfo(type: string): any {
    return {
      cost: (_game: unknown, player: ViewPlayer) => {
        try {
          return this.view.unitInfo(type).cost(this.view, player.view);
        } catch {
          return 0n;
        }
      },
    };
  }
  hasUnitNearby(tile: number, range: number, type: string): boolean {
    return this.view.hasUnitNearby?.(tile, range, type) ?? false;
  }
  nearbyUnits(tile: number, range: number, types: string[]): { distSquared: number }[] {
    return this.view.nearbyUnits?.(tile, range, types) ?? [];
  }
  isShore(tile: number): boolean {
    return this.view.isShore(tile);
  }
  // The view does not expose water components; label them from its map so
  // a move only goes to warships that can actually sail there.
  getWaterComponent(tile: number): number | null {
    const c = waterComponents(this.view)[tile];
    return c === 0 ? null : c;
  }
  hasWaterComponent(tile: number, component: number): boolean {
    return waterComponents(this.view)[tile] === component;
  }

  wrapUnit(u: any): any {
    return {
      id: () => u.id(),
      type: () => u.type(),
      tile: () => u.tile(),
      owner: () => this.wrap(u.owner()),
      level: () => u.level(),
      isActive: () => u.isActive(),
      isUnderConstruction: () => u.isUnderConstruction(),
      isInCooldown: () => u.isInCooldown?.() ?? false,
      targetTile: () => u.targetTile?.(),
      reachedTarget: () => u.reachedTarget?.() ?? false,
    };
  }
}

export class ViewPlayer {
  constructor(
    private readonly game: ViewGame,
    readonly view: any,
  ) {}

  private get isMe(): boolean {
    return this.view.smallID() === this.game.view.myPlayer()?.smallID();
  }

  isPlayer(): boolean {
    return true;
  }
  id(): string {
    return this.view.id();
  }
  smallID(): number {
    return this.view.smallID();
  }
  type(): string {
    return this.view.type();
  }
  clientID(): string | null {
    return this.view.clientID();
  }
  name(): string {
    return this.view.name();
  }
  troops(): number {
    return this.view.troops();
  }
  gold(): bigint {
    return BigInt(this.view.gold());
  }
  numTilesOwned(): number {
    return this.view.numTilesOwned();
  }
  isAlive(): boolean {
    return this.view.isAlive();
  }
  isTraitor(): boolean {
    return this.view.isTraitor();
  }
  isFriendly(other: ViewPlayer): boolean {
    return this.view.isFriendly(other.view);
  }
  isAlliedWith(other: ViewPlayer): boolean {
    return this.view.isAlliedWith(other.view);
  }
  hasEmbargoAgainst(other: ViewPlayer): boolean {
    return this.view.hasEmbargoAgainst(other.view);
  }
  allianceWith(other: ViewPlayer): unknown {
    return this.view.alliances().find((a: any) => a.other === other.id()) ?? null;
  }
  alliances(): unknown[] {
    return this.view.alliances();
  }
  incomingAllianceRequests(): { requestor: () => ViewPlayer }[] {
    return this.game
      .players()
      .filter((p) => p !== this && p.view.isRequestingAllianceWith(this.view))
      .map((p) => ({ requestor: () => p }));
  }
  incomingAttacks(): any[] {
    return this.view.incomingAttacks().map((a: any) => this.attack(a));
  }
  outgoingAttacks(): any[] {
    return this.view.outgoingAttacks().map((a: any) => this.attack(a));
  }
  private attack(a: any): any {
    return {
      id: () => a.id,
      troops: () => a.troops,
      attacker: () => this.game.bySmallID(a.attackerID),
      target: () => this.game.bySmallID(a.targetID),
    };
  }
  units(types?: any, ...rest: any[]): any[] {
    const list = types === undefined ? [] : Array.isArray(types) ? types : [types, ...rest];
    return this.view.units(...list).map((u: any) => this.game.wrapUnit(u));
  }
  unitCount(type: string): number {
    return this.view.units(type).length;
  }
  borderTiles(): ReadonlySet<number> {
    return this.isMe ? this.game.myBorder : new Set();
  }
  *tiles(): IterableIterator<number> {
    if (this.isMe) yield* this.game.scan.myTiles;
  }

  // ---- engine queries, answered by the client's worker -------------------
  private interaction(other: ViewPlayer): any {
    const tile = this.game.scan.tileOf.get(other.smallID());
    if (tile === undefined || !this.isMe) return undefined;
    const me = this.view;
    return this.game.cache.get(`a:${other.smallID()}`, () => me.actions(tile, []), undefined as any);
  }
  canAttackPlayer(other: ViewPlayer): boolean {
    return this.interaction(other)?.canAttack === true;
  }
  canTarget(other: ViewPlayer): boolean {
    return this.interaction(other)?.interaction?.canTarget === true;
  }
  canSendAllianceRequest(other: ViewPlayer): boolean {
    return this.interaction(other)?.interaction?.canSendAllianceRequest === true;
  }
  canDonateGold(other: ViewPlayer): boolean {
    return this.interaction(other)?.interaction?.canDonateGold === true;
  }
  canDonateTroops(other: ViewPlayer): boolean {
    return this.interaction(other)?.interaction?.canDonateTroops === true;
  }
  buildableUnits(tile: number, types: string[]): any[] {
    const me = this.view;
    return types.map((type) =>
      this.game.cache.get(
        `b:${tile}:${type}`,
        async () => (await me.buildables(tile, [type]))[0],
        { type, canBuild: false, canUpgrade: false },
      ) ?? { type, canBuild: false, canUpgrade: false },
    );
  }
  canBuild(type: string, tile: number): number | false {
    return this.buildableUnits(tile, [type])[0].canBuild;
  }
  bestTransportShipSpawn(tile: number): number | false {
    const me = this.view;
    return this.game.cache.get(`s:${tile}`, () => me.bestTransportShipSpawn(tile), false) ?? false;
  }
}
