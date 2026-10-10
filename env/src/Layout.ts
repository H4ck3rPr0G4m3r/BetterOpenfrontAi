/**
 * Observation and action layout, schema v2. One source of truth for the
 * encoder, the codec, the bridge, and (through HELLO) the Python side.
 *
 * Observation per seat, two parts:
 *   raster   uint8[NUM_PLANES * RASTER_CELLS]  — planes in [0,1] scaled to 0..255
 *   features float32[FEATURE_SIZE]            — slot table, build table, globals
 *
 * Masks per seat, one flat uint8 block in this order:
 *   intent  [NUM_INTENTS]
 *   target  [NUM_INTENTS * NUM_SLOTS]   (row per intent: which slots that intent may use)
 *   ratio   [NUM_RATIO_BUCKETS]
 *   spatial [NUM_SPATIAL_KINDS * RASTER_CELLS]  (0 = boat landing, 1 = warship water)
 *   site    [NUM_SITE_KINDS * MAX_BUILD_SITES]  (0 = build, 1 = upgrade)
 *
 * Every head except intent is conditioned on the sampled intent: the policy
 * samples the intent first, then each head that intent uses with that
 * intent's mask row. Heads an intent does not use are ignored entirely
 * (not sampled, not part of the log-probability). See HEAD_USAGE.
 */

export const SCHEMA_VERSION = 3;

export const RASTER_WIDTH = 128;
export const RASTER_HEIGHT = 64;
export const RASTER_CELLS = RASTER_WIDTH * RASTER_HEIGHT;

export enum Plane {
  Land = 0,
  Self = 1,
  Ally = 2,
  Enemy = 3,
  Unowned = 4,
  SelfTroops = 5,
  EnemyTroops = 6,
  /** Land of players currently attacking this seat. */
  Threat = 7,
  /** Land of players this seat is currently attacking. */
  Targets = 8,
  OwnStructures = 9,
  EnemyStructures = 10,
}
export const NUM_PLANES = 11;
export const RASTER_SIZE = NUM_PLANES * RASTER_CELLS;

/** Slot 0 is terra nullius (expansion), slots 1..16 are players. */
export const NUM_SLOTS = 17;
export enum SlotField {
  Valid = 0,
  Tiles = 1,
  TileShareVsUs = 2,
  Troops = 3,
  StrengthVsUs = 4,
  Density = 5,
  Allied = 6,
  SharesBorder = 7,
  AttackingUs = 8,
  WeAttack = 9,
  IsBot = 10,
  IsNation = 11,
  IsHuman = 12,
  Traitor = 13,
  EmbargoAgainstUs = 14,
  RequestsAlliance = 15,
}
export const SLOT_FIELDS = 16;

export const MAX_BUILD_SITES = 32;
export enum SiteField {
  Valid = 0,
  /** One-hot over BUILD_TYPES occupies TypeStart .. TypeStart+6. */
  TypeStart = 1,
  X = 8,
  Y = 9,
  Border = 10,
  /** A City/Port here would join the rail network (a factory in range); a
   *  Factory here would connect existing cities, ports or factories. */
  RailLink = 11,
}
export const SITE_FIELDS = 12;

export enum GlobalField {
  TimeFraction = 0,
  AliveFraction = 1,
  Rank = 2,
  Tiles = 3,
  LandShare = 4,
  Troops = 5,
  TroopRatio = 6,
  GrowthRate = 7,
  OutgoingTroops = 8,
  IncomingTroops = 9,
  IncomingCount = 10,
  OutgoingCount = 11,
  Gold = 12,
  CityAffordable = 13,
  Cities = 14,
  Ports = 15,
  DefensePosts = 16,
  SAMs = 17,
  Silos = 18,
  Warships = 19,
  BoatsAtSea = 20,
  Alive = 21,
  Traitor = 22,
  Allies = 23,
  Embargoed = 24,
  InSpawnPhase = 25,
  AllianceRequests = 26,
  Factories = 27,
}
export const NUM_GLOBALS = 28;

export const SLOT_TABLE_SIZE = NUM_SLOTS * SLOT_FIELDS;
export const SITE_TABLE_SIZE = MAX_BUILD_SITES * SITE_FIELDS;
export const FEATURE_SIZE = SLOT_TABLE_SIZE + SITE_TABLE_SIZE + NUM_GLOBALS;
export const SLOT_OFFSET = 0;
export const SITE_OFFSET = SLOT_TABLE_SIZE;
export const GLOBAL_OFFSET = SLOT_TABLE_SIZE + SITE_TABLE_SIZE;

export enum IntentType {
  Noop = 0,
  Attack = 1,
  Boat = 2,
  BuildUnit = 3,
  UpgradeStructure = 4,
  MoveWarship = 5,
  TargetPlayer = 6,
  AllianceRequest = 7,
  AllianceReject = 8,
  BreakAlliance = 9,
  Embargo = 10,
  DonateGold = 11,
  DonateTroops = 12,
  CancelAttack = 13,
}
export const NUM_INTENTS = 14;

/** Troop/gold fraction per ratio bucket. */
export const RATIO_BUCKETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
export const NUM_RATIO_BUCKETS = RATIO_BUCKETS.length;

export const NUM_SPATIAL_KINDS = 2; // 0 boat landing, 1 warship water
export const NUM_SITE_KINDS = 2; // 0 build, 1 upgrade

/** Unit types offered on build sites, in one-hot order. */
export const BUILD_TYPES = [
  "City",
  "Defense Post",
  "Port",
  "SAM Launcher",
  "Missile Silo",
  "Factory",
  "Warship",
] as const;

export const MASK_INTENT_OFFSET = 0;
export const MASK_TARGET_OFFSET = NUM_INTENTS;
export const MASK_RATIO_OFFSET = MASK_TARGET_OFFSET + NUM_INTENTS * NUM_SLOTS;
export const MASK_SPATIAL_OFFSET = MASK_RATIO_OFFSET + NUM_RATIO_BUCKETS;
export const MASK_SITE_OFFSET = MASK_SPATIAL_OFFSET + NUM_SPATIAL_KINDS * RASTER_CELLS;
export const MASK_SIZE = MASK_SITE_OFFSET + NUM_SITE_KINDS * MAX_BUILD_SITES;

/**
 * Which conditional heads each intent consumes. `spatial` / `site` name the
 * mask kind (row) used, or -1 when the head is unused.
 */
export interface HeadUsage {
  target: boolean;
  ratio: boolean;
  spatial: number;
  site: number;
}
const none: HeadUsage = { target: false, ratio: false, spatial: -1, site: -1 };
export const HEAD_USAGE: HeadUsage[] = [
  none, // noop
  { ...none, target: true, ratio: true }, // attack
  { ...none, ratio: true, spatial: 0 }, // boat
  { ...none, site: 0 }, // build
  { ...none, site: 1 }, // upgrade
  { ...none, spatial: 1 }, // move warship
  { ...none, target: true }, // target player
  { ...none, target: true }, // alliance request
  { ...none, target: true }, // alliance reject
  { ...none, target: true }, // break alliance
  { ...none, target: true }, // embargo
  { ...none, target: true, ratio: true }, // donate gold
  { ...none, target: true, ratio: true }, // donate troops
  none, // cancel attack
];

/** Flat mask buffer for one seat, with typed views per head. */
export interface Masks {
  flat: Uint8Array;
  intent: Uint8Array;
  /** Row for intent k: target.subarray(k*NUM_SLOTS, (k+1)*NUM_SLOTS). */
  target: Uint8Array;
  ratio: Uint8Array;
  spatial: Uint8Array;
  site: Uint8Array;
}

export function createMasks(flat: Uint8Array = new Uint8Array(MASK_SIZE)): Masks {
  if (flat.length !== MASK_SIZE) throw new Error(`mask buffer must be ${MASK_SIZE} bytes`);
  return {
    flat,
    intent: flat.subarray(MASK_INTENT_OFFSET, MASK_TARGET_OFFSET),
    target: flat.subarray(MASK_TARGET_OFFSET, MASK_RATIO_OFFSET),
    ratio: flat.subarray(MASK_RATIO_OFFSET, MASK_SPATIAL_OFFSET),
    spatial: flat.subarray(MASK_SPATIAL_OFFSET, MASK_SITE_OFFSET),
    site: flat.subarray(MASK_SITE_OFFSET, MASK_SIZE),
  };
}

export function targetRow(masks: Masks, intent: number): Uint8Array {
  return masks.target.subarray(intent * NUM_SLOTS, (intent + 1) * NUM_SLOTS);
}
export function spatialRow(masks: Masks, kind: number): Uint8Array {
  return masks.spatial.subarray(kind * RASTER_CELLS, (kind + 1) * RASTER_CELLS);
}
export function siteRow(masks: Masks, kind: number): Uint8Array {
  return masks.site.subarray(kind * MAX_BUILD_SITES, (kind + 1) * MAX_BUILD_SITES);
}

/** Layout summary sent in HELLO so the Python side can assert agreement. */
export function layoutDescription() {
  return {
    version: SCHEMA_VERSION,
    rasterPlanes: NUM_PLANES,
    rasterHeight: RASTER_HEIGHT,
    rasterWidth: RASTER_WIDTH,
    slots: NUM_SLOTS,
    slotFields: SLOT_FIELDS,
    buildSites: MAX_BUILD_SITES,
    siteFields: SITE_FIELDS,
    globals: NUM_GLOBALS,
    featureSize: FEATURE_SIZE,
    intents: NUM_INTENTS,
    ratioBuckets: NUM_RATIO_BUCKETS,
    spatialKinds: NUM_SPATIAL_KINDS,
    siteKinds: NUM_SITE_KINDS,
    maskSize: MASK_SIZE,
  };
}
