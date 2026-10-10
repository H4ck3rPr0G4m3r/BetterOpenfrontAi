import type { TileRef } from "../../OpenFrontIO/src/core/game/GameMap";
import { RASTER_CELLS, RASTER_HEIGHT, RASTER_WIDTH } from "./Layout";

/**
 * Letterbox mapping between map tiles and the 128x64 raster shared by the
 * encoder (observation) and codec (spatial head): one scale factor
 * s = max(width/128, height/64) on both axes, map centred. Geometry stays
 * isometric on every map; padding cells are always masked.
 */
export function letterboxTransform(mapWidth: number, mapHeight: number) {
  const scale = Math.max(mapWidth / RASTER_WIDTH, mapHeight / RASTER_HEIGHT);
  return {
    scale,
    offsetX: Math.floor((RASTER_WIDTH - mapWidth / scale) / 2),
    offsetY: Math.floor((RASTER_HEIGHT - mapHeight / scale) / 2),
  };
}

export interface CellRect {
  x0: number;
  x1: number; // exclusive
  y0: number;
  y1: number; // exclusive
}

/** Static per-map-size geometry, computed once. */
export interface CellGeometry {
  w: number;
  h: number;
  /** Raster cell for each tile column / row (row value is cell row * RASTER_WIDTH). */
  cxForX: Int32Array;
  rowBaseForY: Int32Array;
  /** Representative centre tile per cell, -1 for letterbox padding. */
  centerTile: Int32Array;
}

const geometryCache = new Map<string, CellGeometry>();

export function cellGeometry(w: number, h: number): CellGeometry {
  const key = `${w}x${h}`;
  const cached = geometryCache.get(key);
  if (cached !== undefined) return cached;
  const { scale, offsetX, offsetY } = letterboxTransform(w, h);
  const cxForX = new Int32Array(w);
  for (let x = 0; x < w; x++) {
    cxForX[x] = Math.min(RASTER_WIDTH - 1, Math.max(0, offsetX + Math.floor(x / scale)));
  }
  const rowBaseForY = new Int32Array(h);
  for (let y = 0; y < h; y++) {
    rowBaseForY[y] = Math.min(RASTER_HEIGHT - 1, Math.max(0, offsetY + Math.floor(y / scale))) * RASTER_WIDTH;
  }
  const centerTile = new Int32Array(RASTER_CELLS);
  for (let cell = 0; cell < RASTER_CELLS; cell++) {
    const cx = cell % RASTER_WIDTH;
    const cy = Math.floor(cell / RASTER_WIDTH);
    const x = (cx - offsetX + 0.5) * scale;
    const y = (cy - offsetY + 0.5) * scale;
    centerTile[cell] = x < 0 || x >= w || y < 0 || y >= h ? -1 : Math.floor(y) * w + Math.floor(x);
  }
  const geometry = { w, h, cxForX, rowBaseForY, centerTile };
  geometryCache.set(key, geometry);
  return geometry;
}

export function cellRect(cell: number, w: number, h: number): CellRect {
  const cx = cell % RASTER_WIDTH;
  const cy = Math.floor(cell / RASTER_WIDTH);
  const { scale, offsetX, offsetY } = letterboxTransform(w, h);
  return {
    x0: Math.max(0, Math.floor((cx - offsetX) * scale)),
    x1: Math.min(w, Math.ceil((cx - offsetX + 1) * scale)),
    y0: Math.max(0, Math.floor((cy - offsetY) * scale)),
    y1: Math.min(h, Math.ceil((cy - offsetY + 1) * scale)),
  };
}

/** Raster cell of a tile. */
export function tileCell(tile: TileRef, geometry: CellGeometry): number {
  const x = tile % geometry.w;
  const y = Math.floor(tile / geometry.w);
  return geometry.rowBaseForY[y] + geometry.cxForX[x];
}
