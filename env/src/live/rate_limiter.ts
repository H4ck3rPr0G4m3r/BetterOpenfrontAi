/**
 * Client intent rate budget, mirroring the official server's
 * `ClientMsgRateLimiter.ts`: 10 intents/second, 150 intents/minute,
 * 2000 bytes per intent, 5MB total per client. The driver enforces its own
 * budget BELOW the server's (8/s bursts, 120/min) so it discovers the
 * ceiling by counting, never by being kicked mid-game (§10.5).
 */
export class RateLimiter {
  private secondWindow: number[] = [];
  private minuteWindow: number[] = [];
  private bytesTotal = 0;

  constructor(
    private readonly perSecond = 8,
    private readonly perMinute = 120,
    private readonly maxBytesPerIntent = 2000,
    private readonly maxBytesTotal = 5 * 1024 * 1024,
  ) {}

  /** Milliseconds until `size`-byte intent may be sent, 0 when now. */
  tryAcquire(nowMs: number, sizeBytes: number): number {
    if (sizeBytes > this.maxBytesPerIntent) return Number.POSITIVE_INFINITY;
    if (this.bytesTotal + sizeBytes > this.maxBytesTotal) return Number.POSITIVE_INFINITY;
    this.secondWindow = this.secondWindow.filter((t) => nowMs - t < 1000);
    this.minuteWindow = this.minuteWindow.filter((t) => nowMs - t < 60_000);
    if (this.secondWindow.length >= this.perSecond) {
      return this.secondWindow[0] + 1000 - nowMs;
    }
    if (this.minuteWindow.length >= this.perMinute) {
      return this.minuteWindow[0] + 60_000 - nowMs;
    }
    this.secondWindow.push(nowMs);
    this.minuteWindow.push(nowMs);
    this.bytesTotal += sizeBytes;
    return 0;
  }

  get sentTotal(): number {
    return this.secondWindow.length;
  }
}

/**
 * Version-parity gate (§10.3): the served page injects the server's own
 * commit as `window.BOOTSTRAP_CONFIG.gitCommit`. Refuse to join on
 * mismatch — a confidently wrong shadow sim is worse than no game.
 * NOTE: `GET /desktop/version.json` is NOT the right source (Steam-asset
 * version, different semantics from the web bundle commit).
 */
export function checkVersionParity(pageCommit: string, pinnedCommit: string): { ok: boolean; reason: string } {
  if (!pageCommit) return { ok: false, reason: "live page exposed no git commit, refusing to join" };
  if (pageCommit === "DEV") return { ok: true, reason: "dev server: commit check skipped" };
  if (pageCommit !== pinnedCommit) {
    return { ok: false, reason: `engine mismatch: live ${pageCommit} vs pinned ${pinnedCommit}, refusing to join` };
  }
  return { ok: true, reason: "engine commit matches pinned training commit" };
}
