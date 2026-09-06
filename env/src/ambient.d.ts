/**
 * Ambient declarations `GameRunner.ts`'s runtime dependency on
 * `src/client/hud/NameBoxCalculator.ts` drags in transitively, normally
 * supplied by OpenFrontIO's own `src/client/Main.ts` (Window.showPage) and
 * Vite's client types (import.meta.hot, `?worker&inline` module suffix).
 * Neither is reachable from our narrower program (see tsconfig.json's
 * comment) — declared here so `tsc --noEmit` is a real, clean gate instead
 * of a documented exception.
 */
interface Window {
  showPage?: (pageId: string) => void;
}

interface ImportMeta {
  hot?: {
    accept: (cb?: (mod: unknown) => void) => void;
    dispose: (cb: (data: unknown) => void) => void;
    on: (event: string, cb: (...args: unknown[]) => void) => void;
  };
}

declare module "*?worker&inline" {
  const WorkerFactory: new () => Worker;
  export default WorkerFactory;
}
