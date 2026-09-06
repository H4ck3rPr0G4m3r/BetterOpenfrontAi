import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vitest/config";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      // OpenFrontIO/src/core code uses the bare specifier "resources/*"
      // (see OpenFrontIO/vite.config.ts, which aliases it the same way) —
      // e.g. TribeNames.ts imports "resources/tribeNameThemes.json".
      resources: path.resolve(__dirname, "../OpenFrontIO/resources"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    // Full-game tests construct a real World-map GameRunner; give them room.
    testTimeout: 30_000,
  },
});
