/**
 * Turnstile probe (spec §10.6 step 2): gates ALL of live play.
 *
 * Loads the official openfront.io front page in headless Chromium and
 * reports whether a Cloudflare Turnstile token materializes by ordinary
 * means. No solvers, no stealth plugins, no fingerprint work — either the
 * ordinary challenge passes headless or it does not. If it does not, the
 * correct next step is the sanctioned route: ask the operators for an
 * `ADMIN_BOT_API_KEY` seat (`src/server/AdminBotRoutes.ts`).
 *
 * Usage: `node probe_turnstile.mjs [--url https://openfront.io]`
 * Needs `playwright` (`npx playwright install chromium` first). Exits 0
 * with a JSON verdict on stdout; exit 2 when playwright is unavailable.
 */
let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  console.log(JSON.stringify({ probe: "turnstile", available: false, reason: "playwright not installed" }));
  process.exit(2);
}

const url = process.argv.includes("--url")
  ? process.argv[process.argv.indexOf("--url") + 1]
  : "https://openfront.io";

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  // The join flow renders the Turnstile widget inside an iframe; a token
  // appears as a hidden input once the challenge passes.
  await page.waitForTimeout(15_000);
  const token = await page.evaluate(() => {
    const input = document.querySelector('input[name="cf-turnstile-response"]');
    return input ? String(input.value || "") : "";
  });
  const commit = await page.evaluate(() => window.BOOTSTRAP_CONFIG?.gitCommit ?? "");
  console.log(
    JSON.stringify({
      probe: "turnstile",
      available: true,
      tokenPresent: token.length > 0,
      serverCommit: commit || null,
      verdict:
        token.length > 0
          ? "headless token materializes; live lobby play may proceed to staging step 1"
          : "no headless token; use the sanctioned ADMIN_BOT_API_KEY route",
    }),
  );
} finally {
  await browser.close();
}
