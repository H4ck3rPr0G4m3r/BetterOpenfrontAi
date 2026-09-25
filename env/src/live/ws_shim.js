/**
 * WebSocket shim for live openfront.io lobbies (spec §10.2, chosen transport).
 *
 * Install with Playwright's `addInitScript` BEFORE any page script runs. It
 * wraps `window.WebSocket`, captures the socket the real client opens to the
 * game worker, and exposes both directions to Node:
 *
 * - Inbound: every server frame is forwarded out of the page via the
 *   `__openfrontFrame` binding (registered with `exposeBinding`).
 * - Outbound: Node calls `window.__openfrontSend(bytes)` to emit zbin
 *   intents on that same socket. Intents sent this way are
 *   indistinguishable from the client's own: the server stamps `clientID`
 *   onto the intent itself (`IntentAuthorization.ts`).
 *
 * Explicit non-goal: no anti-detection, no stealth plugins, no Turnstile
 * solvers. Either headless Chromium passes the ordinary challenge or the
 * sanctioned route (`ADMIN_BOT_API_KEY`, `AdminBotRoutes.ts`) is used.
 */
(() => {
  const OriginalWebSocket = window.WebSocket;
  let gameSocket = null;
  const queue = [];

  function isGameSocket(url) {
    return typeof url === "string" && url.includes("/game");
  }

  // @ts-ignore — installed before the bundle; the page has no such binding yet.
  window.WebSocket = function (url, protocols) {
    const socket = protocols
      ? new OriginalWebSocket(url, protocols)
      : new OriginalWebSocket(url);
    if (isGameSocket(typeof url === "string" ? url : url.toString())) {
      gameSocket = socket;
      socket.binaryType = "arraybuffer";
      socket.addEventListener("message", (event) => {
        const data = event.data;
        const bytes = data instanceof ArrayBuffer ? Array.from(new Uint8Array(data)) : null;
        if (bytes && window.__openfrontFrame) {
          window.__openfrontFrame({ bytes }).catch(() => {});
        }
      });
      for (const pending of queue.splice(0)) {
        socket.send(pending);
      }
    }
    return socket;
  };
  window.WebSocket.prototype = OriginalWebSocket.prototype;

  // @ts-ignore — called by Node to emit intents on the captured socket.
  window.__openfrontSend = function (bytes) {
    const payload = new Uint8Array(bytes).buffer;
    if (gameSocket && gameSocket.readyState === 1) {
      gameSocket.send(payload);
      return true;
    }
    queue.push(payload);
    return false;
  };

  // @ts-ignore — lets Node wait for the game socket before joining.
  window.__openfrontGameSocketReady = function () {
    return gameSocket !== null && gameSocket.readyState === 1;
  };
})();
