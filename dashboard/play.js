// Versus page: polls the match snapshot, renders the 64x32 board, posts moves.
const COLORS = ["#0b2a4a", "#3a4a3a", "#c0392b", "#2980b9", "#e67e22"];
const board = document.getElementById("board");
const ctx = board.getContext("2d");
const statusEl = document.getElementById("status");
const cellEl = document.getElementById("cell");
let aimCell = 0;

function draw(snapshot) {
  const { w, h, cells } = snapshot;
  const sx = board.width / w;
  const sy = board.height / h;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      ctx.fillStyle = COLORS[cells[y * w + x]] || "#000";
      ctx.fillRect(Math.floor(x * sx), Math.floor(y * sy), Math.ceil(sx), Math.ceil(sy));
    }
  }
}

function showState(prefix, data) {
  const snap = data.snapshot;
  if (!snap) {
    statusEl.textContent = `${prefix}: ${JSON.stringify(data)}`;
    return;
  }
  draw(snap);
  statusEl.textContent =
    `${prefix} moves=${data.moves ?? "-"} ticks=${data.ticks ?? snap.ticks} ` +
    `done=${snap.done} terminal=${snap.terminal}\n` +
    `you: tiles=${snap.human.tiles} troops=${snap.human.troops} alive=${snap.human.alive}\n` +
    `ai: tiles=${snap.ai.tiles} troops=${snap.ai.troops} alive=${snap.ai.alive}` +
    (data.ai_action_text ? `\nai last move: ${data.ai_action_text}` : "");
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

document.getElementById("start").onclick = async () => {
  const val = (id) => document.getElementById(id).value;
  try {
    const data = await post("/api/play/start", {
      seed: val("seed"),
      bots: Number(val("bots")),
      nations: val("nations"),
      predictor: val("predictor"),
      model: val("model"),
    });
    showState("started", { ...data, moves: 0 });
  } catch (err) {
    statusEl.textContent = `start failed: ${err.message}`;
  }
};

document.getElementById("stop").onclick = async () => {
  await post("/api/play/stop", {});
  statusEl.textContent = "stopped";
};

board.onclick = (ev) => {
  const rect = board.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - rect.left) / rect.width) * 64);
  const y = Math.floor(((ev.clientY - rect.top) / rect.height) * 32);
  // Board cell (64x32) -> spatial head cell (128x64): nearest raster cell.
  aimCell = Math.min(8191, (Math.min(63, y * 2) * 128) + Math.min(127, x * 2));
  cellEl.textContent = String(aimCell);
};

document.getElementById("move").onclick = async () => {
  const val = (id) => document.getElementById(id).value;
  try {
    const data = await post("/api/play/move", {
      action: {
        intentType: Number(val("intent")),
        target: Number(val("target")),
        ratioBucket: Number(val("ratio")),
        spatialCell: aimCell,
        buildSite: Number(val("site")),
      },
    });
    showState("moved", data);
  } catch (err) {
    statusEl.textContent = `move failed: ${err.message}`;
  }
};

// Light polling so the board stays fresh while thinking.
setInterval(async () => {
  try {
    const res = await fetch("/api/play/state");
    const data = await res.json();
    if (data.active && data.snapshot) showState("state", data);
  } catch {
    /* server may be restarting; next tick retries */
  }
}, 2000);
