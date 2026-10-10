"use strict";

const $ = (id) => document.getElementById(id);
const NONE = "";

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function ago(iso) {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function showError(box, err) {
  box.textContent = err ? String(err.message || err) : "";
  box.hidden = !err;
}

/* ---------------- catalog: models, recipes, runs ---------------- */

let catalog = { checkpoints: [], agents: [], configs: [], runs: [] };

function fillModelSelect(select) {
  const keep = select.value;
  select.innerHTML = "";
  if (select.dataset.optional) select.append(el("option", { value: NONE }, "(empty)"));
  const group = el("optgroup", { label: "Trained checkpoints" });
  for (const c of catalog.checkpoints) {
    group.append(el("option", { value: c.path }, c.path.replace(/^checkpoints\//, "")));
  }
  if (catalog.checkpoints.length) select.append(group);
  const scripted = el("optgroup", { label: "Built-in" });
  for (const a of catalog.agents) scripted.append(el("option", { value: a }, a));
  select.append(scripted);
  if ([...select.options].some((o) => o.value === keep)) select.value = keep;
}

async function loadCatalog() {
  catalog = await api("GET", "/api/catalog");
  document.querySelectorAll(".model-select").forEach(fillModelSelect);
  const cfg = $("train-config");
  const keep = cfg.value || "selfplay.yaml";
  cfg.innerHTML = "";
  catalog.configs.forEach((c) => cfg.append(el("option", { value: c }, c)));
  cfg.value = catalog.configs.includes(keep) ? keep : catalog.configs[0] || "";
  const run = $("metrics-run");
  const keepRun = run.value;
  run.innerHTML = "";
  catalog.runs.forEach((r) => run.append(el("option", { value: r.name }, `${r.name} (${ago(r.modified)})`)));
  if (catalog.runs.some((r) => r.name === keepRun)) run.value = keepRun;
}

/* ---------------- jobs ---------------- */

const STATUS_TEXT = { running: "running", done: "finished", error: "failed", stopped: "stopped" };
const openLogs = new Set();

function jobCard(job) {
  const status = el("span", { class: `pill ${job.status}` }, STATUS_TEXT[job.status] || job.status);
  const action = job.status === "running"
    ? el("button", { type: "button", onclick: () => stopJob(job.id) }, job.kind === "lobby" ? "Leave" : "Stop")
    : el("button", { type: "button", class: "quiet", onclick: () => stopJob(job.id) }, "Dismiss");
  const head = el("div", { class: "job-head" },
    el("strong", {}, job.title), status, el("span", { class: "hint" }, ago(job.started)), action);
  const card = el("div", { class: "job-card" }, head);
  if (job.result) card.append(evalTable(job.result));
  const lines = job.log_tail || [];
  if (lines.length) {
    const log = el("pre", { class: "log" }, lines.join("\n"));
    const more = el("details", {}, el("summary", {}, `Output (${job.id})`), log);
    if (openLogs.has(job.id)) more.open = true;
    more.addEventListener("toggle", () => (more.open ? openLogs.add(job.id) : openLogs.delete(job.id)));
    card.append(more);
  } else if (job.status === "running") {
    card.append(el("div", { class: "hint" }, "starting"));
  }
  return card;
}

function evalTable(result) {
  const table = el("table", { class: "compact" });
  table.append(el("thead", {}, el("tr", {}, ...["AI", "games", "survive", "win", "tiles", "placement", "kills"]
    .map((h) => el("th", {}, h)))));
  const body = el("tbody");
  for (const [spec, s] of Object.entries(result.predictors || {})) {
    body.append(el("tr", {},
      el("td", {}, spec.split(/[\\/]/).pop().replace(/\.pt$/, "")),
      el("td", {}, String(s.episodes)),
      el("td", {}, `${Math.round(s.survival_rate * 100)}%`),
      el("td", {}, `${Math.round(s.win_rate * 100)}%`),
      el("td", {}, Math.round(s.mean_tiles).toLocaleString()),
      el("td", {}, s.mean_placement.toFixed(2)),
      el("td", {}, s.mean_kills.toFixed(2))));
  }
  table.append(body);
  const wrap = el("div", {}, table);
  const pairs = Object.entries(result.head_to_head || {});
  if (pairs.length) {
    wrap.append(el("div", { class: "hint" }, pairs.map(([p, r]) => `${p}: ${Math.round(r * 100)}%`).join("   ")));
  }
  return wrap;
}

function renderJobs(jobs) {
  const boxes = { lobby: $("lobby-jobs"), train: $("train-jobs"), eval: $("eval-jobs") };
  for (const [kind, box] of Object.entries(boxes)) {
    const mine = jobs.filter((j) => j.kind === kind);
    box.replaceChildren(...mine.map(jobCard));
  }
}

async function stopJob(id) {
  try {
    await api("DELETE", `/api/jobs/${id}`);
  } finally {
    refreshState();
  }
}

async function refreshState() {
  const status = $("server-status");
  try {
    const st = await api("GET", "/api/state");
    renderJobs(st.jobs);
    const running = st.jobs.filter((j) => j.status === "running").length;
    status.textContent = running ? `${running} running` : "idle";
    status.className = "status ok";
  } catch {
    status.textContent = "server offline";
    status.className = "status bad";
  }
}

function submitter(formId, errorBox, build) {
  $(formId).addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const button = ev.target.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const [path, body] = build();
      await api("POST", path, body);
      showError(errorBox, null);
      await refreshState();
    } catch (err) {
      showError(errorBox, err);
    } finally {
      button.disabled = false;
    }
  });
}

/* ---------------- metrics ---------------- */

const CHARTS = [
  ["rollout/survival_rate", "Survival rate"],
  ["rollout/ep_rew_mean", "Episode return"],
  ["rollout/final_tiles", "Final tiles"],
  ["selfplay/learner_beats_pool", "Beats past selves"],
  ["train/explained_variance", "Value fit (explained variance)"],
  ["time/steps_per_second", "Steps per second"],
];

function drawChart(container, title, points) {
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const last = values[values.length - 1];
  const head = el("div", { class: "chart-title" }, el("span", {}, title),
    el("span", { class: "chart-last" }, Math.abs(last) >= 100 ? Math.round(last).toLocaleString() : last.toFixed(3)));
  const canvas = el("canvas", { class: "chart", width: "520", height: "140" });
  container.append(el("div", { class: "chart-box" }, head, canvas));
  const ctx = canvas.getContext("2d");
  const span = max - min || 1;
  const x = (i) => (i / Math.max(1, points.length - 1)) * (canvas.width - 8) + 4;
  const y = (v) => canvas.height - 8 - ((v - min) / span) * (canvas.height - 16);
  ctx.fillStyle = "rgba(111, 194, 135, 0.10)";
  ctx.beginPath();
  ctx.moveTo(x(0), canvas.height);
  points.forEach((p, i) => ctx.lineTo(x(i), y(p.value)));
  ctx.lineTo(x(points.length - 1), canvas.height);
  ctx.fill();
  ctx.strokeStyle = "#6fc287";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.value)) : ctx.moveTo(x(i), y(p.value))));
  ctx.stroke();
}

function drawIntents(container, intents) {
  const entries = Object.entries(intents).filter(([, v]) => v > 0.001).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return;
  const box = el("div", { class: "chart-box wide" }, el("div", { class: "chart-title" }, el("span", {}, "What the AI does (latest update)")));
  for (const [name, value] of entries) {
    box.append(el("div", { class: "bar" },
      el("span", { class: "name" }, name.replace(/_/g, " ")),
      el("span", { class: "fill", style: `width:${Math.max(2, value * 360)}px` }),
      el("span", { class: "n" }, `${(value * 100).toFixed(1)}%`)));
  }
  container.append(box);
}

async function loadMetrics() {
  const run = $("metrics-run").value;
  const box = $("charts");
  if (!run) {
    $("metrics-note").textContent = "No runs yet. Start a training run above.";
    box.replaceChildren();
    return;
  }
  $("metrics-note").textContent = "loading";
  try {
    const data = await api("GET", `/api/runstats?run=${encodeURIComponent(run)}`);
    box.replaceChildren();
    for (const [key, title] of CHARTS) {
      if (data.series[key]?.length) drawChart(box, title, data.series[key]);
    }
    drawIntents(box, data.intents || {});
    $("metrics-note").textContent = data.last_step
      ? `${data.updates} updates, ${data.last_step.toLocaleString()} steps`
      : data.note || "";
  } catch (err) {
    $("metrics-note").textContent = String(err.message || err);
  }
}

/* ---------------- replays ---------------- */

let replays = [];

function renderReplays() {
  const filter = $("replay-filter").value.trim().toLowerCase();
  const body = $("replay-table").querySelector("tbody");
  const shown = replays.filter((r) => !filter || r.path.toLowerCase().includes(filter)).slice(0, 200);
  if (!shown.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: "4", class: "hint" },
      replays.length ? "No replay matches the filter." : "No replays yet. Training records every 100th game; Evaluate records all of them.")));
    return;
  }
  body.replaceChildren(...shown.map((r) => {
    const outcome = /-alive\.json$/.test(r.path) ? "survived" : /-dead\.json$/.test(r.path) ? "eliminated" : "";
    const watch = el("button", { type: "button", onclick: async (ev) => {
      ev.target.disabled = true;
      try {
        await api("POST", "/api/jobs/watch", { replay: r.path });
        ev.target.textContent = "Opening";
      } catch (err) {
        ev.target.textContent = "Failed";
        ev.target.title = String(err.message || err);
      }
      setTimeout(() => { ev.target.disabled = false; ev.target.textContent = "Watch"; }, 4000);
    } }, "Watch");
    return el("tr", {},
      el("td", { class: "mono" }, r.path),
      el("td", { class: outcome === "survived" ? "good" : "" }, outcome),
      el("td", {}, ago(r.modified)),
      el("td", { class: "right" }, watch));
  }));
}

async function loadReplays() {
  replays = (await api("GET", "/api/replays")).replays;
  renderReplays();
}

/* ---------------- boot ---------------- */

async function boot() {
  submitter("lobby-form", $("lobby-error"), () => ["/api/jobs/lobby", {
    lobby: $("lobby-id").value,
    host: $("lobby-host").value,
    model: $("lobby-model").value,
    name: $("lobby-name").value,
  }]);
  // A pasted openfront.io link implies the real server.
  $("lobby-id").addEventListener("input", (ev) => {
    if (/openfront\.io/.test(ev.target.value)) $("lobby-host").value = "openfront";
    else if (/localhost|127\.0\.0\.1/.test(ev.target.value)) $("lobby-host").value = "local";
  });
  const trainError = el("div", { class: "error", hidden: "" });
  $("train-form").after(trainError);
  submitter("train-form", trainError, () => ["/api/jobs/train", {
    config: $("train-config").value,
    run_name: $("train-name").value,
    total_steps: Number($("train-steps").value),
    workers: Number($("train-workers").value),
    resume: $("train-resume").checked,
  }]);
  const evalError = el("div", { class: "error", hidden: "" });
  $("eval-form").after(evalError);
  submitter("eval-form", evalError, () => ["/api/jobs/eval", {
    seats: [...document.querySelectorAll(".eval-seat")].map((s) => s.value).filter(Boolean),
    games: Number($("eval-games").value),
    bots: Number($("eval-bots").value),
    minutes: Number($("eval-minutes").value),
    record: $("eval-record").checked,
  }]);
  $("metrics-load").addEventListener("click", loadMetrics);
  $("metrics-run").addEventListener("change", loadMetrics);
  $("replay-refresh").addEventListener("click", loadReplays);
  $("replay-filter").addEventListener("input", renderReplays);

  await Promise.all([loadCatalog().catch(() => {}), refreshState(), loadReplays().catch(() => {})]);
  loadMetrics();
  setInterval(refreshState, 2000);
  setInterval(() => loadCatalog().catch(() => {}), 30000);
}

boot();
