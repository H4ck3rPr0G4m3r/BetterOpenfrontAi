"use strict";

const $ = (id) => document.getElementById(id);
const NONE = "";
const ACCENT = "#6fc287";

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
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function fmt(v) {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  if (Math.abs(v) >= 100 || Number.isInteger(v)) return v.toFixed(0);
  return v.toFixed(2);
}
const pct = (v) => (v === undefined ? "-" : `${Math.round(v * 100)}%`);
const NAMES = { vs_bots: "Vs bots", selfplay_ffa: "Self-play FFA", close_rivals: "Close rivals", crowded: "Crowded", easy: "Easy" };
const pretty = (name) => NAMES[name] || name.replace(/_/g, " ");

function showError(box, err) {
  box.textContent = err ? String(err.message || err) : "";
  box.hidden = !err;
}

/* ---------------- tabs ---------------- */

const TABS = ["play", "training", "eval", "replays"];

function showTab(name) {
  const tab = TABS.includes(name) ? name : "play";
  for (const t of TABS) $(t).hidden = t !== tab;
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  if (tab === "training") loadMetrics();
  if (tab === "replays") loadReplays().catch(() => {});
}

/* ---------------- catalog: models, recipes, runs ---------------- */

let catalog = { checkpoints: [], agents: [], configs: [], runs: [] };

function fillModelSelect(select) {
  const keep = select.value;
  select.innerHTML = "";
  if (select.dataset.optional) select.append(el("option", { value: NONE }, "Empty seat"));
  if (catalog.checkpoints.length) {
    const group = el("optgroup", { label: "Trained models" });
    for (const c of catalog.checkpoints) {
      group.append(el("option", { value: c.path }, c.path.replace(/^checkpoints\//, "").replace(/\.pt$/, "")));
    }
    select.append(group);
  }
  const builtin = el("optgroup", { label: "Built-in" });
  for (const a of catalog.agents) builtin.append(el("option", { value: a }, a));
  select.append(builtin);
  if ([...select.options].some((o) => o.value === keep)) select.value = keep;
}

async function loadCatalog() {
  catalog = await api("GET", "/api/catalog");
  document.querySelectorAll(".model-select").forEach(fillModelSelect);
  const cfg = $("train-config");
  const keep = cfg.value || "main.yaml";
  cfg.innerHTML = "";
  catalog.configs.forEach((c) => cfg.append(el("option", { value: c }, c.replace(/\.yaml$/, ""))));
  cfg.value = catalog.configs.includes(keep) ? keep : catalog.configs[0] || "";
  const run = $("metrics-run");
  const keepRun = run.value;
  run.innerHTML = "";
  catalog.runs.forEach((r) => run.append(el("option", { value: r.name }, `${r.name}  (${ago(r.modified)})`)));
  if (catalog.runs.some((r) => r.name === keepRun)) run.value = keepRun;
}

/* ---------------- jobs ---------------- */

const STATUS_TEXT = { running: "Running", done: "Finished", error: "Failed", stopped: "Stopped" };
const openLogs = new Set();

function jobCard(job) {
  const action = job.status === "running"
    ? el("button", { type: "button", class: "small", onclick: () => stopJob(job.id) }, job.kind === "lobby" ? "Leave" : "Stop")
    : el("button", { type: "button", class: "small ghost", onclick: () => stopJob(job.id) }, "Dismiss");
  const head = el("div", { class: "job-head" },
    el("strong", {}, job.title),
    el("span", { class: `pill ${job.status}` }, STATUS_TEXT[job.status] || job.status),
    el("span", { class: "hint" }, ago(job.started)),
    action);
  const card = el("div", { class: `job-card ${job.status}` }, head);
  if (job.result) card.append(evalTable(job.result));
  const lines = job.log_tail || [];
  if (lines.length) {
    const more = el("details", {}, el("summary", {}, "Output"), el("pre", { class: "log", "data-job": job.id }, lines.join("\n")));
    if (openLogs.has(job.id)) more.open = true;
    more.addEventListener("toggle", () => (more.open ? openLogs.add(job.id) : openLogs.delete(job.id)));
    card.append(more);
  } else if (job.status === "running") {
    card.append(el("div", { class: "hint" }, "Starting up"));
  }
  return card;
}

function evalTable(result) {
  const table = el("table", { class: "compact" });
  table.append(el("thead", {}, el("tr", {}, ...["AI", "Games", "Survived", "Won", "Land", "Placement", "Kills"]
    .map((h) => el("th", {}, h)))));
  const body = el("tbody");
  for (const [spec, s] of Object.entries(result.predictors || {})) {
    body.append(el("tr", {},
      el("td", {}, spec.split(/[\\/]/).pop().replace(/\.pt$/, "")),
      el("td", { class: "num" }, String(s.episodes)),
      el("td", { class: "num" }, pct(s.survival_rate)),
      el("td", { class: "num" }, pct(s.win_rate)),
      el("td", { class: "num" }, Math.round(s.mean_tiles).toLocaleString()),
      el("td", { class: "num" }, s.mean_placement.toFixed(2)),
      el("td", { class: "num" }, s.mean_kills.toFixed(2))));
  }
  table.append(body);
  const wrap = el("div", {}, table);
  const pairs = Object.entries(result.head_to_head || {});
  if (pairs.length) {
    wrap.append(el("div", { class: "hint" }, pairs.map(([p, r]) => `${p}: ${pct(r)}`).join("   ")));
  }
  return wrap;
}

function renderJobs(jobs) {
  // Logs are rebuilt on every poll: keep each one where it was scrolled, and
  // keep following the end when it was at the bottom.
  const scroll = new Map();
  document.querySelectorAll("pre.log[data-job]").forEach((pre) => {
    scroll.set(pre.dataset.job, { top: pre.scrollTop, end: pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4 });
  });
  const boxes = { lobby: $("lobby-jobs"), train: $("train-jobs"), eval: $("eval-jobs"), match: $("match-jobs") };
  for (const [kind, box] of Object.entries(boxes)) {
    box.replaceChildren(...jobs.filter((j) => j.kind === kind).map(jobCard));
  }
  document.querySelectorAll("pre.log[data-job]").forEach((pre) => {
    const was = scroll.get(pre.dataset.job);
    pre.scrollTop = !was || was.end ? pre.scrollHeight : was.top;
  });
  // A finished quick match has a new replay to show.
  const done = jobs.filter((j) => j.kind === "match" && j.status === "done").map((j) => j.id).join();
  if (done !== lastMatchesDone) {
    if (lastMatchesDone !== null) loadReplays().catch(() => {});
    lastMatchesDone = done;
  }
}
let lastMatchesDone = null;

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
    status.textContent = running ? `${running} running` : "Idle";
    status.className = "status ok";
  } catch {
    status.textContent = "Server offline";
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

/* ---------------- public openfront.io lobbies ---------------- */

const KINDS = { ffa: "Free for all", team: "Teams", special: "Special", hosted: "Player hosted" };
let publicList = { rows: [], at: 0 };
let publicTimer = null;

function lobbyBody(extra) {
  return { model: $("lobby-model").value, name: $("lobby-name").value, ...extra };
}

async function joinPublic(body, button) {
  button.disabled = true;
  try {
    await api("POST", "/api/jobs/lobby", lobbyBody(body));
    showError($("public-error"), null);
    await refreshState();
  } catch (err) {
    showError($("public-error"), err);
  } finally {
    button.disabled = false;
  }
}

function renderPublic() {
  const body = $("public-table").querySelector("tbody");
  const elapsed = (Date.now() - publicList.at) / 1000;
  const rows = publicList.rows.filter((r) => r.kind !== "hosted");
  $("public-wrap").hidden = false;
  if (!rows.length) {
    body.replaceChildren(el("tr", { class: "empty-row" }, el("td", { colspan: "5" }, "No public lobbies are open right now.")));
    return;
  }
  body.replaceChildren(...rows.map((r) => {
    const left = r.starts_in === null ? null : Math.round(r.starts_in - elapsed);
    const started = left !== null && left <= 0;
    const full = r.max && r.players >= r.max;
    const join = el("button", { type: "button", class: "small", onclick: (ev) => joinPublic({ lobby: r.id, host: "openfront" }, ev.currentTarget) }, "Join");
    if (started || full) join.disabled = true;
    return el("tr", {},
      el("td", {}, r.map || "?"),
      el("td", {}, KINDS[r.kind] || r.kind),
      el("td", { class: "num" }, r.max ? `${r.players} / ${r.max}` : String(r.players)),
      el("td", { class: "num" }, left === null ? "waiting" : started ? "started" : `${left}s`),
      el("td", {}, join));
  }));
}

async function findPublic() {
  const button = $("public-find");
  button.disabled = true;
  $("public-note").textContent = "Opening openfront.io in a browser window. Click the Cloudflare check there if it asks.";
  try {
    const data = await api("POST", "/api/public-lobbies");
    publicList = { rows: data.lobbies || [], at: Date.now() };
    showError($("public-error"), null);
    $("public-note").textContent = `Listed ${new Date().toLocaleTimeString()}.`;
    renderPublic();
    clearInterval(publicTimer);
    publicTimer = setInterval(renderPublic, 1000);
  } catch (err) {
    showError($("public-error"), err);
    $("public-note").textContent = "";
  } finally {
    button.disabled = false;
  }
}

/* ---------------- charts ---------------- */

const CHARTS = [
  ["rollout/survival_rate", "Survival rate", pct],
  ["rollout/ep_rew_mean", "Episode return", fmt],
  ["rollout/final_tiles", "Land at the end", fmt],
  ["selfplay/learner_beats_pool", "Beats past selves", pct],
  ["train/explained_variance", "Value fit", fmt],
  ["time/steps_per_second", "Steps per second", fmt],
];

/** A crisp line chart sized to its box, with min/max shown under it. */
function drawChart(container, title, points, format = fmt, mini = false) {
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const canvas = el("canvas", { class: "chart" });
  const box = el("div", { class: "chart-box" },
    el("div", { class: "chart-title" }, el("span", {}, title), el("span", { class: "chart-last" }, format(values[values.length - 1]))),
    canvas);
  if (!mini) box.append(el("div", { class: "chart-range" }, el("span", {}, `low ${format(min)}`), el("span", {}, `high ${format(max)}`)));
  container.append(box);

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = 3;
  const span = max - min || 1;
  const x = (i) => pad + (i / Math.max(1, points.length - 1)) * (w - 2 * pad);
  const y = (v) => h - pad - ((v - min) / span) * (h - 2 * pad);

  ctx.strokeStyle = "rgba(141, 155, 171, 0.12)";
  ctx.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(0, Math.round(h * f) + 0.5);
    ctx.lineTo(w, Math.round(h * f) + 0.5);
    ctx.stroke();
  }
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(111, 194, 135, 0.22)");
  grad.addColorStop(1, "rgba(111, 194, 135, 0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(x(0), h);
  points.forEach((p, i) => ctx.lineTo(x(i), y(p.value)));
  ctx.lineTo(x(points.length - 1), h);
  ctx.fill();
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 1.75;
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.value)) : ctx.moveTo(x(i), y(p.value))));
  ctx.stroke();
  ctx.fillStyle = ACCENT;
  ctx.beginPath();
  ctx.arc(x(points.length - 1), y(values[values.length - 1]), 2.5, 0, Math.PI * 2);
  ctx.fill();
}

function renderStats(data) {
  const last = data.last || {};
  const stage = last["curriculum/stage"];
  const stats = [
    ["Steps trained", fmt(data.last_step), `${data.updates} updates`],
    ["Survival", pct(last["rollout/survival_rate"]), "recent games"],
    ["Episode return", fmt(last["rollout/ep_rew_mean"]), "recent games"],
    ["Land at the end", fmt(last["rollout/final_tiles"]), "tiles"],
    ["Curriculum stage", stage === undefined ? "-" : String(stage + 1), "of the recipe's stages"],
  ];
  $("run-stats").replaceChildren(...stats.map(([label, value, sub]) =>
    el("div", { class: "stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value), el("div", { class: "sub" }, sub))));
}

function renderIntents(intents) {
  const entries = Object.entries(intents).filter(([, v]) => v > 0.001).sort((a, b) => b[1] - a[1]);
  $("intent-block").hidden = !entries.length;
  const top = entries.length ? entries[0][1] : 1;
  $("intent-bars").replaceChildren(...entries.map(([name, value]) => {
    const label = name === "noop" ? "wait" : pretty(name);
    return el("div", { class: "bar" },
      el("span", { class: "name" }, label.charAt(0).toUpperCase() + label.slice(1)),
      el("span", { class: "track" }, el("span", { class: "fill", style: `display:block;width:${Math.max(1, (value / top) * 100)}%` })),
      el("span", { class: "n" }, `${(value * 100).toFixed(1)}%`));
  }));
}

async function loadMetrics() {
  const run = $("metrics-run").value;
  const charts = $("charts");
  if (!run) {
    $("metrics-note").textContent = "No runs yet. Start one above.";
    for (const id of ["charts", "run-stats", "scenario-charts", "intent-bars"]) $(id).replaceChildren();
    $("scenario-block").hidden = $("intent-block").hidden = true;
    return;
  }
  $("metrics-note").textContent = "Loading";
  try {
    const data = await api("GET", `/api/runstats?run=${encodeURIComponent(run)}`);
    renderStats(data);
    charts.replaceChildren();
    for (const [key, title, format] of CHARTS) {
      if (data.series[key]?.length > 1) drawChart(charts, title, data.series[key], format);
    }
    const minis = $("scenario-charts");
    minis.replaceChildren();
    for (const [key, points] of Object.entries(data.series)) {
      const m = key.match(/^scenario\/(.+)\/survival_rate$/);
      if (m && points.length > 1) drawChart(minis, pretty(m[1]), points, pct, true);
    }
    $("scenario-block").hidden = !minis.children.length;
    renderIntents(data.intents || {});
    $("metrics-note").textContent = data.note || "";
  } catch (err) {
    $("metrics-note").textContent = String(err.message || err);
  }
}

/* ---------------- replays ---------------- */

let replays = [];
const picked = new Set();
let shownPaths = [];

/** Phosphor "trash" (regular), MIT, from @phosphor-icons/core 2.1.1. */
const TRASH_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M216,48H176V40a24,24,0,0,0-24-24H104A24,24,0,0,0,80,40v8H40a8,8,0,0,0,0,16h8V208a16,16,0,0,0,16,16H192a16,16,0,0,0,16-16V64h8a8,8,0,0,0,0-16ZM96,40a8,8,0,0,1,8-8h48a8,8,0,0,1,8,8v8H96Zm96,168H64V64H192ZM112,104v64a8,8,0,0,1-16,0V104a8,8,0,0,1,16,0Zm48,0v64a8,8,0,0,1-16,0V104a8,8,0,0,1,16,0Z"/></svg>';

function trashIcon() {
  const span = el("span", { class: "icon" });
  span.innerHTML = TRASH_SVG;
  return span;
}

/** Resolves true when the user confirms in the dialog. */
function confirmDialog(title, text) {
  const dialog = $("confirm-dialog");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise((resolve) => dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true }));
}

async function deleteReplays(paths) {
  if (!paths.length) return;
  if (paths.length > 1) {
    const ok = await confirmDialog(`Delete ${paths.length} replays?`, "The replay files are removed from disk. This cannot be undone.");
    if (!ok) return;
  }
  try {
    await api("POST", "/api/replays/delete", { paths });
    paths.forEach((p) => picked.delete(p));
    showError($("replay-error"), null);
  } catch (err) {
    showError($("replay-error"), err);
  }
  await loadReplays().catch(() => {});
}

function updatePickControls() {
  const button = $("replay-delete");
  button.disabled = picked.size === 0;
  button.lastChild.textContent = picked.size ? `Delete ${picked.size}` : "Delete selected";
  const all = $("replay-all");
  const shownPicked = shownPaths.filter((p) => picked.has(p)).length;
  all.checked = shownPaths.length > 0 && shownPicked === shownPaths.length;
  all.indeterminate = shownPicked > 0 && shownPicked < shownPaths.length;
}

function renderReplays() {
  const filter = $("replay-filter").value.trim().toLowerCase();
  const body = $("replay-table").querySelector("tbody");
  const rows = replays.map((r) => {
    const outcome = /-alive\.json$/.test(r.path) ? "Survived" : /-dead\.json$/.test(r.path) ? "Eliminated" : "";
    const parts = r.path.split("/");
    return { ...r, outcome, run: parts.length > 1 ? parts[0] : "", file: parts[parts.length - 1] };
  });
  const shown = rows.filter((r) => !filter || `${r.path} ${r.outcome}`.toLowerCase().includes(filter)).slice(0, 300);
  $("replay-count").textContent = replays.length ? `${shown.length} of ${replays.length}` : "";
  const existing = new Set(replays.map((r) => r.path));
  [...picked].forEach((p) => existing.has(p) || picked.delete(p));
  shownPaths = shown.map((r) => r.path);
  updatePickControls();
  if (!shown.length) {
    body.replaceChildren(el("tr", { class: "empty-row" }, el("td", { colspan: "6" },
      replays.length ? "No replay matches the filter." : "No replays yet. Training records one game in 20; evaluation and quick matches record every game.")));
    return;
  }
  body.replaceChildren(...shown.map((r) => {
    const watch = el("button", { type: "button", class: "small", onclick: async (ev) => {
      const b = ev.currentTarget;
      b.disabled = true;
      try {
        await api("POST", "/api/jobs/watch", { replay: r.path });
        b.textContent = "Opening";
      } catch (err) {
        b.textContent = "Failed";
        b.title = String(err.message || err);
      }
      setTimeout(() => { b.disabled = false; b.textContent = "Watch"; b.title = ""; }, 4000);
    } }, "Watch");
    const pick = el("input", { type: "checkbox", "aria-label": `Select ${r.file}` });
    pick.checked = picked.has(r.path);
    pick.addEventListener("change", () => {
      if (pick.checked) picked.add(r.path); else picked.delete(r.path);
      row.classList.toggle("picked", pick.checked);
      updatePickControls();
    });
    const remove = el("button", { type: "button", class: "small ghost icon-only danger", title: "Delete replay",
      "aria-label": `Delete ${r.file}`, onclick: () => deleteReplays([r.path]) }, trashIcon());
    const row = el("tr", { class: picked.has(r.path) ? "picked" : "" },
      el("td", { class: "pick" }, pick),
      el("td", { class: "file" }, r.file.replace(/\.json$/, "")),
      el("td", {}, r.run),
      el("td", {}, r.outcome ? el("span", { class: `badge ${r.outcome === "Survived" ? "good" : ""}` }, r.outcome) : ""),
      el("td", { class: "hint" }, ago(r.modified)),
      el("td", { class: "row-actions" }, watch, remove));
    return row;
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
  // A pasted link picks its own server.
  $("lobby-id").addEventListener("input", (ev) => {
    if (/openfront\.io/.test(ev.target.value)) $("lobby-host").value = "openfront";
    else if (/localhost|127\.0\.0\.1/.test(ev.target.value)) $("lobby-host").value = "local";
  });
  submitter("match-form", $("match-error"), () => ["/api/jobs/match", {
    model: $("match-model").value,
    mode: $("match-mode").value,
    minutes: Number($("match-minutes").value),
  }]);
  $("public-find").addEventListener("click", findPublic);
  $("public-next").addEventListener("click", (ev) => joinPublic({ public: $("public-kind").value }, ev.currentTarget));
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
  $("replay-refresh").addEventListener("click", () => loadReplays().catch(() => {}));
  $("replay-filter").addEventListener("input", renderReplays);
  $("replay-delete").prepend(trashIcon());
  $("replay-delete").addEventListener("click", () => deleteReplays([...picked]));
  $("replay-all").addEventListener("change", (ev) => {
    for (const p of shownPaths) if (ev.target.checked) picked.add(p); else picked.delete(p);
    renderReplays();
  });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (!$("training").hidden) loadMetrics(); }, 200);
  });

  await Promise.all([loadCatalog().catch(() => {}), refreshState()]);
  showTab(location.hash.slice(1));
  setInterval(refreshState, 2000);
  setInterval(() => loadCatalog().catch(() => {}), 30000);
}

boot();
