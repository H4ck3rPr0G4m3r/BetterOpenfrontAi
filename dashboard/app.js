"use strict";

const INTENT_NAMES = [
  "noop", "attack", "boat", "build_unit", "upgrade_structure",
  "move_warship", "target_player", "alliance_request", "alliance_reject",
  "break_alliance", "embargo", "donate_gold", "donate_troops", "cancel_attack",
];
const BUILD_TYPES = [
  "City", "Defense Post", "Port", "SAM Launcher",
  "Missile Silo", "Factory", "Warship",
];

const $ = (id) => document.getElementById(id);

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

/* ---------------- behavior form ---------------- */

function aggressionLabel(v) {
  return v <= 33 ? "turtle" : v <= 66 ? "balanced" : "berserker";
}

function ratioFloorFor(v) {
  return v <= 33 ? 0 : v <= 66 ? 2 : 4;
}

function readBehavior() {
  const disabledIntents = [];
  document.querySelectorAll("#intent-toggles input").forEach((box) => {
    if (!box.checked) disabledIntents.push(Number(box.value));
  });
  const disabledBuilds = [];
  document.querySelectorAll("#build-toggles input").forEach((box) => {
    if (!box.checked) disabledBuilds.push(box.value);
  });
  const rawWeights = $("reward-weights").value.trim();
  return {
    name: $("preset-name").value.trim() || "custom",
    aggression: Number($("aggression").value),
    min_attack_ratio: Number($("min-ratio").value),
    disabled_intents: disabledIntents,
    disabled_builds: disabledBuilds,
    reward_weights: rawWeights ? rawWeights.split(",").map(Number) : null,
    env_overrides: {},
    ppo_overrides: {},
  };
}

function writeBehavior(b) {
  $("aggression").value = b.aggression ?? 50;
  $("min-ratio").value = b.min_attack_ratio ?? ratioFloorFor(b.aggression ?? 50);
  $("reward-weights").value = b.reward_weights ? b.reward_weights.join(",") : "";
  const offIntents = new Set(b.disabled_intents || []);
  document.querySelectorAll("#intent-toggles input").forEach((box) => {
    box.checked = !offIntents.has(Number(box.value));
    box.closest("label").classList.toggle("off", !box.checked);
  });
  const offBuilds = new Set(b.disabled_builds || b.disabled_build_names || []);
  document.querySelectorAll("#build-toggles input").forEach((box) => {
    box.checked = !offBuilds.has(box.value);
    box.closest("label").classList.toggle("off", !box.checked);
  });
  refreshAggression();
}

function refreshAggression() {
  const v = Number($("aggression").value);
  $("aggro-label").textContent = aggressionLabel(v);
  const floor = ratioFloorFor(v);
  $("aggro-map").textContent =
    `attack ratio floor ${[5, 10, 20, 30, 50, 70, 90, 100][floor]}% (bucket ${floor})`;
}

function buildToggles() {
  const intentBox = $("intent-toggles");
  INTENT_NAMES.forEach((name, i) => {
    if (i === 0) return;
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.value = String(i);
    box.addEventListener("change", () => label.classList.toggle("off", !box.checked));
    label.append(box, document.createTextNode(name));
    intentBox.append(label);
  });
  const buildBox = $("build-toggles");
  BUILD_TYPES.forEach((name) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.value = name;
    box.addEventListener("change", () => label.classList.toggle("off", !box.checked));
    label.append(box, document.createTextNode(name));
    buildBox.append(label);
  });
}

/* ---------------- presets ---------------- */

let presets = [];
let presetsInitialized = false;

function renderPresets() {
  const sel = $("preset-select");
  const previous = sel.value;
  sel.innerHTML = "";
  presets.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.name || p.file;
    opt.textContent = p.error ? `${p.file} (invalid)` : p.name;
    sel.append(opt);
  });
  // Polls refresh the list but must never touch the form: keep the
  // user's selection when it still exists, otherwise fall back to first.
  if (previous && [...sel.options].some((o) => o.value === previous)) {
    sel.value = previous;
  }
  showPresetDesc();
  if (!presetsInitialized) {
    presetsInitialized = true;
    const current = presets.find((p) => !p.error);
    if (current) {
      sel.value = current.name || current.file;
      writeBehavior(current);
      showPresetDesc();
    }
  }
}

function showPresetDesc() {
  const p = presets.find((x) => (x.name || x.file) === $("preset-select").value);
  $("preset-desc").textContent = p && !p.error ? p.description || "" : "";
}

/* ---------------- jobs ---------------- */

let jobs = [];
let logSource = null;

function renderJobs() {
  const body = $("jobs").querySelector("tbody");
  body.innerHTML = "";
  jobs.forEach((job) => {
    const tr = document.createElement("tr");
    const cells = [job.id, job.kind, job.status, job.started || ""];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    });
    const watchTd = document.createElement("td");
    const watch = document.createElement("button");
    watch.textContent = "Watch";
    watch.addEventListener("click", () => watchJob(job.id));
    watchTd.append(watch);
    tr.append(watchTd);
    const stopTd = document.createElement("td");
    if (job.status === "running") {
      const stop = document.createElement("button");
      stop.textContent = "Stop";
      stop.addEventListener("click", async () => {
        await api("DELETE", `/api/jobs/${job.id}`);
        await refreshState();
      });
      stopTd.append(stop);
    }
    tr.append(stopTd);
    body.append(tr);
  });
}

function watchJob(id) {
  if (logSource) logSource.close();
  $("log-job").textContent = id;
  const view = $("log-view");
  view.textContent = "";
  const push = (line) => {
    view.textContent += line + "\n";
    const lines = view.textContent.split("\n");
    if (lines.length > 600) view.textContent = lines.slice(-600).join("\n");
    view.scrollTop = view.scrollHeight;
  };
  logSource = new EventSource(`/api/jobs/${id}/logs?follow=1`);
  logSource.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.line !== undefined) push(msg.line);
    if (msg.done) {
      push(`--- job ${msg.status} ---`);
      logSource.close();
      refreshState();
    }
  };
  logSource.onerror = () => logSource.close();
}

/* ---------------- metrics ---------------- */

function drawLineChart(container, title, points) {
  const head = document.createElement("div");
  head.className = "chart-title";
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  head.textContent = `${title} (min ${min.toFixed(3)}, max ${max.toFixed(3)}, n ${points.length})`;
  const canvas = document.createElement("canvas");
  canvas.className = "chart";
  canvas.width = 900;
  canvas.height = 160;
  container.append(head, canvas);
  const ctx = canvas.getContext("2d");
  const span = max - min || 1;
  ctx.strokeStyle = "#7fd08a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / Math.max(1, points.length - 1)) * (canvas.width - 8) + 4;
    const y = canvas.height - 8 - ((p.value - min) / span) * (canvas.height - 16);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawBars(container, title, entries) {
  const head = document.createElement("div");
  head.className = "chart-title";
  head.textContent = title;
  container.append(head);
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  entries.forEach(([name, value]) => {
    const row = document.createElement("div");
    row.className = "bar";
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${Math.max(2, (value / total) * 300)}px`;
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = value.toFixed(3);
    row.append(label, fill, n);
    container.append(row);
  });
}

async function loadMetrics() {
  const box = $("charts");
  box.innerHTML = "";
  const data = await api(
    "GET",
    `/api/metrics?tb_dir=${encodeURIComponent($("metrics-dir").value)}` +
      `&run=${encodeURIComponent($("metrics-run").value)}`
  );
  if (data.note) {
    box.textContent = data.note;
    return;
  }
  const scalars = data.scalars || {};
  const watched = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "time/steps_per_second",
  ];
  watched.forEach((tag) => {
    if (scalars[tag]) drawLineChart(box, tag, scalars[tag]);
  });
  Object.keys(scalars)
    .filter((t) => !watched.includes(t) && !t.startsWith("terminals/"))
    .slice(0, 6)
    .forEach((tag) => drawLineChart(box, tag, scalars[tag]));
  const terms = Object.keys(scalars)
    .filter((t) => t.startsWith("terminals/"))
    .map((t) => [t.replace("terminals/", ""), scalars[t][scalars[t].length - 1].value]);
  if (terms.length) drawBars(box, "terminal mix (latest)", terms);
}

/* ---------------- eval ---------------- */

async function refreshCheckpoints() {
  const data = await api(
    "GET",
    `/api/checkpoints?dir=${encodeURIComponent($("ckpt-dir").value)}`
  );
  const sel = $("eval-model");
  sel.innerHTML = "";
  data.checkpoints.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.path} (${(c.size / 1e6).toFixed(1)}MB)`;
    sel.append(opt);
  });
  const liveModel = $("live-model");
  if (!liveModel.value && data.checkpoints.length) {
    liveModel.placeholder = data.checkpoints[data.checkpoints.length - 1].path;
  }
}

function parseEvalResult(lines) {
  const rows = [];
  lines.forEach((line) => {
    const m = line.match(/^(policy|baseline): return ([^,]+), length ([^,]+), win rate ([^,]+), terminals (.*)$/);
    if (m) rows.push({ name: m[1], ret: m[2], len: m[3], win: m[4], terms: m[5] });
  });
  return rows;
}

/* ---------------- live ---------------- */

let liveSource = null;

function renderLiveStep(step) {
  $("live-action").textContent = step.action_text;
  $("live-meta").textContent =
    `step ${step.n} · ep ${step.episode} · reward ${step.reward.toFixed(3)} · ` +
    `total ${step.total_reward.toFixed(3)} · ticks ${step.ticks}` +
    (step.terminal ? ` · ended: ${step.terminal}` : "");
  const known = window._liveCounts || (window._liveCounts = {});
  known[step.action_name] = (known[step.action_name] || 0) + 1;
  renderLiveCounts(known);
  const masks = step.masks || {};
  $("live-masks").innerHTML = Object.entries(masks)
    .map(([k, v]) => `${k}: ${v.open}/${v.total}`)
    .join("<br>");
  const tbody = $("live-steps").querySelector("tbody");
  const tr = document.createElement("tr");
  [step.n, step.action_text, step.reward.toFixed(3), step.total_reward.toFixed(2),
   step.ticks, step.terminal || ""].forEach((text) => {
    const td = document.createElement("td");
    td.textContent = text;
    tr.append(td);
  });
  tbody.prepend(tr);
  while (tbody.rows.length > 30) tbody.deleteRow(-1);
}

function renderLiveCounts(counts) {
  const box = $("live-intents");
  box.innerHTML = "";
  drawBars(box, "", Object.entries(counts).sort((a, b) => b[1] - a[1]));
}

function renderLiveHello(msg) {
  window._liveCounts = {};
  (msg.recent || []).forEach((s) => {
    window._liveCounts[s.action_name] = (window._liveCounts[s.action_name] || 0) + 1;
  });
  renderLiveCounts(window._liveCounts);
  const tbody = $("live-steps").querySelector("tbody");
  tbody.innerHTML = "";
  (msg.recent || []).slice().reverse().forEach((s) => {
    const tr = document.createElement("tr");
    [s.n, s.action_text, s.reward.toFixed(3), s.total_reward.toFixed(2),
     s.ticks, s.terminal || ""].forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    });
    tbody.append(tr);
  });
  if (msg.latest) {
    $("live-action").textContent = msg.latest.action_text;
    $("live-meta").textContent =
      `step ${msg.latest.n} · ep ${msg.latest.episode} · ticks ${msg.latest.ticks}`;
  }
  if (msg.error) {
    const err = $("live-error");
    err.hidden = false;
    err.textContent = msg.error;
  }
}

function connectLiveStream() {
  if (liveSource) liveSource.close();
  if (!window._phasePoll) window._phasePoll = setInterval(refreshLivePhase, 3000);
  liveSource = new EventSource("/api/live/stream");
  liveSource.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.hello) {
      renderLiveHello(msg);
    }
    if (msg.step) {
      $("live-error").hidden = true;
      renderLiveStep(msg.step);
    }
  };
  liveSource.onerror = () => {
    liveSource.close();
    setTimeout(connectLiveStream, 3000);
  };
}

async function refreshLivePhase() {
  try {
    const st = await api("GET", "/api/live/state");
    const phase = $("live-phase");
    if (st.running || st.phase === "booting" || st.phase === "failed") {
      phase.textContent =
        `phase: ${st.phase}${st.phase_detail ? " — " + st.phase_detail : ""}` +
        ` · steps: ${st.step_count} · episodes: ${st.episodes}`;
    } else if (!st.latest) {
      phase.textContent = "idle";
    }
    if (st.error) {
      const box = $("live-error");
      box.hidden = false;
      box.textContent = st.error + (st.error_detail ? "\n" + st.error_detail : "");
    }
  } catch (err) {
    /* server polling is best effort; the stream carries the steps */
  }
}

/* ---------------- boot ---------------- */

async function refreshState() {
  const data = await api("GET", "/api/state");
  jobs = data.jobs || [];
  presets = data.presets || [];
  renderJobs();
  renderPresets();
  const status = $("server-status");
  status.textContent = `server ok · ${jobs.length} jobs · ${presets.length} presets`;
  status.className = "status ok";
}

async function boot() {
  buildToggles();
  refreshAggression();
  $("aggression").addEventListener("input", () => {
    refreshAggression();
    $("min-ratio").value = String(ratioFloorFor(Number($("aggression").value)));
  });
  $("preset-select").addEventListener("change", showPresetDesc);
  $("preset-apply").addEventListener("click", () => {
    const p = presets.find((x) => (x.name || x.file) === $("preset-select").value);
    if (p && !p.error) {
      writeBehavior(p);
      $("preset-name").value = p.name;
    }
  });
  $("preset-save").addEventListener("click", async () => {
    const saved = await api("POST", "/api/presets", readBehavior());
    await refreshState();
    $("preset-select").value = saved.name;
  });
  $("preset-delete").addEventListener("click", async () => {
    await api("DELETE", `/api/presets/${encodeURIComponent($("preset-select").value)}`);
    await refreshState();
  });
  $("train-start").addEventListener("click", async () => {
    const job = await api("POST", "/api/jobs/train", {
      base: $("train-base").value,
      run_name: $("train-name").value,
      total_timesteps: Number($("train-steps").value),
      behavior: readBehavior(),
      config_override: {
        env: {
          bots: Number($("train-bots").value),
          num_workers: Number($("train-workers").value),
          envs_per_worker: Number($("train-envs").value),
        },
        ppo: {
          learning_rate: Number($("ppo-lr").value),
          ent_coef: Number($("ppo-ent").value),
        },
        run: { device: $("train-device").value },
      },
    });
    await refreshState();
    watchJob(job.id);
  });
  $("metrics-load").addEventListener("click", loadMetrics);
  $("ckpt-refresh").addEventListener("click", refreshCheckpoints);
  $("eval-start").addEventListener("click", async () => {
    const job = await api("POST", "/api/jobs/eval", {
      model: $("eval-model").value,
      base: "default.yaml",
      episodes: Number($("eval-episodes").value),
      seed: $("eval-seed").value,
      baseline: $("eval-baseline").checked,
      behavior: readBehavior(),
    });
    $("eval-result").textContent = `eval started as ${job.id}, watching log…`;
    await refreshState();
    watchJob(job.id);
    const poll = setInterval(async () => {
      const detail = await api("GET", `/api/jobs/${job.id}`);
      const rows = parseEvalResult(detail.log_tail || []);
      if (rows.length) {
        $("eval-result").innerHTML = rows
          .map((r) => `${r.name}: return ${r.ret}, length ${r.len}, win ${r.win}, ${r.terms}`)
          .join("<br>");
      }
      if (detail.status !== "running") clearInterval(poll);
    }, 3000);
  });
  $("live-start").addEventListener("click", async () => {
    window._liveCounts = {};
    $("live-steps").querySelector("tbody").innerHTML = "";
    try {
      const modelInput = $("live-model").value.trim();
      await api("POST", "/api/live/start", {
        predictor: $("live-predictor").value,
        model: modelInput || $("live-model").placeholder || null,
        map: $("live-map").value,
        bots: Number($("live-bots").value),
        seed: $("live-seed").value,
        behavior: readBehavior(),
      });
      $("live-error").hidden = true;
    } catch (err) {
      const box = $("live-error");
      box.hidden = false;
      box.textContent = String(err.message || err);
    }
  });
  $("live-stop").addEventListener("click", async () => {
    await api("POST", "/api/live/stop", {});
  });
  try {
    await refreshState();
  } catch (err) {
    const status = $("server-status");
    status.textContent = `server unreachable: ${err.message}`;
    status.className = "status bad";
  }
  try {
    await refreshCheckpoints();
  } catch (err) {
    /* checkpoints dir may not exist yet */
  }
  connectLiveStream();
  setInterval(refreshState, 10000);
}

document.addEventListener("DOMContentLoaded", boot);
