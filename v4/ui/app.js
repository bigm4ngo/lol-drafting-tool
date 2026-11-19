"use strict";
const EMPTY = -1, SKIPPED = -2, SIDES = ["blue", "red"];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + "%";
const pts = (x) => (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + " pts";

let meta, byId = new Map(), byName = new Map();
let state, history = [], selected = null, roleFilter = null, rev = 0, liveTimer = null, lastLive = "";
let pools = {};

const store = {
  get(k, d) { try { const v = localStorage.getItem("dl4." + k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("dl4." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

function blankState(fmt, unavailable = [], first = "blue") {
  return { fmt, first: fmt === "ranked" ? "blue" : first, bans: [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], picks: [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
           roles: [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]], unavailable };
}
// Imported files and saved browser data are untrusted: keep only known roles and numeric champion ids.
const cleanId = (x) => (byId.has(+x) || +x === EMPTY || +x === SKIPPED ? +x : EMPTY);
function cleanPools(raw) {
  return Object.fromEntries(Object.entries(raw || {}).filter(([r]) => meta.roles.includes(r))
    .map(([r, ids]) => [r, (Array.isArray(ids) ? ids : []).map(Number).filter((x) => byId.has(x))]));
}
function cleanState(raw) {
  if (!raw || !meta.formats[raw.fmt]) return null;
  const rows = (v, f) => [0, 1].map((i) => Array.from({ length: 5 }, (_, j) => f((v || [])[i]?.[j] ?? EMPTY)));
  return { fmt: raw.fmt, first: raw.first === "red" && raw.fmt !== "ranked" ? "red" : "blue", bans: rows(raw.bans, cleanId), picks: rows(raw.picks, (x) => (cleanId(x) === SKIPPED ? EMPTY : cleanId(x))),
           roles: rows(raw.roles, (x) => (Number.isInteger(+x) && +x >= 0 && +x < 5 ? +x : EMPTY)),
           unavailable: (Array.isArray(raw.unavailable) ? raw.unavailable : []).map(Number).filter((x) => byId.has(x)) };
}
const steps = () => meta.formats[state.fmt][state.first === "red" ? "steps_red_first" : "steps"];
const slotVal = (s) => state[s.kind === "ban" ? "bans" : "picks"][SIDES.indexOf(s.side)][s.slot];
const nextStep = () => steps().find((s) => slotVal(s) === EMPTY) || null;
const target = () => selected || nextStep();
const same = (a, b) => a && b && a.side === b.side && a.kind === b.kind && a.slot === b.slot;

function legal(champ, tgt) {
  if (!tgt) return false;
  const picked = new Set([...state.picks[0], ...state.picks[1]].filter((c) => c >= 0));
  const current = slotVal(tgt);
  if (champ === current) return true;
  if (picked.has(champ) || state.unavailable.includes(champ)) return false;
  const banned = [...state.bans[0], ...state.bans[1]].includes(champ);
  if (!banned) return true;
  // ranked bans are simultaneous: the other team may ban the same champion
  return tgt.kind === "ban" && state.fmt === "ranked" && !state.bans[SIDES.indexOf(tgt.side)].includes(champ);
}

function commit(mutator) {
  history.push(JSON.stringify(state));
  if (history.length > 200) history.shift();
  mutator();
  selected = null;
  store.set("state", state);
  render();
  if ($("auto").checked) analyze();
}

function place(champ) {
  const tgt = target();
  if (!tgt || !legal(champ, tgt)) return;
  commit(() => {
    const side = SIDES.indexOf(tgt.side);
    state[tgt.kind === "ban" ? "bans" : "picks"][side][tgt.slot] = champ;
  });
}

// ------------------------------------------------------------------ rendering
function face(id, cls = "face") {
  const c = byId.get(id);
  if (!c) return `<div class="${cls}"></div>`;
  // initials sit under the portrait and show through if the image can't load (offline)
  return `<div class="${cls}">${esc(c.name.slice(0, 2))}<img src="${esc(c.image)}" alt="" loading="lazy" onerror="this.style.display='none'"></div>`;
}

function renderStrip() {
  const list = steps(), now = nextStep(), tgt = target();
  $("strip").style.setProperty("--n", list.length);
  $("strip").innerHTML = list.map((s, i) => {
    const v = slotVal(s), c = byId.get(v);
    const cls = ["tick", s.side, s.kind, v !== EMPTY ? "filled" : "", same(s, now) ? "now" : "", selected && same(s, tgt) ? "selected" : ""].join(" ");
    const label = `${s.side} ${s.kind} ${s.slot + 1}${c ? ": " + c.name : v === SKIPPED ? ": no ban" : ""}`;
    return `<button class="${cls}" data-i="${i}" title="${esc(label)}" aria-label="${esc(label)}">${c ? `<img src="${esc(c.image)}" alt="">` : i + 1}</button>`;
  }).join("");
}

function renderTeam(side) {
  const sideIdx = SIDES.indexOf(side), tgt = target(), now = nextStep();
  const mark = (s) => (same(s, now) ? " now" : "") + (selected && same(s, tgt) ? " selected" : "");
  const bans = state.bans[sideIdx].map((v, slot) => {
    const s = { side, kind: "ban", slot };
    return `<div class="slot ban${v === SKIPPED ? " skipped" : ""}${mark(s)}" data-side="${side}" data-kind="ban" data-slot="${slot}" tabindex="0"
      title="${esc(byId.get(v)?.name || "Ban " + (slot + 1))}">${v >= 0 ? `<img src="${esc(byId.get(v).image)}" alt="${esc(byId.get(v).name)}">` : ""}</div>`;
  }).join("");
  const picks = state.picks[sideIdx].map((v, slot) => {
    const s = { side, kind: "pick", slot }, c = byId.get(v), role = state.roles[sideIdx][slot];
    const opts = [`<option value="-1">Role: auto</option>`].concat(meta.roles.map((r, i) => `<option value="${i}" ${i === role ? "selected" : ""}>${r[0] + r.slice(1).toLowerCase()}</option>`)).join("");
    return `<div class="slot pick${mark(s)}" data-side="${side}" data-kind="pick" data-slot="${slot}" tabindex="0">
      ${face(v)}<div>${c ? `<div class="name">${esc(c.name)}</div>` : `<div class="empty">Pick ${slot + 1}</div>`}
      <select data-role="${side}:${slot}" aria-label="Role for ${side} pick ${slot + 1}">${opts}</select></div></div>`;
  }).join("");
  const label = $("our-side").value === side ? "<small>advising</small>" : "";
  $("team-" + side).innerHTML = `<h2>${side === "blue" ? "Blue" : "Red"} side ${label}</h2><div class="bans">${bans}</div>${picks}`;
}

function renderGrid() {
  const q = $("search").value.trim().toLowerCase(), tgt = target();
  const poolSet = new Set(Object.values(pools).flat());
  $("grid").innerHTML = meta.champions.filter((c) => (!q || c.name.toLowerCase().includes(q)) && (!roleFilter || c.roles.includes(roleFilter)))
    .map((c) => {
      const ok = legal(c.id, tgt);
      return `<button class="champ${poolSet.has(c.id) ? " pool" : ""}" data-id="${c.id}" ${ok ? "" : "disabled"} title="${esc(c.name)} — ${esc(c.roles.join("/"))}${c.traits.length ? "; " + esc(c.traits.join(", ")) : ""}">
        ${face(c.id)}<span>${esc(c.name)}</span></button>`;
    }).join("");
  $("skip-ban").disabled = !tgt || tgt.kind !== "ban";
}

function renderPools() {
  $("pools").innerHTML = meta.roles.map((r) => `<div class="pool-row"><span>${r[0] + r.slice(1).toLowerCase()}</span><div>
    <input type="text" list="champ-names" data-pool="${r}" placeholder="Add champion" aria-label="Add ${r} pool champion">
    <div class="pool-chips">${(pools[r] || []).map((id) => `<button data-unpool="${esc(r)}:${+id}" title="Remove">${esc(byId.get(id)?.name || id)} ×</button>`).join("")}</div></div></div>`).join("");
}

function render() {
  const fearless = meta.formats[state.fmt].fearless;
  $("fmt").value = state.fmt;
  $("first").value = state.first;
  $("first-wrap").hidden = state.fmt === "ranked";
  $("next-game").hidden = !fearless;
  $("fearless-panel").hidden = !fearless;
  $("unavailable").innerHTML = state.unavailable.length ? state.unavailable.map((id) => `<span>${esc(byId.get(id)?.name || id)}</span>`).join("") : `<span>None yet</span>`;
  renderStrip(); renderTeam("blue"); renderTeam("red"); renderGrid();
  const now = nextStep();
  if (!$("status").classList.contains("error")) $("status").textContent = now ? `${cap(now.side)} ${now.kind} ${now.slot + 1}${selected ? " (editing a different slot)" : ""}` : "Draft complete";
}
const cap = (s) => s[0].toUpperCase() + s.slice(1);

// ------------------------------------------------------------------ analysis
async function analyze() {
  const my = ++rev;
  $("analyze").disabled = true;
  setStatus("Analysing…");
  const prefs = { our_side: $("our-side").value, plan: $("plan").value || null, pools, pool_only: $("pool-only").checked,
                  budget_ms: +$("budget").value };
  try {
    const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ state, prefs, rev: my }) });
    const res = await r.json();
    if (my !== rev) return; // a newer state was analysed meanwhile
    if (!r.ok) throw new Error(res.error || r.statusText);
    renderAnalysis(res);
    setStatus(`Analysed in ${res.elapsed_ms} ms`);
  } catch (e) {
    if (my === rev) setStatus("Analysis failed: " + e.message, true);
  } finally {
    if (my === rev) $("analyze").disabled = false;
  }
}

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function champLink(x) { return x ? esc(x.name) : "—"; }

function lineCard(l, first) {
  const c = byId.get(l.champion);
  const p = l.punish;
  const rows = [
    l.likely_reply && ["Likely reply", `${esc(l.likely_reply.kind)} ${champLink(l.likely_reply)} <small>(${Math.round(l.likely_reply.frequency * 100)}% of branches)</small>`],
    p && ["Credible punish", `${esc(p.kind)} ${champLink(p)} <span class="delta-neg">${pts(p.delta)}</span>${p.pivot_to ? ` → pivot to ${champLink(p.pivot_to)}` : ""}`],
    l.follow_up && ["Follow-up", champLink(l.follow_up)],
    l.preserves && l.preserves.flex_roles.length && ["Flex", l.preserves.flex_roles.map((r) => r.toLowerCase()).join(" / ")],
    l.substitute && ["Substitute", `${champLink(l.substitute)} (${l.substitute.role.toLowerCase()})`],
    l.delta_vs_typical_ban !== undefined && ["vs typical ban", `<span class="${l.delta_vs_typical_ban < 0 ? "delta-neg" : "delta-pos"}">${pts(l.delta_vs_typical_ban)}</span>`],
    ["Plan", `${esc(l.plan.label)} — ${Math.round(l.plan.fit * 100)}% of requirements met`],
    l.plan.often_unmet.length && ["Often missing", esc(l.plan.often_unmet.join("; "))],
    l.plan.risks.length && ["Watch", esc(l.plan.risks.join("; "))],
  ].filter(Boolean);
  return `<article class="line${first ? " first" : ""}">
    <header>${c ? `<img src="${esc(c.image)}" alt="">` : ""}<div><div class="title">${l.action === "ban" ? "Ban" : "Pick"} ${esc(l.name)}</div>
    <div class="role">${l.role ? esc(l.role.toLowerCase()) : ""}${l.close_to_best && !first ? " · within noise of the top line" : ""}</div></div>
    <span class="grow"></span><button data-apply="${l.champion}" title="Place this in the current slot">Use</button></header>
    <div class="tags">${l.reasons.map((r) => `<span>${esc(r)}</span>`).join("")}</div>
    <div class="nums"><div><b>${pct(l.win)}</b><small>average outcome</small></div><div><b>${pct(l.downside)}</b><small>bad-branch outcome</small></div>
      <div><b>${Math.round(l.coverage * 100)}%</b><small>data coverage</small></div></div>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
    <p class="why">${esc(l.explain || "")}</p></article>`;
}

function renderAnalysis(res) {
  const el = $("analysis");
  const notes = (res.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
  const ver = `Policy ${esc(res.versions.policy)}, value ${esc(res.versions.value)}, ${res.versions.matches} matches`;
  if (res.kind === "recommend") {
    el.innerHTML = `<h3>${cap(res.our_side)} ${esc(res.step.kind)} ${res.step.slot + 1}: ${res.lines.length} lines worth considering</h3>
      <p class="sub">${res.candidates} candidates searched, ${res.rollouts_per_line} simulated drafts each. ${ver}.</p>
      <div class="lines">${res.lines.map((l, i) => lineCard(l, i === 0)).join("")}</div>
      ${res.also.length ? `<div class="also">Also checked: ${res.also.map((a) => `<button data-apply="${a.champion}">${esc(a.name)} ${pct(a.win)}</button>`).join("")}</div>` : ""}
      ${notes ? `<ul class="notes">${notes}</ul>` : ""}`;
  } else if (res.kind === "forecast") {
    const max = Math.max(...res.forecast.map((f) => f.probability), 0.01);
    el.innerHTML = `<h3>${cap(res.step.side)} to ${esc(res.step.kind)}: what they might do</h3>
      <p class="sub">${esc(res.message)} ${ver}.</p>
      <table class="fc"><tr><th>Their option</th><th>Likelihood</th><th>Your outlook if they do</th><th></th></tr>
      ${res.forecast.map((f) => `<tr><td>${esc(f.name)}</td><td class="bar-cell"><i style="width:${(f.probability / max) * 100}%"></i><b>${pct(f.probability)}</b></td>
        <td>${pct(f.our_win_if)}${res.biggest_threat && f.champion === res.biggest_threat.champion ? ' <span class="warn">biggest threat</span>' : ""}</td>
        <td><button data-apply="${f.champion}">Place</button></td></tr>`).join("")}</table>`;
  } else if (res.kind === "complete") {
    const team = (side) => {
      const t = res.teams[side];
      return `<div class="team-card"><h3>${cap(side)}: ${pct(side === "blue" ? res.win_blue : 1 - res.win_blue)}</h3>
        <p class="sub">${t.lineup.map((x) => `${esc(x.name)} (${x.role.toLowerCase()})`).join(", ")}</p>
        ${t.plans.map((p) => `<p><b>${esc(p.label)}</b> ${Math.round(p.fit * 100)}%${p.unmet.length ? ` — missing: ${esc(p.unmet.join("; "))}` : ""}${p.risks.length ? `<br><small class="warn">Watch: ${esc(p.risks.join("; "))}</small>` : ""}</p>`).join("")}
        ${t.role_uncertainty.length ? `<p class="sub">Role uncertainty: ${t.role_uncertainty.map((u) => `${esc(u.champion)} ${Object.entries(u.roles).map(([r, p]) => `${r.toLowerCase()} ${Math.round(p * 100)}%`).join("/")}`).join("; ")}</p>` : ""}</div>`;
    };
    el.innerHTML = `<h3>Final draft</h3><p class="sub">Estimated outcome from the value model (historical association, not a guarantee). Data coverage ${Math.round(res.coverage * 100)}%. ${ver}.</p>
      <div class="final">${team("blue")}${team("red")}</div>`;
  } else {
    el.innerHTML = `<h3>No recommendation</h3><p class="sub">${esc(res.message || "")}</p>`;
  }
}

// ------------------------------------------------------------------ live client
function setMode(live) {
  $("mode-manual").setAttribute("aria-selected", String(!live));
  $("mode-live").setAttribute("aria-selected", String(live));
  clearInterval(liveTimer);
  if (!live) { setStatus("Manual mode"); return; }
  const poll = async () => {
    try {
      const res = await (await fetch("/api/live")).json();
      if (!res.available) { setStatus(res.message, true); return; }
      const key = JSON.stringify(res.state);
      if (key === lastLive) return;
      lastLive = key;
      state = res.state;
      $("our-side").value = res.our_side;
      setStatus("Live: synced from the League client");
      render();
      analyze();
    } catch (e) { setStatus("Live polling failed: " + e.message, true); }
  };
  poll();
  liveTimer = setInterval(poll, 1500);
}

// ------------------------------------------------------------------ events
function slotFromEl(el) {
  const s = el.closest("[data-kind]");
  return s ? { side: s.dataset.side, kind: s.dataset.kind, slot: +s.dataset.slot } : null;
}

function bind() {
  $("grid").addEventListener("click", (e) => { const b = e.target.closest(".champ"); if (b && !b.disabled) place(+b.dataset.id); });
  for (const side of SIDES) {
    $("team-" + side).addEventListener("click", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "OPTION") return;
      const s = slotFromEl(e.target);
      if (s) { selected = same(s, selected) ? null : s; render(); }
    });
    $("team-" + side).addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.target.click(); } });
    $("team-" + side).addEventListener("change", (e) => {
      const key = e.target.dataset.role;
      if (!key) return;
      const [sd, slot] = key.split(":");
      commit(() => { state.roles[SIDES.indexOf(sd)][+slot] = +e.target.value; });
    });
  }
  $("strip").addEventListener("click", (e) => {
    const b = e.target.closest(".tick"); if (!b) return;
    const s = steps()[+b.dataset.i];
    selected = same(s, selected) ? null : s; render();
  });
  $("analysis").addEventListener("click", (e) => { const b = e.target.closest("[data-apply]"); if (b) place(+b.dataset.apply); });
  $("search").addEventListener("input", renderGrid);
  $("search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { const b = $("grid").querySelector(".champ:not(:disabled)"); if (b) { place(+b.dataset.id); $("search").value = ""; renderGrid(); } }
  });
  $("role-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    roleFilter = roleFilter === b.dataset.role ? null : b.dataset.role;
    for (const x of $("role-filter").children) x.setAttribute("aria-pressed", String(x.dataset.role === roleFilter));
    renderGrid();
  });
  $("skip-ban").addEventListener("click", () => {
    const tgt = target(); if (!tgt || tgt.kind !== "ban") return;
    commit(() => { state.bans[SIDES.indexOf(tgt.side)][tgt.slot] = SKIPPED; });
  });
  $("undo").addEventListener("click", () => { if (!history.length) return; state = JSON.parse(history.pop()); selected = null; store.set("state", state); render(); });
  $("clear-slot").addEventListener("click", () => {
    const tgt = selected; if (!tgt) { setStatus("Select a filled slot first", true); return; }
    commit(() => {
      const side = SIDES.indexOf(tgt.side);
      state[tgt.kind === "ban" ? "bans" : "picks"][side][tgt.slot] = EMPTY;
      if (tgt.kind === "pick") state.roles[side][tgt.slot] = EMPTY;
    });
  });
  $("swap").addEventListener("click", () => commit(() => { for (const k of ["bans", "picks", "roles"]) state[k].reverse(); }));
  $("reset").addEventListener("click", () => commit(() => { state = blankState(state.fmt, state.unavailable, state.first); }));
  $("first").addEventListener("change", () => commit(() => { state.first = $("first").value; }));
  $("next-game").addEventListener("click", () => commit(() => {
    const used = [...state.picks[0], ...state.picks[1]].filter((c) => c >= 0);
    state = blankState(state.fmt, [...new Set([...state.unavailable, ...used])], state.first);
  }));
  $("fmt").addEventListener("change", () => commit(() => {
    const keep = meta.formats[$("fmt").value].fearless ? state.unavailable : [];
    state = { ...blankState($("fmt").value, keep, state.first), bans: state.bans, picks: state.picks, roles: state.roles };
  }));
  for (const id of ["our-side", "plan", "budget", "pool-only"]) $(id).addEventListener("change", () => { store.set("prefs", prefsSnapshot()); render(); analyze(); });
  $("analyze").addEventListener("click", analyze);
  $("export").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify({ state, pools, prefs: prefsSnapshot() }, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `draft-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json` });
    a.click(); URL.revokeObjectURL(a.href);
  });
  $("import").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      const imported = cleanState(data.state);
      if (!imported) throw new Error("not a Draft Lab export");
      commit(() => { state = imported; if (data.pools) { pools = cleanPools(data.pools); store.set("pools", pools); renderPools(); } });
    } catch (err) { setStatus("Import failed: " + err.message, true); }
    e.target.value = "";
  });
  $("pools").addEventListener("change", (e) => {
    const role = e.target.dataset.pool; if (!role) return;
    const c = byName.get(e.target.value.trim().toLowerCase());
    if (!c) { setStatus(`No champion called "${e.target.value}"`, true); return; }
    pools[role] = [...new Set([...(pools[role] || []), c.id])];
    store.set("pools", pools); renderPools(); renderGrid(); setStatus(`Added ${c.name} to ${role.toLowerCase()} pool`);
  });
  $("pools").addEventListener("click", (e) => {
    const b = e.target.closest("[data-unpool]"); if (!b) return;
    const [role, id] = b.dataset.unpool.split(":");
    pools[role] = (pools[role] || []).filter((x) => x !== +id);
    store.set("pools", pools); renderPools(); renderGrid();
  });
  $("mode-manual").addEventListener("click", () => setMode(false));
  $("mode-live").addEventListener("click", () => setMode(true));
  document.addEventListener("keydown", (e) => {
    if (e.key === "z" && (e.ctrlKey || e.metaKey) && document.activeElement.tagName !== "INPUT") { e.preventDefault(); $("undo").click(); }
    if (e.key === "/" && document.activeElement.tagName !== "INPUT") { e.preventDefault(); $("search").focus(); }
  });
}

const prefsSnapshot = () => ({ our_side: $("our-side").value, plan: $("plan").value, budget: $("budget").value, pool_only: $("pool-only").checked });

function renderStatus() {
  const s = meta.status, g = s.gpu;
  $("engine-status").innerHTML = `<p>Draft policy: <b>${esc(s.versions.policy)}</b><br>Composition value: <b>${esc(s.versions.value)}</b><br>
    Evidence: <b>${s.versions.matches}</b> matches (${esc(s.versions.priors)})<br>Champion data: <b>${esc(s.versions.catalog)}</b><br>
    Training GPU: <b>${g.cuda ? esc(g.device) : "not available"}</b>${g.torch ? ` (torch ${esc(g.torch)})` : ""}</p>
    ${(s.warnings || []).map((w) => `<p class="warn">${esc(w)}</p>`).join("")}`;
}

async function init() {
  try {
    meta = await (await fetch("/api/meta")).json();
  } catch (e) {
    document.body.innerHTML = `<p style="padding:24px">Can't reach the Draft Lab engine. Start it with <code>python -m draft4.server</code> and reload.</p>`;
    return;
  }
  for (const c of meta.champions) { byId.set(c.id, c); byName.set(c.name.toLowerCase(), c); byName.set(c.key.toLowerCase(), c); }
  $("fmt").innerHTML = Object.entries(meta.formats).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
  $("plan").innerHTML += Object.entries(meta.plans).map(([k, v]) => `<option value="${k}" title="${esc(v.win)}">${esc(v.label)}</option>`).join("");
  $("role-filter").innerHTML = meta.roles.map((r) => `<button data-role="${r}" aria-pressed="false">${r[0] + r.slice(1).toLowerCase()}</button>`).join("");
  $("champ-names").innerHTML = meta.champions.map((c) => `<option value="${esc(c.name)}">`).join("");
  state = cleanState(store.get("state", null)) || blankState("pro");
  pools = cleanPools(store.get("pools", {}));
  const p = store.get("prefs", null);
  if (p) { $("our-side").value = p.our_side; $("plan").value = p.plan; $("budget").value = p.budget; $("pool-only").checked = p.pool_only; }
  bind(); renderPools(); renderStatus(); render();
  if ($("auto").checked) analyze();
}
init();
