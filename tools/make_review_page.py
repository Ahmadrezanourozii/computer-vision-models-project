"""Build the 'Mooney Review' web page (blind face check + manual top-10 pick).

    python tools/make_review_page.py --data data -o work/review/mooney_review.html

Images are embedded as data URIs; answers are stored in the page's `db` (doc review/state).
"""
import argparse
import base64
import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image


def data_uri(path, size, fmt, quality=82):
    im = Image.open(path)
    im = im.convert("L") if fmt == "PNG" else im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "PNG":
        im.save(buf, "PNG", optimize=True)
    else:
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = r"""<title>Mooney Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Archivo:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap">
<style>
/* Layout: one working column; a focused single-image flow for the blind check, a dense grid for the pick */
:root{
  --paper:#fbfbf9; --ink:#121314; --muted:#5f6670; --line:#dfe1e3; --tile:#f0f1f2;
  --amber:#c27a12; --amber-soft:#fff1d9; --good:#1f7a4d; --bad:#b23a2e;
  --display:"Bricolage Grotesque",Georgia,serif; --body:"Archivo",system-ui,sans-serif; --mono:"JetBrains Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#121314;--ink:#eef0f1;--muted:#a3aab2;--line:#2c2f33;--tile:#1c1e21;--amber:#f2b052;--amber-soft:#2e2414;--good:#5cc28d;--bad:#ef7a6c;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#121314;--ink:#eef0f1;--muted:#a3aab2;--line:#2c2f33;--tile:#1c1e21;--amber:#f2b052;--amber-soft:#2e2414;--good:#5cc28d;--bad:#ef7a6c;color-scheme:dark}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{background:var(--paper);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
main{max-width:1180px;margin:0 auto;display:grid;gap:22px}
h1{font:800 clamp(28px,4vw,40px)/1.05 var(--display);margin:0;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:68ch;margin:6px 0 0}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{font:600 14px var(--body);padding:9px 14px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.tab[aria-selected="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.tab:focus-visible,.btn:focus-visible,.card button:focus-visible{outline:3px solid var(--amber);outline-offset:2px}
.status{font:500 13px var(--mono);color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}
.status b{color:var(--ink)}
.sync{font:500 12px var(--mono);color:var(--muted)}
/* blind check */
.blind{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,300px);gap:24px;align-items:start}
.stage{background:#000;border-radius:14px;overflow:hidden;aspect-ratio:1/1;max-width:100%;display:grid;place-items:center}
.stage img{width:100%;height:100%;object-fit:contain;image-rendering:auto}
.panel{display:grid;gap:12px}
.q{font:700 20px/1.25 var(--display)}
.btn{font:600 15px var(--body);padding:13px 16px;border-radius:12px;border:1.5px solid var(--line);background:var(--tile);color:var(--ink);cursor:pointer;text-align:left;display:flex;justify-content:space-between;align-items:center}
.btn kbd{font:500 12px var(--mono);color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.btn.face{border-color:var(--good)} .btn.none{border-color:var(--bad)}
.btn:hover{background:var(--amber-soft)}
.nav{display:flex;gap:8px}
.nav .btn{flex:1;justify-content:center;padding:9px}
.hint{font-size:13px;color:var(--muted)}
.progress{height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.progress i{display:block;height:100%;background:var(--amber);width:0}
/* pick grid */
.filters{display:flex;gap:6px;flex-wrap:wrap}
.chip{font:500 13px var(--body);padding:6px 11px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.chip[aria-pressed="true"]{background:var(--amber-soft);border-color:var(--amber)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.card{border:1.5px solid var(--line);border-radius:12px;overflow:hidden;background:var(--tile);display:grid}
.card.sel{border-color:var(--amber);box-shadow:0 0 0 2px var(--amber) inset}
.pair{display:grid;grid-template-columns:1fr 1fr;background:#000}
.pair img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.meta{display:flex;justify-content:space-between;align-items:center;padding:7px 9px;gap:6px}
.id{font:500 12px var(--mono)}
.tag{font:500 11px var(--mono);color:var(--muted)}
.card button{font:600 13px var(--body);border:none;background:var(--ink);color:var(--paper);border-radius:8px;padding:6px 10px;cursor:pointer}
.card.sel button{background:var(--amber);color:#111}
.blindmark{font:500 11px var(--mono)}
.blindmark.face{color:var(--good)} .blindmark.none{color:var(--bad)} .blindmark.unsure{color:var(--muted)}
.fallback{border:1px dashed var(--line);border-radius:12px;padding:12px;display:grid;gap:8px}
.fallback textarea{width:100%;min-height:90px;font:12px var(--mono);background:var(--tile);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}
@media (max-width:760px){.blind{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<main>
  <header>
    <h1>Mooney Review</h1>
    <p class="lede">Step 1 is blind: you see only the two-tone image and decide whether you see a face. Step 2 reveals the photographs so you can pick the ten best Mooney images for the slides. Answers save automatically.</p>
  </header>
  <div class="bar">
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" id="tab-blind" aria-selected="true">1 · Blind face check</button>
      <button class="tab" role="tab" id="tab-pick" aria-selected="false">2 · Pick top 10</button>
    </div>
    <div class="status"><span>checked <b id="n-blind">0</b>/<span id="n-total"></span></span><span>selected <b id="n-sel">0</b>/10</span><span class="sync" id="sync">connecting…</span></div>
  </div>

  <section id="view-blind" class="blind">
    <div class="stage"><img id="blind-img" alt="Mooney image under review"></div>
    <div class="panel">
      <div class="progress" aria-hidden="true"><i id="blind-bar"></i></div>
      <div class="tag" id="blind-pos"></div>
      <div class="q">Do you see a face in this image?</div>
      <button class="btn face" id="ans-face">Yes, a face <kbd>F</kbd></button>
      <button class="btn" id="ans-unsure">Not sure <kbd>U</kbd></button>
      <button class="btn none" id="ans-none">No face <kbd>N</kbd></button>
      <div class="nav"><button class="btn" id="prev">← Back</button><button class="btn" id="next">Skip →</button></div>
      <p class="hint">Look for a few seconds, then answer with your first honest impression. The order is shuffled. Please do this step before opening step 2.</p>
    </div>
  </section>

  <section id="view-pick" hidden>
    <div class="bar">
      <div class="filters" id="filters"></div>
      <div class="hint">Left: Mooney · right: original. Click “Select” on the ten best Mooney images.</div>
    </div>
    <div class="grid" id="grid"></div>
  </section>

  <section class="fallback" id="fallback" hidden>
    <b>Saving is not available in this view.</b>
    <span class="hint">Your answers are kept on this page only. Copy the text below and paste it into the chat.</span>
    <textarea id="dump" readonly></textarea>
  </section>
</main>
<script>
const ITEMS = __ITEMS__;          // [{id, light, m, o}]
const ORDER = __ORDER__;          // shuffled ids for the blind check
const byId = Object.fromEntries(ITEMS.map(x => [x.id, x]));
let state = {blind: {}, top10: []};
let pos = 0, view = "blind", filter = "all", db = null, docRef = null, saveTimer = null, writing = false, dirty = false;
const $ = s => document.getElementById(s);
$("n-total").textContent = ITEMS.length;

function firstOpen() { const i = ORDER.findIndex(id => !state.blind[id]); return i < 0 ? 0 : i; }
function renderCounts() {
  $("n-blind").textContent = Object.keys(state.blind).length;
  $("n-sel").textContent = state.top10.length;
  $("blind-bar").style.width = (100 * Object.keys(state.blind).length / ITEMS.length) + "%";
}
function renderBlind() {
  const id = ORDER[pos];
  $("blind-img").src = byId[id].m;
  const a = state.blind[id];
  $("blind-pos").textContent = `image ${pos + 1} of ${ORDER.length}` + (a ? ` · your answer: ${a}` : "");
  renderCounts();
}
function answer(a) {
  state.blind[ORDER[pos]] = a;
  if (pos < ORDER.length - 1) pos++;
  renderBlind(); queueSave();
}
function renderPick() {
  const g = $("grid"); g.textContent = "";
  ITEMS.filter(x => filter === "all" || (filter === "selected" ? state.top10.includes(x.id) : x.light === filter)).forEach(x => {
    const sel = state.top10.includes(x.id);
    const c = document.createElement("div"); c.className = "card" + (sel ? " sel" : "");
    const pair = document.createElement("div"); pair.className = "pair";
    for (const [src, alt] of [[x.m, "Mooney " + x.id], [x.o, "Original " + x.id]]) {
      const im = document.createElement("img"); im.src = src; im.alt = alt; im.loading = "lazy"; pair.append(im);
    }
    const meta = document.createElement("div"); meta.className = "meta";
    const left = document.createElement("div");
    const idl = document.createElement("div"); idl.className = "id"; idl.textContent = x.id;
    const tg = document.createElement("div"); tg.className = "tag"; tg.textContent = "light " + x.light;
    const b = state.blind[x.id];
    if (b) { const bm = document.createElement("span"); bm.className = "blindmark " + b; bm.textContent = " · blind: " + b; tg.append(bm); }
    left.append(idl, tg);
    const btn = document.createElement("button"); btn.id = "sel-" + x.id;
    btn.textContent = sel ? "Selected" : "Select";
    btn.setAttribute("aria-pressed", sel ? "true" : "false");
    btn.onclick = () => toggle(x.id);
    meta.append(left, btn); c.append(pair, meta); g.append(c);
  });
  renderCounts();
}
function toggle(id) {
  const i = state.top10.indexOf(id);
  if (i >= 0) state.top10.splice(i, 1);
  else if (state.top10.length < 10) state.top10.push(id);
  else { $("sync").textContent = "10 already selected - deselect one first"; return; }
  renderPick(); queueSave();
}
function setView(v) {
  view = v;
  $("tab-blind").setAttribute("aria-selected", v === "blind"); $("tab-pick").setAttribute("aria-selected", v === "pick");
  $("view-blind").hidden = v !== "blind"; $("view-pick").hidden = v !== "pick";
  v === "blind" ? renderBlind() : renderPick();
}
function buildFilters() {
  const f = $("filters");
  for (const k of ["all", "left", "right", "top", "bottom", "selected"]) {
    const c = document.createElement("button"); c.className = "chip"; c.id = "f-" + k; c.textContent = k;
    c.setAttribute("aria-pressed", k === filter); c.onclick = () => { filter = k;
      document.querySelectorAll(".chip").forEach(e => e.setAttribute("aria-pressed", e.id === "f-" + k)); renderPick(); };
    f.append(c);
  }
}
function dumpLocal() { $("dump").value = JSON.stringify(state); }
function queueSave() {
  dumpLocal();
  if (!docRef) return;
  dirty = true; clearTimeout(saveTimer); saveTimer = setTimeout(flush, 700);
}
async function flush() {
  if (writing || !dirty) return;
  writing = true; dirty = false; $("sync").textContent = "saving…";
  try { await docRef.set({blind: state.blind, top10: state.top10, updatedAt: new Date().toISOString()});
        $("sync").textContent = "saved"; }
  catch (e) { $("sync").textContent = "not saved (" + (e && e.code || "error") + ")"; $("fallback").hidden = false; }
  writing = false; if (dirty) flush();
}
document.addEventListener("keydown", e => {
  if (view !== "blind" || e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "f") answer("face"); else if (k === "u") answer("unsure"); else if (k === "n") answer("none");
  else if (e.key === "ArrowRight") { pos = Math.min(ORDER.length - 1, pos + 1); renderBlind(); }
  else if (e.key === "ArrowLeft") { pos = Math.max(0, pos - 1); renderBlind(); }
});
$("ans-face").onclick = () => answer("face"); $("ans-unsure").onclick = () => answer("unsure"); $("ans-none").onclick = () => answer("none");
$("next").onclick = () => { pos = Math.min(ORDER.length - 1, pos + 1); renderBlind(); };
$("prev").onclick = () => { pos = Math.max(0, pos - 1); renderBlind(); };
$("tab-blind").onclick = () => setView("blind"); $("tab-pick").onclick = () => setView("pick");
buildFilters(); renderBlind(); dumpLocal();

(async () => {
  const timer = setTimeout(() => { $("sync").textContent = "saving unavailable"; $("fallback").hidden = false; }, 10000);
  try { db = window.claude ? await window.claude.use("db") : null; } catch (e) { db = null; }
  if (!db) { clearTimeout(timer); $("sync").textContent = "saving unavailable"; $("fallback").hidden = false; return; }
  docRef = db.doc("review/state");
  let first = true;
  docRef.onSnapshot(snap => {
    clearTimeout(timer);
    if (snap.exists && first && !dirty) {
      const d = snap.data() || {};
      state = {blind: Object.assign({}, d.blind || {}), top10: Array.from(d.top10 || [])};
      pos = firstOpen(); view === "blind" ? renderBlind() : renderPick(); dumpLocal();
    }
    if (!snap.metadata.fromCache) { first = false; $("sync").textContent = "saved"; }
  }, err => { clearTimeout(timer); $("sync").textContent = "saving unavailable"; $("fallback").hidden = false; });
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("-o", "--out", default="work/review/mooney_review.html")
    a = ap.parse_args()
    data = Path(a.data)
    meta = pd.read_csv(data / "metadata.csv")
    items = []
    for _, r in meta.iterrows():
        items.append(dict(id=r["id"], light=r["light"],
                          m=data_uri(data / "Mooney" / f"{r['id']}.png", 512, "PNG"),
                          o=data_uri(data / "Original" / f"{r['id']}.jpg", 320, "JPEG")))
    order = [x["id"] for x in items]
    random.Random(11).shuffle(order)
    html = PAGE.replace("__ITEMS__", json.dumps(items)).replace("__ORDER__", json.dumps(order))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(items)} items)")


if __name__ == "__main__":
    main()
