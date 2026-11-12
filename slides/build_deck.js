// Builds the Part-1 presentation from work/deck_input.json (written by slides/prepare_deck.py).
//   NODE_PATH=$(npm root -g) node slides/build_deck.js work/deck_input.json out.pptx
// Plain academic layout: white slides, one sans-serif family, short text, long explanations in the notes.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const IN = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3] || "deck.pptx";
const SKILL = process.env.PPTX_SKILL || "";

const THEME = {
  name: "CompVP",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "04316A", lt2: "F2F4F7",
    accent1: "04316A", accent2: "8C9FB1", accent3: "C8102E", accent4: "5B6770",
    accent5: "7A8590", accent6: "2F6DB5", hlink: "04316A", folHlink: "5B6770",
  },
};
const HEX = THEME.colors;
const BLUE = HEX.dk2, GREY = HEX.accent2, MUTED = HEX.accent5;
const ORIG = GREY, MOON = BLUE;                     // chart convention: Original = grey, Mooney = blue

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                        // 13.333 x 7.5 in
pres.author = "Ahmadreza Nourozi";
pres.title = "Do Modern Vision Systems Perceive Faces in Mooney Images? - CompVP Project Part 1";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const W = 13.333, H = 7.5, MX = 0.7, TOP = 1.55, BOT = 6.75;
const FOOT = "Ahmadreza Nourozi  |  Computational Visual Perception, Project Part 1  |  FAU Erlangen-Nürnberg";

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.lt1 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.45, w: W - 2 * MX, h: 0.8, fontSize: 30, bold: true,
        color: BLUE, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: H - 0.45, w: 10, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------------------------------------------------------- helpers
function img(slide, p, o) { if (p && fs.existsSync(p)) slide.addImage({ path: p, ...o }); }
function imgFit(slide, p, x, y, w, h, extra = {}) {   // keep aspect ratio inside box, top-left aligned
  if (!p || !fs.existsSync(p)) return { w: 0, h: 0 };
  const sz = IN.sizes && IN.sizes[p];
  let ww = w, hh = h;
  if (sz) { const r = sz[0] / sz[1]; if (w / h > r) { ww = h * r; } else { hh = w / r; } }
  slide.addImage({ path: p, x, y, w: ww, h: hh, ...extra });
  return { w: ww, h: hh };
}
function bullets(slide, lines, o) {
  slide.addText(lines.map((t, k) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: k < lines.length - 1 } })),
    Object.assign({ fontSize: 18, color: HEX.dk1, valign: "top", paraSpaceAfter: 10, margin: 0, isTextBox: true }, o));
}
function conclusion(slide, text, y = 6.05) {
  slide.addText(text, { x: MX, y, w: W - 2 * MX, h: 0.6, fontSize: 18, bold: true, color: BLUE, margin: 0, valign: "middle",
    isTextBox: true });
}
function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.3, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true });
}
const COLORS = { orig: ORIG, moon: MOON, dark: HEX.accent4, teal: HEX.accent6, grey: "C5CCD3", red: HEX.accent3 };
function colorize(spec) { if (spec && spec.series) spec.series.forEach(x => { x.color = COLORS[x.color] || x.color || ORIG; }); return spec; }
function barChart(slide, spec, o) {
  const series = spec.series;
  const data = series.map(x => ({ name: x.name, labels: spec.labels, values: x.values }));
  const fmt = spec.fmt || "0%";
  slide.addChart(pres.charts.BAR, data, Object.assign({
    barDir: "col", barGrouping: "clustered", chartColors: series.map(x => x.color || ORIG), showLegend: series.length > 1,
    legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.dk1, showValue: true,
    dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1,
    dataLabelFormatCode: fmt, valAxisMinVal: spec.min ?? 0, valAxisMaxVal: spec.max || 1, valAxisLabelFormatCode: fmt,
    valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.dk1, valGridLine: { color: "E1E5EA", size: 0.75 },
    catGridLine: { style: "none" }, barGapWidthPct: 70, showTitle: !!spec.title, title: spec.title || "", titleFontSize: 13,
    titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  }, o));
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide({ masterName: "TITLE" });
  s.addText("Computational Visual Perception  |  Project Part 1", { x: MX, y: 1.0, w: 7.2, h: 0.4, fontSize: 16, color: MUTED,
    margin: 0, isTextBox: true });
  s.addText(IN.title, { x: MX, y: 1.6, w: 7.2, h: 2.2, fontSize: 36, bold: true, color: BLUE, margin: 0, valign: "top",
    isTextBox: true });
  s.addText([
    { text: "Ahmadreza Nourozi", options: { fontSize: 22, bold: true, color: HEX.dk1, breakLine: true } },
    { text: "Student number 23726011  |  M.Sc. Artificial Intelligence", options: { fontSize: 16, color: HEX.dk1, breakLine: true } },
    { text: "Friedrich-Alexander-Universität Erlangen-Nürnberg", options: { fontSize: 16, color: HEX.dk1, breakLine: true } },
    { text: IN.date || "November 2026", options: { fontSize: 16, color: MUTED } },
  ], { x: MX, y: 4.3, w: 7.2, h: 1.9, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 6 });
  const t = (IN.top10 || []).slice(0, 2), ts = 2.3, g = 0.12, x0 = W - MX - 2 * ts - g, y0 = 1.35;
  t.forEach((q, i) => {
    img(s, q.orig, { x: x0, y: y0 + i * (ts + g), w: ts, h: ts });
    img(s, q.mooney, { x: x0 + ts + g, y: y0 + i * (ts + g), w: ts, h: ts });
  });
  caption(s, "Original and Mooney image", x0, y0 + 2 * ts + g + 0.08, 2 * ts + g);
}

// ---------------------------------------------------------------- 2 approach (one page)
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(IN.approach_title, { placeholder: "title" });
  const cols = IN.approach_cols, gap = 0.5, cw = (W - 2 * MX - 2 * gap) / 3;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + gap);
    s.addText(c.head, { x, y: TOP, w: cw, h: 0.45, fontSize: 20, bold: true, color: HEX.dk1, margin: 0, isTextBox: true });
    bullets(s, c.lines, { x, y: TOP + 0.65, w: cw, h: 3.2, fontSize: 17 });
  });
  s.addText([{ text: "Question. ", options: { bold: true, color: BLUE } }, { text: IN.approach_question, options: { color: HEX.dk1 } }],
    { x: MX, y: 5.5, w: W - 2 * MX, h: 0.95, fontSize: 18, margin: 0, valign: "top", isTextBox: true });
  if (IN.approach_note) s.addNotes(IN.approach_note);
}

// ---------------------------------------------------------------- 3 selected 10 + 10
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Ten selected originals and their Mooney images", { placeholder: "title" });
  const n = 5, gap = 0.1, pair = 0.04, ts = 1.06, bw = 2 * ts + pair, x0 = MX + (W - 2 * MX - n * bw - (n - 1) * 0.28) / 2;
  (IN.top10 || []).forEach((t, i) => {
    const r = Math.floor(i / n), c = i % n, x = x0 + c * (bw + 0.28), y = TOP + 0.05 + r * (ts + 0.55);
    img(s, t.orig, { x, y, w: ts, h: ts });
    img(s, t.mooney, { x: x + ts + pair, y, w: ts, h: ts });
    s.addText(t.label.replace("·", "|"), { x, y: y + ts + 0.04, w: bw, h: 0.26, fontSize: 11, color: MUTED, align: "center",
      margin: 0, isTextBox: true });
  });
  s.addText(IN.top10_note, { x: MX, y: 5.35, w: W - 2 * MX, h: 0.7, fontSize: 18, color: HEX.dk1, margin: 0, valign: "top",
    isTextBox: true });
}

// ---------------------------------------------------------------- 4 declaration of vision systems
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Vision systems used", { placeholder: "title" });
  const hdr = ["", "System", "Model", "Output analysed"].map(t => ({ text: t,
    options: { bold: true, color: HEX.dk1, fontSize: 15, border: [{ type: "none" }, { type: "none" }, { type: "solid", pt: 1.25, color: HEX.dk1 }, { type: "none" }] } }));
  const line = [{ type: "none" }, { type: "none" }, { type: "solid", pt: 0.5, color: "C5CCD3" }, { type: "none" }];
  const rows = [hdr].concat(IN.systems.map((r, i) => [
    { text: `${i + 1}`, options: { border: line } }, { text: r.system, options: { bold: true, border: line } },
    { text: r.model.replace(/\s*\([^)]*\/[^)]*\)/, ""), options: { border: line } }, { text: r.output, options: { border: line } },
  ]));
  s.addTable(rows, { x: MX, y: TOP, w: W - 2 * MX, colW: [0.5, 3.2, 4.1, 4.133], fontSize: 15, color: HEX.dk1,
    valign: "middle", margin: [0.06, 0.08, 0.06, 0.08], rowH: [0.5, 0.82, 0.82, 0.82, 0.82] });
  s.addText(IN.systems_note, { x: MX, y: 5.75, w: W - 2 * MX, h: 0.7, fontSize: 16, color: HEX.dk1, margin: 0, valign: "top",
    isTextBox: true });
}

// ---------------------------------------------------------------- 5-8 results per system
function resultSlide(r) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(r.title, { placeholder: "title" });
  const lw = 7.0, lh = 4.0;
  const d = imgFit(s, r.panel, MX, TOP, lw, lh);
  if (r.panel_caption) caption(s, r.panel_caption, MX, TOP + d.h + 0.08, lw);
  const rx = MX + lw + 0.45, rw = W - MX - rx, fh = 0.45 * (r.facts || []).length + 0.1;
  bullets(s, r.facts || [], { x: rx, y: TOP, w: rw, h: fh, fontSize: 16, paraSpaceAfter: 6 });
  if (r.chart) barChart(s, colorize(r.chart), { x: rx - 0.1, y: TOP + fh + 0.1, w: rw + 0.1, h: 5.8 - TOP - fh });
  conclusion(s, r.takeaway);
  if (r.note) s.addNotes(r.note);
}
(IN.results || []).forEach(resultSlide);

// ---------------------------------------------------------------- light direction (hypothesis H2)
if (IN.light_slide) {
  const L = IN.light_slide;
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(L.title, { placeholder: "title" });
  barChart(s, colorize(L.chart), { x: MX - 0.1, y: TOP - 0.05, w: 6.9, h: 4.3 });
  bullets(s, L.points, { x: MX + 7.3, y: TOP + 0.2, w: W - 2 * MX - 7.3, h: 3.9, fontSize: 17 });
  conclusion(s, L.takeaway);
  if (L.note) s.addNotes(L.note);
}

// ---------------------------------------------------------------- summary across systems
if (IN.summary) {
  const Sm = IN.summary;
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(Sm.title, { placeholder: "title" });
  barChart(s, colorize(Sm.main), { x: MX - 0.1, y: TOP - 0.05, w: 6.9, h: 4.3 });
  bullets(s, Sm.points || [], { x: MX + 7.3, y: TOP + 0.2, w: W - 2 * MX - 7.3, h: 3.9, fontSize: 17 });
  conclusion(s, Sm.takeaway);
  if (Sm.note) s.addNotes(Sm.note);
}

// ---------------------------------------------------------------- tool screenshot
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Screenshot of the Mooney tool", { placeholder: "title" });
  const ln = { color: "C5CCD3", width: 0.75 };
  imgFit(s, IN.shot_gui, MX, TOP, 7.4, 4.6, { line: ln });
  const rx = MX + 7.65, rw = W - MX - rx;
  const a = imgFit(s, IN.shot_cli, rx, TOP, rw, 2.0, { line: ln });
  imgFit(s, IN.shot_batch, rx, TOP + a.h + 0.15, rw, 4.6 - a.h - 0.15, { line: ln });
  s.addText(IN.tool_text, { x: MX, y: 6.2, w: W - 2 * MX, h: 0.45, fontSize: 14, color: HEX.dk1, margin: 0, valign: "top",
    isTextBox: true });
}

// ---------------------------------------------------------------- code screenshot
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Screenshot of the code that generates the Mooney images", { placeholder: "title" });
  imgFit(s, IN.shot_code, MX, TOP, 8.2, 5.0);
  s.addText(IN.code_text, { x: MX + 8.6, y: TOP, w: W - 2 * MX - 8.6, h: 3.5, fontSize: 17, color: HEX.dk1, margin: 0,
    valign: "top", isTextBox: true, paraSpaceAfter: 12 });
}

// ---------------------------------------------------------------- AI usage
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Declaration of AI usage", { placeholder: "title" });
  const gap = 0.6, cw = (W - 2 * MX - gap) / 2;
  IN.ai_cols.forEach((c, i) => {
    const x = MX + i * (cw + gap);
    s.addText(c.head, { x, y: TOP, w: cw, h: 0.45, fontSize: 19, bold: true, color: HEX.dk1, margin: 0, isTextBox: true });
    bullets(s, c.lines, { x, y: TOP + 0.65, w: cw, h: 3.4, fontSize: 15, paraSpaceAfter: 9 });
  });
  s.addText(IN.ai_note, { x: MX, y: 5.75, w: W - 2 * MX, h: 0.7, fontSize: 14, color: HEX.dk1, margin: 0, valign: "top",
    isTextBox: true });
}

// ---------------------------------------------------------------- references
if (IN.references) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("References", { placeholder: "title" });
  s.addText(IN.references.map((t, k) => ({ text: t, options: { breakLine: k < IN.references.length - 1 } })),
    { x: MX, y: TOP, w: W - 2 * MX, h: 5.1, fontSize: 12.5, color: HEX.dk1, valign: "top", paraSpaceAfter: 5, margin: 0,
      isTextBox: true });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
