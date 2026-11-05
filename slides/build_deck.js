// Builds the Part-1 presentation from work/deck_input.json (written by slides/prepare_deck.py).
//   NODE_PATH=$(npm root -g) node slides/build_deck.js work/deck_input.json out.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const IN = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3] || "deck.pptx";
const SKILL = process.env.PPTX_SKILL || "";

const THEME = {
  name: "Mooney",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "111315", lt1: "FFFFFF", dk2: "2B2F33", lt2: "F1F3F5",
    accent1: "F2A93B", accent2: "4A5560", accent3: "2F7F8A", accent4: "C0392B",
    accent5: "8E9AA3", accent6: "1F4E5F", hlink: "2F7F8A", folHlink: "4A5560",
  },
};
const HEX = THEME.colors;
const ORIG = HEX.accent2, MOON = HEX.accent1;      // chart convention: Original = slate, Mooney = amber

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                        // 13.333 x 7.5 in
pres.author = "Ahmadreza Nourozi";
pres.title = "Do Vision Models See Mooney Faces? - CompVP Project Part 1";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.6;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.5, w: W - 2 * MX, h: 0.75, fontFace: THEME.headFontFace,
        fontSize: 28, bold: true, color: HEX.dk1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "A. Nourozi  ·  CompVP Project Part 1", options: { x: MX, y: H - 0.42, w: 6, h: 0.3, fontSize: 10,
        color: HEX.accent5, fontFace: THEME.bodyFontFace, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.5, w: W - 2 * MX, h: 0.75, fontFace: THEME.headFontFace,
        fontSize: 28, bold: true, color: HEX.lt1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - MX - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
});

// ---------------------------------------------------------------- helpers
function tag(slide, text, dark = false) {
  slide.addText(text.toUpperCase(), { x: MX, y: 0.2, w: 8, h: 0.28, fontSize: 11, bold: true, charSpacing: 3,
    color: dark ? HEX.accent1 : "B9781A", margin: 0, isTextBox: true, objectName: "section tag" });
}
function card(slide, x, y, w, h, fill = HEX.lt2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill },
    objectName: "card" });
}
function stat(slide, x, y, w, big, label, color = HEX.dk1, sub = null) {
  slide.addText(big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace, margin: 0,
    isTextBox: true });
  slide.addText(label, { x, y: y + 0.75, w, h: 0.5, fontSize: 13, color: HEX.dk2, margin: 0, valign: "top", isTextBox: true });
  if (sub) slide.addText(sub, { x, y: y + 1.2, w, h: 0.35, fontSize: 11, color: HEX.accent5, margin: 0, isTextBox: true });
}
function pct(v) { return (v === null || v === undefined || Number.isNaN(v)) ? "–" : `${Math.round(100 * v)}%`; }
function num(v, d = 2) { return (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d); }
function img(slide, p, o) { if (p && fs.existsSync(p)) slide.addImage({ path: p, ...o }); }
function imgFit(slide, p, x, y, w, h, extra = {}) {   // keep aspect ratio inside box (x,y,w,h), centred
  if (!p || !fs.existsSync(p)) return;
  const sz = IN.sizes && IN.sizes[p];
  let ww = w, hh = h;
  if (sz) { const r = sz[0] / sz[1]; if (w / h > r) { ww = h * r; } else { hh = w / r; } }
  slide.addImage({ path: p, x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh, ...extra });
}
function barChart(slide, spec, o) {
  // spec: {labels, series:[{name, values, color}], fmt, max, title}  (legacy: orig/moon arrays)
  const series = spec.series || [{ name: "Original", values: spec.orig, color: ORIG }, { name: "Mooney", values: spec.moon, color: MOON }];
  const data = series.map(x => ({ name: x.name, labels: spec.labels, values: x.values }));
  const fmt = spec.fmt || "0%";
  slide.addChart(pres.charts.BAR, data, Object.assign({
    barDir: "col", barGrouping: "clustered", chartColors: series.map(x => x.color || ORIG), showLegend: series.length > 1,
    legendPos: "t", legendFontSize: 11, legendFontFace: "+mn-lt", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk2, dataLabelFormatCode: fmt,
    valAxisMinVal: 0, valAxisMaxVal: spec.max || 1, valAxisLabelFormatCode: fmt, valAxisLabelFontSize: 10,
    catAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2,
    valAxisLabelColor: HEX.accent5, valGridLine: { color: "E3E6E8", size: 0.75 }, catGridLine: { style: "none" },
    barGapWidthPct: 60, showTitle: !!spec.title, title: spec.title || "", titleFontSize: 12, titleColor: HEX.dk2,
    titleFontFace: "+mn-lt",
  }, o));
}
const COLORS = { orig: ORIG, moon: MOON, dark: HEX.dk2, teal: HEX.accent3, grey: HEX.accent5, red: HEX.accent4 };
function colorize(spec) { if (spec && spec.series) spec.series.forEach(x => { x.color = COLORS[x.color] || x.color || ORIG; }); return spec; }

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide({ masterName: "DARK" });
  // right: Mooney mosaic
  const tiles = IN.title_tiles || [];
  const ts = 1.62, g = 0.08, x0 = W - MX - 3 * ts - 2 * g, y0 = 1.0;
  tiles.slice(0, 9).forEach((p, i) => img(s, p, { x: x0 + (i % 3) * (ts + g), y: y0 + Math.floor(i / 3) * (ts + g), w: ts, h: ts,
    rounding: false }));
  s.addText("COMPUTATIONAL VISUAL PERCEPTION  ·  PROJECT PART 1", { x: MX, y: 1.0, w: 6.9, h: 0.3, fontSize: 12, bold: true,
    charSpacing: 3, color: HEX.accent1, margin: 0, isTextBox: true });
  s.addText(IN.title || "Do Modern Vision Systems Perceive Faces in Mooney Images?", { x: MX, y: 1.45, w: 6.9, h: 1.95, fontSize: 36, bold: true,
    fontFace: THEME.headFontFace, color: HEX.lt1, margin: 0, valign: "top", isTextBox: true });
  s.addText(IN.subtitle || "", { x: MX, y: 3.5, w: 6.6, h: 0.9, fontSize: 16, color: "C9CED3", margin: 0, valign: "top",
    isTextBox: true });
  s.addText([
    { text: "Ahmadreza Nourozi", options: { fontSize: 22, bold: true, color: HEX.lt1, breakLine: true } },
    { text: "M.Sc. Artificial Intelligence  ·  Student no. 23726011", options: { fontSize: 14, color: "C9CED3", breakLine: true } },
    { text: "FAU Erlangen-Nürnberg  ·  " + (IN.date || "November 2026"), options: { fontSize: 14, color: "C9CED3" } },
  ], { x: MX, y: 4.75, w: 6.6, h: 1.2, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 4 });
  s.addText("Team: Ahmadreza Nourozi (single-person team)", { x: MX, y: 6.55, w: 6.6, h: 0.3, fontSize: 11,
    color: HEX.accent5, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 2 approach (one page)
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Approach and ideas");
  s.addText(IN.approach_title, { placeholder: "title" });
  const cols = IN.approach_cols;               // [{head, lines:[...]}, x3]
  const cw = (W - 2 * MX - 2 * 0.3) / 3, cy = 1.45, ch = 4.5;
  cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, cy, cw, ch);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: cy + 0.25, w: 0.5, h: 0.5, fill: { color: i === 1 ? HEX.dk1 : HEX.accent1 },
      line: { color: i === 1 ? HEX.dk1 : HEX.accent1 } });
    s.addText(String(i + 1), { x: x + 0.25, y: cy + 0.25, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16,
      bold: true, color: i === 1 ? HEX.lt1 : HEX.dk1, margin: 0, isTextBox: true });
    s.addText(c.head, { x: x + 0.9, y: cy + 0.25, w: cw - 1.1, h: 0.5, fontSize: 18, bold: true, valign: "middle",
      fontFace: THEME.headFontFace, color: HEX.dk1, margin: 0, isTextBox: true });
    s.addText(c.lines.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < c.lines.length - 1 } })),
      { x: x + 0.25, y: cy + 0.95, w: cw - 0.45, h: ch - 1.15, fontSize: 13, color: HEX.dk2, valign: "top",
        paraSpaceAfter: 6, margin: 0, isTextBox: true });
  });
  card(s, MX, 6.15, W - 2 * MX, 0.78, "FFF4E2");
  s.addText(IN.approach_question, { x: MX + 0.25, y: 6.15, w: W - 2 * MX - 0.5, h: 0.78, fontSize: 13, color: HEX.dk1,
    margin: 0, valign: "middle", isTextBox: true });
}

// ---------------------------------------------------------------- 3 selected 10 + 10
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Subset: manually selected top 10");
  s.addText("10 selected originals and their Mooney images", { placeholder: "title" });
  const n = 10, gap = 0.08, ts = (W - 2 * MX - (n - 1) * gap) / n;
  const y1 = 1.55, y2 = y1 + ts + 0.12;
  (IN.top10 || []).forEach((t, i) => {
    const x = MX + i * (ts + gap);
    img(s, t.orig, { x, y: y1, w: ts, h: ts });
    img(s, t.mooney, { x, y: y2, w: ts, h: ts });
    s.addText(t.label, { x, y: y2 + ts + 0.05, w: ts, h: 0.3, fontSize: 10, color: HEX.accent5, align: "center", margin: 0,
      isTextBox: true });
  });
  s.addText("Original", { x: MX - 0.02, y: y1 - 0.02, w: 1, h: 0.25, fontSize: 9, bold: true, color: HEX.lt1, margin: 0.03,
    fill: { color: HEX.dk2 }, isTextBox: true });
  s.addText("Mooney", { x: MX - 0.02, y: y2 - 0.02, w: 1, h: 0.25, fontSize: 9, bold: true, color: HEX.dk1, margin: 0.03,
    fill: { color: HEX.accent1 }, isTextBox: true });
  const tc = IN.top10_criteria || [];
  const cw = (W - 2 * MX - (tc.length - 1) * 0.3) / Math.max(1, tc.length);
  tc.forEach((c, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, 5.05, cw, 1.75);
    s.addText(c.head, { x: x + 0.25, y: 5.15, w: cw - 0.5, h: 0.4, fontSize: 14, bold: true, color: HEX.dk1,
      fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText(c.text, { x: x + 0.25, y: 5.55, w: cw - 0.5, h: 1.15, fontSize: 12, color: HEX.dk2, margin: 0, valign: "top",
      isTextBox: true });
  });
}

// ---------------------------------------------------------------- 4 declaration of vision systems
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Declaration of used vision systems");
  s.addText("Four vision systems, one input protocol", { placeholder: "title" });
  const hdr = ["", "System", "Model / checkpoint", "Output we analyse", "Source · licence"].map(t => ({ text: t,
    options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 12 } }));
  const rows = [hdr].concat(IN.systems.map((r, i) => [
    { text: `R${i + 1}`, options: { bold: true, color: HEX.dk1, fill: { color: HEX.accent1 }, align: "center" } },
    { text: r.system, options: { bold: true } }, { text: r.model }, { text: r.output }, { text: r.source, options: { fontSize: 10.5 } },
  ]));
  s.addTable(rows, { x: MX, y: 1.5, w: W - 2 * MX, colW: [0.55, 2.1, 3.25, 3.75, 2.48], fontSize: 11.5, color: HEX.dk2,
    border: { type: "solid", pt: 0.75, color: "D5D9DD" }, valign: "middle", margin: [0.06, 0.08, 0.06, 0.08],
    rowH: [0.42, 0.95, 0.95, 0.95, 0.95] });
  s.addText(IN.systems_note, { x: MX, y: 6.0, w: W - 2 * MX, h: 0.75, fontSize: 12, color: HEX.dk2, margin: 0,
    valign: "top", isTextBox: true });
}

// ---------------------------------------------------------------- 5-8 results per system
function resultSlide(r, idx) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, `Results · system ${idx} · ${r.system}`);
  s.addText(r.title, { placeholder: "title" });
  const lx = MX, ly = 1.45, lw = 7.6, lh = 4.15;
  imgFit(s, r.panel, lx, ly, lw, lh);
  if (r.panel_caption) s.addText(r.panel_caption, { x: lx, y: ly + lh + 0.05, w: lw, h: 0.3, fontSize: 10.5, color: HEX.accent5,
    margin: 0, isTextBox: true });
  // right column: stats + chart
  const rx = MX + lw + 0.35, rw = W - MX - rx;
  (r.stats || []).forEach((st, i) => stat(s, rx + i * (rw / r.stats.length), 1.45, rw / r.stats.length - 0.15, st.big, st.label,
    st.color === "moon" ? "B9781A" : HEX.dk1));
  if (r.chart) barChart(s, colorize(r.chart), { x: rx - 0.1, y: 2.85, w: rw + 0.1, h: 2.95 });
  card(s, MX, 6.0, W - 2 * MX, 0.82, "FFF4E2");
  s.addText(r.takeaway, { x: MX + 0.25, y: 6.0, w: W - 2 * MX - 0.5, h: 0.82, fontSize: 13.5, color: HEX.dk1, margin: 0,
    valign: "middle", isTextBox: true });
  return s;
}
(IN.results || []).forEach((r, i) => resultSlide(r, i + 1));

// ---------------------------------------------------------------- light-from-above slide (hypothesis H2)
if (IN.light_slide) {
  const L = IN.light_slide;
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Results · light from above vs. below (H2)");
  s.addText(L.title, { placeholder: "title" });
  barChart(s, colorize(L.chart), { x: MX - 0.1, y: 1.4, w: 8.3, h: 4.45, showValue: false });
  card(s, MX + 8.45, 1.45, W - 2 * MX - 8.45, 4.4);
  s.addText(L.points.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < L.points.length - 1 } })),
    { x: MX + 8.65, y: 1.6, w: W - 2 * MX - 8.85, h: 4.1, fontSize: 12.5, color: HEX.dk2, valign: "top", paraSpaceAfter: 6,
      margin: 0, isTextBox: true });
  card(s, MX, 6.0, W - 2 * MX, 0.82, "FFF4E2");
  s.addText(L.takeaway, { x: MX + 0.25, y: 6.0, w: W - 2 * MX - 0.5, h: 0.82, fontSize: 13.5, color: HEX.dk1, margin: 0,
    valign: "middle", isTextBox: true });
}

// ---------------------------------------------------------------- 9 summary across systems
if (IN.summary) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Results · all systems");
  s.addText(IN.summary.title, { placeholder: "title" });
  barChart(s, colorize(IN.summary.main), { x: MX - 0.1, y: 1.4, w: 7.4, h: 4.5 });
  if (IN.summary.by_light) barChart(s, colorize(IN.summary.by_light), { x: 7.75, y: 1.4, w: W - MX - 7.75, h: 4.5,
    showValue: false });
  card(s, MX, 6.0, W - 2 * MX, 0.82, "FFF4E2");
  s.addText(IN.summary.takeaway, { x: MX + 0.25, y: 6.0, w: W - 2 * MX - 0.5, h: 0.82, fontSize: 13.5, color: HEX.dk1,
    margin: 0, valign: "middle", isTextBox: true });
}

// ---------------------------------------------------------------- 10 tool screenshot
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Screenshot of the tool");
  s.addText("The Mooney tool: GUI with live controls + batch CLI", { placeholder: "title" });
  imgFit(s, IN.shot_gui, MX, 1.4, 8.3, 5.3, { line: { color: "D5D9DD", width: 0.75 } });
  imgFit(s, IN.shot_cli, MX + 8.55, 1.4, W - 2 * MX - 8.55, 3.3, { line: { color: "D5D9DD", width: 0.75 } });
  s.addText(IN.tool_text, { x: MX + 8.55, y: 4.85, w: W - 2 * MX - 8.55, h: 1.9, fontSize: 12, color: HEX.dk2, margin: 0,
    valign: "top", isTextBox: true, paraSpaceAfter: 4 });
}

// ---------------------------------------------------------------- 11 code screenshot
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "Screenshot of the code that generates Mooneys");
  s.addText(IN.code_title || "The code that generates the Mooney images", { placeholder: "title" });
  imgFit(s, IN.shot_code, MX, 1.4, 8.6, 5.4);
  s.addText(IN.code_text, { x: MX + 8.85, y: 1.45, w: W - 2 * MX - 8.85, h: 5.3, fontSize: 12.5, color: HEX.dk2, margin: 0,
    valign: "top", isTextBox: true, paraSpaceAfter: 6 });
}

// ---------------------------------------------------------------- 12 AI usage
{
  const s = pres.addSlide({ masterName: "DARK" });
  tag(s, "Declaration of AI usage", true);
  s.addText("How generative AI was used in this project", { placeholder: "title" });
  const cw = (W - 2 * MX - 0.3) / 2;
  IN.ai_cols.forEach((c, i) => {
    const x = MX + i * (cw + 0.3);
    card(s, x, 1.5, cw, 4.6, "1E2124");
    s.addText(c.head, { x: x + 0.3, y: 1.7, w: cw - 0.6, h: 0.45, fontSize: 17, bold: true, color: HEX.accent1,
      fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText(c.lines.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < c.lines.length - 1 } })),
      { x: x + 0.3, y: 2.3, w: cw - 0.6, h: 3.65, fontSize: 13, color: "E3E6E8", valign: "top", paraSpaceAfter: 6, margin: 0,
        isTextBox: true });
  });
  s.addText(IN.ai_note, { x: MX, y: 6.3, w: W - 2 * MX, h: 0.5, fontSize: 12, color: "C9CED3", margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 13 references
if (IN.references) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  tag(s, "References and reproducibility");
  s.addText("References", { placeholder: "title" });
  s.addText(IN.references.map((t, k) => ({ text: t, options: { breakLine: k < IN.references.length - 1 } })),
    { x: MX, y: 1.45, w: W - 2 * MX, h: 5.3, fontSize: 12, color: HEX.dk2, valign: "top", paraSpaceAfter: 5, margin: 0,
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
