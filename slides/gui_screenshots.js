// Screenshots of the running Mooney GUI for the slides.
//   NODE_PATH=$(npm root -g) node slides/gui_screenshots.js <url> <image> <mesh> <texture> <inDir> <outDir> <shotDir> <moonDir>
const { chromium } = require("playwright");
const [, , url, image, mesh, texture, inDir, outDir, shotDir, moonDir] = process.argv;

async function openTab(page, name) {
  await page.getByRole("tab", { name }).click();
  await page.waitForTimeout(700);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1.5 });
  await page.goto(url, { waitUntil: "load", timeout: 60000 });
  await page.waitForSelector("input[type=file]", { state: "attached", timeout: 60000 });

  // 1) Image -> Mooney with a real portrait
  await page.locator("input[type=file]").first().setInputFiles(image);
  await page.getByText("threshold", { exact: false }).first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${shotDir}/gui_image.png`, fullPage: false });

  // 2) Batch tab: convert the whole folder
  await openTab(page, "Batch");
  await page.getByLabel("Input folder").fill(inDir);
  await page.getByLabel("Output folder").fill(outDir);
  await page.getByRole("button", { name: "Run batch" }).click();
  await page.getByText("Converted", { exact: false }).first().waitFor({ timeout: 240000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${shotDir}/gui_batch.png`, fullPage: false });

  // 3) 3D mesh tab
  await openTab(page, "3D mesh → Mooney");
  const files = page.locator("input[type=file]");
  const n = await files.count();
  for (let i = 0; i < n; i++) {                       // the first visible file input of this tab takes the mesh
    if (await files.nth(i).evaluate(e => e.accept && e.accept.includes(".glb"))) { await files.nth(i).setInputFiles(mesh); break; }
  }
  await page.getByText("light = (", { exact: false }).first().waitFor({ timeout: 120000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${shotDir}/gui_mesh.png`, fullPage: false });

  // 4) Select-top-10 tab
  await openTab(page, "Select top 10");
  await page.getByLabel("Mooney folder").fill(moonDir);
  await page.getByRole("button", { name: "Load" }).click();
  await page.waitForTimeout(4000);
  await page.screenshot({ path: `${shotDir}/gui_select.png`, fullPage: false });
  await browser.close();
  console.log("screenshots ->", shotDir);
})();
