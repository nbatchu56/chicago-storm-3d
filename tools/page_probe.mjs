#!/usr/bin/env node
// Headless checks of a built LakeView page (world.html + data/ beside it) with Playwright's Chromium (software WebGL).
//
//   node tools/page_probe.mjs --site out/ --harbor                 # the harbour mask against the survey polygon
//   node tools/page_probe.mjs --site out/ --shot harbor out/h.png  # a screenshot of a named view (or lat,lon,km,heading,pitch)
//   node tools/page_probe.mjs --site out/ --harbor-waves           # Hs at the basin centre for a north wind
//   node tools/page_probe.mjs --site out/ --eval q.js              # print the value of a JS expression run in the page
//
// Exit code 1 when a check fails. Views: harbor, lagoon, mouth, outlook.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");   // resolves through NODE_PATH too
import { createServer } from "node:http";
import { readFileSync, statSync, existsSync } from "node:fs";
import { join, extname, resolve } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const has = (k) => args.includes(k);
const site = resolve(opt("--site", "."));
const page_ = opt("--page", "world.html");
const VIEWS = {
  harbor: [41.9295, -87.6335, 0.9, 0, 89.9],      // straight down over the basin and the channel
  lagoon: [41.9215, -87.6308, 1.1, 0, 89.9],
  mouth: [41.9322, -87.6318, 0.26, 245, 15],
  outlook: [41.93255, -87.63027, 0.319, 149.6, 16.4],
};
const MIME = { ".html": "text/html", ".json": "application/json", ".jpg": "image/jpeg", ".png": "image/png", ".js": "text/javascript" };

const server = createServer((req, res) => {
  const p = join(site, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || page_);
  if (!existsSync(p) || !statSync(p).isFile()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": MIME[extname(p)] || "application/octet-stream" }); res.end(readFileSync(p));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/${page_}`;

const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 1 });
const logs = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`); });
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: "load" });
await page.waitForFunction(() => window.__world && window.__world.MAP && window.__world.MAP.ok, null, { timeout: 180000 });
await page.waitForTimeout(1500);
let failed = false;
const fail = (m) => { console.log("FAIL " + m); failed = true; };

if (has("--harbor")) {
  // sample 3 m inside and 3 m outside the surveyed shoreline of every survey polygon; the mask should say harbour water
  // (2) inside and land (0) outside, except where the polygon meets the open lake (the LSD inlet stub)
  const r = await page.evaluate(() => {
    const W = window.__world, sv = W.BATHY.survey; if (!sv || !sv.polys) return null;
    const out = {};
    for (const poly of sv.polys) {
      const ring = poly.rings[0]; let area = 0; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) area += (ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1]);
      const ccw = area < 0;   // shoelace sign in a y-up frame
      const want = /pond/i.test(poly.name || "") ? 3 : 2;   // ponds are class 3, harbours 2
      let nIn = 0, okIn = 0, nOut = 0, okOut = 0; const badIn = [], badOut = [];
      for (let i = 0; i < ring.length - 1; i++) {
        const [ax, ay] = ring[i], [bx, by] = ring[i + 1], L = Math.hypot(bx - ax, by - ay); if (L < 1) continue;
        const ux = (bx - ax) / L, uy = (by - ay) / L, nx = ccw ? uy : -uy, ny = ccw ? -ux : ux;   // outward normal
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        const inM = W.maskAt((mx - nx * 3) / 1000, (my - ny * 3) / 1000), outM = W.maskAt((mx + nx * 3) / 1000, (my + ny * 3) / 1000);
        nIn++; if (inM === want) okIn++; else badIn.push([Math.round(mx), Math.round(my), inM]);
        nOut++; if (outM === 0) okOut++; else badOut.push([Math.round(mx), Math.round(my), outM]);
      }
      out[poly.name] = { inside: okIn / nIn, outside: okOut / nOut, n: nIn, badIn: badIn.slice(0, 8), badOut: badOut.slice(0, 8) };
    }
    out._map = { lidarWater: W.MAP.lidarWater, marina: W.MAP.marina, waterways: W.MAP.waterways, waterAreas: W.MAP.waterAreas, specks: W.MAP.specks, survey: W.MAP.survey };
    return out;
  });
  if (!r) fail("no survey polygons in the page (BATHY.survey)");
  else {
    for (const [k, v] of Object.entries(r)) {
      if (k === "_map") { console.log("map:", JSON.stringify(v)); continue; }
      console.log(`${k}: inside water ${(v.inside * 100).toFixed(1)}%  outside land ${(v.outside * 100).toFixed(1)}%  (${v.n} edges)`);
      if (v.badIn.length) console.log("   first inside misses [x, y, class]:", JSON.stringify(v.badIn));
      if (v.badOut.length) console.log("   first outside misses [x, y, class]:", JSON.stringify(v.badOut));
      if (k === "Diversey Harbor" && (v.inside < 0.99 || v.outside < 0.95)) fail(`${k} edge: need >= 99% / 95%`);
    }
  }
}

if (has("--harbor-waves")) {
  const r = await page.evaluate(() => { const W = window.__world; const [x, y] = W.ll(41.9295, -87.6335);
    const c = W.conditions ? W.conditions(x, y, Date.now(), true) : null; return c && { systems: c.systems, src: c.src, harbor: c.harbor, depth: c.depth }; });
  console.log("basin:", JSON.stringify(r));
}

if (has("--eval")) {   // run a JS expression file in the page (window.__world is the page's API) and print its JSON
  const r = await page.evaluate(readFileSync(opt("--eval"), "utf8")); console.log(JSON.stringify(r, null, 1));
}

if (has("--shot")) {
  const v = VIEWS[opt("--shot")] || opt("--shot").split(",").map(Number); const out = args[args.indexOf("--shot") + 2] || "shot.png";
  await page.evaluate(([la, lo, d, h, p]) => { const W = window.__world; const [x, y] = W.ll(la, lo); W.flyTo(x, y, d, h, p, true); }, v);
  const ov = opt("--ov"); if (ov) await page.evaluate((k) => { const b = document.querySelector(`[data-ov="${k}"]`); if (b) b.click(); }, ov);
  await page.waitForTimeout(4000);
  await page.screenshot({ path: out, timeout: 150000 }); console.log("wrote " + out);
}

if (logs.length) console.log("console:", logs.slice(0, 12).join("\n  "));
await browser.close(); server.close();
process.exit(failed ? 1 : 0);
