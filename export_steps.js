#!/usr/bin/env node
// Capture all narration-step states of interactive slides as vector PDFs.
// Uses page.pdf() so Plotly SVG charts are embedded as vectors, not pixels.
// Run via:  node export_steps.js   (requires export_pdf.sh to have been run once)
//
// Dependencies: puppeteer (from decktape npx cache), pdfunite (poppler)

'use strict';

const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const { spawnSync } = require('child_process');

// ── Find puppeteer from the decktape npx cache ──────────────────────────────
const PUPPETEER_PATH = (() => {
  const base = path.join(process.env.HOME, '.npm', '_npx');
  for (const dir of fs.readdirSync(base)) {
    const p = path.join(base, dir, 'node_modules', 'puppeteer');
    if (fs.existsSync(p)) return p;
  }
  throw new Error('puppeteer not found in npx cache — run export_pdf.sh first');
})();
const puppeteer = require(PUPPETEER_PATH);

const PORT   = 8767;
const SLIDES = path.resolve(__dirname);
const TMP    = path.join(SLIDES, '_step_pdfs');
const W = 1600, H = 900;

// ── Interactive slides to capture ────────────────────────────────────────────
// slideIndex: 0-based horizontal slide index in Reveal.js
// buttonId:   id of the "Next" button to click
// nClicks:    number of button presses (total pages = nClicks + 1)
// outName:    output PDF filename
// waitMs:     ms to wait for Plotly to re-render after each click
const STEP_SLIDES = [
  {
    slideIndex: 16,
    buttonId:   'lr-story-next',
    nClicks:    4,
    outName:    'lecture1_slide17_steps.pdf',
    waitMs:     1800,
  },
];

// ── Helpers ──────────────────────────────────────────────────────────────────
function startServer() {
  const mime = { '.html':'text/html', '.js':'application/javascript',
                 '.css':'text/css',   '.png':'image/png', '.jpg':'image/jpeg' };
  const srv = http.createServer((req, res) => {
    const fp = path.join(SLIDES, req.url.split('?')[0]);
    if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
      res.writeHead(404); res.end(); return;
    }
    res.writeHead(200, { 'Content-Type': mime[path.extname(fp)] || 'application/octet-stream' });
    fs.createReadStream(fp).pipe(res);
  });
  srv.listen(PORT);
  return srv;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function capturePdf(page, file) {
  const buf = await page.pdf({
    width:  `${W}px`,
    height: `${H}px`,
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
  });
  fs.writeFileSync(file, buf);
  console.log('  captured', path.basename(file));
}

function mergePdfs(inputs, outPath) {
  const r = spawnSync('pdfunite', [...inputs, outPath], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error('pdfunite failed:\n' + r.stderr);
  console.log('  merged →', path.basename(outPath));
}

// ── Main ─────────────────────────────────────────────────────────────────────
(async () => {
  if (!fs.existsSync(TMP)) fs.mkdirSync(TMP);

  const server = startServer();
  await sleep(500);

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', `--window-size=${W},${H}`],
  });

  try {
    for (const slide of STEP_SLIDES) {
      console.log(`\nCapturing slide ${slide.slideIndex + 1} (${slide.nClicks + 1} pages) → ${slide.outName}`);

      const page = await browser.newPage();
      await page.setViewport({ width: W, height: H });
      await page.goto(`http://localhost:${PORT}/lecture1.html`, { waitUntil: 'networkidle2', timeout: 30000 });

      await page.waitForFunction('window.Reveal && Reveal.isReady()');
      await page.evaluate(idx => Reveal.slide(idx, 0, -1), slide.slideIndex);
      await sleep(slide.waitMs * 1.5);  // extra wait for initial Plotly render

      const pdfs = [];

      const f0 = path.join(TMP, `step0.pdf`);
      await capturePdf(page, f0);
      pdfs.push(f0);

      for (let i = 1; i <= slide.nClicks; i++) {
        await page.click(`#${slide.buttonId}`);
        await sleep(slide.waitMs);
        const fi = path.join(TMP, `step${i}.pdf`);
        await capturePdf(page, fi);
        pdfs.push(fi);
      }

      await page.close();
      mergePdfs(pdfs, path.join(SLIDES, slide.outName));
    }
  } finally {
    await browser.close();
    server.close();
    for (const f of fs.readdirSync(TMP)) fs.unlinkSync(path.join(TMP, f));
    fs.rmdirSync(TMP);
  }
})();
