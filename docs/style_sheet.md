# ML4APP lecture deck — style sheet

Conventions distilled from the lecture1b.html rework. As of June 2026 the
shared CSS and boot JS live in `docs/lib/`, so a new lecture only has to
include three small references and a `data-lecture` attribute. Items marked
**[required]** must match exactly; **[guideline]** items can flex if a slide
genuinely needs to.

---

## 1. Page boilerplate

### `<head>` [required]

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lecture N: <descriptive title></title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/white.css">
<link rel="stylesheet" href="lib/lecture.css">
<!-- Optional: per-lecture <style> block for additions on top of lecture.css -->
</head>
<body data-lecture="N">
```

The `data-lecture` attribute sets the footer label (`Lecture N`). Lectures
1b, 2, 3 use this pattern.

### Script tags at end of `<body>` [required]

```html
<script src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.0/dist/plotly.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/plugin/math/math.js"></script>
<script src="lib/lecture-helpers.js"></script>
<script src="lib/lecture-boot.js"></script>
<script>
  /* per-lecture demo code here */
</script>
```

Load order matters: Plotly → Reveal → math plugin → helpers → boot →
lecture code. `lecture-boot.js` calls `Reveal.initialize`, installs the
MathJax retypeset hook, and attaches the footer.

### What's in `lib/`

- **`lib/lecture.css`** — all shared `.reveal`/`.box`/`.slide-footer`/
  `.ctrl-row` rules. Section 2 below reproduces the contents for reference.
- **`lib/lecture-boot.js`** — `Reveal.initialize`, `slidechanged` MathJax
  retypeset hook, footer injector (reads `<body data-lecture="...">`).
- **`lib/lecture-helpers.js`** — `createRNG`, `gaussN`, `matVec`, `solve`,
  `polyDesign`, `polyEval`, `fitPoly`. Per-lecture `baseLayout` and demo
  constants stay in the lecture file.

Do NOT pass a custom `math:` config to `Reveal.initialize` — leave MathJax
3 on its default CHTML output. SVG output and `mjx-container` CSS
overrides BOTH made math glitch worse, not better, in our experiments.

---

## 2. Stylesheet block (reference)

This is the content of `lib/lecture.css`. New lectures should **not**
inline it — link the file. The block is reproduced here so the conventions
are visible without opening another file.

```css
.reveal .slides section {
  background: #ffffff;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  padding: 30px 40px;
  box-sizing: border-box;
  height: 700px !important;
}
.reveal .slides section.stack {
  background: none;
  box-shadow: none;
  padding: 0;
  height: auto !important;
}
.reveal { font-family: 'Inter', sans-serif; font-weight: 400; }
.reveal h1, .reveal h2, .reveal h3 { color: #2c3e50; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
.reveal h2 { font-size: 1.2em; margin-bottom: 0.3em; text-align: left; }
.reveal .subtitle { font-size: 0.6em; color: #7f8c8d; margin-top: 0.5em; }
.reveal section { font-size: 0.66em; }
.reveal section.stack { font-size: 1em; }
.reveal p { margin: 0.3em 0; }
.reveal ul { text-align: left; margin: 0.3em 0; }
.reveal li { margin: 0.15em 0; }
.reveal .highlight { color: #e74c3c; font-weight: bold; }
.reveal .MathJax { font-size: 0.95em; }
.reveal .box, .reveal .box-blue, .reveal .box-orange,
.reveal .box-green, .reveal .box-purple {
  background: #ecf0f1; border-radius: 10px; padding: 12px 16px;
  margin: 10px 0; text-align: left;
}
.reveal .box-blue   { border-left: 4px solid #2980b9; }
.reveal .box-orange { border-left: 4px solid #e67e22; }
.reveal .box-green  { border-left: 4px solid #27ae60; }
.reveal .box-purple { border-left: 4px solid #8e44ad; }
.slide-footer {
  position: absolute; bottom: 8px; left: 40px; right: 40px;
  font-size: 10px; color: #bdc3c7;
  display: flex; justify-content: space-between; pointer-events: none;
}
pre code { font-family: 'Menlo','Monaco',monospace; }
.ctrl-row { display:flex; gap:14px; font-size:0.9em; align-items:center; margin-top:4px; flex-wrap:wrap; }
.ctrl-row label { white-space: nowrap; }
.ctrl-row input[type=range] { vertical-align: middle; }
.ctrl-row button { padding:3px 12px; cursor:pointer; }
.ctrl-row input[type=checkbox] { vertical-align: middle; transform: scale(1.2); margin-right:4px; }
```

### What was removed deliberately

- `text-transform: none` override on headings: we keep Reveal white's default
  uppercase for h1/h2/h3.
- `.reveal mjx-container { ... }` overrides: caused more rendering glitches
  than they fixed; lec4 has none and works fine.
- Dark-red math color (`color: #8b1a1a`): math is now default black.
- Navy strong color (`color: #1a3a6e` on `<strong>`): bolds default to black.
- `.sec-btn`: roadmap tiles now use `.box .box-blue` (etc.) directly.

---

## 3. Color palette

| Use | Color |
|---|---|
| Headings | `#2c3e50` (dark slate) |
| Subtle text, captions | `#7f8c8d` |
| Section accent — Section 1 | `#2980b9` (blue) |
| Section accent — Section 2 | `#e67e22` (orange) |
| Section accent — Section 3 | `#27ae60` (green) |
| Section accent — Section 4 | `#8e44ad` (purple) |
| Red highlight / observed data marker | `#e74c3c` / `#c0392b` |
| Box background | `#ecf0f1` |
| Slide background | `#ffffff` |

Red is reserved for the `highlight` class and the observed-data marker
(red ✕ on plots). Don't use red for ordinary emphasis — it cheapens both.

---

## 4. Required slide order

Every lecture deck contains, in this order:

1. **Title slide** [required]
2. **Today's Roadmap** [required]
3. **Section divider** + content slides for Section 1
4. (Section divider) + content for Section 2, etc.
5. **Key Takeaways** [required]
6. **Summary & What's Next** [required]

### 4.0 Slide title convention [required]

`h1`, `h2`, `h3` are prose only — **no MathJax expressions in titles.** Equations
belong in slide bodies, not headings. Reveal's white-theme uppercase mangles
`\(...\)` glyphs, MathJax re-typesetting is slower per-nav when headings carry
math, and a sentence-form title reads cleanly in a table of contents.

If a section-divider title is long, prefer `font-size: 1.6em` with `<br>` to wrap
rather than shrinking further:

```html
<h1 style="font-size:1.8em; margin:0;">Going beyond univariate parameters<br>and homoscedastic noise</h1>
```

Aim for 3–6 words on body slides; section dividers can be a touch longer.

### 4.1 Title slide [required]

```html
<section>
  <h1 style="font-size:1.5em;">Machine Learning for Astroparticle Physics: <br> A Crash-course in SBI</h1>
  <h3>Lecture N - <Topic>></h3>
  <p class="subtitle">Christoph Weniger &mdash; University of Amsterdam (GRAPPA)</p>
</section>
```

Course tagline lives in the h1, not in a separate subtitle. The only
`.subtitle` line on the title slide is the author affiliation.

### 4.2 Today's Roadmap [required]

Three section tiles separated by → arrows, plus a "thread" callout.
Use `box box-blue/orange/green` (NOT `.sec-btn`).

```html
<section>
  <h2>Today's Roadmap</h2>
  <div style="display:flex; gap:8px; margin-top:20px;">
    <div class="box box-blue" style="flex:1; text-align:center;">
      <strong style="color:#2980b9;">1</strong><br>
      <Section 1 title><br><span style="font-size:0.85em;"><one-line scope></span>
    </div>
    <div style="align-self:center; font-size:1.4em; color:#bdc3c7;">&rarr;</div>
    <div class="box box-orange" style="flex:1; text-align:center;">
      <strong style="color:#e67e22;">2</strong><br>
      <Section 2 title><br><span style="font-size:0.85em;"><scope></span>
    </div>
    <div style="align-self:center; font-size:1.4em; color:#bdc3c7;">&rarr;</div>
    <div class="box box-green" style="flex:1; text-align:center;">
      <strong style="color:#27ae60;">3</strong><br>
      <Section 3 title><br><span style="font-size:0.85em;"><scope></span>
    </div>
  </div>
  <div class="box fragment" style="margin-top:25px; text-align:center; font-size:1.0em;">
    <strong>The thread:</strong> <one-sentence narrative arc through the three sections>.
  </div>
</section>
```

If you have four sections, add a fourth tile with `box box-purple` and a
purple `<strong>`. Don't try to fit five — split the lecture instead.

**Section titles must match exactly across four touchpoints:** the roadmap tile,
the section divider h1, the Key Takeaways bullet header, and the Summary & What's
Next box header. Renaming a section means updating all four — easy to miss.

### 4.3 Section divider [required, one per section]

A full-bleed centered card with gradient bars and a section title.

```html
<section>
  <div style="display:flex; align-items:center; justify-content:center; height:100%;">
    <div style="text-align:center;">
      <div style="width:200px; height:4px; background:linear-gradient(90deg,#2980b9,#e74c3c); margin:0 auto 25px;"></div>
      <h1 style="font-size:2em; margin:0;"><Section title></h1>
      <p style="font-size:1.1em; color:#7f8c8d; margin-top:15px;"><one-line tagline>.</p>
      <div style="width:200px; height:4px; background:linear-gradient(90deg,#2980b9,#e74c3c); margin:25px auto 0;"></div>
    </div>
  </div>
</section>
```

Gradient direction and colors are the same on every divider — visual
consistency, not section coding.

#### Transition slides (NOT section dividers) [guideline]

Reserve the section-divider pattern (gradient bars + h1 + tagline) for top-level
transitions that match the roadmap. For *rhetorical pivots inside a section*
("now what if we stack more layers?"), use a title-free centered prose slide —
no bars, no h1, just one centred sentence:

```html
<section>
  <div style="display:flex; align-items:center; justify-content:center; height:100%;">
    <p style="font-size:1.4em; color:#2c3e50; text-align:center; max-width:820px; margin:0;">
      One hidden layer is a universal approximator. But what happens when we stack more?
    </p>
  </div>
</section>
```

Promoting a rhetorical pivot to a section divider misleads the reader about the
deck's structure — the roadmap promises N sections and they should see exactly N
dividers.

### 4.4 Key Takeaways [required]

Fragmented bullets, one per section + a closing conceptual point.

```html
<section>
  <h2>Key Takeaways</h2>
  <ul style="font-size:0.92em; line-height:1.6;">
    <li class="fragment"><strong><Section 1 name>:</strong> <one-sentence what-and-why>.</li>
    <li class="fragment"><strong><Section 2 name>:</strong> ...</li>
    <li class="fragment"><strong><Section 3 name>:</strong> ...</li>
    <li class="fragment"><strong>The conceptual jump:</strong> <bridge to the next lecture>.</li>
  </ul>
</section>
```

### 4.5 Summary & What's Next [required]

Three boxes (one per section), key equations strip, one-line handoff to
the next lecture.

```html
<section>
  <h2>Summary &amp; What's Next</h2>
  <div style="display:flex; gap:15px; font-size:0.9em;">
    <div class="box" style="flex:1;">
      <h3 style="margin-top:0; color:#2980b9;"><Section 1 name></h3>
      <ul style="font-size:0.9em; line-height:1.5;">
        <li><bullet></li><li><bullet></li><li><bullet></li>
      </ul>
    </div>
    <div class="box" style="flex:1;">
      <h3 style="margin-top:0; color:#e67e22;"><Section 2 name></h3>
      <ul>...</ul>
    </div>
    <div class="box" style="flex:1;">
      <h3 style="margin-top:0; color:#27ae60;"><Section 3 name></h3>
      <ul>...</ul>
    </div>
  </div>
  <div class="box" style="text-align:center; margin-top:12px; padding:8px; font-size:0.85em;">
    <strong>Key equations:</strong> &nbsp;
    <eq1> &nbsp;|&nbsp; <eq2> &nbsp;|&nbsp; <eq3>
  </div>
  <p style="text-align:center; margin-top:10px; font-size:0.85em; color:#7f8c8d;">
    Next lecture: <one-line preview>.
  </p>
</section>
```

---

## 5. Box conventions

### Plain box

For prose callouts, probability models, definitions, etc.

```html
<div class="box" style="margin:8px auto; max-width:520px; text-align:center;">
  \[ <equation> \]
</div>
```

For full-width prose boxes drop `max-width` and `margin:auto`.

### Coloured-border box

`box-blue / orange / green / purple` add a single 4 px left border. Use
sparingly: at most one or two per slide. Don't mix more than two colours
on a slide.

```html
<div class="box box-blue" style="padding:8px 12px;">
  Body text.
</div>
```

The lec4 convention (inline `border-left:4px solid #...` instead of a
class) also works; classes are cleaner.

### "Key equations" recap box

Used only on the Summary & What's Next slide. Single line, equations
separated by `&nbsp;|&nbsp;`.

---

## 6. Two-column slides with a Plotly figure [guideline]

The recipe that survives Reveal scaling without clipping or overflow:

```html
<div style="display:flex; gap:14px;">
  <div style="flex:0 0 32%; text-align:left;">
    <!-- text, equations, notes -->
  </div>
  <div style="flex:1 1 0; min-width:0; overflow:hidden;">
    <div id="myplot" style="width:100%; max-width:100%; height:380px;"></div>
    <div class="ctrl-row" style="justify-content:center; margin-top:8px;">
      <label>... slider ...</label>
    </div>
    <div class="ctrl-row" style="justify-content:center; margin-top:6px;">
      <button>... resample ...</button>
    </div>
  </div>
</div>
```

Key invariants:

- Left column: explicit `flex:0 0 X%` and `text-align:left` (otherwise
  prose inherits Reveal's center alignment).
- Right column: `flex:1 1 0; min-width:0; overflow:hidden`. The
  `min-width:0` is essential — without it, the Plotly SVG's intrinsic
  width overflows the column.
- Plot div: `width:100%; max-width:100%`.
- Controls live BELOW the plot, each row centered. Slider in its own
  row, buttons in their own row. Don't cram sliders and buttons together.

### Plotly call [required]

Always include `autosize:true` in the layout, `responsive:true` in the
config, and follow `Plotly.react` with `Plotly.Plots.resize`:

```js
Plotly.react('myplot', traces, baseLayout({
  autosize: true,
  // ... rest of layout
}), {displayModeBar: false, responsive: true});
Plotly.Plots.resize('myplot');
```

Without these, the plot sometimes initialises at the wrong width when the
slide was hidden at page load.

---

## 7. Controls

### .ctrl-row

```html
<div class="ctrl-row" style="justify-content:center; margin-top:8px;">
  <label>
    <input type="range" id="my-slider" min="0" max="1" step="0.01" value="0.5"
           style="width:220px; vertical-align:middle;">
    &nbsp;\(\theta\) = <span id="my-slider-val" style="display:inline-block; min-width:2.4em;">0.5</span>
  </label>
</div>
```

### Buttons

Use the `↻` glyph (`&#x21bb;`) for resample. Keep labels short:

```html
<button id="my-resample">&#x21bb; Resample v &amp; noise</button>
```

When there are two resample buttons (e.g. partial vs full resample), put
them on the SAME centered row.

---

## 8. Math

- Default MathJax 3 (CHTML). No config, no CSS overrides on
  `mjx-container`. Re-typeset on `slidechanged` (see §1).
- Use `\(...\)` for inline math, `\[...\]` for display.
- Use `\mathrm{}` for multi-letter subscripts: `\sigma_\mathrm{obs}`.
- Prefer symbolic constants over numeric values in equations; quote the
  numbers in a small grey line beneath the box:
  ```
  \[ p(v) = \mathcal{N}(v; \mu_v, \sigma_v^2) \]
  <p style="font-size:0.85em; color:#555; margin:4px 0 0;">
    with \(\mu_v = 10\) m/s, \(\sigma_v = 0.2\) m/s.
  </p>
  ```
- Include units inside `\mathcal{N}` arguments when the equation stands
  alone: `\mathcal{N}(v; 10\ \text{m/s}, (0.2\ \text{m/s})^2)`.
- Centre display equations by putting them in a `.box` with
  `max-width:` and `margin:... auto`.

### Parameter notation [guideline]

When the workflow is closed-form linear algebra (linear-basis MLE), use the
familiar `\(\mathbf{w}_{\mathrm{ML}} = (\boldsymbol\Phi^T\boldsymbol\Phi)^{-1}\boldsymbol\Phi^T \boldsymbol\theta\)`.
When the workflow is gradient descent, bundle every trainable quantity into one
parameter vector and write the update with `\(\nabla_{\!\boldsymbol\phi}\)`:

```
\[ \boldsymbol\phi = (\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \sigma_\theta, \ldots) \]
\[ \boldsymbol\phi \leftarrow \boldsymbol\phi - \eta\,\nabla_{\!\boldsymbol\phi}\,E(\boldsymbol\phi) \]
```

Mixing the two notations in the same lecture is fine *if* the transition is
named. Naming the transition (e.g. on a "No Closed-Form Solution" slide) is the
moment to swap from `\(\mathbf{w}\)` to `\(\boldsymbol\phi\)`.

---

## 9. Plotly conventions

### Shared layout helper

Every script defines:

```js
const LAY_BASE = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{size:13}, margin:{t:30, r:20, b:40, l:55},
  legend:{x:0.02, y:0.98, bgcolor:'rgba(255,255,255,0.6)'}
};
function baseLayout(opts={}) { return Object.assign({}, LAY_BASE, opts); }
```

### Observed-data marker [required]

A red ✕ at the observed value:

```js
{ x:[x_obs], y:[0], mode:'markers',
  name:'x_obs = '+x_obs.toFixed(2)+' m',
  marker:{size:18, color:'#c0392b', symbol:'x-thin',
          line:{color:'#c0392b', width:4}} }
```

Don't use a filled red circle. Don't surround it with a ±σ band. The cross
is the observation; the uncertainty lives in the model.

### Reference / posterior curve

Red line `#c0392b`, width 3, filled to zero with low-alpha red:

```js
{ x:THETA_GRID, y:posterior, mode:'lines',
  line:{color:'#c0392b', width:3},
  fill:'tozeroy', fillcolor:'rgba(192,57,43,0.12)' }
```

### Simulation cloud / accepted samples

- All sims: small grey markers, `marker:{size:3, color:'rgba(127,140,141,0.45)'}`
- Accepted subset: coloured (orange for "kept" in rejection ABC, green for
  the summary branch in Demo 2, etc.), size 5–8.

### Uncertainty bands [required]

Convention: **±σ** (not ±2σ). Light grey fill `rgba(127,140,141,0.25)`. Drawn
**first** in the Plotly trace array so the band sits behind data markers,
ground-truth dashes, and the fit line.

```js
Plotly.react('myplot', [
  { x: xLine.concat(xLine.slice().reverse()),
    y: upper.concat(lower.slice().reverse()),
    fill:'toself', fillcolor:'rgba(127,140,141,0.25)', line:{width:0},
    name:'fit ± σ', hoverinfo:'skip' },
  // ... then truth, data, fit ...
], ...);
```

Same convention for both fixed-σ bands (constant width) and learned σ(x) bands
(envelope follows the curve). When the band is per-x, draw it as
`upper.concat(lower.slice().reverse())` of the *exact same x grid* twice.

---

## 10. Interactive demos — lifecycle and import [required]

Interactive demos are where decks accumulate technical debt fastest. Three rules
catch the recurring landmines.

### 10.1 IIFEs must self-clean on `slidechanged`

If a demo schedules a `setTimeout(trainLoop, …)` that reschedules itself, the
*leave-the-slide* branch is essential. Without it, pressing **Train** and
navigating away leaks CPU until the deck is reloaded — visible as transitions
getting visibly chunkier over a long session.

```js
function init(e) {
  if (e.currentSlide && e.currentSlide.querySelector('#my-plot')) {
    reset();                              // entering the slide
  } else if (animId) {
    clearTimeout(animId); animId = null;  // leaving the slide
  }
}
Reveal.on('slidechanged', init);
```

Same pattern with `clearInterval(timer)` if the demo uses `setInterval`.

### 10.2 Top-level DOM lookups in IIFEs are landmines

Lines like

```js
document.getElementById('foo').addEventListener('input', draw);
```

at the top level of an IIFE throw a `TypeError` if `#foo` is no longer in the
DOM (e.g. you deleted the slide). Because IIFEs evaluate at script-block load,
**one missing element kills every IIFE that comes after it in the same
`<script>` block.** Symptom: completely unrelated demos stop rendering.

When you delete a slide, *audit its JS counterpart*: either remove the IIFE
entirely or wrap the lookups in `if (el) el.addEventListener(...)`.

### 10.3 Importing demos from other decks

Before pasting an IIFE from `old/lecture*.html` (or another deck):

1. **Grep for ID collisions:**
   ```bash
   grep "id=['\"]<prefix>" target.html
   ```
   If any of the imported IDs collide, prefix the imports (e.g. `mlpreg-*` →
   `mlpreg2-*`) and rename them in both HTML and JS.

2. **Alias differing helper names** with one line rather than touching every
   call site. Example: old/lecture3 uses `boxMuller(rng)`; lecture2 has the
   same algorithm under `gaussN(rng)`. At the top of the imported JS block:

   ```js
   var boxMuller = gaussN;  // alias so imported demos run unmodified
   ```

3. **Skip the `Details` and `Pseudocode` vertical sub-slides** unless you
   actually want them. The originals were lecture-3 specific.

---

## 11. Writing style

- **No em dashes** in prose. Use commas, parentheses, semicolons, or
  separate sentences instead. (Em dashes in bullet-list separators are
  fine.)
- Concise titles. Trim to 3–6 words where possible.
- Drop defensive verbiage. Lines like "this is correct, not a bug" or
  "as expected" undermine the slide; if a result needs defending,
  rephrase the slide.
- Avoid "we" + abstraction. Prefer "the simulator samples θ" over
  "we sample θ from the simulator".
- Captions under plots: small (`font-size:0.85em`), grey
  (`color:#7f8c8d`), centered, no period at end if it's a fragment.
- No emojis anywhere unless the user explicitly asks.

---

## 12. Per-lecture checklist

When porting an existing deck to this style, walk through:

- [ ] CSS block matches §2 exactly
- [ ] `Reveal.initialize` matches §1 (no `math:` config, no CHTML→SVG switch)
- [ ] MathJax re-typeset hook present (§1)
- [ ] Footer label has correct lecture number (§1)
- [ ] Title slide reformatted per §4.1
- [ ] **No LaTeX in slide titles** (§4.0)
- [ ] Today's Roadmap slide added (§4.2)
- [ ] One section divider per section (§4.3); rhetorical pivots inside a
      section use title-free transition slides, NOT dividers (§4.3)
- [ ] Section titles identical across roadmap, divider, Key Takeaways,
      Summary boxes (§4.2)
- [ ] Key Takeaways slide added (§4.4)
- [ ] Summary & What's Next slide added (§4.5)
- [ ] All boxes use `.box` or `.box-*`, no inline `background:#f4f6f8`
- [ ] No `text-transform: none` override on headings
- [ ] Two-column slides follow the flex recipe (§6)
- [ ] All Plotly calls include `autosize`, `responsive`, and
      `Plotly.Plots.resize` (§6)
- [ ] Red ✕ for observed data, red curve for reference posterior (§9)
- [ ] **Uncertainty bands are ±σ, light grey, drawn first in trace order** (§9)
- [ ] **Interactive demos clear timers on `slidechanged` (leave branch)** (§10.1)
- [ ] **No orphan IIFEs after deleting slides; top-level
      `getElementById(...).addEventListener` guarded or removed** (§10.2)
- [ ] When SGD is in play, use `φ` for the parameter bundle and `∇_φ E` (§8)
- [ ] Em dashes purged from prose (§11)

---

## 13. Known-good reference

`docs/lecture1b.html` is the canonical reference implementation. When in
doubt, copy from there.

`docs/lec4.html` is the source the style was distilled from but predates
some refinements (e.g. it uses `.sec-btn` for the roadmap; we standardised
on `.box box-blue` etc. afterwards). Use lecture1b as the template, not
lec4.
