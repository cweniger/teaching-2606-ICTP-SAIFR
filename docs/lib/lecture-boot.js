/* ML4APP lecture deck — Reveal boot + footer + MathJax retypeset hook.
   Identical across lectures; the per-lecture label is read from
   <body data-lecture="1b"> (or whatever the file sets). */

Reveal.initialize({
  hash: true, center: false, transition: 'none', slideNumber: true,
  width: 1214, height: 700, margin: 0.04, minScale: 0.2, maxScale: 2.0,
  plugins: [ RevealMath.MathJax3 ]
});

// MathJax3 typesets all slides at page load using whatever container
// dimensions exist at that moment. Slides not visible then cache wrong
// glyph positions and render clipped. Re-typeset the becoming-current
// slide on every navigation.
Reveal.on('slidechanged', function(e) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([e.currentSlide]).catch(function(){});
  }
});

(function attachFooters(){
  const label = (document.body.dataset.lecture || '').trim();
  const lectureText = label ? ('ML4APP 2026 &mdash; Lecture ' + label) : 'ML4APP 2026';
  document.querySelectorAll('.reveal .slides > section').forEach(function(slide) {
    const footer = document.createElement('div');
    footer.className = 'slide-footer';
    footer.innerHTML =
      '<a href="index.html" style="color:#bdc3c7; text-decoration:none; pointer-events:auto;">&larr; Index</a>' +
      '<span>' + lectureText + '</span>' +
      '<span>Christoph Weniger &mdash; GRAPPA / UvA</span>';
    slide.appendChild(footer);
  });
})();
