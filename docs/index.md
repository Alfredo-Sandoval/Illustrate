---
hide:
  - navigation
  - toc
---

<div class="hero-stage">
  <div class="hero" markdown>
    <h1>Illustrate</h1>
    <p>Render PDB structures as shaded spheres with the desktop app, CLI, or Python.</p>

    <div class="buttons">
      <a href="getting-started/" class="button primary">Get Started</a>
      <a href="https://github.com/Alfredo-Sandoval/Illustrate" class="button secondary">GitHub</a>
    </div>
  </div>

  <div class="hero-render">
    <img src="images/01_classic_goodsell.png" alt="Illustrate render of hemoglobin">
    <p class="hero-render-caption">Hemoglobin (PDB 2HHB)</p>
  </div>
</div>

<div class="preset-gallery">
  <div class="preset-thumb" data-href="guide/presets/">
    <img src="images/03_cool_blues.png" alt="Cool Blues">
    <span>Cool Blues</span>
  </div>
  <div class="preset-thumb" data-href="guide/presets/">
    <img src="images/05_high_contrast.png" alt="High Contrast">
    <span>High Contrast</span>
  </div>
  <div class="preset-thumb" data-href="guide/presets/">
    <img src="images/06_pen_and_ink.png" alt="Pen &amp; Ink">
    <span>Pen &amp; Ink</span>
  </div>
  <div class="preset-thumb" data-href="guide/presets/">
    <img src="images/04_earth_tones.png" alt="Earth Tones">
    <span>Earth Tones</span>
  </div>
</div>

<script>
document.querySelectorAll('.preset-thumb[data-href]').forEach(function(el) {
  el.style.cursor = 'pointer';
  el.addEventListener('click', function() { location.href = el.getAttribute('data-href'); });
});
</script>

<p class="preset-browse-link">
  <a href="guide/presets/">Using presets &rarr;</a>
</p>
