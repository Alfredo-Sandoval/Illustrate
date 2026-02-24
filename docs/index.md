---
hide:
  - navigation
  - toc
---

<div class="hero-stage">
  <div class="hero" markdown>
    <h1>Illustrate</h1>
    <p>Molecular rendering with sphere-based geometry, soft shadows, and outlines.<br>
    Based on <a href="https://mgl.scripps.edu/people/goodsell/">David S. Goodsell</a>'s rendering algorithm.</p>

    <div class="buttons">
      <a href="getting-started/" class="button primary">Get Started</a>
      <a href="https://github.com/Alfredo-Sandoval/Illustrate" class="button secondary">GitHub</a>
    </div>
  </div>

  <div class="hero-render">
    <img src="images/01_classic_goodsell.png" alt="Hemoglobin rendered with the Default preset">
    <p class="hero-render-caption">Human hemoglobin (PDB 2HHB) - Default preset</p>
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
  <a href="guide/presets/">Browse all presets &rarr;</a>
</p>

## Quickstart

```bash
git clone https://github.com/Alfredo-Sandoval/Illustrate.git
cd Illustrate
pip install -e .
```

```python
from illustrate.fetch import fetch_pdb
from illustrate import render, write_png
from illustrate.presets import render_params_from_preset

pdb_path = fetch_pdb("2hhb")
params = render_params_from_preset("Default", str(pdb_path))
result = render(params)
write_png("hemoglobin.png", result.rgb)
```

## What It Does

Python implementation of the Goodsell sphere-based molecular renderer. Reads a PDB file, applies selection rules to assign colors and radii, and produces an image with fog, shadows, and outlines.

- Selection rules match atoms by record type, name pattern, and residue range. First match wins.
- 8 presets cover common styles (white/dark background, pen & ink, earth tones, etc.).
- Available as a CLI (`illustrate-py`), desktop GUI, Jupyter workflow, or FastAPI service.
