# Texture Padding Lab

Small R&D project for exploring UV texture padding algorithms.

This repository focuses on understanding padding behavior and underlying
image-processing concepts, rather than delivering a final
production-ready solution.

The goal is to study how different approaches behave, compare results,
and build a solid mental model before moving toward more complex methods.

---

## uv_pad_nearest

Baseline UV padding implementation using **nearest island pixels**
(Voronoi-style behavior).

This version is intentionally simple and deterministic, serving as a
reference point for all further experiments.

### Characteristics

- Island is defined by `alpha > 0`
- Each background pixel copies color from the nearest island pixel
- Distance measured in pixel units (L2 / Euclidean)
- No blur
- No diffusion
- No heuristic tweaks

---

## Repository Structure
core/
uv_pad_nearest.py # baseline algorithm (kept stable)

examples/
input.png # sample RGBA input
output_nearest_r64.png # output with pad radius = 64 px


- `core/` contains minimal, stable reference implementations.
- `examples/` contains input/output images used only to document behavior.

---

## Usage

```bash
python core/uv_pad_nearest.py \
  -i examples/input.png \
  -o output.png \
  --pad-radius 64
```
## Notes

- The implementation under core/ should not be modified directly.

- All variations, experiments, and alternative approaches should be
implemented as separate files.

- Example images are provided only as behavioral reference, not as
visual targets.

- This repository is treated as an exploration log rather than a library.