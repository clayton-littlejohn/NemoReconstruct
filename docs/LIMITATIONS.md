# Known Limitations

## DGX Spark / GB10 (aarch64) Platform

### USDZ Conversion — Stub `pxr` Package in fVDB Env

**Affects:** fVDB workflows only (fixed)

The fVDB conda environment (Python 3.12, aarch64) ships a stub `pxr` package —
`usd-core` is not available for this platform/Python combination. `Usd.Stage.CreateInMemory()`
does not exist in the stub, causing `frgs convert` (PLY → USDZ) to crash with
`AttributeError`.

- **Fix applied:** A standalone PLY → USDZ converter (`backend/app/services/convert_ply_to_usdz.py`)
  runs under the 3DGRUT conda environment (Python 3.11), which has a working `usd-core 25.11`.
- **3DGRUT unaffected:** 3DGRUT produces USDZ natively during training (`export_usdz.enabled=true`).

### PyTorch Compute Capability Warning

PyTorch warns that CC 12.1 exceeds its maximum supported CC 12.0. JIT-compiled CUDA
extensions (e.g., 3DGRUT's ray-tracing kernels) still build and run correctly despite
the warning.

---

## Collision Mesh

Both fVDB and 3DGRUT workflows use alpha-shape collision mesh generation on Gaussian
centroids via `trimesh`. This produces a usable collision mesh (~50K faces) suitable
for Isaac Sim physics.

---

## Ollama on DGX Spark

`nemotron-3-nano` crashes during model load with a GGML CUDA assertion
(`ggml_nbytes(src0) <= INT_MAX`). Use `glm-4.7-flash` instead for local inference.
