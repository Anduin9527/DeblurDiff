# DeblurDiff Python/ML Development Guidelines

> Project-specific conventions for the DeblurDiff research codebase.

---

## Overview

DeblurDiff is a Python/PyTorch image restoration research repository, not a
web backend. This `backend/` layer is retained because Trellis uses that name
for non-frontend implementation context; treat it as the authoritative
Python/ML code-spec layer for this project.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Architecture Map](./architecture-map.md) | Paper-to-code mapping and model execution flow | Filled |
| [Directory Structure](./directory-structure.md) | Repository layout and ownership boundaries | Filled |
| [Quality Guidelines](./quality-guidelines.md) | Research-code quality, verification, and known caveats | Filled |

---

## Pre-Development Checklist

- Read [Architecture Map](./architecture-map.md) before changing model,
  diffusion, sampler, inference, or training behavior.
- Read [Directory Structure](./directory-structure.md) before adding files or
  moving code.
- Read [Quality Guidelines](./quality-guidelines.md) before modifying training,
  inference, configs, or CUDA/CuPy paths.
- If changing cross-file contracts such as checkpoint format, config schema,
  dataset tuple shape, or inference CLI arguments, also read
  `../guides/cross-layer-thinking-guide.md`.

## Removed Template Layers

The initial Trellis fullstack bootstrap generated frontend, database, API
error-handling, and logging specs. They were removed because this repository
does not contain a frontend app, database layer, API server, or structured
logging framework. Recreate a dedicated spec only when such a subsystem is
actually added.

---

**Language**: All project spec documentation should be written in **English**.
