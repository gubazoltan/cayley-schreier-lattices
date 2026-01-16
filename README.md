# Cayley-Schreier Lattices

[![arXiv](https://img.shields.io/badge/arXiv-2509.25316-b31b1b.svg)](https://arxiv.org/abs/2509.25316)

Code and data for the research paper  
**“Topological non-abelian lattice gauge structures in Cayley-Schreier lattices.”**

---

## Installation (conda)

Create the environment from the provided `environment.yml` (includes Python and Jupyter tooling):
```bash
conda env create -f environment.yml
conda activate cayley-schreier-env
```

Make the project importable (so notebooks can `import cayley_schreier`):

```bash
python -m pip install -e .
```

Run that command from the repository root (the folder containing `pyproject.toml`).

---

## Reproducing Figures

Run the project notebooks (located in `notebooks/`) to regenerate all figures and data. Execute each notebook top-to-bottom.

---

## License

MIT License (see `LICENSE`).

---
