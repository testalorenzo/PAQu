<p align="center">
  <img src="PAQu.png" alt="PAQu logo" width="220"/>
</p>

<h1 align="center">PAQu — Protein Isoform Abundance Quantification</h1>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.64898/2026.04.20.719668v1"><img src="https://img.shields.io/badge/paper-biorXiv-red" alt="Paper"/></a>
  <a href="https://testalorenzo.github.io/PAQu_web/"><img src="https://img.shields.io/badge/website-PAQu-blue" alt="Website"/></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

---

## Overview

**PAQu** is a Bayesian method for estimating protein isoform abundances from mass-spectrometry proteomics data. It integrates mass spectrometry and gene expression data to resolve ambiguity when peptide sequences match multiple protein isoforms. The model decomposes observed peptide intensities into isoform-level signals using a Gibbs sampler, while accounting for sample covariates, transcript-level information, and condition effects. PAQu provides uncertainty quantification and a unified multiomics framework for improved accuracy.

It was applied to study protein isoform differences in schizophrenia, revealing elevated C4A levels — but not C4B — in affected individuals versus controls.

For more details, visit the [PAQu website](https://testalorenzo.github.io/PAQu_web/).

The method is described in:

> Testa L., Klei L., Rengle A., Yocum A., Lewis D.A., Devlin B., Roeder K., MacDonald M.L. (2026). *Estimating protein isoform abundances with PAQu*. biorXiv. https://doi.org/10.64898/2026.04.20.719668

---

## How it works

PAQu frames isoform quantification as a constrained matrix factorisation problem:

```
I  =  I₀  +  T × W  +  A × D  +  noise
P  ≈  I × Z  +  noise
```

| Symbol | Meaning | Dimensions | Estimated |
|--------|---------|------------|:---------:|
| **I** | Latent isoform abundances | n samples × q isoforms | ✅ |
| **I₀** | Baseline isoform abundance (intercept) | q isoforms | ✅ |
| **T** | Transcript-level expression covariates | n samples × q isoforms | ❌ |
| **W** | Transcript effect on isoform abundances | q isoforms | ✅ |
| **A** | Binary condition/treatment indicator | n samples | ❌ |
| **D** | Condition effect on isoform abundances | q isoforms | ✅ |
| **P** | Observed peptide intensities | n samples × r peptides | ❌ |
| **Z** | Peptide-to-isoform mapping weights | q isoforms × r peptides | ✅ |

Parameters are inferred via **Gibbs sampling** with conjugate priors. Posterior uncertainty is summarised through the **Local False Sign Rate (LFSR)**.

Optional inputs enrich the model:
- **Xᵢ** — sample-level covariates believed to affect isoform abundances
- **Xₚ** — sample-level covariates believed to affect peptide intensities

---

## Repository structure

| File | Description |
|------|-------------|
| `PAQu/` | Installable package — `PAQu` class (model + Gibbs sampler) |
| `vignette.ipynb` | Interactive walkthrough with pre-rendered figures |

### `application/`

| File | Description |
|------|-------------|
| `PAQu_application.py` | Real-data analysis pipeline |
| `format_upload.py` | Post-processing helper to format output files |

### `simulations/`

| File | Description |
|------|-------------|
| `synthetic_simulations.py` | Fully synthetic simulation study from the paper |

#### `simulations/realistic_simulations/`

Semi-synthetic simulation study using real proteomics data. Data files (`annotations.csv`, `P_values.csv`, `T_values.csv`, `meta_peptide.csv`) are expected two levels above this folder (i.e. at the repository root). Run scripts from within this folder.

| File | Description |
|------|-------------|
| `_blocks.py` | Shared helper: loads annotations, builds bipartite blocks and collapsed-isoform dictionaries |
| `_utils.py` | Shared helper: ROC curve utilities |
| `select_subset.py` | One-time step: selects a ~10,000-peptide block-aligned subset of the real dataset |
| `generate_data.py` | Per-replicate data perturbation: samples control subjects, draws ~15 % of isoforms as truly DE, and shifts their peptide intensities by D_size |
| `run_paqu.py` | Fits PAQu on every (seed, D_size) replicate |
| `run_isobayes.py` | Fits IsoBayes per sample on every replicate |
| `run_isobayes_persample.R` | R helper called by `run_isobayes.py`: fits IsoBayes to a single sample |
| `run_de_methods.R` | R helper: applies edgeR, DESeq2 and limma-voom to an IsoBayes-derived counts matrix |
| `run_peptide_baselines.py` | Computes naive peptide-abundance baselines (per-isoform t-test on average / sum / max of mapped peptides) |
| `compare_methods.py` | ROC analysis comparing PAQu, IsoBayes and peptide baselines |
| `compare_methods_combined.py` | Produces the combined 10-vs-10 and 5-vs-5 comparison figures |

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/testalorenzo/PAQu.git
```

Or clone and install in editable mode (recommended if you want to inspect or modify the source):

```bash
git clone https://github.com/testalorenzo/PAQu.git
cd PAQu
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `tqdm`) are installed automatically. Optional extras:

```bash
pip install "PAQu[plot]"        # matplotlib — for plot_convergence() and the vignette
pip install "PAQu[app]"         # pandas, networkx, joblib — for the application scripts
pip install "PAQu[plot,app]"    # everything
```

---

## Quick start

For a full walkthrough with figures, see [`vignette.ipynb`](vignette.ipynb).

```python
import numpy as np
from PAQu import PAQu

# Minimal inputs
n, q, r = 50, 3, 10           # samples, isoforms, peptides
P = np.random.randn(n, r)     # observed peptide intensities
M = np.ones((q, r))           # peptide-isoform mask (1 = compatible)
A = np.array([0]*25 + [1]*25) # condition label (0/1)

model = PAQu(A=A, M=M, P=P, T=q)   # T=q → isoform count (no transcript data)
model.fit(n_iter=2000, prior_D='Spike-and-Slab')

# Posterior summaries
I_hat             = model.I_storer[1000:].mean(axis=0)        # isoform abundances
D_hat             = model.D_storer[1000:].mean(axis=0)        # condition effects
lfsr_D            = model.LFSR('D_storer', burn_in=1000)      # local false sign rates
ci_lo, ci_hi      = model.credible_interval('D_storer', 1000) # 95% credible intervals

# Convergence diagnostics
fig = model.plot_convergence(burn_in=1000)
```

---

## Citation

If you use PAQu in your research, please cite:

```bibtex
@article{testa2026paqu,
  title   = {Estimating protein isoform abundances with {PAQu}},
  author  = {Testa, L. and Klei, L. and Rengle, A. and Yocum, A. and Lewis, D. A. and Devlin, B. and Roeder, K. and MacDonald, M. L.},
  journal = {biorXiv},
  year    = {2026},
  doi     = {10.64898/2026.04.20.719668},
  url     = {https://www.biorxiv.org/content/10.64898/2026.04.20.719668v1}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
