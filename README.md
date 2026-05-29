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

The method is described in:

> Testa L., Klei L., Rengle A., Yocum A., Lewis D.A., Devlin B., Roeder K., MacDonald M.L. (2026). *Estimating protein isoform abundances with PAQu*. biorXiv. https://doi.org/10.64898/2026.04.20.719668

---

## How it works

PAQu frames isoform quantification as a constrained matrix factorisation problem:

```
I  =  I₀  +  T × W  +  A × D  +  noise
P  ≈  I × Z  +  noise
```

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| **P** | Observed peptide intensities | n samples × r peptides |
| **I** | Latent isoform abundances (estimated) | n samples × q isoforms |
| **Z** | Peptide-to-isoform mapping weights | q isoforms × r peptides |
| **A** | Binary condition/treatment indicator | n samples |
| **D** | Condition effect on isoform abundances | q isoforms |
| **T** | Transcript-level expression covariates | n samples × q isoforms |
| **W** | Transcript effect on isoform abundances | q isoforms |

Parameters are inferred via **Gibbs sampling** with conjugate priors. Posterior uncertainty is summarised through the **Local False Sign Rate (LFSR)**.

Optional inputs enrich the model:
- **Xi / Xp** — additional sample-level or peptide-level covariates

---

## Repository structure

| File | Description |
|------|-------------|
| `PAQu.py` | Core `PAQu` class — model definition and Gibbs sampler |
| `PAQu_simulations.py` | Simulation study from the paper |
| `PAQu_application.py` | Real-data analysis pipeline |
| `format_upload.py` | Post-processing helper to format output files |

---

## Installation

There is no package to install. Clone the repository and make sure the dependencies are available:

```bash
git clone https://github.com/testalorenzo/PAQu.git
cd PAQu
pip install numpy scipy tqdm pandas joblib networkx
```

---

## Quick start

```python
import numpy as np
from PAQu import PAQu

# Minimal inputs
n, q, r = 50, 3, 10          # samples, isoforms, peptides
P = np.random.randn(n, r)    # observed peptide intensities
M = np.ones((q, r))          # peptide-isoform mask (1 = compatible)
A = np.array([0]*25 + [1]*25) # condition label (0/1)

model = PAQu(A=A, M=M, P=P, T=q)   # T=q → isoform count (no transcript data)
model.fit(n_iter=2000, prior_D='Spike-and-Slab')

# Posterior summaries
I_hat  = model.I_storer[1000:].mean(axis=0)   # isoform abundances
D_hat  = model.D_storer[1000:].mean(axis=0)   # condition effects
lfsr_D = model.LFSR('D_storer', burn_in=1000)  # local false sign rates
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
