# prediction-performance-ci (`mabt`)

Python package implementing **multiplicity-adjusted bootstrap tilting (MABT)** lower confidence bounds for **prediction performance after model selection** (currently: **accuracy** for binary classification).

This repository is the Python port (R → Python) of code developed for the dissertation:

- Pascal Rink (2025). *Confidence Limits for Prediction Performance*. Doctoral thesis, University of Bremen. https://doi.org/10.26092/elib/3822

Related paper:
- Pascal Rink & Werner Brannath (2025). *Post-selection confidence bounds for prediction performance*. *Machine Learning*. https://link.springer.com/article/10.1007/s10994-024-06632-w

---

## Why post-selection confidence bounds are needed

In many applied ML workflows, multiple candidate models are trained (e.g., many LASSO variants). A subset of “promising” candidates may be identified using cross-validation on the training data, and then a final model is selected by maximizing performance on an evaluation dataset.

If you compute a standard confidence interval/bound **after selecting the best-performing candidate**, the reported performance is typically **too optimistic** (selection bias / winner’s curse). The selection event (best-of-`k`) is part of the randomness and must be accounted for in inference.

**MABT** addresses this by producing a **lower (1 − α) confidence bound** for the **conditional** prediction performance of the *selected* model, while adjusting for selection over multiple candidates.

---

## What this package provides

### Core function

- `mabt.mabt_ci(true_labels, pred_labels, alpha=0.05, B=10000, seed=None)`

Returns:

- `bound`: **MABT lower confidence bound** for the selected model’s accuracy (level `1 - alpha`)
- `tau`: calibrated **tilting parameter**
- `t0`: **naive** point estimate for the selected model (optimistic “best observed” accuracy)

---

## Installation

From GitHub:

```bash
pip install git+https://github.com/pascalrink/prediction-performance-ci.git
```

For development:

```bash
git clone https://github.com/pascalrink/prediction-performance-ci.git
cd prediction-performance-ci
pip install -e .
```

---

## Quickstart

### Minimal (NumPy)

```python
import numpy as np
from mabt import mabt_ci

true_labels = np.array([0, 1, 0, 1])
pred_labels = np.array([
    [0, 1],  # sample 1 predictions from model 1 and 2
    [1, 1],
    [0, 0],
    [1, 0],
])

bound, tau, t0 = mabt_ci(true_labels, pred_labels, alpha=0.05, B=10_000, seed=123)

print("MABT lower bound:", bound)
print("Tilting parameter:", tau)
print("Point estimate (optimistic):", t0)
```

### Reproducible CSV example (matches this repository)

The repository currently includes one end-to-end example:

```bash
python examples/run_from_csv.py
```

`examples/run_from_csv.py` does the following:

1. loads `examples/data/labels.csv` (true labels) and `examples/data/predictions.csv` (predicted labels for multiple models),
2. calls `mabt_ci(...)`,
3. prints `bound`, `tau`, and `t0`.

---

## Input format and assumptions

### `true_labels`
- shape `(n,)`
- binary labels encoded as `{0, 1}`

### `pred_labels`
- shape `(n,)` for a single model or `(n, k)` for `k` candidate models
- **hard** class predictions in `{0, 1}` (not probabilities)
- each column corresponds to one candidate model

### Typical usage pattern (pipeline)

- Candidate models are trained on **training data** (outside this repo).
- “Promising” candidates are identified using (e.g.) cross-validation on training data.
- On a **separate evaluation dataset**, you collect:
  - true labels `y`
  - predicted labels for each candidate model
- On this evaluation dataset you:
  1) select the best candidate by accuracy, and  
  2) compute a post-selection lower confidence bound for the selected model’s conditional accuracy.

---

## Interpreting `bound`, `tau`, and `t0`

- `t0` is the empirical accuracy of the “best” model (maximum over candidates).  
  **By construction this is optimistic** because it ignores selection.
- `bound` is a conservative lower guarantee that remains valid even when the model was selected among multiple candidates.
- `tau` is the exponential-tilting parameter; in practice it mainly serves as a diagnostic (how strongly reweighting was needed).

---

## Method overview (short, but substantive)

The Python implementation follows the MABT approach in the dissertation, in particular:

- Part II: “Multiplicity-Adjusted Bootstrap Tilting” (Chapter 5)
- Appendix C.1: “MABT algorithm” (Algorithm C.1: Accuracy)  
  https://doi.org/10.26092/elib/3822

### Core idea in six steps (accuracy, binary)

1. **Correctness matrix**: convert labels into indicators  
   `ỹ_ij = 1{ŷ_ij == y_i}` for each sample `i` and model `j`.  
   The empirical accuracy per model is `θ̂_j = mean_i(ỹ_ij)`.

2. **Selection**: choose `s = argmax_j θ̂_j` (best observed).  
   The naive estimate is `t0 = θ̂_s`.

3. **Multivariate bootstrap**: resample evaluation observations jointly across all models
   to approximate the joint distribution of the model-wise statistics.

4. **Simultaneity / maxT-style adjustment**: transform bootstrap statistics using their
   empirical CDFs (“uniformization”) and aggregate via a maximum construction.
   This leverages dependence between models without assuming independence.

5. **Bootstrap tilting (exponential reweighting)**: use weights of the form  
   `w_i(τ) ∝ exp(τ · U_i)`,  
   where `U_i` is derived from the selected model’s data (for accuracy this is directly tied
   to the selected model’s correctness indicator).

6. **Calibrate `τ` and compute the bound**: find `τ < 0` such that the multiplicity-adjusted,
   bootstrap-based p-value equals `α`. The bound is then the weighted accuracy of the selected model:  
   `bound = Σ_i w_i(τ) · ỹ_is`.

The construction is designed such that the resulting lower bound achieves nominal coverage
asymptotically (see the theory in Chapter 5 of the dissertation).

---

## Limitations (current port status)

- currently implemented: **accuracy** for **binary classification** with hard labels `{0,1}`
- other performance measures (e.g., AUC) are discussed in the dissertation (with applications)

---

## Testing

```bash
pytest
```

---

## How to cite

If you use this package in academic work, cite the dissertation:

```bibtex
@phdthesis{rink2025confidence,
  title   = {Confidence Limits for Prediction Performance},
  author  = {Rink, Pascal},
  school  = {University of Bremen},
  year    = {2025},
  doi     = {10.26092/elib/3822}
}
```

---

## License

MIT (see `LICENSE`).
