# mabt

Python implementation of multiplicity-adjusted bootstrap tilting lower confidence bounds based on 
Pascal Rink (2025). Confidence Limits for Prediction Performance. Dissertation, University of Bremen, https://doi.org/10.26092/elib/3822

## Installation

From GitHub (replace with your URL):

```bash
pip install git+https://github.com/pascalrink/prediction-performance-ci.git
```

## For development

```bash
git clone https://github.com/pascalrink/prediction-performance-ci.git
cd mabt
pip install -e .
```

## Usage

```python
import numpy as np
from mabt_ci import mabt_ci

true_labels = np.array([0, 1, 0, 1])
pred_labels = np.array([[0, 1],
                        [1, 1],
                        [0, 0],
                        [1, 0]])

bound, tau, t0 = mabt_ci(true_labels, pred_labels, alpha=0.05, B=10000, seed=123)
print(bound, tau, t0)
```

## Examples

```bash
python examples/run_from_csv.py
```

## Testing

```bash
pytest
```