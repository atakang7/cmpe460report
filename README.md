# KAN vs. MLP on MNIST and Fashion-MNIST

CMPE 460 notebooks compare MLP, KAN, and FastKAN at roughly 110,000 parameters per model on two image datasets ([report](plots/cmpe460.pdf)).

```text
+-------+  samples  +------------+  figures  +--------+
| data/ | --------> | main.ipynb | --------> | plots/ |
+-------+           +------------+           +--------+
```

## Run

Requires Python 3.10.12 and Poetry.

```bash
git clone https://github.com/atakang7/cmpe460report.git
cd cmpe460report
poetry env use 3.10.12
poetry install --no-root
poetry run jupyter notebook main.ipynb
```

Run all cells in Jupyter; for grid-size and depth experiments, open `ablation.ipynb` with `poetry run jupyter notebook ablation.ipynb`.

## Config

| var | default | what |
| --- | --- | --- |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `128` | Images per batch |
| `LEARNING_RATE` | `0.001` | Adam learning rate |
| `device` | CUDA if available | Falls back to CPU |

Settings are in both notebooks; datasets use `data/`, checkpoints use `saved_models/`, and logs go to `experiment_results.txt`.

## CI

```text
+---------------------+  triggers  +-----------------+  runs  +-------------------------+
| push / pull_request | ---------> | Notebook checks | -----> | tests/test_notebooks.py |
+---------------------+            +-----------------+        +-------------------------+
```

```bash
python -m unittest discover -s tests -v
```

## Limits

- Results cover MNIST and Fashion-MNIST (28 x 28 pixels), with no repeated-trial error bars.
- Existing `saved_models/*.pth` files skip training; move them elsewhere to rerun changed settings.
- Runtime depends on hardware; CI checks notebook syntax and checkpoint loading, not benchmark accuracy.
