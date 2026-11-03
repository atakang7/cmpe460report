# cmpe460report

Jupyter benchmarks for CMPE 460 readers comparing MLP, KAN, and FastKAN on MNIST and Fashion-MNIST.

```text
+-------+  samples  +------------------+  checkpoints  +---------------+
| data/ | --------> | main.ipynb       | ------------> | saved_models/ |
|       |           | ablation.ipynb   |               +---------------+
+-------+           +------------------+
                             | figures
                             v
                          +--------+
                          | plots/ |
                          +--------+
```

## Quick start

Use Python 3.10.12 and Poetry. Open each notebook from the repository root.

```sh
poetry install --no-root
poetry run jupyter notebook main.ipynb
poetry run jupyter notebook ablation.ipynb
```

## Layout

| Path | Contents |
| --- | --- |
| [main.ipynb](main.ipynb) | MLP, KAN, and FastKAN runs on both datasets |
| [ablation.ipynb](ablation.ipynb) | KAN grid-size and MLP depth experiments |
| [data/](data/) | MNIST and Fashion-MNIST image files |
| [saved_models/](saved_models/) | Checkpoints for the trained models |
| [plots/](plots/) | Figures and the course report |
| [pyproject.toml](pyproject.toml), [poetry.lock](poetry.lock) | Dependencies and locked versions |

## Docs

[Course report](plots/cmpe460.pdf) and [recorded experiment output](experiment_results.txt).

## Limits

- Existing checkpoints skip training even when epochs, batch size, or learning rate change.
- Training uses CUDA when available; CPU runs take longer.
- CI checks notebook syntax and does not execute the experiments.
