# KAN vs. MLP on MNIST and Fashion-MNIST

This CMPE 460 experiment compares a multilayer perceptron (MLP), a B-spline Kolmogorov-Arnold Network (KAN), and an RBF-based FastKAN at roughly 110,000 trainable parameters each. It records how accuracy and training time differ when model size is held close to that budget.

[Read the report (PDF)](plots/cmpe460.pdf).

## Quick start

Install Python 3.10.12 and Poetry, then run:

```bash
git clone https://github.com/atakang7/cmpe460report.git
cd cmpe460report
poetry env use 3.10.12
poetry install --no-root
poetry run jupyter notebook main.ipynb
```

In Jupyter, select **Run All Cells**. To run the depth and grid-size experiments, open the second notebook:

```bash
poetry run jupyter notebook ablation.ipynb
```

The first run may download MNIST and Fashion-MNIST. GPU acceleration is optional; the notebooks use CUDA when available and CPU otherwise.

## Configuration

The experiment settings are near the end of the training section in each notebook:

| Setting | Default |
| --- | --- |
| Epochs | 10 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Random seed | 42 |
| Device | CUDA, otherwise CPU |

Datasets are read from `data/`. Model checkpoints are read from or written to `saved_models/`; figures go to `plots/`, and training logs go to `experiment_results.txt`. No credentials or environment variables are required.

**Existing checkpoints are loaded instead of training again.** Move the relevant `.pth` files out of `saved_models/` before rerunning an experiment with different settings. Changing the epoch count or learning rate alone will not invalidate a checkpoint.

## How it works

`main.ipynb` trains three models on each dataset and compares validation accuracy and average seconds per epoch. `ablation.ipynb` runs those comparisons before testing a smaller KAN spline grid and a deeper MLP on Fashion-MNIST.

The saved Fashion-MNIST results from the original RTX 3060 run were:

| Model | Parameters | Best accuracy | Mean epoch time |
| --- | ---: | ---: | ---: |
| MLP | 109,386 | 88.11% | 18.19 s |
| KAN | 107,190 | 86.90% | 20.29 s |
| FastKAN | 108,034 | 86.15% | 19.34 s |

See [training curves](plots/training_curves.png), [dataset comparison](plots/dataset_comparison.png), and [ablation figures](plots/ablation_analysis.png). The captured execution logs are in `experiment_results.txt`.

## Usage and checks

These are local research notebooks; there is no API or deployment target. You can inspect the checked-in plots and report without installing Python.

To validate notebook JSON, Python syntax, and safe checkpoint loading without starting model training:

```bash
python -m unittest discover -s tests -v
```

The same check runs in GitHub Actions on pull requests and pushes to `main`.

## Limitations

The comparison covers two 28×28 grayscale datasets, ten epochs, and the recorded runs rather than repeated trials with error bars. Timings depend on hardware and PyTorch versions. Both notebooks contain the model definitions so each can run on its own; the ablation notebook repeats the base experiment before its extra studies. The CI check does not train models or verify benchmark accuracy.

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [Kolmogorov-Arnold Networks are Radial Basis Function Networks (FastKAN)](https://arxiv.org/abs/2405.06721)
- [Full course report](plots/cmpe460.pdf)

Licensed under [MIT](LICENSE).
