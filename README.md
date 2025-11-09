# KAN vs. MLP: Computer Vision Benchmark under Strict Parameter Constraints

A rigorous empirical benchmark comparing **Kolmogorov-Arnold Networks (KAN)** and **FastKAN** against standard **Multi-Layer Perceptrons (MLP)** on basic computer vision tasks.

Unlike theoretical studies that often use large models, this project enforces a strict **~100k parameter budget** across all architectures to simulate real-world resource-constrained environments (e.g., edge devices).

See Report: [Report](./plots/cmpe460.pdf)

## Findings

Based on 10-epoch training runs on an NVIDIA RTX 3060:

1.  **MLP is Strictly Superior at this Scale:** The standard MLP outperformed both KAN and FastKAN in every metric (accuracy, speed, convergence rate) on both MNIST and Fashion-MNIST.
2.  **KAN Wastes Parameters on Simple Tasks:** Reducing KAN's spline grid from $G=5$ to $G=3$ *improved* Fashion-MNIST accuracy (+0.33%), proving that standard KANs over-allocate limited parameters to complex edge functions rather than useful network width.
3.  **Vectorization Doesn't Fix the Speed Gap:** Even highly optimized `EfficientKAN` implementations were **~15% slower** per epoch than standard MLPs due to the mathematical complexity of spline evaluation.

| Model | Fashion-MNIST Acc | Parameters | Training Time (RTX 3060) |
| :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | **88.11%** | **109,386** | **18.19s / epoch** |
| KAN (Efficient) | 86.90% | 107,190 | 20.29s / epoch |
| FastKAN | 86.15% | 108,034 | 19.34s / epoch |

## Visualizations

### 1. Training Dynamics (MLP vs. KANs)
MLP (blue) demonstrates significantly faster initial convergence and maintains a higher accuracy plateau throughout training.
![Training Curves](./plots/training_curves.png)

### 2. Aggregate Performance Comparison
The performance gap widens slightly on the more complex Fashion-MNIST dataset (bottom row), highlighting KAN's struggle with richer visual features under tight budgets.
![Dataset Comparison](./plots/dataset_comparison.png)

### 3. Ablation Studies (Fashion-MNIST)
Testing grid sensitivity (left dashed lines) and structural depth (right bars) confirms MLP superiority is robust and architectural, not just a hyperparameter fluke.
![Ablation Analysis](./plots/ablation_analysis.png)

## 🛠️ Reproduction

This project uses **Poetry** for exact dependency management and **Jupyter** for experiments.

### Installation
```bash
git clone [https://github.com/atakang7/cmpe460report.git](https://github.com/atakang7/cmpe460report.git)
cd cmpe460report
poetry install
```

### Running Experiments
Activate the environment and open the notebooks:
```bash
poetry shell
```

* **`main.ipynb`**: Runs the core 6 benchmarks (3 models x 2 datasets), generates standard training curves, and saves checkpoints.
* **`ablation.ipynb`**: Executes the mandatory ablation studies (Grid Sensitivity Analysis and Structural Depth Tests) on Fashion-MNIST.

## 📂 Repository Structure

```
.
├── main.ipynb                 # Primary benchmark runner
├── ablation.ipynb             # Supplementary ablation studies
├── experiment_results.txt     # Raw execution logs verifying all claims
├── data/                      # Cached datasets (MNIST/Fashion-MNIST)
├── saved_models/              # PyTorch state dictionaries (.pth)
├── plots/                     # Generated figures for report and readme
│   ├── training_curves.png
│   ├── dataset_comparison.png
│   ├── ablation_analysis.png
│   └── comparison_table.png
├── pyproject.toml             # Poetry dependency specification
└── poetry.lock                # Exact dependency lockfile
```

## 📜 References Used in Report

1.  **[Hornik et al., 1989]** *Multilayer feedforward networks are universal approximators.* Neural Networks, 2(5).
2.  **[Li, 2024]** *Kolmogorov-Arnold Networks are Radial Basis Function Networks (FastKAN).* arXiv:2405.06721.
3.  **[Liu et al., 2024]** *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756.
4.  **[Poeta et al., 2024]** *A benchmarking study of Kolmogorov-Arnold Networks on tabular data.* arXiv:2406.14529.
5.  **[Prince, 2023]** *Understanding Deep Learning.* MIT Press.
6.  **[Yu et al., 2024]** *KAN or MLP: A Fairer Comparison.* arXiv:2407.16674.