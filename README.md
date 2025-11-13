# λβ-VAE Disentanglement Experiments
This repository implements β-VAE and λβ-VAE models to evaluate disentanglement metrics on the dSprites, Shapes3D, and MPI3D datasets. It includes both linear and nonlinear implementations. The β-VAE baselines are trained first, followed by continued training with an additional λ term in the loss function to assess its impact on reconstruction quality and disentanglement performance.

## Features
- **Models**: Linear and convolutional encoder-decoder VAE architectures.
- **Loss Functions**: β-VAE (reconstruction + β × KL divergence) and λβ-VAE (reconstruction + β × KL divergence + λ × L2 loss).
- **Datasets**: dSprites, Shapes3D, and MPI3D (included in the `data` folder).
- **Metrics**: Negative Log-Likelihood (NLL), Mutual Information Gap (MIG), Separated Attribute Predictability (SAP), and $I_m$ score.
- **Visualizations**: Image reconstruction grids, latent traversal GIFs, mutual information heatmaps, boxplots and heatmaps for all metrics, and interactive Plotly heatmaps.
- **Reproducibility**: Multi-seed experiments with fixed random seeds.

## Installation
1. Download and unzip the repository archive from GitHub.
2. Ensure the `data` folder contains the datasets.
3. Install dependencies using Python 3.8+:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the experiments in a linear sequence: start with linear models, then proceed to nonlinear datasets, and finally generate interactive visualizations. This ensures compatibility and allows for comparison between linear and nonlinear results.

### Running All Experiments in Sequence
To execute everything step by step:
```
python linear_betavae.py
python main_beta.py
python main_lambda.py
python interactive_heatmap_visualizer.py
```

### Running Linear β-VAE and λβ-VAE Experiments
The linear experiments use two methods (fixed-point iteration and AdamW optimization) within the script. Run this before nonlinear experiments:
```
python linear_betavae.py
```
- Results are saved to `./linear_betavae_results/`.
- Pre-computed results: Download `linear_betavae_results.zip` attached in the repository main.

### Running Nonlinear β-VAE Experiments
Train and evaluate nonlinear β-VAE models:
```
python main_beta.py
```
- Results are saved to `./betavae_results/<dataset>/`.
- Includes trained models, metrics, visualizations, and summary files.
- Aggregated boxplots: `./betavae_results/metric_boxplots_across_datasets.png`.

### Running Nonlinear λβ-VAE Experiments
Continue training from β-VAE checkpoints (run `main_beta.py` first):
```
python main_lambda.py
```
- Results are saved to `./lambda_betavae_results/<dataset>/`.
- Includes updated models, metrics, visualizations, and summary files.
- Aggregated heatmaps: `./lambda_betavae_results/metric_heatmaps_across_datasets.png`.

### Plotting Interactive Heatmaps
After running nonlinear experiments, load metrics from .npz files, calculate percentage changes vs. baseline for NLL, SAP, MIG, and $I_m$, and generate static Matplotlib heatmaps showing these changes. Also, create interactive Plotly heatmaps per dataset with a slider to balance reconstruction (NLL) and disentanglement (MIG), highlighting optimal beta-lambda pairs:
```
python interactive_heatmap_visualizer.py
```
- Results are saved to `./lambda_betavae_results/` including "percent_heatmaps_across_datasets.png" and HTML files (e.g., `interactive_tchebycheff_heatmap_<dataset>.html`).
- Use the slider to adjust weights (0-100% reconstruction weight) and find the best β-λ pair for different priorities.
- Pre-computed results: Download `interactive_heatmap_results.zip` attached in the repository main.

## Configuration
Customize hyperparameters (e.g., β values, λ values, number of seeds, training steps) in `config.py`.

## Datasets
We do not redistribute datasets. Please download them from the official sources below and place the files in the `data` folder so loaders work correctly.
- `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`: dSprites (shapes with variations in shape, scale, orientation, position).
  - Source: https://github.com/deepmind/dsprites-dataset
- `3dshapes.h5`: Shapes3D (3D shapes with variations in hue, shape, scale, orientation).
  - Source: https://github.com/deepmind/3d-shapes
- `mpi3d_real.npz`: MPI3D (realistic 3D objects with variations in color, shape, size, camera position).
  - Source: https://github.com/rr-learning/disentanglement_dataset
Expected layout:
```text
data/
├─ dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
├─ 3dshapes.h5
└─ mpi3d_real.npz
```

## Results
- **Metrics**: Reconstruction loss (NLL), MIG, SAP, $I_m$.
- **Visualizations**:
  - Original and reconstructed image grids.
  - Latent traversal animations (GIFs).
  - Mutual information heatmaps.
  - Boxplots and heatmaps for all metrics.
  - Interactive weighted score heatmaps.
- Summary statistics (mean ± std across seeds) in text files per dataset.

### Pre-trained Models and Results
Pre-trained models, metrics, and visualizations from training runs are available for download. These can be used to reproduce or analyze results without re-training.
- β-VAE (nonlinear): https://drive.google.com/file/d/1ofhdRkt4kaD7aLhTqq9FE2KIMQksmbyi/view?usp=drive_link
- λβ-VAE (nonlinear): https://drive.google.com/file/d/1AMGOE_MrNIQjRhlMHL73AQMQe4R7Qd9B/view?usp=drive_link

## Project Structure
- `config.py`: Configuration settings.
- `datasets.py`: Dataset loading.
- `models.py`: VAE model definitions.
- `losses.py`: Loss functions and training utilities.
- `metrics.py`: Disentanglement metric calculations.
- `utils.py`: Utility functions for seeding and data handling.
- `visualizations.py`: Functions for generating plots and images.
- `main_beta.py`: Script for nonlinear β-VAE training and evaluation.
- `main_lambda.py`: Script for nonlinear λβ-VAE continuation and evaluation.
- `linear_betavae.py`: Main script for linear β-VAE and λβ-VAE experiments.
- `interactive_heatmap_visualizer.py`: Script for generating static percentage change heatmaps and interactive Plotly heatmaps.
- `requirements.txt`: List of dependencies.

## Acknowledgments
- Datasets and metrics inspired by the Disentanglement Library: https://github.com/google-research/disentanglement_lib.
