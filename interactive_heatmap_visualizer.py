# interactive_heatmap_visualizer.py
"""Interactive Plotly heatmaps per dataset with a slider to balance reconstruction (NLL) and disentanglement (MIG), highlighting optimal beta-lambda pairs."""

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configuration for beta and lambda values
class Config:
    betas = [1, 4, 8, 16, 32]
    lambdas = [0, 2, 4, 6, 8]

# Define datasets and paths
datasets = ['dSprites', 'Shapes3D', 'MPI3D']
save_path = './lambda_betavae_results'

# Load results from .npz files
results_dict = {}
num_seeds = 10
metrics = ["NLL", "SAP", "MIG", "Im"]
for dataset_name in datasets:
    dataset_path = os.path.join(save_path, dataset_name)
    results_dict[dataset_name] = {}
    for beta in Config.betas:
        results_dict[dataset_name][beta] = {}
        for lambda_ in Config.lambdas:
            lambda_results = []
            for seed in range(num_seeds):
                file_path = os.path.join(
                    dataset_path, f"metrics_beta_{beta}_lambda_{lambda_}_seed_{seed}.npz"
                )
                if os.path.exists(file_path):
                    npz = np.load(file_path)
                    result = {m: npz.get(m, np.nan) for m in metrics}
                    lambda_results.append(result)
            if lambda_results:
                results_dict[dataset_name][beta][lambda_] = lambda_results

def plot_metric_heatmaps(results_dict, datasets, save_path):
    """
    Plot heatmaps showing the percentage change in key metrics across beta and lambda values for each dataset.
    """
    metrics = ["NLL", "SAP", "MIG", "Im"]
    metric_labels = [
        "% Change in Reconstruction Error",
        "% Change in SAP Score",
        "% Change in MIG Score",
        r"% Change in $I_m$ Score",
    ]
    recon_cmap = LinearSegmentedColormap.from_list('custom_red', ['#FFE6E6', '#A93226'])
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E6F3FF', '#4682B4'])
    cmap_dict = {"NLL": recon_cmap, "SAP": blue_cmap, "MIG": blue_cmap, "Im": blue_cmap}

    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    baseline_beta = 1 if 1 in Config.betas else Config.betas[0]
    baseline_lambda = 0 if 0 in Config.lambdas else Config.lambdas[0]

    for col, dataset_name in enumerate(datasets):
        results = results_dict.get(dataset_name, {})
        baseline_results = results.get(baseline_beta, {}).get(baseline_lambda, [])
        baseline_values = {}
        if baseline_results:
            for metric in metrics:
                values = [r[metric] for r in baseline_results if not np.isnan(r[metric])]
                baseline_values[metric] = np.mean(values) if values else np.nan

        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            heatmap_data = np.full((len(Config.lambdas), len(Config.betas)), np.nan)
            baseline = baseline_values.get(metric, np.nan)
            for i, lambda_ in enumerate(Config.lambdas):
                for j, beta in enumerate(Config.betas):
                    lambda_results = results.get(beta, {}).get(lambda_, [])
                    if lambda_results:
                        values = [r[metric] for r in lambda_results if not np.isnan(r[metric])]
                        mean_value = np.mean(values) if values else np.nan
                        if not np.isnan(mean_value) and not np.isnan(baseline) and baseline != 0:
                            percent_change = ((mean_value - baseline) / baseline) * 100
                            heatmap_data[i, j] = percent_change

            im = ax.imshow(heatmap_data, cmap=cmap_dict[metric], aspect='auto')

            for i in range(len(Config.lambdas)):
                for j in range(len(Config.betas)):
                    val = heatmap_data[i, j]
                    text = f'{val:.2f}' if not np.isnan(val) else ''
                    ax.text(j, i, text, ha='center', va='center', color='black', fontsize=15)
                    if Config.lambdas[i] == baseline_lambda and Config.betas[j] == baseline_beta:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))

            ax.set_xticks(np.arange(len(Config.betas)))
            ax.set_yticks(np.arange(len(Config.lambdas)))

            if row == len(metrics) - 1:
                ax.set_xticklabels(Config.betas, fontsize=13)
                ax.set_xlabel(r'$\beta$', fontsize=13)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_yticklabels(Config.lambdas, fontsize=13)
                ax.set_ylabel(r'$\lambda$', fontsize=13)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_label, fontsize=13)

            if row == 0:
                ax.set_title(dataset_name, fontsize=15)

            ax.set_ylim(len(Config.lambdas) - 0.5, -0.5)

    plt.tight_layout()
    heatmaps_path = os.path.join(save_path, "percent_change_heatmaps_across_datasets.png")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(heatmaps_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_cell_bounds(values):
    """
    Compute cell boundaries for heatmap cells based on provided values.
    """
    if not values:
        return []
    n = len(values)
    bounds = []
    for i in range(n + 1):
        if i == 0:
            delta = abs(values[1] - values[0]) if n > 1 else 1
            sign = 1 if (n > 1 and values[1] > values[0]) else -1
            bounds.append(values[0] - sign * (delta / 2))
        elif i == n:
            delta = abs(values[-1] - values[-2]) if n > 1 else 1
            sign = 1 if (n > 1 and values[-1] > values[-2]) else -1
            bounds.append(values[-1] + sign * (delta / 2))
        else:
            bounds.append((values[i - 1] + values[i]) / 2)
    return bounds

def plot_interactive_weighted_score_heatmap(results_dict, datasets, save_path):
    """
    Create an interactive heatmap displaying weighted scores across beta and lambda values,
    with a slider to adjust the weight on reconstruction quality versus disentanglement.
    """
    for dataset_name in datasets:
        results = results_dict.get(dataset_name, {})
        # Collect mean NLL and MIG for each (beta, lambda) pair
        mean_dict = {}
        for beta in Config.betas:
            for lambda_ in Config.lambdas:
                lambda_results = results.get(beta, {}).get(lambda_, [])
                if lambda_results:
                    mean_nll = np.mean([r['NLL'] for r in lambda_results if not np.isnan(r['NLL'])])
                    mean_mig = np.mean([r['MIG'] for r in lambda_results if not np.isnan(r['MIG'])])
                    if not np.isnan(mean_nll) and not np.isnan(mean_mig):
                        mean_dict[(beta, lambda_)] = {'NLL': mean_nll, 'MIG': mean_mig}

        # Compute global min/max for normalization
        all_nll = [d['NLL'] for d in mean_dict.values()]
        all_mig = [d['MIG'] for d in mean_dict.values()]
        min_nll, max_nll = min(all_nll), max(all_nll)
        min_mig, max_mig = min(all_mig), max(all_mig)
        nll_range = max_nll - min_nll if max_nll != min_nll else 1
        mig_range = max_mig - min_mig if max_mig != min_mig else 1

        # Precompute score matrices and best pairs for each reconstruction weight percentage
        recon_weights_pct = range(101)
        score_matrices = []
        best_betas = []
        best_lambdas = []
        max_scores = []
        for recon_weight_pct in recon_weights_pct:
            w_recon = recon_weight_pct / 100.0
            w_disent = (100 - recon_weight_pct) / 100.0
            heatmap_data = np.full((len(Config.lambdas), len(Config.betas)), np.nan)
            for i, lambda_ in enumerate(Config.lambdas):
                for j, beta in enumerate(Config.betas):
                    key = (beta, lambda_)
                    if key in mean_dict:
                        d = mean_dict[key]
                        norm_nll = (max_nll - d['NLL']) / nll_range  # Invert since lower NLL is better
                        norm_mig = (d['MIG'] - min_mig) / mig_range
                        score = w_recon * norm_nll + w_disent * norm_mig
                        heatmap_data[i, j] = score
            score_matrices.append(heatmap_data)

            if np.all(np.isnan(heatmap_data)):
                best_betas.append(None)
                best_lambdas.append(None)
                max_scores.append(np.nan)
            else:
                valid_heatmap = np.nan_to_num(heatmap_data, nan=-np.inf)
                best_i, best_j = np.unravel_index(np.argmax(valid_heatmap), valid_heatmap.shape)
                best_betas.append(Config.betas[best_j])
                best_lambdas.append(Config.lambdas[best_i])
                max_scores.append(np.max(heatmap_data))

        # Create interactive Plotly figure
        x_ticks = list(range(len(Config.betas)))
        y_ticks = list(range(len(Config.lambdas)))
        initial_idx = 50  # Start at 50% reconstruction weight
        initial_z = score_matrices[initial_idx]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=initial_z,
                x=x_ticks,
                y=y_ticks,
                colorscale='viridis',
                colorbar=dict(title='Score'),
                texttemplate="%{z:.3f}",
                textfont={"size": 12, "color": "black"},
                showlegend=False,
                hoverinfo='none'
            )
        )

        fig.update_xaxes(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=Config.betas,
            title_text="β"
        )
        fig.update_yaxes(
            tickmode='array',
            tickvals=y_ticks,
            ticktext=Config.lambdas,
            autorange='reversed',
            title_text="λ"
        )

        bounds_x = get_cell_bounds(x_ticks)
        bounds_y = get_cell_bounds(y_ticks)

        initial_best_beta = best_betas[initial_idx]
        initial_best_lambda = best_lambdas[initial_idx]
        initial_best_j = Config.betas.index(initial_best_beta)
        initial_best_i = Config.lambdas.index(initial_best_lambda)
        initial_x0 = bounds_x[initial_best_j]
        initial_x1 = bounds_x[initial_best_j + 1]
        initial_y0 = bounds_y[initial_best_i]
        initial_y1 = bounds_y[initial_best_i + 1]

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    x0=initial_x0,
                    x1=initial_x1,
                    y0=initial_y0,
                    y1=initial_y1,
                    line=dict(color="red", width=2)
                )
            ]
        )

        # Create slider steps for updating heatmap and highlight
        steps = []
        for i, recon_weight_pct in enumerate(recon_weights_pct):
            if best_betas[i] is None:
                continue
            z_data = score_matrices[i]
            best_j = Config.betas.index(best_betas[i])
            best_i = Config.lambdas.index(best_lambdas[i])
            x0 = bounds_x[best_j]
            x1 = bounds_x[best_j + 1]
            y0 = bounds_y[best_i]
            y1 = bounds_y[best_i + 1]
            shape_dict = dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="red", width=2)
            )
            step = dict(
                method='update',
                label=f'{recon_weight_pct}',
                args=[
                    {'z': [z_data]},
                    {'shapes': [shape_dict]}
                ]
            )
            steps.append(step)

        sliders = [dict(
            active=initial_idx,
            pad={"t": 50},
            currentvalue={"prefix": "Reconstruction Weight: ", "suffix": "%"},
            steps=steps
        )]

        fig.update_layout(sliders=sliders)
        fig.update_layout(
            title=f"Interactive Weighted Score Heatmap for {dataset_name}<br>",
            height=500,
            width=600,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        # Save as interactive HTML
        plot_path = os.path.join(save_path, f"interactive_weighted_score_heatmap_{dataset_name}.html")
        fig.write_html(plot_path)

# Execute the plotting functions
plot_metric_heatmaps(results_dict, datasets, save_path)
plot_interactive_weighted_score_heatmap(results_dict, datasets, save_path)