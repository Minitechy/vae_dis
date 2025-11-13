# interactive_heatmap_visualizer.py
"""Interactive Plotly heatmaps per dataset with a slider to balance reconstruction (NLL) and disentanglement (MIG), highlighting optimal beta-lambda pairs."""

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go

class Config:
    """
    Configuration class defining beta and lambda values for the grid search.
    """
    betas = [1, 4, 8, 16, 32]
    lambda_values = [0, 2, 4, 6, 8]

# Define datasets
datasets = ['dSprites', 'Shapes3D', 'MPI3D']

# Path to results
save_path = './lambda_betavae_results'

# Load data from .npz files
results_dict = {}
num_seeds = 10
metrics = ["NLL", "SAP", "MIG", "Im"]
for ds_name in datasets:
    ds_path = os.path.join(save_path, ds_name)
    results_dict[ds_name] = {}
    for beta in Config.betas:
        results_dict[ds_name][beta] = {}
        for lam in Config.lambda_values:
            lambda_results = []
            for seed in range(num_seeds):
                file_path = os.path.join(ds_path, f"metrics_beta_{beta}_lambda_{lam}_seed_{seed}.npz")
                if os.path.exists(file_path):
                    npz = np.load(file_path)
                    result = {m: npz.get(m, np.nan) for m in metrics}
                    lambda_results.append(result)
            results_dict[ds_name][beta][lam] = lambda_results

def plot_metric_heatmaps(results_dict, datasets, save_path):
    """
    Generate static heatmaps showing percentage changes in key metrics across beta and lambda values for each dataset.
    """
    metrics = ["NLL", "SAP", "MIG", "Im"]
    metric_labels = ["% Change in Reconstruction Error", "% Change in SAP Score",
                     "% Change in MIG Score", r"% Change in $I_m$ Score"]
    recon_cmap = LinearSegmentedColormap.from_list('custom_red', ['#FFE6E6', '#A93226'])
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E6F3FF', '#4682B4'])
    cmap_dict = {"NLL": recon_cmap, "SAP": blue_cmap, "MIG": blue_cmap, "Im": blue_cmap}
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    beta_idx = Config.betas.index(1) if 1 in Config.betas else 0
    lambda_idx = Config.lambda_values.index(0) if 0 in Config.lambda_values else 0
    for col, ds_name in enumerate(datasets):
        results = results_dict[ds_name]
        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            heatmap_data = np.zeros((len(Config.lambda_values), len(Config.betas)))
            baseline_results = results.get(1, {}).get(0, [])
            baseline = np.mean([r[metric] for r in baseline_results if not np.isnan(r[metric])]) \
                if baseline_results else np.nan
            for i, lam in enumerate(Config.lambda_values):
                for j, beta in enumerate(Config.betas):
                    lambda_results = results.get(beta, {}).get(lam, [])
                    mean_value = np.mean([r[metric] for r in lambda_results if not np.isnan(r[metric])]) \
                        if lambda_results else np.nan
                    if not np.isnan(mean_value) and not np.isnan(baseline) and baseline != 0:
                        percent = ((mean_value - baseline) / baseline) * 100
                        heatmap_data[i, j] = percent
                    else:
                        heatmap_data[i, j] = np.nan
            im = ax.imshow(heatmap_data, cmap=cmap_dict[metric], aspect='auto')
            for i, j in np.ndindex(len(Config.lambda_values), len(Config.betas)):
                val = heatmap_data[i, j]
                text = f'{val:.2f}' if not np.isnan(val) else ''
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=15)
                if i == lambda_idx and j == beta_idx:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                                           linewidth=2, edgecolor='red', facecolor='none'))
            ax.set_xticks(np.arange(len(Config.betas)))
            ax.set_yticks(np.arange(len(Config.lambda_values)))
            if row == 3:
                ax.set_xticklabels([f'{b}' for b in Config.betas], fontsize=13)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels([f'{l}' for l in Config.lambda_values], fontsize=13)
                ax.tick_params(axis='y', which='both', labelsize=12, length=5, left=True, labelleft=True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_label, fontsize=13)
            if row == 0:
                ax.set_title(ds_name, fontsize=15)
            if col == 0:
                ax.set_ylabel(r'$\lambda$', fontsize=13)
            if row == 3:
                ax.set_xlabel(r'$\beta$', fontsize=13)
            ax.set_ylim(len(Config.lambda_values) - 0.5, -0.5)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    heatmaps_path = os.path.join(save_path, "percent_heatmaps_across_datasets.png")
    plt.tight_layout()
    plt.savefig(heatmaps_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_bounds(vals):
    """
    Compute cell boundaries for heatmap cells in interactive plots.
    """
    if not vals:
        return []
    n = len(vals)
    bounds = []
    for i in range(n + 1):
        if i == 0:
            if n > 1:
                delta = abs(vals[1] - vals[0])
                sign = 1 if vals[1] > vals[0] else -1
            else:
                delta = 1
                sign = 1
            bounds.append(vals[0] - sign * (delta / 2))
        elif i == n:
            delta = abs(vals[-1] - vals[-2])
            sign = 1 if vals[-1] > vals[-2] else -1
            bounds.append(vals[-1] + sign * (delta / 2))
        else:
            bounds.append((vals[i - 1] + vals[i]) / 2)
    return bounds

def get_mean_nll_mig(results):
    """
    Compute mean NLL and MIG values for each (beta, lambda) pair by averaging across seeds.
    """
    mean_dict = {}
    for beta in Config.betas:
        for lam in Config.lambda_values:
            lambda_results = results.get(beta, {}).get(lam, [])
            if not lambda_results:
                continue
            nll_vals = [r['NLL'] for r in lambda_results if not np.isnan(r['NLL'])]
            mig_vals = [r['MIG'] for r in lambda_results if not np.isnan(r['MIG'])]
            if nll_vals and mig_vals:
                mean_NLL = np.mean(nll_vals)
                mean_MIG = np.mean(mig_vals)
                mean_dict[(beta, lam)] = {'NLL': mean_NLL, 'MIG': mean_MIG}
    return mean_dict

def compute_pareto_front(points):
    """
    Identify Pareto-optimal points using dominance checks.
    """
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (q['NLL'] <= p['NLL'] and q['MIG'] >= p['MIG'] and
               (q['NLL'] < p['NLL'] or q['MIG'] > p['MIG'])):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda d: d['NLL'])
    return pareto

def plot_pareto_front(results_dict, datasets, save_path):
    """
    Generate a plot of the Pareto front for NLL vs. MIG trade-offs for each dataset.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[3.0, 1.5],
        height_ratios=[1, 1, 1],
        wspace=0.10,
        hspace=0.25,
        left=0.08, right=0.98,
        top=0.95, bottom=0.05
    )
    for i, ds_name in enumerate(datasets):
        results = results_dict[ds_name]
        mean_dict = get_mean_nll_mig(results)
        if not mean_dict:
            print(f"No valid data for {ds_name}")
            continue
        points = []
        for (beta, lam), d in mean_dict.items():
            points.append({
                'beta': beta,
                'lam': lam,
                'NLL': float(d['NLL']),
                'MIG': float(d['MIG'])
            })
        pareto = compute_pareto_front(points)
        if not pareto:
            print(f"No Pareto-optimal points for {ds_name}")
            continue
        pareto_sorted = sorted(pareto, key=lambda p: p['NLL'])
        all_NLL = [p['NLL'] for p in points]
        all_MIG = [p['MIG'] for p in points]
        pf_NLL = [p['NLL'] for p in pareto_sorted]
        pf_MIG = [p['MIG'] for p in pareto_sorted]
        baseline_point = None
        if (1, 0) in mean_dict:
            d0 = mean_dict[(1, 0)]
            baseline_point = {'NLL': float(d0['NLL']),
                              'MIG': float(d0['MIG'])}
        ax = fig.add_subplot(gs[i, 0])
        ax_table = fig.add_subplot(gs[i, 1])
        ax_table.axis('off')
        # Pareto scatter
        ax.scatter(
            all_NLL, all_MIG,
            s=20,
            c='#D3D3D3',
            edgecolors='none',
            label = r'All $(\beta,\lambda)$'
        )
        ax.plot(
            pf_NLL, pf_MIG,
            '-', color='#A93226', linewidth=1.0,
            label='Pareto front'
        )
        ax.scatter(
            pf_NLL, pf_MIG,
            s=20,
            c='#A93226'
        )
        if baseline_point is not None:
            ax.scatter(
                baseline_point['NLL'], baseline_point['MIG'],
                s=20,
                c='#4682B4',
                marker='D',
                label=r'$(\beta,\lambda)=(1,0)$'
            )
        ax.set_xlabel("NLL", fontsize=10)
        ax.set_ylabel("MIG", fontsize=10)
        ax.set_title(f"{ds_name}", fontsize=10, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.4)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(
            loc='best',
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor='#BBBBBB'
        )
        if all_NLL and all_MIG:
            x_min, x_max = min(all_NLL), max(all_NLL)
            y_min, y_max = min(all_MIG), max(all_MIG)
            x_pad = 0.035 * (x_max - x_min if x_max > x_min else 1.0)
            y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        # Pareto table
        col_labels = [r'$\beta$', r'$\lambda$', 'NLL', 'MIG']
        cell_text = [
            [
                f"{p['beta']}",
                f"{p['lam']}",
                f"{p['NLL']:.2f}",
                f"{p['MIG']:.3f}",
            ]
            for p in pareto_sorted
        ]
        n_rows = len(cell_text) + 1
        table_height = min(0.88, 0.055 * n_rows)
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='center',
            colLoc='center',
            bbox=[0.0, 1.0 - table_height, 1.0, table_height],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.02, 1.0)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#F7F7F7')
            cell.set_linewidth(0.25)
        ax_table.set_title(
            "Pareto-optimal configurations",
            fontsize=9,
            pad=4
        )
        for spine in ax_table.spines.values():
            spine.set_visible(False)
    out_path = os.path.join(save_path, "pareto_fronts_across_datasets.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_interactive_tchebycheff_heatmap(results_dict, datasets, save_path, rho=0.001):
    """
    Generate an interactive heatmap using augmented Tchebycheff scalarization for each dataset.
    """
    for ds_name in datasets:
        results = results_dict[ds_name]
        mean_dict = get_mean_nll_mig(results)
        if not mean_dict:
            print(f"No valid data for {ds_name}")
            continue
        all_NLL = [d['NLL'] for d in mean_dict.values()]
        all_MIG = [d['MIG'] for d in mean_dict.values()]
        min_NLL, max_NLL = min(all_NLL), max(all_NLL)
        min_MIG, max_MIG = min(all_MIG), max(all_MIG)
        nll_range = max_NLL - min_NLL if max_NLL != min_NLL else 1.0
        mig_range = max_MIG - min_MIG if max_MIG != min_MIG else 1.0
        # Utopia point from observed configs
        utopia_NLL = min_NLL
        utopia_MIG = max_MIG
        w_pcts = range(101)
        score_matrices = []
        best_pairs = []
        min_scores = []
        for w_pct in w_pcts:
            w1 = w_pct / 100.0  # weight for NLL
            w2 = 1 - w1  # weight for MIG
            heatmap_data = np.full((len(Config.lambda_values), len(Config.betas)), np.nan)
            for i, lam in enumerate(Config.lambda_values):
                for j, beta in enumerate(Config.betas):
                    key = (beta, lam)
                    if key in mean_dict:
                        d = mean_dict[key]
                        dev_NLL = (d['NLL'] - utopia_NLL) / nll_range
                        dev_MIG = (utopia_MIG - d['MIG']) / mig_range
                        max_dev = max(w1 * dev_NLL, w2 * dev_MIG)
                        score = max_dev + rho * (w1 * dev_NLL + w2 * dev_MIG)
                        heatmap_data[i, j] = score
            score_matrices.append(heatmap_data)
            if not np.all(np.isnan(heatmap_data)):
                i_min, j_min = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
                best_pairs.append((Config.betas[j_min], Config.lambda_values[i_min]))
                min_scores.append(np.nanmin(heatmap_data))
            else:
                best_pairs.append((None, None))
                min_scores.append(np.nan)
        x_ticks = list(range(len(Config.betas)))
        y_ticks = list(range(len(Config.lambda_values)))
        initial_idx = 50
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=score_matrices[initial_idx],
                x=x_ticks,
                y=y_ticks,
                colorscale='viridis_r',
                colorbar=dict(title='Score'),
                texttemplate="%{z:.3f}",
                textfont={"size": 12, "color": "black"},
                hoverinfo='none'
            )
        )
        fig.update_xaxes(tickmode='array', tickvals=x_ticks, ticktext=Config.betas, title_text="β")
        fig.update_yaxes(tickmode='array', tickvals=y_ticks, ticktext=Config.lambda_values,
                         autorange='reversed', title_text="λ")
        bounds_x = get_bounds(x_ticks)
        bounds_y = get_bounds(y_ticks)
        initial_best_beta, initial_best_lam = best_pairs[initial_idx]
        if initial_best_beta is not None:
            initial_j = Config.betas.index(initial_best_beta)
            initial_i = Config.lambda_values.index(initial_best_lam)
            initial_x0 = bounds_x[initial_j]
            initial_x1 = bounds_x[initial_j + 1]
            initial_y0 = bounds_y[initial_i]
            initial_y1 = bounds_y[initial_i + 1]
            fig.update_layout(
                shapes=[dict(type="rect", x0=initial_x0, x1=initial_x1,
                             y0=initial_y0, y1=initial_y1,
                             line=dict(color="red", width=2))]
            )
        steps = []
        for idx, w_pct in enumerate(w_pcts):
            best_beta, best_lam = best_pairs[idx]
            if best_beta is None:
                continue
            z_data = score_matrices[idx]
            j = Config.betas.index(best_beta)
            i = Config.lambda_values.index(best_lam)
            x0 = bounds_x[j]
            x1 = bounds_x[j + 1]
            y0 = bounds_y[i]
            y1 = bounds_y[i + 1]
            shape = dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                         line=dict(color="red", width=2))
            step = dict(
                method='update',
                label=f'{w_pct}',
                args=[{'z': [z_data]}, {'shapes': [shape]}]
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
            title=f"Tchebycheff Scalarization Heatmap for {ds_name}",
            height=500, width=600,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_path = os.path.join(save_path, f"interactive_tchebycheff_heatmap_{ds_name}.html")
        fig.write_html(plot_path)

def plot_static_tchebycheff_heatmaps(results_dict, datasets, save_path, rho=0.001):
    """
    Generate a single static heatmap figure showing Tchebycheff scores across weights (10%, 50%, 90%).
    """
    w_pcts = [10, 50, 90]
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for col, ds_name in enumerate(datasets):
        results = results_dict[ds_name]
        mean_dict = get_mean_nll_mig(results)
        if not mean_dict:
            print(f"No valid data for {ds_name}")
            continue
        all_NLL = [d['NLL'] for d in mean_dict.values()]
        all_MIG = [d['MIG'] for d in mean_dict.values()]
        min_NLL, max_NLL = min(all_NLL), max(all_NLL)
        min_MIG, max_MIG = min(all_MIG), max(all_MIG)
        nll_range = max_NLL - min_NLL if max_NLL != min_NLL else 1.0
        mig_range = max_MIG - min_MIG if max_MIG != min_MIG else 1.0
        utopia_NLL = min_NLL
        utopia_MIG = max_MIG
        for row, w_pct in enumerate(w_pcts):
            ax = axes[row, col]
            w1 = w_pct / 100.0
            w2 = 1 - w1
            heatmap_data = np.zeros((len(Config.lambda_values), len(Config.betas)))
            for i, lam in enumerate(Config.lambda_values):
                for j, beta in enumerate(Config.betas):
                    key = (beta, lam)
                    if key in mean_dict:
                        d = mean_dict[key]
                        dev_NLL = (d['NLL'] - utopia_NLL) / nll_range
                        dev_MIG = (utopia_MIG - d['MIG']) / mig_range
                        max_dev = max(w1 * dev_NLL, w2 * dev_MIG)
                        score = max_dev + rho * (w1 * dev_NLL + w2 * dev_MIG)
                        heatmap_data[i, j] = score
                    else:
                        heatmap_data[i, j] = np.nan
            im = ax.imshow(heatmap_data, cmap='viridis_r', aspect='auto')
            for i, j in np.ndindex(heatmap_data.shape):
                val = heatmap_data[i, j]
                text = f'{val:.3f}' if not np.isnan(val) else ''
                ax.text(j, i, text, ha='center', va='center', color='black', fontsize=15)
            if not np.all(np.isnan(heatmap_data)):
                i_min, j_min = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
                ax.add_patch(Rectangle((j_min - 0.5, i_min - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))
            ax.set_xticks(np.arange(len(Config.betas)))
            ax.set_yticks(np.arange(len(Config.lambda_values)))
            if row == 2:
                ax.set_xticklabels([f'{b}' for b in Config.betas], fontsize=13)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels([f'{l}' for l in Config.lambda_values], fontsize=13)
                ax.tick_params(axis='y', which='both', labelsize=12, length=5, left=True, labelleft=True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', fontsize=13)
            if row == 0:
                ax.set_title(ds_name, fontsize=15)
            if col == 0:
                ax.text(-1.1, (len(Config.lambda_values)-1)/2, f'Reconstruction Weight: {w_pct}%', va='center', ha='right', rotation=90, fontsize=13)
            if row == 2:
                ax.set_xlabel(r'$\beta$', fontsize=13)
            ax.set_ylim(len(Config.lambda_values) - 0.5, -0.5)
            if col == 0:
                ax.set_ylabel(r'$\lambda$', fontsize=13)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    heatmaps_path = os.path.join(save_path, "static_tchebycheff_heatmaps_across_datasets.png")
    plt.tight_layout()
    plt.savefig(heatmaps_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
        
plot_metric_heatmaps(results_dict, datasets, save_path)
plot_pareto_front(results_dict, datasets, save_path)
plot_interactive_tchebycheff_heatmap(results_dict, datasets, save_path)
plot_static_tchebycheff_heatmaps(results_dict, datasets, save_path)