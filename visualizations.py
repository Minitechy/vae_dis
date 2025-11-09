# visualizations.py
"""Visualization functions for images, reconstructions, traversals, and heatmaps."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from config import CONFIG

def save_distinct_images(dataset, indices, save_path, channels, seed):
    """Save distinct images from dataset."""
    if len(indices) != CONFIG['num_distinct_images']:
        raise ValueError(f"Expected {CONFIG['num_distinct_images']} indices, got {len(indices)}")
    os.makedirs(save_path, exist_ok=True)
    for i, idx in enumerate(indices):
        image, _ = dataset[idx]
        image_save_path = os.path.join(save_path, f"image_{i+1}_seed_{seed}.png")
        if not os.path.exists(image_save_path):
            plt.imsave(image_save_path, image.squeeze() if channels == 1 else image.permute(1, 2, 0).numpy(),
                       cmap='gray' if channels == 1 else None)
        plt.close()

def visualize_originals_grid(fixed_batch, save_path, channels, seed, num_images=64):
    """Visualize a grid of original images from a fixed batch (once per seed)."""
    images = fixed_batch[:num_images].cpu().numpy()
    img_dim = fixed_batch.shape[-1]
    grid_shape = (8 * img_dim, 8 * img_dim, 3) if channels == 3 else (8 * img_dim, 8 * img_dim)
    original_grid = np.ones(grid_shape)
    for row in range(8):
        for col in range(8):
            idx = row * 8 + col
            if idx < num_images:
                img = images[idx]
                if channels == 3:
                    img = img.transpose(1, 2, 0)
                else:
                    img = img.squeeze()
                original_grid[row*img_dim:(row+1)*img_dim, col*img_dim:(col+1)*img_dim] = img
    plt.figure(figsize=(10, 10))
    plt.imshow(original_grid, cmap='gray' if channels == 1 else None)
    plt.axis('off')
    original_path = os.path.join(save_path, f"original_images_seed_{seed}.png")
    if not os.path.exists(original_path):
        plt.savefig(original_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_reconstructions_grid(model, fixed_batch, beta, save_path, channels, seed, lambda_=None, num_images=64):
    """Visualize a grid of reconstructed images from a fixed batch."""
    model.eval()
    images = fixed_batch[:num_images]
    with torch.no_grad():
        mu, _ = model.encoder(images)
        recon_images = torch.clamp(model.decoder(mu), 0, 1).cpu().numpy()
    img_dim = images.shape[-1]
    grid_shape = (8 * img_dim, 8 * img_dim, 3) if channels == 3 else (8 * img_dim, 8 * img_dim)
    reconstructed_grid = np.ones(grid_shape)
    for row in range(8):
        for col in range(8):
            idx = row * 8 + col
            if idx < num_images:
                recon = recon_images[idx]
                if channels == 3:
                    recon = recon.transpose(1, 2, 0)
                else:
                    recon = recon.squeeze()
                reconstructed_grid[row*img_dim:(row+1)*img_dim, col*img_dim:(col+1)*img_dim] = recon
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_grid, cmap='gray' if channels == 1 else None)
    plt.axis('off')
    if lambda_ is None:
        recon_path = os.path.join(save_path, f"reconstructed_images_beta_{beta}_seed_{seed}.png")
    else:
        recon_path = os.path.join(save_path, f"reconstructed_images_beta_{beta}_lambda_{lambda_}_seed_{seed}.png")
    if not os.path.exists(recon_path):
        plt.savefig(recon_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_latent_traversal_combined(model, gif_path, indices, dataset, channels, latent_dim, num_traversals=11):
    """Generate a combined GIF of latent traversals for distinct images."""
    model.eval()
    interpolation = torch.linspace(-2, 2, num_traversals).to(CONFIG['device'])
    combined_gif_frames_all = [[] for _ in range(num_traversals)]
    for row, idx in enumerate(indices):
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(CONFIG['device'])
        mu, _ = model.encoder(image)
        combined_gif_frames = [[] for _ in range(num_traversals)]
        for latent_idx in range(latent_dim):
            z = mu.clone()
            for i, alpha in enumerate(interpolation):
                z_copy = z.clone()
                z_copy[:, latent_idx] = alpha
                with torch.no_grad():
                    generated = torch.clamp(model.decoder(z_copy), 0, 1)
                img = generated[0].cpu().numpy()
                img = img.transpose(1, 2, 0) if channels == 3 else img.squeeze()
                combined_gif_frames[i].append(np.uint8(255 * img))
        gif_frames_with_columns = [np.hstack(frame_list) for frame_list in combined_gif_frames]
        for i in range(num_traversals):
            combined_gif_frames_all[i].append(gif_frames_with_columns[i])
    final_gif_frames = [np.vstack(frame_list) for frame_list in combined_gif_frames_all]
    save_combined_gif(final_gif_frames, f"{gif_path}_combined.gif", channels, latent_dim)

def save_combined_gif(frames, output_path, channels, latent_dim):
    """Save latent traversal frames as a GIF with dimension labels."""
    gif_frames = []
    for frame in frames:
        if frame is not None and frame.size > 0:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(frame, cmap='gray' if channels == 1 else None, vmin=0, vmax=255)
            ax.axis('off')
            for i in range(latent_dim):
                x_pos = (frame.shape[1] / latent_dim) * (i + 0.5)
                ax.text(x_pos, -10, f'$x_{{{i + 1}}}$', fontsize=15, ha='center', va='bottom')
            fig.canvas.draw()
            gif_frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
            gif_frames.append(np.uint8(gif_frame))
            plt.close(fig)
    if gif_frames and not os.path.exists(output_path):
        imageio.mimsave(output_path, gif_frames, fps=5, loop=0)

def visualize_mi_heatmap(mi_matrix, factor_names, save_path, beta, seed, lambda_=None):
    """Visualize the mutual information heatmap between latents and factors."""
    plt.figure(figsize=(10, len(factor_names)))
    ax = sns.heatmap(mi_matrix.T, annot=True, fmt=".2f", cmap='Blues', annot_kws={'size': 15})
    ax.set_xlabel('Latent Dimensions', fontsize=13)
    ax.set_ylabel('Ground Truth Factors', fontsize=13)
    ax.set_xticks(np.arange(mi_matrix.shape[0]) + 0.5)
    ax.set_yticks(np.arange(len(factor_names)) + 0.5)
    ax.set_xticklabels([f'$x_{{{i+1}}}$' for i in range(mi_matrix.shape[0])], fontsize=12)
    ax.set_yticklabels(factor_names, fontsize=12)
    ax.collections[0].colorbar.set_label('Mutual Information', fontsize=13)
    plt.tight_layout()
    if lambda_ is None:
        heatmap_path = os.path.join(save_path, f"mi_heatmap_beta_{beta}_seed_{seed}.png")
    else:
        heatmap_path = os.path.join(save_path, f"mi_heatmap_beta_{beta}_lambda_{lambda_}_seed_{seed}.png")
    if not os.path.exists(heatmap_path):
        plt.savefig(heatmap_path)
    plt.close()

def plot_metric_boxplots(results_dict, datasets, save_path):
    """Generate box plots for β-VAE metrics across all datasets."""
    dataset_display_map = {
        "dSprites": "dSprites",
        "Shapes3D": "Shapes3D",
        "MPI3D": "MPI3D"
    }
    metrics = ["NLL", "SAP", "MIG", "Im"]
    metric_labels = ["Reconstruction Error", "SAP Score", "MIG Score", r"$I_m$ Score"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    boxprops = dict(facecolor="#E6F3FF", edgecolor="black", linewidth=1)
    positions = np.arange(len(CONFIG['betas']))
    for col, ds_name in enumerate(datasets):
        results = results_dict[ds_name]
        data_dict = {
            metric: [[r[metric] for r in results[beta]] for beta in CONFIG['betas']]
            for metric in metrics
        }
        dataset_display = dataset_display_map[ds_name]
        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            ax.boxplot(
                data_dict[metric], positions=positions, widths=0.4, showmeans=True,
                patch_artist=True, boxprops=boxprops
            )
            ax.set_ylabel(metric_label)
            if row == 0:
                ax.set_title(dataset_display, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(positions)
            if row == 3:
                ax.set_xticklabels([f"{b}" for b in CONFIG['betas']], fontsize=12)
            else:
                ax.set_xticklabels([])
            if row == 3:
                ax.set_xlabel(r"$\beta$", fontsize=12)
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color="orange", linewidth=2, label="Median"),
                    plt.Line2D([0], [0], marker="^", color="green", linestyle="None", markersize=6, label="Mean")
                ],
                loc="best", fontsize=8, frameon=True, edgecolor="black", framealpha=0.9
            )
    plt.tight_layout()
    boxplots_path = os.path.join(save_path, "metric_boxplots_across_datasets.png")
    if not os.path.exists(boxplots_path):
        plt.savefig(boxplots_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_heatmaps(results_dict, datasets, save_path):
    """Plot heatmaps for key metrics across betas, lambdas, and datasets."""
    metrics = ["NLL", "SAP", "MIG", "Im"]
    metric_labels = ["Mean Reconstruction Error", "Mean SAP Score", "Mean MIG Score", r"Mean $I_m$ Score"]
    recon_cmap = LinearSegmentedColormap.from_list('custom_red', ['#FFE6E6', '#A93226'])
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E6F3FF', '#4682B4'])
    cmap_dict = {"NLL": recon_cmap, "SAP": blue_cmap, "MIG": blue_cmap, "Im": blue_cmap}
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    beta_idx = CONFIG['betas'].index(1) if 1 in CONFIG['betas'] else 0
    lambda_idx = CONFIG['lambdas'].index(0) if 0 in CONFIG['lambdas'] else 0
    for col, ds_name in enumerate(datasets):
        results = results_dict[ds_name]
        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            heatmap_data = np.zeros((len(CONFIG['lambdas']), len(CONFIG['betas'])))
            for i, lambda_ in enumerate(CONFIG['lambdas']):
                for j, beta in enumerate(CONFIG['betas']):
                    lambda_results = results[beta][lambda_]
                    mean_value = np.mean([r[metric] for r in lambda_results])
                    heatmap_data[i, j] = mean_value
            im = ax.imshow(heatmap_data, cmap=cmap_dict[metric], aspect='auto')
            fmt = ".2f" if metric == "NLL" else ".3f"
            for i, j in np.ndindex(len(CONFIG['lambdas']), len(CONFIG['betas'])):
                ax.text(j, i, f'{heatmap_data[i, j]:{fmt}}', ha='center', va='center', color='black', fontsize=15)
                if i == lambda_idx and j == beta_idx:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))
            ax.set_xticks(np.arange(len(CONFIG['betas'])))
            ax.set_yticks(np.arange(len(CONFIG['lambdas'])))
            if row == 3:
                ax.set_xticklabels([f'{b}' for b in CONFIG['betas']], fontsize=13)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels([f'{l}' for l in CONFIG['lambdas']], fontsize=13)
                ax.tick_params(axis='y', which='both', labelsize=12, length=5, left=True, labelleft=True)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', labelsize=12, length=5, left=True, labelleft=False)
            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_label, fontsize=13)
            if row == 0:
                ax.set_title(ds_name, fontsize=15)
            if col == 0:
                ax.set_ylabel(r'$\lambda$', fontsize=13)
            if row == 3:
                ax.set_xlabel(r'$\beta$', fontsize=13)
            ax.set_ylim(len(CONFIG['lambdas']) - 0.5, -0.5)
    plt.tight_layout()
    heatmaps_path = os.path.join(save_path, "metric_heatmaps_across_datasets.png")
    if not os.path.exists(heatmaps_path):
        plt.savefig(heatmaps_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_summary(results, dataset_name, save_path, is_lambda=False):
    """Save summary results for metrics to a text file."""
    os.makedirs(save_path, exist_ok=True)
    summary_file = os.path.join(save_path, f"{dataset_name}_results_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Summary Results for {dataset_name} (Mean ± Std)\n")
        f.write("=" * 50 + "\n")
        if not is_lambda:
            for beta in CONFIG['betas']:
                beta_results = results[beta]
                means = {key: np.mean([r[key] for r in beta_results]) for key in beta_results[0]}
                stds = {key: np.std([r[key] for r in beta_results]) for key in beta_results[0]}
                f.write(f"beta = {beta}:\n")
                for key in ["NLL", "KL", "Total loss", "MIG", "SAP", "Im"]:
                    f.write(f" {key}: {means[key]:.4f} ± {stds[key]:.4f}\n")
                f.write("-" * 50 + "\n")
        else:
            for beta in CONFIG['betas']:
                for lambda_ in CONFIG['lambdas']:
                    lambda_results = results[beta][lambda_]
                    means = {key: np.mean([r[key] for r in lambda_results]) for key in lambda_results[0]}
                    stds = {key: np.std([r[key] for r in lambda_results]) for key in lambda_results[0]}
                    f.write(f"beta = {beta}, lambda = {lambda_}:\n")
                    for key in ["NLL", "KL", "L2", "Total loss", "MIG", "SAP", "Im"]:
                        f.write(f" {key}: {means[key]:.4f} ± {stds[key]:.4f}\n")
                    f.write("-" * 50 + "\n")