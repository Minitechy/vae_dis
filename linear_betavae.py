# linear_betavae.py
"""Main script for linear β-VAE and λβ-VAE experiments using fixed-point and AdamW optimization."""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from scipy.linalg import inv
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_spd_matrix
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from tqdm import tqdm
from termcolor import colored
from dataclasses import dataclass, field

# Configure threading for numerical libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# =========================
# Configuration
# =========================
@dataclass
class Config:
    """Configuration for experiment parameters."""
    EPSILON = 5e-5
    NUM_SOLS = 50
    BETA_ARR = np.array([1, 4, 8, 16, 32])
    LAMBDA_ARR = np.array([0, 2, 4, 6, 8])
    N_VALUES = [50, 100]
    MS_MAP = {50: (10, 5), 100: (10, 5)}
    MAX_ITERS_MAP = {
        'fixed_point': {50: 50000, 100: 100000},
        'adamw': {50: 30000, 100: 50000}
    }
    TOL_ERR_MAP = {50: 1e-6, 100: 1e-5}
    GRAD_NORM_MAP = {
        'adamw': {50: {'AB': 1.0, 'LZ_LW': 2.0}, 100: {'AB': 2.0, 'LZ_LW': 3.0}}
    }
    SAVE_PATH = os.path.join(os.getcwd(), "linear_betavae_results")
    METHODS = ['fixed_point', 'adamw']

CONFIG = Config()
os.makedirs(CONFIG.SAVE_PATH, exist_ok=True)

# =========================
# Utilities
# =========================
def generate_diag_cov(s, min_var=0.1, max_var=1.0):
    """Generate diagonal covariance matrix with uniform random variances."""
    return np.diag(np.random.uniform(min_var, max_var, s))

def initialize_inputs(n, m, method, seed=None):
    """Generate initial matrices for optimization."""
    if method == 'fixed_point':
        B = np.random.rand(m, n)
        Sigma_W = make_spd_matrix(m)
        return np.concatenate((B.ravel(), Sigma_W.ravel()))
    else:
        torch.manual_seed(seed)
        A = torch.randn(n, m, dtype=torch.float64)
        B = torch.randn(m, n, dtype=torch.float64)
        Sigma_Z = torch.tensor(make_spd_matrix(n), dtype=torch.float64)
        L_Z = torch.linalg.cholesky(Sigma_Z)
        Sigma_W = torch.tensor(make_spd_matrix(m), dtype=torch.float64)
        L_W = torch.linalg.cholesky(Sigma_W)
        return np.concatenate((A.numpy().ravel(), B.numpy().ravel(), L_Z.numpy().ravel(), L_W.numpy().ravel()))

# =========================
# Loss Computations
# =========================
def compute_recon_err(A, B, Sigma_Z, Sigma_W, n, Sigma_Y, is_torch=False):
    """Compute negative log-likelihood reconstruction error."""
    Sigma_X = B @ Sigma_Y @ B.T + Sigma_W
    inv_Sigma_Z = torch.inverse(Sigma_Z) if is_torch else inv(Sigma_Z)
    pi_term = torch.tensor(2 * np.pi, dtype=torch.float64) if is_torch else 2 * np.pi
    det_func = torch.det if is_torch else np.linalg.det
    trace_func = torch.trace if is_torch else np.trace
    return -0.5 * (
        trace_func(A.T @ inv_Sigma_Z @ Sigma_Y @ B.T) +
        trace_func(inv_Sigma_Z @ A @ B @ Sigma_Y) -
        trace_func(inv_Sigma_Z @ Sigma_Y) -
        trace_func(A.T @ inv_Sigma_Z @ A @ Sigma_X) -
        n * (torch.log(pi_term) if is_torch else np.log(pi_term)) -
        (torch.log(det_func(Sigma_Z)) if is_torch else np.log(det_func(Sigma_Z)))
    )

def compute_kl(Sigma_X, Sigma_W, m, is_torch=False):
    """Compute KL-divergence term."""
    det_func = torch.det if is_torch else np.linalg.det
    trace_func = torch.trace if is_torch else np.trace
    log_term = torch.log(det_func(Sigma_W)) if is_torch else np.log(det_func(Sigma_W))
    return 0.5 * (trace_func(Sigma_X) - log_term - m)

def compute_l2_norm_loss(A, B, Sigma_Y, Sigma_W, n, is_torch=False):
    """Compute L2-norm regularization loss."""
    eye_n = torch.eye(n, dtype=torch.float64) if is_torch else np.eye(n)
    trace_func = torch.trace if is_torch else np.trace
    return trace_func((eye_n - A @ B) @ Sigma_Y @ (eye_n - A @ B).T + A @ Sigma_W @ A.T)

def compute_vae_loss(A, B, Sigma_Z, Sigma_W, beta, n, m, Sigma_Y, lambda_=0, is_torch=False):
    """Compute VAE loss with optional λ regularization."""
    Sigma_X = B @ Sigma_Y @ B.T + Sigma_W
    recon_err = compute_recon_err(A, B, Sigma_Z, Sigma_W, n, Sigma_Y, is_torch)
    kl = compute_kl(Sigma_X, Sigma_W, m, is_torch)
    beta_loss = recon_err + (torch.tensor(beta, dtype=torch.float64) if is_torch else beta) * kl
    total_loss = beta_loss
    if lambda_ != 0:
        l2_norm_loss = compute_l2_norm_loss(A, B, Sigma_Y, Sigma_W, n, is_torch)
        total_loss += (torch.tensor(lambda_, dtype=torch.float64) if is_torch else lambda_) * l2_norm_loss
    return recon_err, total_loss

# =========================
# Metrics
# =========================
def compute_SAP(cov_XV, m, s):
    """Compute SAP score for disentanglement evaluation."""
    cov_XV_block = cov_XV[:m, m:m+s]
    variances_X = np.diag(cov_XV)[:m, None]
    variances_V = np.diag(cov_XV)[m:m+s, None].T
    squared_correlations = (cov_XV_block ** 2) / (variances_X @ variances_V)
    top_indices = np.argsort(squared_correlations, axis=0)[-2:][::-1]
    sap_score = np.mean(
        squared_correlations[top_indices[0], np.arange(s)] -
        squared_correlations[top_indices[1], np.arange(s)]
    )
    return sap_score if not np.isnan(sap_score) else np.nan, squared_correlations

def compute_Im(cov_XV, m, s):
    """Compute maximum mutual information sum using Hungarian algorithm."""
    cov_XV_block = cov_XV[:m, m:m+s]
    variances_X = np.diag(cov_XV)[:m, None]
    variances_V = np.diag(cov_XV)[m:m+s, None].T
    rho_ij_sq = (cov_XV_block ** 2) / (variances_X @ variances_V)
    mi_XV = -0.5 * np.log(1 - rho_ij_sq)
    cost_matrix = -mi_XV
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mi_max = -cost_matrix[row_ind, col_ind].sum()
    return mi_max if not np.isnan(mi_max) else np.nan, mi_XV

# =========================
# Optimization
# =========================
def optimize(inputs, beta, lambda_, n, m, s, Sigma_Y, Gamma, Sigma_V, method, tol_err=None, inv_Sigma_Y=None):
    """Perform optimization for β-VAE or λβ-VAE using specified method."""
    if method == 'fixed_point':
        if inv_Sigma_Y is None:
            raise ValueError("inv_Sigma_Y required for fixed_point method")
        B = inputs[:m*n].reshape(m, n)
        Sigma_W = inputs[m*n:].reshape(m, m)
        current_encoder = inputs.copy()
        flag = True
        for i in range(CONFIG.MAX_ITERS_MAP[method][n]):
            if not flag:
                scaled_inv_Sigma_Z = (inv(Sigma_Z) + 2 * lambda_ * np.eye(n)) / beta if lambda_ != 0 else inv(Sigma_Z) / beta
                B = inv(np.eye(m) + A.T @ scaled_inv_Sigma_Z @ A) @ A.T @ scaled_inv_Sigma_Z
                Sigma_W = inv(np.eye(m) + A.T @ scaled_inv_Sigma_Z @ A)
                flag = True
            B_diff_norm = np.linalg.norm(B - current_encoder[:m*n].reshape(m, n), 'fro')
            Sigma_W_diff_norm = np.linalg.norm(Sigma_W - current_encoder[m*n:].reshape(m, m), 'fro')
            if B_diff_norm <= tol_err and Sigma_W_diff_norm <= tol_err:
                break
            current_encoder = np.concatenate((B.ravel(), Sigma_W.ravel()))
        else:
            inv_Sigma_W = inv(Sigma_W)
            A = inv(inv_Sigma_Y + B.T @ inv_Sigma_W @ B) @ B.T @ inv_Sigma_W
            Sigma_Z = inv(inv_Sigma_Y + B.T @ inv_Sigma_W @ B)
            flag = False
            if i == 0:
                current_decoder = np.concatenate((A.ravel(), Sigma_Z.ravel()))
            else:
                A_diff_norm = np.linalg.norm(A - current_decoder[:n*m].reshape(n, m), 'fro')
                Sigma_Z_diff_norm = np.linalg.norm(Sigma_Z - current_decoder[n*m:].reshape(n, n), 'fro')
                if A_diff_norm <= tol_err and Sigma_Z_diff_norm <= tol_err:
                    break
                current_decoder = np.concatenate((A.ravel(), Sigma_Z.ravel()))
        if lambda_ == 0:
            recon, _ = compute_vae_loss(A, B, Sigma_Z, Sigma_W, beta, n, m, Sigma_Y)
        else:
            recon, _ = compute_vae_loss(A, B, Sigma_Z, Sigma_W, beta, n, m, Sigma_Y, lambda_)
    else:
        A = torch.tensor(inputs[:n*m].reshape(n, m), dtype=torch.float64, requires_grad=True)
        B = torch.tensor(inputs[n*m:n*m+m*n].reshape(m, n), dtype=torch.float64, requires_grad=True)
        L_Z = torch.tensor(inputs[n*m+m*n:n*m+m*n+n*n].reshape(n, n), dtype=torch.float64, requires_grad=True)
        L_W = torch.tensor(inputs[n*m+m*n+n*n:].reshape(m, m), dtype=torch.float64, requires_grad=True)
        Sigma_Y_torch = torch.tensor(Sigma_Y, dtype=torch.float64)
        optimizer = torch.optim.AdamW([A, B, L_Z, L_W], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        best_loss = float('inf')
        best_metrics = None
        for i in range(CONFIG.MAX_ITERS_MAP[method][n]):
            optimizer.zero_grad()
            Sigma_Z = L_Z @ L_Z.T + CONFIG.EPSILON * torch.eye(n, dtype=torch.float64)
            Sigma_W = L_W @ L_W.T + CONFIG.EPSILON * torch.eye(m, dtype=torch.float64)
            recon_err, loss = compute_vae_loss(A, B, Sigma_Z, Sigma_W, beta, n, m, Sigma_Y_torch, lambda_, is_torch=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([A, B], max_norm=CONFIG.GRAD_NORM_MAP[method][n]['AB'])
            torch.nn.utils.clip_grad_norm_([L_Z, L_W], max_norm=CONFIG.GRAD_NORM_MAP[method][n]['LZ_LW'])
            optimizer.step()
            L_Z.data = torch.tril(L_Z.data)
            L_W.data = torch.tril(L_W.data)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_metrics = (
                    A.detach().numpy().copy(),
                    B.detach().numpy().copy(),
                    Sigma_Z.detach().numpy().copy(),
                    Sigma_W.detach().numpy().copy(),
                    recon_err.item()
                )
        A, B, Sigma_Z, Sigma_W, recon = best_metrics
    Sigma_X = B @ Sigma_Y @ B.T + Sigma_W
    Sigma_XV = B @ Gamma @ Sigma_V
    cov_XV = np.block([[Sigma_X, Sigma_XV], [Sigma_XV.T, Sigma_V]])
    SAP, _ = compute_SAP(cov_XV, m, s)
    Im, _ = compute_Im(cov_XV, m, s)
    return float(recon), float(SAP), float(Im)

# =========================
# Visualization
# =========================
def plot_boxplots(n, m, s):
    """Generate box plots for β-VAE metrics across all methods."""
    method_display_map = {
        'fixed_point': 'Fixed Point',
        'adamw': 'AdamW'
    }
    boxprops = dict(facecolor="#E6F3FF", edgecolor="black", linewidth=1)
    positions = np.arange(len(CONFIG.BETA_ARR))
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    lambda_idx = np.where(CONFIG.LAMBDA_ARR == 0)[0][0]
    for col, method in enumerate(CONFIG.METHODS):
        recon_data = globals()[f"{method}_recon_n{n}"]
        sap_data = globals()[f"{method}_SAP_n{n}"]
        im_data = globals()[f"{method}_Im_n{n}"]
        method_display = method_display_map[method]
        data_dict = {
            'recon': [recon_data[i, lambda_idx, :] for i in range(len(CONFIG.BETA_ARR))],
            'sap': [sap_data[i, lambda_idx, :] for i in range(len(CONFIG.BETA_ARR))],
            'im': [im_data[i, lambda_idx, :] for i in range(len(CONFIG.BETA_ARR))]
        }
        for row, (data_key, ylabel, title) in enumerate([
            ('recon', 'Reconstruction Error', 'Reconstruction Error'),
            ('sap', 'SAP Score', 'SAP Score'),
            ('im', r'$I_m$ Score', r'$I_m$ Score')
        ]):
            ax = axes[row, col]
            ax.boxplot(
                data_dict[data_key],
                positions=positions,
                widths=0.4,
                showmeans=True,
                patch_artist=True,
                boxprops=boxprops
            )
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(method_display, fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(positions)
            if row == 2:
                ax.set_xticklabels([f"{b}" for b in CONFIG.BETA_ARR], fontsize=12)
            else:
                ax.set_xticklabels([])
            if row == 2:
                ax.set_xlabel(r"$\beta$", fontsize=12)
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color="orange", linewidth=2, label="Median"),
                    plt.Line2D([0], [0], marker="^", color="green", linestyle="None", markersize=6, label="Mean")
                ],
                loc="best",
                fontsize=8,
                frameon=True,
                edgecolor="black",
                framealpha=0.9
            )
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.SAVE_PATH, f"boxplots_betavae_n{n}_m{m}_s{s}.png"))
    plt.close()

def plot_heatmap(n, m, s):
    """Generate heatmap plots for λβ-VAE metrics across all methods."""
    method_display_map = {
        'fixed_point': 'Fixed Point',
        'adamw': 'AdamW'
    }
    recon_cmap = LinearSegmentedColormap.from_list('custom_red', ['#FFE6E6', '#A93226'])
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#E6F3FF', '#4682B4'])
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    beta_idx = np.where(CONFIG.BETA_ARR == 1)[0][0]
    lambda_idx = np.where(CONFIG.LAMBDA_ARR == 0)[0][0]
    for col, method in enumerate(CONFIG.METHODS):
        metrics = {
            'recon': np.nanmean(globals()[f'{method}_recon_n{n}'], axis=-1).T,
            'sap': np.nanmean(globals()[f'{method}_SAP_n{n}'], axis=-1).T,
            'im': np.nanmean(globals()[f'{method}_Im_n{n}'], axis=-1).T
        }
        method_display = method_display_map[method]
        for row, (mean, title, cmap) in enumerate([
            (metrics['recon'], 'Mean Reconstruction Error', recon_cmap),
            (metrics['sap'], 'Mean SAP Score', blue_cmap),
            (metrics['im'], 'Mean $I_m$ Score', blue_cmap)
        ]):
            ax = axes[row, col]
            im = ax.imshow(mean, cmap=cmap, aspect='auto')
            if row == 0:
                ax.set_title(f"{method_display}", fontsize=13)
            ax.set_yticks(np.arange(len(CONFIG.LAMBDA_ARR)))
            ax.set_yticklabels([f'{l}' for l in CONFIG.LAMBDA_ARR] if col == 0 else [])
            ax.set_xticks(np.arange(len(CONFIG.BETA_ARR)))
            ax.set_xticklabels([f'{b}' for b in CONFIG.BETA_ARR] if row == 2 else [])
            ax.grid(False)
            plt.colorbar(im, ax=ax, label=title)
            for i, j in np.ndindex(len(CONFIG.LAMBDA_ARR), len(CONFIG.BETA_ARR)):
                ax.text(j, i, f'{mean[i, j]:.3f}', ha='center', va='center', color='black', fontsize=12)
                if i == lambda_idx and j == beta_idx:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))
            if col == 0:
                ax.set_ylabel(r'$\lambda$')
            if row == 2:
                ax.set_xlabel(r'$\beta$')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.SAVE_PATH, f'heatmap_lambda_betavae_n{n}_m{m}_s{s}.png'))
    plt.close()

# =========================
# Main
# =========================
if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    for n in CONFIG.N_VALUES:
        m, s = CONFIG.MS_MAP[n]
        print(colored(f'\nn = {n}, m = {m}, s = {s}:', 'blue', attrs=['bold']))
        # Initialize metric arrays
        shape = (len(CONFIG.BETA_ARR), len(CONFIG.LAMBDA_ARR), CONFIG.NUM_SOLS)
        for method in CONFIG.METHODS:
            globals()[f'{method}_recon_n{n}'] = np.full(shape, np.nan)
            globals()[f'{method}_SAP_n{n}'] = np.full(shape, np.nan)
            globals()[f'{method}_Im_n{n}'] = np.full(shape, np.nan)
        # Generate covariance matrices
        Sigma_V = generate_diag_cov(s)
        np.savez(os.path.join(CONFIG.SAVE_PATH, f'Sigma_V_n{n}_m{m}_s{s}.npz'), Sigma_V=Sigma_V)
        Gamma = np.random.randn(n, s)
        Sigma_Y = Gamma @ Sigma_V @ Gamma.T + 0.05 * np.eye(n)
        inv_Sigma_Y = inv(Sigma_Y)
        np.savez(os.path.join(CONFIG.SAVE_PATH, f'Gamma_Sigma_Y_n{n}_m{m}_s{s}.npz'), Gamma=Gamma, Sigma_Y=Sigma_Y)
        # Initialize parameters
        initial_params = {
            'fixed_point': [initialize_inputs(n, m, 'fixed_point') for _ in range(CONFIG.NUM_SOLS)],
            'adamw': [initialize_inputs(n, m, 'adamw', seed=42 + i) for i in range(CONFIG.NUM_SOLS)]
        }
        # Run optimization for each method sequentially
        for method in CONFIG.METHODS:
            print(colored(f'\nRunning {method.upper()} method...', attrs=['bold']))
            for b_idx, beta in enumerate(CONFIG.BETA_ARR):
                for l_idx, lam in enumerate(CONFIG.LAMBDA_ARR):
                    results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(optimize)(
                            inputs, beta, lam, n, m, s, Sigma_Y, Gamma, Sigma_V, method,
                            tol_err=CONFIG.TOL_ERR_MAP[n] if method == 'fixed_point' else None,
                            inv_Sigma_Y=inv_Sigma_Y if method == 'fixed_point' else None
                        ) for inputs in tqdm(initial_params[method], desc=f"β={beta}, λ={lam}", ncols=100)
                    )
                    recon_vals, SAP_vals, Im_vals = zip(*results)
                    globals()[f'{method}_recon_n{n}'][b_idx, l_idx, :len(recon_vals)] = recon_vals
                    globals()[f'{method}_SAP_n{n}'][b_idx, l_idx, :len(SAP_vals)] = SAP_vals
                    globals()[f'{method}_Im_n{n}'][b_idx, l_idx, :len(Im_vals)] = Im_vals
        plot_boxplots(n, m, s)
        plot_heatmap(n, m, s)