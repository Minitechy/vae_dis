# metrics.py
"""Disentanglement metric computations."""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import svm
from sklearn.metrics import mutual_info_score
from torch.utils.data import DataLoader, Subset

from config import CONFIG

def get_representations_and_factors(model, dataset, indices):
    """Extract latent representations and factors for given indices."""
    model.eval()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    mus, factors = [], []
    with torch.no_grad():
        for batch, factor_batch in loader:
            batch = batch.to(CONFIG['device'], non_blocking=True)
            mu, _ = model.encoder(batch)
            mus.append(mu.cpu().numpy())
            factors.append(factor_batch.numpy())
    return np.concatenate(mus, axis=0).T, np.concatenate(factors, axis=0).T

def histogram_discretizer(target, num_bins):
    """Discretize target values into bins."""
    discretized = np.zeros_like(target, dtype=np.int_)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(mus, ys):
    """Compute mutual information between latent variables and factors."""
    num_latents, num_factors = mus.shape[0], ys.shape[0]
    m = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m

def discrete_entropy(ys):
    """Compute entropy for each factor."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h

def discretize_factors(ys_train, ys_test):
    """Discretize factors for non-continuous data."""
    num_factors = ys_train.shape[0]
    ys_train_disc = np.zeros_like(ys_train, dtype=int)
    ys_test_disc = np.zeros_like(ys_test, dtype=int)
    for j in range(num_factors):
        col_train = np.round(ys_train[j, :], decimals=6)
        unique = np.sort(np.unique(col_train))
        ys_train_disc[j, :] = np.searchsorted(unique, col_train)
        col_test = np.round(ys_test[j, :], decimals=6)
        ys_test_disc[j, :] = np.searchsorted(unique, col_test)
    return ys_train_disc, ys_test_disc

def compute_Im(m):
    """
    Compute Im score: max sum of MI over unique latent-factor pairs using Hungarian algorithm.
    """
    if len(m.shape) != 2:
        raise ValueError("m must be a 2D array with shape (num_latents, num_factors)")
  
    cost = -m
    row_ind, col_ind = linear_sum_assignment(cost)
    Im = m[row_ind, col_ind].sum()
    return Im

def compute_metrics(mus_train, ys_train, mus_test, ys_test, num_bins, continuous_factors, seed):
    """Compute disentanglement metrics (MIG, SAP, Im)."""
    metrics = {}
    num_latents, num_factors = mus_train.shape[0], ys_train.shape[0]
    if not continuous_factors:
        ys_train, ys_test = discretize_factors(ys_train, ys_test)
    discretized_mus = histogram_discretizer(mus_train, num_bins)
    m = discrete_mutual_info(discretized_mus, ys_train)
  
    # MIG
    ent = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    metrics["MIG"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], ent, where=ent != 0))
  
    # SAP
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i, y_j = mus_train[i, :], ys_train[j, :]
            if continuous_factors:
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu, var_y = cov_mu_i_y_j[0, 0], cov_mu_i_y_j[1, 1]
                score_matrix[i, j] = (cov_mu_y / (var_mu * var_y)) if var_mu > 1e-12 else 0.0
            else:
                mu_i_test, y_j_test = mus_test[i, :], ys_test[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced", random_state=seed)
                classifier.fit(mu_i[:, np.newaxis], y_j)
                score_matrix[i, j] = np.mean(classifier.predict(mu_i_test[:, np.newaxis]) == y_j_test)
    sorted_matrix = np.sort(score_matrix, axis=0)
    metrics["SAP"] = np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
  
    # Im
    metrics["Im"] = compute_Im(m)
    return metrics, m