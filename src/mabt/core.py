import numpy as np

from scipy.stats import rankdata
from scipy.optimize import root_scalar


def mabt_ci(true_labels, pred_labels, alpha=0.05, B=10000, seed=None):
    """
    Computes the multiplicity-adjusted bootstrap tilting lower confidence bound as in 
    Pascal Rink (2025). Confidence Limits for Prediction Performance. Dissertation, 
    University of Bremen, https://doi.org/10.26092/elib/3822

    Parameters
    ----------
    true_labels : array-like, shape (n,)
        True binary labels (0 or 1).
    pred_labels : array-like, shape (n,) or (n, k)
        Predicted labels from k models.
    alpha : float
        Significance level (default: 0.05). 
    B : int
        Number of bootstrap resamples.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    bound : float
        Multiplicity-adjusted tilting lower (1-alpha) * 100 % confidence bound.
    tau : float
        Corresponding tilting parameter.
    t0 : float
        Point estimate of selected model (optimistic)
    """

    # --- Input validation ---
    true_labels = np.asarray(true_labels).squeeze()
    pred_labels = np.asarray(pred_labels)

    if true_labels.ndim != 1:
        raise ValueError(f"true_labels must be 1D, got shape {true_labels.shape}")

    if pred_labels.ndim == 1:
        pred_labels = pred_labels[:, None]
    elif pred_labels.ndim != 2:
        raise ValueError(f"pred_labels must be 1D or 2D, got shape {pred_labels.shape}")

    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError(
            f"true_labels ({true_labels.shape[0]}) and "
            f"pred_labels ({pred_labels.shape[0]}) have different lengths"
        )

    unique_vals = np.unique(true_labels)
    if not np.array_equal(unique_vals, [0, 1]):
        raise ValueError(f"true_labels must contain exactly {{0, 1}}, got {unique_vals}")

    # --- Helper functions ---
    def _stratified_bootstrap_sample(rng, n):
        idx_all = np.arange(n)
        idx0 = idx_all[true_labels == 0]
        idx1 = idx_all[true_labels == 1]
        boot = np.concatenate([
            rng.choice(idx0, size=len(idx0), replace=True),
            rng.choice(idx1, size=len(idx1), replace=True),
        ])
        rng.shuffle(boot)
        return boot

    def _multidim_bootstrap():
        
        similar_mat = (
            true_labels[:, None].astype(np.int64) == pred_labels.astype(np.int64)
        ).astype(int)

        n, k = similar_mat.shape
        mu0 = similar_mat.mean(axis=0)

        t_mat = np.empty((B, k))
        freq_mat = np.empty((B, n), dtype=int)

        rng = np.random.default_rng(seed)
        for b in range(B):
            idx = _stratified_bootstrap_sample(rng, n)
            freq_mat[b] = np.bincount(idx, minlength=n)
            similar_mat_b = similar_mat[idx]
            mu_b = similar_mat_b.mean(axis=0)
            std_err = np.asarray(similar_mat_b.std(ddof=1, axis=0) / np.sqrt(n))
            std_err = np.where(std_err == 0, 1 / np.sqrt(n), std_err) # avoid division by zero
            t_mat[b] = (mu_b - mu0) / std_err

        return similar_mat, t_mat, freq_mat

    def _tilting_weights(tau):
        w = np.exp(emp_influence_vals * tau)
        return w / w.sum()

    def _estim_p_value(tau):
        w = _tilting_weights(tau)
        xi = np.average(selected_vec, weights=w)
        t0_xi = (t0 - xi) / selected_vec.std(ddof=1)

        # Compute importance weights in log space for numerical stability
        log_importance_weights = freq_mat @ np.log(n * w)  # shape (B,)
        log_importance_weights -= log_importance_weights.max()  # normalize to avoid overflow
        
        importance_weights = np.exp(log_importance_weights)

        sorted_idx = np.argsort(t_mat[:, selected_idx])
        sorted_imp_weights = importance_weights[sorted_idx]
        cum_weights = np.cumsum(sorted_imp_weights)

        tilt_ecdf = lambda z: cum_weights[
            np.searchsorted(t_mat[sorted_idx, selected_idx], z, side="right")
        ] / cum_weights[-1]

        return float(1 - max_ecdf(tilt_ecdf(t0_xi)))

    def _find_tau(tau):
        return _estim_p_value(tau) - alpha

    def _find_bracket(f, lower=-10, upper=0, steps=200):
        vals = np.linspace(lower, upper, steps)
        f_vals = np.array([f(v) for v in vals])
        sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
        if len(sign_changes) == 0:
            raise RuntimeError(
                f"No valid bracket found for root_scalar. "
                f"Check alpha={alpha} and the input data."
            )
        i = sign_changes[0]
        return (vals[i], vals[i + 1])

    # --- Main computation ---
    similar_mat, t_mat, freq_mat = _multidim_bootstrap()

    B, n = freq_mat.shape
    t0_vec = similar_mat.mean(axis=0)
    selected_idx = np.argmax(t0_vec)
    t0 = t0_vec[selected_idx]
    selected_vec = similar_mat[:, selected_idx]
    emp_influence_vals = selected_vec

    unif_transformed = rankdata(t_mat, axis=0) / B
    supp = np.sort(unif_transformed.max(axis=1))
    max_ecdf = lambda t: np.searchsorted(supp, t, side="right") / B

    bracket = _find_bracket(_find_tau)
    tau = root_scalar(_find_tau, bracket=bracket, method="brentq").root
    bound = np.average(selected_vec, weights=_tilting_weights(tau))

    return bound, tau, t0