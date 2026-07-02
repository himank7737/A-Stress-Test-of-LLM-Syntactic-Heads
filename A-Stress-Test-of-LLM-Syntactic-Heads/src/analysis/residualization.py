"""Fix 3: Root-Verb Residualization.

Root tokens behave differently from everything else - they tend to have
lower surprisal (sentence structure "commits" to a root early) and higher
attention confidence (nearly every dependent points at the root eventually,
directly or via chains, so root heads get a lot of practice). Left in
uncorrected, `is_root` acts as a confound that can inflate or deflate the
surprisal-confidence correlation depending on how the sample happens to be
distributed.

We regress both surprisal and confidence on binary is_root via OLS and use
the residuals for the correlation, not the raw values.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr


def residualize(y, covariate):
    """OLS residuals of y on [1, covariate]."""
    X = np.column_stack([np.ones(len(y)), np.array(covariate, dtype=float)])
    y = np.array(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def surprisal_confidence_correlation(df):
    """df needs columns: surprisal, conf_max, is_root.
    Returns dict with residualized r, rho, n, and p-values.
    """
    surp_res = residualize(df["surprisal"], df["is_root"])
    conf_res = residualize(df["conf_max"], df["is_root"])

    r, p_pearson = pearsonr(surp_res, conf_res)
    rho, p_spearman = spearmanr(surp_res, conf_res)

    return {
        "pearson_r": r,
        "pearson_p": p_pearson,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "n": len(df),
        "surp_residual": surp_res,
        "conf_residual": conf_res,
    }
