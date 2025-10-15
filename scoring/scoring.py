import numpy as np
import pandas as pd
from numba import njit

# -------------------------
# Numba-accelerated helpers
# -------------------------


@njit(cache=True, fastmath=True)
def normalize_numba(arr):
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mx, mn):
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


@njit(cache=True, fastmath=True)
def normalize_pos_only_numba(arr):
    out = np.zeros_like(arr)
    pos_mask = arr > 0
    if not np.any(pos_mask):
        return out
    mn = np.nanmin(arr[pos_mask])
    mx = np.nanmax(arr[pos_mask])
    if np.isclose(mx, mn):
        out[pos_mask] = 1.0
        return out
    out[pos_mask] = (arr[pos_mask] - mn) / (mx - mn)
    return out


@njit(cache=True, fastmath=True)
def weighted_score_numba(matrix, weights):
    """Compute row-wise dot product (matrix @ weights)."""
    n = matrix.shape[0]
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(weights.size):
            total += matrix[i, j] * weights[j]
        res[i] = total
    return res


def compute_scores(df_players, df_keepers, df_teams, prob_set, calendario,
                   pesi_gk=None, pesi_out=None):
    """
    Numba-optimized compute_scores (still returns pandas DataFrame)
    """

    # Copy only what we need (avoid overhead)
    df_p = df_players.copy()
    df_k = df_keepers.copy()

    w_gk = np.array([pesi_gk[k] for k in ["Titolarità", "Forma",
                    "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    gk = np.array([df_k[k] for k in ["Titolarità", "Forma",
                                     "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    punteggio_gk = gk.T @ w_gk
    score_gk_norm = normalize_numba(punteggio_gk) * 100

    w_out = np.array([pesi_out[k] for k in ["Titolarità", "Forma",
                     "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    out = np.array([df_p[k] for k in ["Titolarità", "Forma",
                                      "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    punteggio = out.T @ w_out
    score_norm = normalize_numba(punteggio) * 100

    df_p["Punteggio"] = punteggio
    df_p["Score"] = score_norm

    df_k["Punteggio"] = punteggio_gk
    df_k["Score"] = score_gk_norm

    # -----------------------
    # Final concat
    # -----------------------
    cols = ["Player", "Squad", "Pos", "Titolarità", "Forma", "BonusPot",
            "Affidabilità", "Calendario", "Penalità", "Punteggio", "Score", "chiave"]

    df_final = pd.concat([df_p[cols], df_k[cols]], ignore_index=True)
    return df_final
