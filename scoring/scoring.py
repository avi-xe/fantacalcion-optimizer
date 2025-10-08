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
    df_t = df_teams.copy()
    cal = calendario.copy()

    # -----------------------
    # Build opponent mappings (same as before, vectorized)
    # -----------------------
    df_t_idx = df_t.set_index("Squadra", drop=False)
    team_to_xga = {}
    team_to_xg = {}

    for _, row in cal.iterrows():
        casa, ospite = row["Casa"], row["Ospite"]
        if casa not in team_to_xga:
            team_to_xga[casa] = df_t_idx.loc[ospite, "A_xGA"] if ospite in df_t_idx.index else np.nan
        if ospite not in team_to_xga:
            team_to_xga[ospite] = df_t_idx.loc[casa, "H_xGA"] if casa in df_t_idx.index else np.nan
        if casa not in team_to_xg:
            team_to_xg[casa] = df_t_idx.loc[ospite, "A_xG"] if ospite in df_t_idx.index else np.nan
        if ospite not in team_to_xg:
            team_to_xg[ospite] = df_t_idx.loc[casa, "H_xG"] if casa in df_t_idx.index else np.nan

    # -----------------------
    # TITOLARITÀ
    # -----------------------
    if prob_set is not None:
        titolari = set(prob_set["in_campo_casa"].explode().dropna().astype(str)) | \
                   set(prob_set["in_campo_ospite"].explode().dropna().astype(str))
        df_p["Titolarità"] = np.where(df_p["Player"].astype(str).isin(titolari), 1.0, -1)
        df_k["Titolarità"] = np.where(df_k["Player"].astype(str).isin(titolari), 1.0, -1)
    else:
        df_p["Titolarità"] = 1.0
        df_k["Titolarità"] = 1.0

    # -----------------------
    # OUTFIELDERS
    # -----------------------
    xg_plus_xag = (df_p["%xG"].to_numpy(float) + df_p["%xAG"].to_numpy(float))
    forma = normalize_pos_only_numba(xg_plus_xag)
    bonuspot = normalize_pos_only_numba(df_p["%G+A"].to_numpy(float))
    aff = (df_p["Starts"].to_numpy(float) / df_p["MP"].to_numpy(float)) * \
          (df_p["Min"].to_numpy(float) / (df_p["MP"].to_numpy(float) * 90))
    affid = normalize_numba(np.nan_to_num(aff))

    # Calendar
    opp_metric = df_p["Squad"].map(team_to_xga).fillna(1.0).to_numpy(float)
    calendario_norm = normalize_numba(opp_metric)

    pen = df_p["CrdY"].to_numpy(float) * 0.05 + df_p["CrdR"].to_numpy(float) * 0.2
    penalita = normalize_numba(pen)

    out_mat = np.vstack([
        df_p["Titolarità"].to_numpy(float),
        forma, bonuspot, affid, calendario_norm, penalita
    ]).T

    w_out = np.array([pesi_out[k] for k in ["Titolarità", "Forma", "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    punteggio = weighted_score_numba(out_mat, w_out)
    score_norm = normalize_numba(punteggio) * 100

    df_p["Punteggio"] = punteggio
    df_p["Score"] = score_norm

    # -----------------------
    # GOALKEEPERS
    # -----------------------
    forma_gk = normalize_pos_only_numba((df_k["Save%"].to_numpy(float) + df_k["PSxG+/-"].to_numpy(float)) / 2)
    bonuspot_gk = normalize_numba(df_k["CS%"].to_numpy(float))
    affid_gk = normalize_numba((df_k["Starts"].to_numpy(float) / df_k["MP"].to_numpy(float)))
    opp_metric_gk = df_k["Squad"].map(team_to_xg).fillna(1.0).to_numpy(float)
    calendario_gk = normalize_numba(opp_metric_gk)
    penalita_gk = normalize_numba(df_k["GA90"].to_numpy(float))

    gk_mat = np.vstack([
        df_k["Titolarità"].to_numpy(float),
        forma_gk, bonuspot_gk, affid_gk, calendario_gk, penalita_gk
    ]).T

    w_gk = np.array([pesi_gk[k] for k in ["Titolarità", "Forma", "BonusPot", "Affidabilità", "Calendario", "Penalità"]])
    punteggio_gk = weighted_score_numba(gk_mat, w_gk)
    score_gk_norm = normalize_numba(punteggio_gk) * 100

    df_k["Punteggio"] = punteggio_gk
    df_k["Score"] = score_gk_norm

    # -----------------------
    # Final concat
    # -----------------------
    cols = ["Player", "Squad", "Pos", "Titolarità", "Forma", "BonusPot",
            "Affidabilità", "Calendario", "Penalità", "Punteggio", "Score", "chiave"]

    df_final = pd.concat([df_p[cols], df_k[cols]], ignore_index=True)
    return df_final
