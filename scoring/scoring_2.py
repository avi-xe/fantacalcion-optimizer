import numpy as np
import pandas as pd
from numba import njit, prange

# ---------------- Numba helpers ----------------
@njit(fastmath=True, parallel=True)
def normalize_array(arr):
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    diff = max_val - min_val if max_val != min_val else 1.0
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        out[i] = (arr[i] - min_val) / diff
    return out

@njit(parallel=True)
def compute_scores_numba(features, weights):
    """
    features: shape (n_players, n_features)
    weights: shape (n_individuals, n_features)
    returns: shape (n_players, n_individuals)
    """
    n_players, n_features = features.shape
    n_individuals = weights.shape[0]
    out = np.zeros((n_players, n_individuals))
    for i in prange(n_players):
        for j in prange(n_individuals):
            s = 0.0
            for k in range(n_features):
                s += features[i, k] * weights[j, k]
            out[i, j] = s
    return out

# ---------------- Precompute features ----------------
def preprocess_features(df_players, df_keepers, df_teams, prob_set, calendario):
    # Titolarità
    titolari = set(prob_set["in_campo_casa"].explode()).union(
        set(prob_set["in_campo_ospite"].explode())
    )
    df_players["Titolarità"] = np.where(df_players["Player"].isin(titolari), 1.0, 0.3)
    df_keepers["Titolarità"] = np.where(df_keepers["Player"].isin(titolari), 1.0, 0.3)

    # Forma
    df_players["Forma"] = normalize_array((df_players["%xG"].fillna(0) + df_players["%xAG"].fillna(0)).values)
    df_keepers["Forma"] = normalize_array((df_keepers[["Save%", "PSxG+/-"]].mean(axis=1).fillna(0)).values)

    # BonusPot
    df_players["BonusPot"] = normalize_array(df_players["%G+A"].fillna(0).values)
    df_keepers["BonusPot"] = normalize_array(df_keepers["CS%"].fillna(0).values)

    # Affidabilità
    df_players["Affidabilità"] = normalize_array(((df_players["Starts"]/df_players["MP"].replace(0,1)) *
                                                  (df_players["Min"]/(df_players["MP"].replace(0,1)*90))).fillna(0).values)
    df_keepers["Affidabilità"] = normalize_array((df_keepers["Starts"]/df_keepers["MP"].replace(0,1)).fillna(0).values)

    # Penalità
    df_players["Penalità"] = normalize_array((df_players["CrdY"]*0.05 + df_players["CrdR"]*0.2).fillna(0).values)
    df_keepers["Penalità"] = normalize_array(df_keepers["GA90"].fillna(0).values)

    # Calendario (vectorized)
    team_stats = df_teams.set_index("Squadra")
    cal_players = []
    for _, row in df_players.iterrows():
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"]==squadra) | (calendario["Ospite"]==squadra)].head(1)
        if match.empty:
            cal_players.append(1.0)
        else:
            in_casa = match["Casa"].values[0] == squadra
            avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
            if avv in team_stats.index:
                cal_players.append(team_stats.loc[avv, "A_xGA"] if in_casa else team_stats.loc[avv, "H_xGA"])
            else:
                cal_players.append(1.0)
    df_players["Calendario"] = normalize_array(np.array(cal_players))

    cal_keepers = []
    for _, row in df_keepers.iterrows():
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"]==squadra) | (calendario["Ospite"]==squadra)].head(1)
        if match.empty:
            cal_keepers.append(1.0)
        else:
            in_casa = match["Casa"].values[0] == squadra
            avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
            if avv in team_stats.index:
                cal_keepers.append(team_stats.loc[avv, "A_xG"] if in_casa else team_stats.loc[avv, "H_xG"])
            else:
                cal_keepers.append(1.0)
    df_keepers["Calendario"] = normalize_array(np.array(cal_keepers))

    # Return features as arrays (order: Titolarità, Forma, BonusPot, Affidabilità, Calendario, Penalità)
    player_features = df_players[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)
    keeper_features = df_keepers[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)

    return player_features, keeper_features

# ---------------- Evaluate all individuals in one batch ----------------
def evaluate_population(individuals, player_features, keeper_features):
    """
    individuals: list of np.array of length 6+6 weights
    Returns: array of total scores per individual
    """
    n_ind = len(individuals)
    # Build weight matrices
    weights_out = np.array([ind[:6] for ind in individuals], dtype=np.float32)
    weights_gk  = np.array([ind[6:] for ind in individuals], dtype=np.float32)

    scores_out = compute_scores_numba(player_features, weights_out)
    scores_gk  = compute_scores_numba(keeper_features, weights_gk)

    # Take best 11 (vectorized approximation: sum top 11 scores)
    top11_scores = np.sum(np.sort(scores_out, axis=0)[-11:,:], axis=0)
    top_gk_scores = np.max(scores_gk, axis=0)  # pick best keeper

    total_scores = top11_scores + top_gk_scores
    return total_scores
