from numba import njit, prange
import numpy as np

from scraping.scraping_fantacalcio import crea_chiave_giocatore

@njit(fastmath=True, parallel=True)
def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    diff = max_val - min_val if max_val != min_val else 1.0
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        out[i] = (arr[i]-min_val)/diff
    return out

# ---------------- Preprocessing ----------------
def preprocess_features(df_players, df_keepers, df_teams, prob_set, calendario):
    # Convert numeric
    numeric_cols_players = ["%xG","%xAG","%G+A","Starts","MP","Min","CrdY","CrdR"]
    numeric_cols_keepers = ["Save%","PSxG+/-","CS%","Starts","MP","GA90"]
    df_players[numeric_cols_players] = df_players[numeric_cols_players].fillna(0).astype(np.float32)
    df_keepers[numeric_cols_keepers] = df_keepers[numeric_cols_keepers].fillna(0).astype(np.float32)

    # Titolarità
    titolari = set(prob_set["in_campo_casa"].explode()).union(set(prob_set["in_campo_ospite"].explode()))
    df_players = crea_chiave_giocatore(df_players)
    df_keepers = crea_chiave_giocatore(df_keepers)
    df_players["Titolarità"] = np.where(df_players["chiave"].isin(titolari),1,0).astype(np.float32)
    df_keepers["Titolarità"] = np.where(df_keepers["chiave"].isin(titolari),1,0).astype(np.float32)

    # Forma
    df_players["Forma"] = normalize_array((df_players["%xG"].values + df_players["%xAG"].values).astype(np.float32))
    df_keepers["Forma"] = normalize_array(df_keepers[["Save%","PSxG+/-"]].mean(axis=1).values.astype(np.float32))

    # BonusPot
    df_players["BonusPot"] = normalize_array(df_players["%G+A"].values.astype(np.float32))
    df_keepers["BonusPot"] = normalize_array(df_keepers["CS%"].values.astype(np.float32))

    # Affidabilità
    aff_players = (df_players["Starts"].values / np.where(df_players["MP"].values==0,1,df_players["MP"].values)) * \
                  (df_players["Min"].values / (np.where(df_players["MP"].values==0,1,df_players["MP"].values)*90))
    df_players["Affidabilità"] = normalize_array(aff_players.astype(np.float32))

    aff_keepers = df_keepers["Starts"].values / np.where(df_keepers["MP"].values==0,1,df_keepers["MP"].values)
    df_keepers["Affidabilità"] = normalize_array(aff_keepers.astype(np.float32))

    # Penalità
    df_players["Penalità"] = normalize_array((df_players["CrdY"].values*0.05 + df_players["CrdR"].values*0.2).astype(np.float32))
    df_keepers["Penalità"] = normalize_array(df_keepers["GA90"].values.astype(np.float32))

    # Calendario (simple loop)
    team_stats = df_teams.set_index("Squadra")
    def compute_cal(df,xga_col,xg_col):
        cal_scores = np.ones(len(df),dtype=np.float32)
        for i in range(len(df)):
            squadra = df.iloc[i]["Squad"]
            pos = df.iloc[i]["Pos"]
            match = calendario[(calendario["Casa"]==squadra)|(calendario["Ospite"]==squadra)].head(1)
            if match.empty:
                cal_scores[i] = 1.0
            else:
                in_casa = match["Casa"].values[0]==squadra
                avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
                if avv not in team_stats.index:
                    cal_scores[i] = 1.0
                else:
                    if pos!="P":
                        cal_scores[i] = team_stats.loc[avv,xga_col] if in_casa else team_stats.loc[avv,"H_xGA"]
                    else:
                        cal_scores[i] = team_stats.loc[avv,xg_col] if in_casa else team_stats.loc[avv,"H_xG"]
        return normalize_array(cal_scores)

    df_players["Calendario"] = compute_cal(df_players,"A_xGA","A_xG")
    df_keepers["Calendario"] = compute_cal(df_keepers,"A_xGA","A_xG")

    player_features = df_players[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)
    keeper_features = df_keepers[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)

    return player_features, keeper_features, df_players, df_keepers