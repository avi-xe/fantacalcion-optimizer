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


def weighted_metric(df, metrics, min_col="90s"):
    all_values = 0
    for metric in metrics:
        all_values += df[metric].values
    base_forma = all_values.astype(np.float32)
    minutes = df[min_col].values.astype(np.float32)

    # Weighted forma reduces small-sample inflation
    weighted = base_forma * minutes

    return normalize_array(weighted)

# ---------------- Preprocessing ----------------


def normalize(x):
    x = np.array(x, dtype=float)
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(x), endpoint=True)
    return (ranks * 100).astype(np.float32)


def invert(series):
    """Inverte la scala (utile per penalità ecc.)."""
    return 100 - series


def safe_div(x, y):
    """Divisione protetta (evita divisione per zero)."""
    return np.where(y != 0, x / y, 0)


def titolarita(df):
    """Probabilità di giocare titolare (stabilità minuti)."""
    score = (
        0.5 * normalize(df["Starts"]) +
        0.3 * normalize(df["Min%"]) +
        0.2 * normalize(invert(df["Mn/Start"]))
    )
    return score.clip(0, 100)


def forma(df, Pos):
    """Rendimento recente o stagionale (dipende dal ruolo)."""
    if Pos == "FW":
        base = (
            0.45 * normalize(df["G+A_per90"]) +
            0.30 * normalize(df["npxG_per90"]) +
            0.25 * normalize(df["SoT_per90"])
        )
    elif Pos == "MF":
        base = (
            0.40 * normalize(df["G+A_per90"]) +
            0.30 * normalize(df["xAG"]) +
            0.30 * normalize(df["SCA90"])
        )
    else:  # DF
        base = (
            0.40 * normalize(df["Tkl+Int_per90"]) +
            0.30 * normalize(df["Clr"]) +
            0.30 * normalize(df["Blocks"])
        )
    return base.clip(0, 100)


def bonus_pot(df, Pos):
    """Potenziale bonus (gol, assist, chance create)."""
    if Pos == "FW":
        score = (
            0.6 * normalize(df["G+A_per90"]) +
            0.2 * normalize(df["npxG+xAG"]) +
            0.2 * normalize(df["SoT%"])
        )
    elif Pos == "MF":
        score = (
            0.5 * normalize(df["xAG"]) +
            0.3 * normalize(df["KP"]) +
            0.2 * normalize(df["SCA90"])
        )
    else:  # DF
        score = (
            0.5 * normalize(df["PrgC"]) +
            0.3 * normalize(df["Crs"]) +
            0.2 * normalize(df["SCA"])
        )
    return score.clip(0, 100)


def affidabilita(df):
    """Disponibilità e disciplina."""
    score = (
        0.5 * normalize(df["Min%"]) +
        0.25 * normalize(invert(df["CrdY"])) +
        0.25 * normalize(invert(df["Err"]))
    )
    return score.clip(0, 100)


def penalita(df):
    """Ammonizioni, espulsioni, errori, falli, autogol."""
    score = (
        0.35 * normalize(df["CrdY"]) +
        0.35 * normalize(df["CrdR"]) +
        0.30 * normalize(df["Err"] + df["OG"] + df["PKcon"])
    )
    return score.clip(0, 100)


def preprocess_features(df_players, df_keepers, df_teams, prob_set, calendario):
    # Convert numeric
    numeric_cols_players = ["%xG", "%xAG", "%G+A",
                            "Starts", "MP", "Min", "CrdY", "CrdR"]
    numeric_cols_keepers = ["Save%", "PSxG+/-",
                            "CS%", "Starts", "MP", "Min", "GA90"]
    df_players[numeric_cols_players] = df_players[numeric_cols_players].fillna(
        0).astype(np.float32)
    df_keepers[numeric_cols_keepers] = df_keepers[numeric_cols_keepers].fillna(
        0).astype(np.float32)

    # Titolarità
    titolari = set(prob_set["in_campo_casa"].explode()).union(
        set(prob_set["in_campo_ospite"].explode()))
    df_players = crea_chiave_giocatore(df_players)
    df_keepers = crea_chiave_giocatore(df_keepers)
    df_players["Titolarità"] = np.where(
        df_players["chiave"].isin(titolari), 1, 0).astype(np.float32)
    df_keepers["Titolarità"] = np.where(
        df_keepers["chiave"].isin(titolari), 1, 0).astype(np.float32)

    df_players["Forma"] = weighted_metric(df_players, ["%xG", "%xAG"])
    df_keepers["Forma"] = weighted_metric(df_keepers, ["Save%", "PSxG+/-"])

    # BonusPot
    df_players["BonusPot"] = weighted_metric(df_players, ["%G+A"])
    df_keepers["BonusPot"] = weighted_metric(df_keepers, ["CS%"])

    # Affidabilità
    aff_players = (df_players["Starts"].values / np.where(df_players["MP"].values == 0, 1, df_players["MP"].values)) * \
                  (df_players["Min"].values / (np.where(df_players["MP"].values ==
                   0, 1, df_players["MP"].values)*90))
    df_players["Affidabilità"] = normalize_array(
        aff_players.astype(np.float32))

    aff_keepers = (df_keepers["Starts"].values /
                   np.where(df_keepers["MP"].values == 0, 1, df_keepers["MP"].values)) * \
        (df_keepers["Min"].values / (np.where(df_keepers["MP"].values ==
                                              0, 1, df_keepers["MP"].values)*90))
    df_keepers["Affidabilità"] = normalize_array(
        aff_keepers.astype(np.float32))

    # Penalità
    df_players["Penalità"] = normalize_array(
        (df_players["CrdY"].values*0.05 + df_players["CrdR"].values*0.2).astype(np.float32))
    df_keepers["Penalità"] = normalize_array(
        df_keepers["GA90"].values.astype(np.float32))

    # Calendario (simple loop)
    team_stats = df_teams.set_index("Squadra")

    def compute_cal(df, xga_col, xg_col):
        cal_scores = np.ones(len(df), dtype=np.float32)
        for i in range(len(df)):
            squadra = df.iloc[i]["Squad"]
            pos = df.iloc[i]["Pos"]
            match = calendario[(calendario["Casa"] == squadra) | (
                calendario["Ospite"] == squadra)].head(1)
            if match.empty:
                cal_scores[i] = 1.0
            else:
                in_casa = match["Casa"].values[0] == squadra
                avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
                if avv not in team_stats.index:
                    cal_scores[i] = 1.0
                else:
                    if pos != "P":
                        cal_scores[i] = team_stats.loc[avv,
                                                       xga_col] if in_casa else team_stats.loc[avv, "H_xGA"]
                    else:
                        cal_scores[i] = team_stats.loc[avv,
                                                       xg_col] if in_casa else team_stats.loc[avv, "H_xG"]
        return normalize_array(cal_scores)

    df_players["Calendario"] = compute_cal(df_players, "A_xGA", "A_xG")
    df_keepers["Calendario"] = compute_cal(df_keepers, "A_xGA", "A_xG")

    player_features = df_players[["Titolarità", "Forma", "BonusPot",
                                  "Affidabilità", "Calendario", "Penalità"]].values.astype(np.float32)
    keeper_features = df_keepers[["Titolarità", "Forma", "BonusPot",
                                  "Affidabilità", "Calendario", "Penalità"]].values.astype(np.float32)

    return player_features, keeper_features, df_players, df_keepers


def preprocess_features_players(df_players, df_teams, prob_set, calendario):
    """
    Calcola i macroparametri sintetici per i giocatori di movimento (no portieri)
    usando le funzioni modulari definite in precedenza.
    """
    # --- Conversione numerica ---
    numeric_cols_players = [
        "%xG", "%xAG", "%G+A",
        "Starts", "MP", "Min", "CrdY", "CrdR",
        "Mn/Start", "xAG", "G+A_per90", "npxG_per90",
        "SoT_per90", "SoT%", "npxG+xAG", "KP",
        "SCA90", "PrgC", "Crs", "SCA", "Tkl+Int_per90",
        "Clr", "Blocks", "Err", "OG", "PKcon"
    ]
    for col in numeric_cols_players:
        if col not in df_players.columns:
            df_players[col] = 0.0
    df_players[numeric_cols_players] = df_players[numeric_cols_players].fillna(
        0).astype(np.float32)

    # --- Titolarità (usando prob_set come riferimento partenti) ---
    titolari = set(prob_set["in_campo_casa"].explode()).union(
        set(prob_set["in_campo_ospite"].explode())
    )
    df_players = crea_chiave_giocatore(df_players)
    df_players["Titolarità"] = np.where(
        df_players["chiave"].isin(titolari), 1.0, 0.0
    ).astype(np.float32)

    # --- Calcolo macroparametri modulari ---
    # (le funzioni titolarita, forma, bonus_pot, affidabilita, penalita, calendario devono essere già importate)
    tit = titolarita(df_players)
    forma_scores = forma_scores = [forma(df_players.loc[[i]], r)[
        0] for i, r in df_players["Pos"].items()]

    bonus_scores = [bonus_pot(df_players.loc[[i]], r)[0]
                    for i, r in df_players["Pos"].items()]
    aff = affidabilita(df_players)
    pen = penalita(df_players)

    # --- Calendario ---
    team_stats = df_teams.set_index("Squadra")

    def compute_cal(df, xga_col, xg_col):
        cal_scores = np.ones(len(df), dtype=np.float32)
        for i in range(len(df)):
            squadra = df.iloc[i]["Squad"]
            pos = df.iloc[i]["Pos"]
            match = calendario[
                (calendario["Casa"] == squadra) | (
                    calendario["Ospite"] == squadra)
            ].head(1)
            if match.empty:
                cal_scores[i] = 1.0
            else:
                in_casa = match["Casa"].values[0] == squadra
                avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
                if avv not in team_stats.index:
                    cal_scores[i] = 1.0
                else:
                    cal_scores[i] = (
                        team_stats.loc[avv,
                                       xga_col] if in_casa else team_stats.loc[avv, "H_xGA"]
                    )
        return normalize(cal_scores)

    cal = compute_cal(df_players, "A_xGA", "A_xG")

    # --- Aggregazione finale ---
    df_players["Titolarità"] = tit.values.astype(np.float32)
    df_players["Forma"] = np.array(forma_scores, dtype=np.float32)
    df_players["BonusPot"] = np.array(bonus_scores, dtype=np.float32)
    df_players["Affidabilità"] = aff.values.astype(np.float32)
    df_players["Calendario"] = cal.values.astype(np.float32)
    df_players["Penalità"] = pen.values.astype(np.float32)

    player_features = df_players[[
        "Titolarità", "Forma", "BonusPot",
        "Affidabilità", "Calendario", "Penalità"
    ]].values.astype(np.float32)

    return player_features, df_players
