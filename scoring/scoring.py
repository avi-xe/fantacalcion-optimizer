import torch
import pandas as pd

EXCLUSION_PLAYERS = ["Player", "Squad", "Pos", "Nation", "Born"]
EXCLUSION_KEEPERS = ["Player", "Squad", "Pos", "Nation",
                     "Born", "Att", "AvgLen", "Matches", "Launch%"]
EXCLUSION_TEAMS = ["Squadra"]


# --- PESI GIOCATORI ---
PESI_GIOCATORI = {
    "Titolarità": 0.35,
    "Forma": 0.20,
    "BonusPot": 0.20,
    "Affidabilità": 0.15,
    "Calendario": 0.10,
    "Penalità": -0.10
}

PESI_PORTIERI = {
    "Titolarità": 0.40,
    "Forma": 0.20,
    "BonusPot": 0.10,
    "Affidabilità": 0.15,
    "Calendario": 0.20,
    "Penalità": -0.15
}


def normalizza(val, vmin, vmax):
    if vmax == vmin:
        return 0
    return (val - vmin) / (vmax - vmin)


def convert_numeric_columns(df, exclude=[]):
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.fillna(0)


# --- PESI GIOCATORI ---
PESI_GIOCATORI = {
    "Titolarità": 0.35,
    "Forma": 0.20,
    "BonusPot": 0.20,
    "Affidabilità": 0.15,
    "Calendario": 0.10,
    "Penalità": -0.10
}

PESI_PORTIERI = {
    "Titolarità": 0.40,
    "Forma": 0.20,
    "BonusPot": 0.10,
    "Affidabilità": 0.15,
    "Calendario": 0.20,
    "Penalità": -0.15
}


def compute_scores(df_players, df_keepers, df_teams, prob_set, calendario, pesi_gk=PESI_PORTIERI, pesi_out=PESI_GIOCATORI):

    # --- CONVERSIONE COLONNE ---
    df_players = convert_numeric_columns(df_players, exclude=EXCLUSION_PLAYERS)
    df_keepers = convert_numeric_columns(df_keepers, exclude=EXCLUSION_KEEPERS)
    df_teams = convert_numeric_columns(df_teams,   exclude=EXCLUSION_TEAMS)

    # --- TITOLARITÀ ---
    if prob_set is not None:
        titolari = set(prob_set["in_campo_casa"].explode()).union(
            set(prob_set["in_campo_ospite"].explode()))
        df_players["Titolarità"] = df_players["Player"].apply(
            lambda x: 1.0 if x in titolari else 0.3)
    else:
        df_players["Titolarità"] = df_players["Player"].apply(lambda x: 1.0)

    # --- FORMA (xG+xAG/90) ---
    df_players["Forma"] = (df_players["%xG"] + df_players["%xAG"])
    df_players["Forma"] = df_players["Forma"].apply(lambda x: normalizza(
        x, df_players["Forma"].min(), df_players["Forma"].max()) if x > 0 else 0)

    # --- BONUS POTENZIALE (G+A/90) ---
    df_players["BonusPot"] = (df_players["%G+A"]).apply(lambda x: normalizza(
        x, df_players["%G+A"].min(), df_players["%G+A"].max()) if x > 0 else 0)

    # --- AFFIDABILITÀ ---
    df_players["Affidabilità"] = (
        df_players["Starts"] / df_players["MP"]) * (df_players["Min"] / (df_players["MP"] * 90))
    df_players["Affidabilità"] = df_players["Affidabilità"].apply(lambda x: normalizza(
        x, df_players["Affidabilità"].min(), df_players["Affidabilità"].max()))

    # --- CALENDARIO (difesa avversaria) ---
    def prossimo_avversario(row):
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (
            calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            return 1.0
        in_casa = match["Casa"].values[0] == squadra
        avversario = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
        if in_casa:
            xga = df_teams.loc[df_teams["Squadra"]
                               == avversario, "A_xGA"].values[0]
        else:
            xga = df_teams.loc[df_teams["Squadra"]
                               == avversario, "H_xGA"].values[0]
        return xga

    df_players["Calendario_raw"] = df_players.apply(
        prossimo_avversario, axis=1)
    df_players["Calendario"] = df_players["Calendario_raw"].apply(lambda x: normalizza(
        x, df_players["Calendario_raw"].min(), df_players["Calendario_raw"].max()))

    # --- PENALITÀ ---
    df_players["Penalità"] = df_players["CrdY"] * \
        0.05 + df_players["CrdR"] * 0.2
    df_players["Penalità"] = df_players["Penalità"].apply(lambda x: normalizza(
        x, df_players["Penalità"].min(), df_players["Penalità"].max()))

    df_players["Punteggio"] = df_players.apply(lambda r: sum(
        r[k]*w for k, w in pesi_out.items()), axis=1)
    df_players["Score"] = df_players["Punteggio"].apply(lambda x: normalizza(
        x, df_players["Punteggio"].min(), df_players["Punteggio"].max()) * 100)

    # ========================================================================
    # ===================== PORTIERI (df_keepers) ============================
    # ========================================================================
    gk = df_keepers.copy()
    if prob_set is not None:
        gk["Titolarità"] = gk["Player"].apply(
            lambda x: 1.0 if x in titolari else 0.3)
    else:
        gk["Titolarità"] = gk["Player"].apply(
            lambda x: 1.0)

    gk["Forma"] = gk[["Save%", "PSxG+/-"]].mean(axis=1)
    gk["Forma"] = gk["Forma"].apply(lambda x: normalizza(
        x, gk["Forma"].min(), gk["Forma"].max()) if x > 0 else 0)

    gk["BonusPot"] = gk["CS%"].apply(
        lambda x: normalizza(x, gk["CS%"].min(), gk["CS%"].max()))

    gk["Affidabilità"] = (gk["Starts"] / gk["MP"])
    gk["Affidabilità"] = gk["Affidabilità"].apply(lambda x: normalizza(
        x, gk["Affidabilità"].min(), gk["Affidabilità"].max()))

    def prossimo_avversario_gk(row):
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (
            calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            return 1.0
        in_casa = match["Casa"].values[0] == squadra
        avversario = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
        if in_casa:
            xg = df_teams.loc[df_teams["Squadra"]
                              == avversario, "A_xG"].values[0]
        else:
            xg = df_teams.loc[df_teams["Squadra"]
                              == avversario, "H_xG"].values[0]
        return xg

    gk["Calendario_raw"] = gk.apply(prossimo_avversario_gk, axis=1)
    gk["Calendario"] = gk["Calendario_raw"].apply(lambda x: normalizza(
        x, gk["Calendario_raw"].min(), gk["Calendario_raw"].max()))

    gk["Penalità"] = gk["GA90"]
    gk["Penalità"] = gk["Penalità"].apply(lambda x: normalizza(
        x, gk["Penalità"].min(), gk["Penalità"].max()))

    gk["Punteggio"] = gk.apply(lambda r: sum(
        r[k]*w for k, w in pesi_gk.items()), axis=1)
    gk["Score"] = gk["Punteggio"].apply(lambda x: normalizza(
        x, gk["Punteggio"].min(), gk["Punteggio"].max()) * 100)

    # --- OUTPUT RIDOTTO ---
    df_players_out = df_players[["Player", "Squad", "Pos", "Titolarità", "Forma",
                                 "BonusPot", "Affidabilità", "Calendario", "Penalità", "Punteggio", "Score", "chiave"]]
    gk_out = gk[["Player", "Squad", "Pos", "Titolarità", "Forma", "BonusPot",
                 "Affidabilità", "Calendario", "Penalità", "Punteggio", "Score", "chiave"]]

    df_final = pd.concat([df_players_out, gk_out], ignore_index=True)

    return df_final


def compute_scores_torch(df_players, df_keepers, df_teams, calendario, prob_set,
                         feat_cols=["Titolarità", "Forma", "BonusPot",
                                    "Affidabilità", "Calendario", "Penalità"],
                         pesi_gk=None, pesi_out=None):
    """
    Calcola i punteggi come tensori differenziabili.
    Converte automaticamente le colonne in numerico e sostituisce NaN con 0.
    """

    # ---------- FORZA NUMERICHE ----------
    def to_numeric_safe(df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    df_players = to_numeric_safe(
        df_players, ["%xG", "%xAG", "%G+A", "Starts", "MP", "Min", "CrdY", "CrdR"])
    df_keepers = to_numeric_safe(
        df_keepers, ["Save%", "PSxG+/-", "CS%", "Starts", "MP", "GA90"])

    # ---------- TITOLARITÀ ----------
    titolari = set()
    if prob_set is not None:
        titolari.update(prob_set["in_campo_casa"].explode())
        titolari.update(prob_set["in_campo_ospite"].explode())

    def titolarita_tensor(df):
        vals = [1.0 if x in titolari else 0.3 for x in df["Player"]]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(1)

    # ---------- FEATURES PLAYERS ----------
    features_players = []
    features_players.append(titolarita_tensor(df_players))  # Titolarità

    # Forma
    forma = torch.tensor(
        df_players["%xG"].values + df_players["%xAG"].values, dtype=torch.float32).unsqueeze(1)
    forma = (forma - forma.min()) / (forma.max() - forma.min() + 1e-8)
    features_players.append(forma)

    # BonusPot
    bonus = torch.tensor(
        df_players["%G+A"].values, dtype=torch.float32).unsqueeze(1)
    bonus = (bonus - bonus.min()) / (bonus.max() - bonus.min() + 1e-8)
    features_players.append(bonus)

    # Affidabilità
    affid = torch.tensor((df_players["Starts"]/df_players["MP"] * df_players["Min"]/(
        df_players["MP"]*90)).values, dtype=torch.float32).unsqueeze(1)
    affid = (affid - affid.min()) / (affid.max() - affid.min() + 1e-8)
    features_players.append(affid)

    # Calendario
    cal = []
    for idx, row in df_players.iterrows():
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (
            calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            xga = 1.0
        else:
            in_casa = match["Casa"].values[0] == squadra
            avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
            if in_casa:
                xga = df_teams.loc[df_teams["Squadra"]
                                   == avv, "A_xGA"].values[0]
            else:
                xga = df_teams.loc[df_teams["Squadra"]
                                   == avv, "H_xGA"].values[0]
        cal.append(xga)
    cal = torch.tensor(cal, dtype=torch.float32).unsqueeze(1)
    cal = (cal - cal.min()) / (cal.max() - cal.min() + 1e-8)
    features_players.append(cal)

    # Penalità
    pen = torch.tensor((df_players["CrdY"]*0.05 + df_players["CrdR"]
                       * 0.2).values, dtype=torch.float32).unsqueeze(1)
    pen = (pen - pen.min()) / (pen.max() - pen.min() + 1e-8)
    features_players.append(pen)

    X_players = torch.cat(features_players, dim=1)  # [N_players, n_feat]

    if pesi_out is None:
        pesi_out = torch.ones(
            len(feat_cols), dtype=torch.float32, requires_grad=True)
    score_players = X_players @ pesi_out

    # ---------- FEATURES GK ----------
    features_gk = []
    features_gk.append(titolarita_tensor(df_keepers))

    forma_gk = torch.tensor(
        df_keepers[["Save%", "PSxG+/-"]].mean(axis=1).values, dtype=torch.float32).unsqueeze(1)
    forma_gk = (forma_gk - forma_gk.min()) / \
        (forma_gk.max() - forma_gk.min() + 1e-8)
    features_gk.append(forma_gk)

    bonus_gk = torch.tensor(
        df_keepers["CS%"].values, dtype=torch.float32).unsqueeze(1)
    bonus_gk = (bonus_gk - bonus_gk.min()) / \
        (bonus_gk.max() - bonus_gk.min() + 1e-8)
    features_gk.append(bonus_gk)

    affid_gk = torch.tensor(
        (df_keepers["Starts"]/df_keepers["MP"]).values, dtype=torch.float32).unsqueeze(1)
    affid_gk = (affid_gk - affid_gk.min()) / \
        (affid_gk.max() - affid_gk.min() + 1e-8)
    features_gk.append(affid_gk)

    # Calendario GK
    cal_gk = []
    for idx, row in df_keepers.iterrows():
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (
            calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            xg = 1.0
        else:
            in_casa = match["Casa"].values[0] == squadra
            avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
            if in_casa:
                xg = df_teams.loc[df_teams["Squadra"] == avv, "A_xG"].values[0]
            else:
                xg = df_teams.loc[df_teams["Squadra"] == avv, "H_xG"].values[0]
        cal_gk.append(xg)
    cal_gk = torch.tensor(cal_gk, dtype=torch.float32).unsqueeze(1)
    cal_gk = (cal_gk - cal_gk.min()) / (cal_gk.max() - cal_gk.min() + 1e-8)
    features_gk.append(cal_gk)

    pen_gk = torch.tensor(
        df_keepers["GA90"].values, dtype=torch.float32).unsqueeze(1)
    pen_gk = (pen_gk - pen_gk.min()) / (pen_gk.max() - pen_gk.min() + 1e-8)
    features_gk.append(pen_gk)

    X_gk = torch.cat(features_gk, dim=1)

    if pesi_gk is None:
        pesi_gk = torch.ones(
            len(feat_cols), dtype=torch.float32, requires_grad=True)
    score_gk = X_gk @ pesi_gk

    return X_players, score_players, X_gk, score_gk
