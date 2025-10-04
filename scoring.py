import pandas as pd

EXCLUSION_PLAYERS = ["Player", "Squad", "Pos", "Nation", "Born"]
EXCLUSION_KEEPERS = ["Player", "Squad", "Pos", "Nation",
                     "Born", "Att", "AvgLen", "Matches", "Launch%"]
EXCLUSION_TEAMS = ["Squadra"]


def normalizza(val, vmin, vmax):
    if vmax == vmin:
        return 0
    return (val - vmin) / (vmax - vmin)


def convert_numeric_columns(df, exclude=[]):
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.fillna(0)


def compute_scores(df_players, df_keepers, df_teams, prob_set, calendario):

    # --- CONVERSIONE COLONNE ---
    df_players = convert_numeric_columns(df_players, exclude=EXCLUSION_PLAYERS)
    df_keepers = convert_numeric_columns(df_keepers, exclude=EXCLUSION_KEEPERS)
    df_teams = convert_numeric_columns(df_teams,   exclude=EXCLUSION_TEAMS)

    # --- TITOLARITÀ ---
    titolari = set(prob_set["in_campo_casa"].explode()).union(set(prob_set["in_campo_ospite"].explode()))
    df_players["Titolarità"] = df_players["Player"].apply(lambda x: 1.0 if x in titolari else 0.3)

    # --- FORMA (xG+xAG/90) ---
    df_players["Forma"] = (df_players["%xG"] + df_players["%xAG"])
    df_players["Forma"] = df_players["Forma"].apply(lambda x: normalizza(x, df_players["Forma"].min(), df_players["Forma"].max()) if x > 0 else 0)

    # --- BONUS POTENZIALE (G+A/90) ---
    df_players["BonusPot"] = (df_players["%G+A"]).apply(lambda x: normalizza(x, df_players["%G+A"].min(), df_players["%G+A"].max()) if x > 0 else 0)

    # --- AFFIDABILITÀ ---
    df_players["Affidabilità"] = (df_players["Starts"] / df_players["MP"]) * (df_players["Min"] / (df_players["MP"] * 90))
    df_players["Affidabilità"] = df_players["Affidabilità"].apply(lambda x: normalizza(x, df_players["Affidabilità"].min(), df_players["Affidabilità"].max()))

    # --- CALENDARIO (difesa avversaria) ---
    def prossimo_avversario(row):
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            return 1.0
        in_casa = match["Casa"].values[0] == squadra
        avversario = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
        if in_casa:
            xga = df_teams.loc[df_teams["Squadra"] == avversario, "A_xGA"].values[0]
        else:
            xga = df_teams.loc[df_teams["Squadra"] == avversario, "H_xGA"].values[0]
        return xga

    df_players["Calendario_raw"] = df_players.apply(prossimo_avversario, axis=1)
    df_players["Calendario"] = df_players["Calendario_raw"].apply(lambda x: normalizza(x, df_players["Calendario_raw"].min(), df_players["Calendario_raw"].max()))

    # --- PENALITÀ ---
    df_players["Penalità"] = df_players["CrdY"] * 0.05 + df_players["CrdR"] * 0.2
    df_players["Penalità"] = df_players["Penalità"].apply(lambda x: normalizza(x, df_players["Penalità"].min(), df_players["Penalità"].max()))

    # --- PESI GIOCATORI ---
    PESI_GIOCATORI = {
        "Titolarità": 0.35,
        "Forma": 0.20,
        "BonusPot": 0.20,
        "Affidabilità": 0.15,
        "Calendario": 0.10,
        "Penalità": -0.10
    }

    df_players["Punteggio"] = df_players.apply(lambda r: sum(r[k]*w for k,w in PESI_GIOCATORI.items()), axis=1)
    df_players["Score"] = df_players["Punteggio"].apply(lambda x: normalizza(x, df_players["Punteggio"].min(), df_players["Punteggio"].max()) * 100)

    # ========================================================================
    # ===================== PORTIERI (df_keepers) ============================
    # ========================================================================
    gk = df_keepers.copy()
    gk["Titolarità"] = gk["Player"].apply(lambda x: 1.0 if x in titolari else 0.3)

    gk["Forma"] = gk[["Save%", "PSxG+/-"]].mean(axis=1)
    gk["Forma"] = gk["Forma"].apply(lambda x: normalizza(x, gk["Forma"].min(), gk["Forma"].max()) if x > 0 else 0)

    gk["BonusPot"] = gk["CS%"].apply(lambda x: normalizza(x, gk["CS%"].min(), gk["CS%"].max()))

    gk["Affidabilità"] = (gk["Starts"] / gk["MP"])
    gk["Affidabilità"] = gk["Affidabilità"].apply(lambda x: normalizza(x, gk["Affidabilità"].min(), gk["Affidabilità"].max()))

    def prossimo_avversario_gk(row):
        squadra = row["Squad"]
        match = calendario[(calendario["Casa"] == squadra) | (calendario["Ospite"] == squadra)].head(1)
        if match.empty:
            return 1.0
        in_casa = match["Casa"].values[0] == squadra
        avversario = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
        if in_casa:
            xg = df_teams.loc[df_teams["Squadra"] == avversario, "A_xG"].values[0]
        else:
            xg = df_teams.loc[df_teams["Squadra"] == avversario, "H_xG"].values[0]
        return xg

    gk["Calendario_raw"] = gk.apply(prossimo_avversario_gk, axis=1)
    gk["Calendario"] = gk["Calendario_raw"].apply(lambda x: normalizza(x, gk["Calendario_raw"].min(), gk["Calendario_raw"].max()))

    gk["Penalità"] = gk["GA90"]
    gk["Penalità"] = gk["Penalità"].apply(lambda x: normalizza(x, gk["Penalità"].min(), gk["Penalità"].max()))

    PESI_PORTIERI = {
        "Titolarità": 0.40,
        "Forma": 0.20,
        "BonusPot": 0.10,
        "Affidabilità": 0.15,
        "Calendario": 0.20,
        "Penalità": -0.15
    }

    gk["Punteggio"] = gk.apply(lambda r: sum(r[k]*w for k,w in PESI_PORTIERI.items()), axis=1)
    gk["Score"] = gk["Punteggio"].apply(lambda x: normalizza(x, gk["Punteggio"].min(), gk["Punteggio"].max()) * 100)

    # --- OUTPUT RIDOTTO ---
    df_players_out = df_players[["Player","Squad","Pos","Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità","Punteggio","Score"]]
    gk_out         = gk[["Player","Squad","Pos","Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità","Punteggio","Score"]]

    df_final = pd.concat([df_players_out, gk_out], ignore_index=True)

    return df_final
