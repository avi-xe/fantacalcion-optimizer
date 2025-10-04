import pandas as pd

def assegna_titolarita(df: pd.DataFrame, prob_set: set, col_player: str = "Player") -> pd.DataFrame:
    """
    Assegna la colonna 'Titolarità' al dataframe df (esteso) in base al prob_set (abbreviato).

    - df: DataFrame con colonna dei nomi completi (es. 'Lorenzo Rossi')
    - prob_set: set o lista di nomi abbreviati (es. {'Rossi L.', 'Rossi Man.'})
    - col_player: nome della colonna in df che contiene i nomi completi
    """

    def parse_abbrev(nome):
        # "Rossi L." -> ("Rossi", "L")
        parts = nome.replace(".", "").split()
        return parts[0], parts[1] if len(parts) > 1 else ""

    def parse_esteso(nome):
        # "Lorenzo Rossi" -> ("Rossi", "Lorenzo")
        parti = nome.split()
        return parti[-1], " ".join(parti[:-1])

    # Costruiamo il set dei giocatori titolari (matchati)
    tit_set = set()
    for in_campo_casa in prob_set["in_campo_casa"]:
        for abbrev in in_campo_casa:
            cognome_abbrev, iniziale = parse_abbrev(abbrev)
            for player in df[col_player]:
                cognome, nome = parse_esteso(player)
                if cognome.lower() == cognome_abbrev.lower() and nome.lower().startswith(iniziale.lower()):
                    tit_set.add(player)
    for in_campo_casa in prob_set["in_campo_casa"]:
        for abbrev in in_campo_casa:
            cognome_abbrev, iniziale = parse_abbrev(abbrev)
            for player in df[col_player]:
                cognome, nome = parse_esteso(player)
                if cognome.lower() == cognome_abbrev.lower() and nome.lower().startswith(iniziale.lower()):
                    tit_set.add(player)

    # Aggiunge colonna Titolarità
    df = df.copy()
    df["Titolarità"] = df[col_player].apply(lambda x: 2 if x in tit_set else 0)
    return df


def normalizza(val, vmin, vmax):
    return 2 * (val - vmin) / (vmax - vmin) if vmax > vmin else 1

def compute_scores(df_players, df_teams, prob_set, calendario):
    df = df_players.copy()

    # Titolarità
    df = assegna_titolarita(df=df, prob_set=prob_set)

    # Forma (gol+assist/90)
    df["Forma"] = (df["Gls"] + df["Ast"]) / df["Min"]
    df["Forma"] = df["Forma"].apply(lambda x: normalizza(x, 0, 1))

    # Bonus potenziale
    df["BonusPot"] = (df["Gls"] + df["Ast"]).apply(lambda x: normalizza(x, 0, 10))

    # Affidabilità (minuti giocati)
    max_min = df["Min"].max()
    df["Affidabilità"] = df["Min"].apply(lambda x: normalizza(x, 0, max_min))

    # Calendario: avversario (semplificazione: usa GolSubiti squadra avversaria media)
    # Per ora assegniamo valore neutro, ma puoi estenderlo con i dati di calendario
    df["Calendario"] = 1.0

    # Totale
    df["Punteggio"] = df[["Titolarità","Forma","BonusPot","Affidabilità","Calendario"]].sum(axis=1)

    return df

if __name__ == "__main__":
    print("Modulo scoring pronto: usa compute_scores()")
