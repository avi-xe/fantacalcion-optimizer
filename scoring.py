import pandas as pd

def normalizza(val, vmin, vmax):
    return 2 * (val - vmin) / (vmax - vmin) if vmax > vmin else 1

def compute_scores(df_players, df_teams, prob_set, calendario):
    df = df_players.copy()

    # Titolarità
    df["Titolarità"] = df["Nome"].apply(lambda x: 2 if x in prob_set else 0)

    # Forma (gol+assist/90)
    df["Forma"] = (df["Gol"] + df["Assist"]) / df["Nineties"]
    df["Forma"] = df["Forma"].apply(lambda x: normalizza(x, 0, 1))

    # Bonus potenziale
    df["BonusPot"] = (df["Gol"] + df["Assist"]).apply(lambda x: normalizza(x, 0, 10))

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
