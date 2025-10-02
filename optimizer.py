import pandas as pd

def best_eleven(df):
    df_sorted = df.sort_values("Punteggio", ascending=False)
    squadre_scelte = set()
    formazione = []
    for _, row in df_sorted.iterrows():
        if row["Squadra"] not in squadre_scelte and len(formazione) < 11:
            formazione.append(row)
            squadre_scelte.add(row["Squadra"])
    return pd.DataFrame(formazione)

if __name__ == "__main__":
    print("Modulo optimizer pronto: usa best_eleven(df)")
