import pandas as pd

def best_eleven(df):
    df_sorted = df.sort_values("Punteggio", ascending=False)
    squadre_scelte = set()
    formazione = []
    for _, row in df_sorted.iterrows():
        if row["Squad"] not in squadre_scelte and len(formazione) < 11:
            formazione.append(row)
            squadre_scelte.add(row["Squad"])
    return pd.DataFrame(formazione)

def best_eleven_343(df):
    """
    Restituisce la miglior formazione 3-4-3 basata sul punteggio, rispettando il numero di giocatori per ruolo
    e una sola scelta per squadra.
    """
    df_sorted = df.sort_values("Punteggio", ascending=False)
    
    formazione = []
    squadre_scelte = set()
    
    # Obiettivi per ruolo
    ruoli_target = {"GK": 1, "DF": 3, "MF": 4, "FW": 3}
    ruoli_selezionati = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    
    # Mappatura dei ruoli secondari verso ruolo principale
    ruolo_map = {
        "GK": "GK",
        "DF": "DF", "FB": "DF", "LB": "DF", "RB": "DF", "CB": "DF",
        "MF": "MF", "DM": "MF", "CM": "MF", "LM": "MF", "RM": "MF", "WM": "MF", "AM": "MF",
        "FW": "FW", "LW": "FW", "RW": "FW"
    }
    
    for _, row in df_sorted.iterrows():
        squad = row["Squad"]
        ruolo_orig = row["Pos"]
        ruolo = ruolo_map.get(ruolo_orig, None)
        
        if ruolo is None:
            continue  # ruoli non riconosciuti
        
        # controllo numero max per ruolo e una sola squadra
        if ruoli_selezionati[ruolo] < ruoli_target[ruolo] and squad not in squadre_scelte:
            formazione.append(row)
            ruoli_selezionati[ruolo] += 1
            squadre_scelte.add(squad)
        
        # Se abbiamo completato la formazione, interrompi
        totale_giocatori = sum(ruoli_selezionati.values())
        if totale_giocatori >= 11:
            break
    
    return pd.DataFrame(formazione)


if __name__ == "__main__":
    print("Modulo optimizer pronto: usa best_eleven(df)")
