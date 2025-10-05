from scraping.scraping_fbref_players_gemini import giocatori, portieri
from scraping.scraping_fbref_squads_gemini import get_squads, get_schedule
from scraping.scraping_sosfanta import get_probabili
from scraping.calendario import get_next_matches
from scoring.scoring import compute_scores
from optimizer.optimizer import best_eleven, best_eleven_343
import pandas as pd
from scraping.scraping_pianetafanta import scrape_pianetafanta
from train.optimizer_trainer import train_weights, prepare_tensors_for_day
import torch

import pandas as pd
import torch
import torch.optim as optim


def calcola_fanta_voto(x):
    score = float(''.join([c for c in str(x.Voto) if c != '*'])) + x.Gf * 3 - x.Gs + x.Rp*3 - \
        x.Rs*3 + x.Rf*2 - x.Amm*.5 - x.Esp + x.Ass
    return score


def load_fantacalcio_votes(xlsx_path):
    # Leggi il foglio saltando le prime 4 righe (così la riga 5 è la prima squadra)
    df_raw = pd.read_excel(
        xlsx_path, sheet_name="Fantacalcio", skiprows=4, header=None)

    teams_data = []
    current_team = None
    current_block = []

    for _, row in df_raw.iterrows():
        # Rimuove valori NaN e stringhe vuote
        non_nulls = [str(x).strip()
                     for x in row if pd.notna(x) and str(x).strip() != ""]

        # Caso 1 → Riga con solo un valore = nome squadra
        if len(non_nulls) == 1:
            # Se c'era un blocco precedente, salvalo
            if current_team and current_block:
                teams_data.append((current_team, pd.DataFrame(current_block)))
            # Nuova squadra
            current_team = non_nulls[0]
            current_block = []
            continue

        # Caso 2 → Riga giocatore o allenatore
        if len(non_nulls) > 2:
            current_block.append(row.values.tolist())

        # Caso 3 → Riga ALL (fine squadra)
        if len(non_nulls) > 1 and non_nulls[1] == "ALL":
            # Salva il blocco squadra (senza includere ALL)
            if current_team and current_block:
                df_team = pd.DataFrame(current_block)
                teams_data.append((current_team, df_team))
            current_team = None
            current_block = []

    # Combina tutti i blocchi in un unico DataFrame
    all_players = []
    for team_name, df_team in teams_data:
        df_team = df_team.reset_index(drop=True)

        # 🔹 Usa la prima riga come intestazione
        df_team.columns = df_team.iloc[0]
        df_team = df_team[1:].reset_index(drop=True)
        df_team["FantaVoto"] = df_team.apply(
            lambda x: calcola_fanta_voto(x), axis=1)

        # Se ha le colonne che ci servono
        if {'Nome', 'Ruolo', 'Voto', 'Gf', 'Gs', 'Rp', 'Rs', 'Rf', 'Au', 'Amm', 'Esp', 'Ass', 'FantaVoto'}.issubset(df_team.columns):
            temp = df_team[['Nome', 'Ruolo']].copy()
            temp['Voto'] = df_team["FantaVoto"]
            temp['Squad'] = team_name
            all_players.append(temp)

    df_final = pd.concat(all_players, ignore_index=True)
    df_final.rename(columns={'Nome': 'Player', 'Ruolo': 'Pos'}, inplace=True)
    df_final['Voto'] = pd.to_numeric(df_final['Voto'], errors='coerce')
    df_final.dropna(subset=['Voto'], inplace=True)

    return df_final.reset_index(drop=True)


def crea_chiave_giocatore(df, col_nome="Player"):
    """
    Crea chiave univoca per i giocatori:
      - Cognome + abbreviazione nome se ci sono più nomi diversi con lo stesso cognome
      - Solo cognome se il cognome è unico
    L'abbreviazione prende le lettere necessarie per distinguere i nomi all'interno dello stesso cognome.
    """
    preposizioni = {"de", "del", "della", "dei",
                    "di", "da", "dal", "dalla", "lo", "la"}

    df = df.copy()
    df["tokens"] = df[col_nome].str.split()

    # Estrae NomeProprio e Cognome
    def estrai_nome_cognome(tokens):
        if len(tokens) == 1:
            return "", tokens[0]
        cognome = [tokens[-1]]
        for t in reversed(tokens[:-1]):
            if t.lower() in preposizioni:
                cognome.insert(0, t)
            else:
                break
        nome = tokens[: len(tokens) - len(cognome)]
        return " ".join(nome), " ".join(cognome)

    df[["NomeProprio", "Cognome"]] = df["tokens"].apply(
        lambda x: pd.Series(estrai_nome_cognome(x)))

    # Raggruppa per cognome e ottieni tutti i nomi unici per quel cognome
    gruppi = df.groupby("Cognome")["NomeProprio"].unique().to_dict()

    chiavi = {}

    for cognome, nomi in gruppi.items():
        # Se c'è un solo nome per quel cognome → chiave = cognome
        if len(nomi) == 1:
            chiavi.update({(cognome, nomi[0]): cognome})
        else:
            # Altrimenti genera abbreviazioni uniche tra i nomi
            abbrev_usate = {}
            for nome in nomi:
                # Determina la lunghezza minima per distinguere questo nome dagli altri
                l = 1
                while True:
                    abbrev = nome[:l]
                    # controlla se abbrev è unico tra gli altri nomi
                    collision = any(abbrev == abbrev_usate.get(n, "")
                                    for n in abbrev_usate)
                    if not collision:
                        break
                    l += 1
                abbrev_usate[nome] = abbrev
                chiavi[(cognome, nome)] = f"{cognome} {abbrev}."

    # Applica le chiavi al dataframe
    df["chiave"] = df.apply(
        lambda row: chiavi[(row["Cognome"], row["NomeProprio"])], axis=1)
    return df


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


def main(matchweek, train=True):
    print("🔄 Recupero dati da FBref...")
    df_keepers = pd.read_parquet("./train/fbref/portieri.parquet")
    df_players = pd.read_parquet("./train/fbref/giocatori.parquet")
    df_teams = pd.read_parquet("./train/fbref/squads.parquet")

    print("🔄 Recupero calendario prossima giornata...")
    calendario = pd.read_parquet(f"./train/fbref/schedule_{matchweek}.parquet")

    print("⚙️ Calcolo punteggi iniziali...")
    pesi_gk_initial = [0.4, 0.2, 0.1, 0.15, 0.2, -0.15]
    pesi_gk_initial_dict = {
        "Titolarità": pesi_gk_initial[0],
        "Forma": pesi_gk_initial[1],
        "BonusPot": pesi_gk_initial[2],
        "Affidabilità": pesi_gk_initial[3],
        "Calendario": pesi_gk_initial[4],
        "Penalità": pesi_gk_initial[5]
    }

    pesi_out_initial = [0.35, 0.2, 0.2, 0.15, 0.1, -0.1]
    pesi_out_initial_dict = {
        "Titolarità": pesi_out_initial[0],
        "Forma": pesi_out_initial[1],
        "BonusPot": pesi_out_initial[2],
        "Affidabilità": pesi_out_initial[3],
        "Calendario": pesi_out_initial[4],
        "Penalità": pesi_out_initial[5]
    }
    df_scored = compute_scores(df_players=df_players, df_keepers=df_keepers,
                               df_teams=df_teams, prob_set=None, calendario=calendario, pesi_gk=pesi_gk_initial_dict, pesi_out=pesi_out_initial_dict)
    # --- df_scored contiene le feature della giornata corrente ---
    feat_cols = ['Titolarità', 'Forma', 'BonusPot',
                 'Affidabilità', 'Calendario', 'Penalità']

    # Separazione giocatori
    mask_gk = df_scored['Pos'].str.startswith('G')
    mask_out = ~mask_gk

    # # --- Carica CSV voti fantacalcio.it ---
    url = f"./train/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
    df_votes = load_fantacalcio_votes(url)
    df_votes.rename(columns={'Nome': 'Player', 'Ruolo': 'Pos'}, inplace=True)

    # 2️⃣ Crea chiave abbreviata
    df_scored = crea_chiave_giocatore(df_scored)
    df_votes = crea_chiave_giocatore(df_votes)

    # 3️⃣ Definiamo le colonne per il match
    match_cols = ["chiave", "Squad"]  # o "City" se preferisci

    # 4️⃣ Subset dei DataFrame da usare per il merge
    df_scored_gk = df_scored.loc[mask_gk, match_cols]
    df_scored_out = df_scored.loc[mask_out, match_cols]
    df_votes_subset = df_votes[match_cols + ["Voto"]]

    # 5️⃣ Merge per ottenere i voti
    merged_gk = df_scored_gk.merge(df_votes_subset, on=match_cols, how="left")
    # merged_gk = merged_gk[merged_gk["Voto"].notnull()]
    merged_out = df_scored_out.merge(
        df_votes_subset, on=match_cols, how="left")
    # merged_out = merged_out[merged_out["Voto"].notnull()]

    # 1️⃣ Converti feature in tensori
    X_gk = torch.tensor(
        df_scored.loc[mask_gk, feat_cols].values, dtype=torch.float32, requires_grad=False)
    X_out = torch.tensor(
        df_scored.loc[mask_out, feat_cols].values, dtype=torch.float32, requires_grad=False)

    # 6️⃣ Convertiamo i voti in tensori
    y_gk = torch.tensor(merged_gk["Voto"].values,
                        dtype=torch.float32, requires_grad=False)
    y_out = torch.tensor(
        merged_out["Voto"].values, dtype=torch.float32, requires_grad=False)

    # Trova gli indici dei voti validi
    mask_valid_gk = ~torch.isnan(y_gk)
    mask_valid_out = ~torch.isnan(y_out)

    # Applica la maschera
    X_gk = X_gk[mask_valid_gk]
    y_gk = y_gk[mask_valid_gk]

    X_out = X_out[mask_valid_out]
    y_out = y_out[mask_valid_out]

    # --- Pesabili ---
    pesi_gk = torch.tensor(
        [0.4, 0.2, 0.1, 0.15, 0.2, -0.15], requires_grad=True)
    pesi_out = torch.tensor(
        [0.35, 0.2, 0.2, 0.15, 0.1, -0.1], requires_grad=True)

    optimizer = optim.Adam([pesi_gk, pesi_out], lr=0.05)

    # ... il codice di setup rimane invariato ...
    for epoch in range(200):
        optimizer.zero_grad()

        # 1. Score Lineare: Traccia il gradiente!
        score_gk = X_gk.matmul(pesi_gk)
        score_out = X_out.matmul(pesi_out)

        # 2. Selezione Top-K
        k_gk = min(1, len(score_gk))
        k_out = min(10, len(score_out))

        # Estraiamo i punteggi TOP e i loro INDICI
        # Non modifichiamo questi, li usiamo solo per la selezione
        top_score_gk, idx_gk = torch.topk(score_gk, k_gk)
        top_score_out, idx_out = torch.topk(score_out, k_out)

        # 3. CORREZIONE: Stacchiamo gli indici e estraiamo i voti.
        # Questa riga estrae i voti reali (Y) utilizzando gli indici
        # ottenuti dallo Score, ma la variabile risultante (voti_selezionati)
        # è un tensore DATO FISSO, SENZA grad_fn.
        voti_gk_selezionati = y_gk[idx_gk].detach().sum()  # Somma e stacca
        voti_out_selezionati = y_out[idx_out].detach().sum()

        # 4. RICOSTRUZIONE DELLA LOSS:
        # L'errore si verifica perché la Loss non ha un grad_fn.
        # Definiamo la Loss come un tensore che TRACCIA il gradiente,
        # pur basandosi sui valori reali estratti al punto 3.

        # Vogliamo massimizzare i voti reali, ma la Loss non può essere
        # direttamente definita da questi voti staccati.

        # Soluzione avanzata per Ranking: Usare i punteggi top stessi
        # come proxy differenziabile per la Loss, pesandoli con i voti reali.

        # Approccio 5 (Il più probabile per funzionare):
        # -----------------------------------------------------------
        # Utilizziamo una funzione di approssimazione per la Top-K selection
        # (Softmax). Siccome vogliamo il Top-K *rigido* e non la soft-selection,
        # la soluzione è forzare l'uso dei valori top_score (che tracciano il gradiente)
        # nel calcolo della Loss, ignorando y_gk/y_out nel grafo.

        # In questo modo, l'ottimizzatore impara a rendere il punteggio (top_score)
        # proporzionale al voto reale, massimizzando il punteggio stesso.

        # Loss: Vogliamo massimizzare i voti reali dei selezionati.
        # Ciò implica che il modello deve assegnare punteggi più alti ai giocatori
        # che hanno avuto voti reali più alti.

        # Definiamo il contributo del Voto Reale come un tensore costante (staccato)
        # La Loss finale sarà il prodotto tra il Voto Reale e lo Score del modello.

        # Voti reali estratti (staccati dal grafo)
        voti_reali_gk = y_gk[idx_gk].detach()
        voti_reali_out = y_out[idx_out].detach()

        # Score del modello per quei giocatori (tracciano il gradiente)
        score_selezionato_gk = top_score_gk
        score_selezionato_out = top_score_out

        # Definiamo la Loss come il negativo del prodotto tra Voti Reali e Score.
        # Questo costringe il modello ad aumentare lo Score dei giocatori
        # con un alto Voto Reale.
        loss_gk = - (score_selezionato_gk * voti_reali_gk).sum()
        loss_out = - (score_selezionato_out * voti_reali_out).sum()

        loss = loss_gk + loss_out

        loss.backward()  # <-- Ora dovrebbe funzionare
        optimizer.step()

        # Stampiamo la Loss basata sui voti reali per il monitoraggio:
        voti_totali_reali = voti_reali_gk.sum().item() + voti_reali_out.sum().item()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}, Voti Reali Selezionati: {voti_totali_reali:.2f}, pesi_out {pesi_out.data.numpy()}, pesi_gk {pesi_gk.data.numpy()}")

    print("Pesi ottimizzati:")
    print("OUT:", pesi_out.data.numpy())
    print("GK:", pesi_gk.data.numpy())


if __name__ == "__main__":
    main(1, True)
