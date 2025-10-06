from optimizer.best_eleven import best_eleven  # funzione che ritorna miglior 11
import torch
from torch import optim
from optimizer.best_eleven import best_eleven
from scoring.scoring import compute_scores
import pandas as pd

from scraping.scraping_fantacalcio import load_fantacalcio_votes



feat_cols = ["Titolarità", "Forma", "BonusPot",
             "Affidabilità", "Calendario", "Penalità"]


def train_all_weeks_sum_votes(weeks=range(1, 39), feat_cols=feat_cols, k_gk=1, k_out=10):
    all_records = []

    # ---------- RACCOLTA DATI ----------
    for matchweek in weeks:
        try:
            df_keepers = pd.read_parquet("./train/fbref/portieri.parquet")
            df_players = pd.read_parquet("./train/fbref/giocatori.parquet")
            df_teams = pd.read_parquet("./train/fbref/squads.parquet")
            calendario = pd.read_parquet(
                f"./train/fbref/schedule_{matchweek}.parquet")

            df_votes = load_fantacalcio_votes(
                f"./train/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
            )
            df_votes.rename(columns={"Nome": "Player",
                            "Ruolo": "Pos"}, inplace=True)
            df_votes = crea_chiave_giocatore(df_votes)

            # prob_set solo per giocatori a voto
            prob_set = pd.DataFrame({"in_campo_casa": df_votes["chiave"].values,
                                     "in_campo_ospite": ['']*len(df_votes)})

            df_scores = compute_scores(
                df_players=df_players,
                df_keepers=df_keepers,
                df_teams=df_teams,
                prob_set=prob_set,
                calendario=calendario,
                pesi_gk={f: 1/6 for f in feat_cols},
                pesi_out={f: 1/6 for f in feat_cols}
            )

            df_scores = crea_chiave_giocatore(df_scores)
            df_scores = df_scores.merge(
                df_votes[["chiave", "Voto"]], on="chiave", how="left")
            df_scores["Giornata"] = matchweek

            # sostituire NaN con 0
            df_scores["Voto"] = df_scores["Voto"].fillna(0.0)

            all_records.append(df_scores)
            print(
                f"✅ Giornata {matchweek} aggiunta con {len(df_scores)} record.")

        except Exception as e:
            print(f"⚠️ Errore giornata {matchweek}: {e}")

    # unisci tutto
    df_all = pd.concat(all_records, ignore_index=True)
    print(f"Totale righe per training: {len(df_all)}")

    # ---------- PREPARAZIONE TENSOR ----------
    mask_gk = df_all["Pos"].str.startswith("G")
    mask_out = ~mask_gk

    X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df_all["Voto"].values, dtype=torch.float32)

    # inizializzazione pesi
    torch.manual_seed(42)
    pesi_gk = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)
    pesi_out = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)

    optimizer = optim.Adam([pesi_gk, pesi_out], lr=0.02)
    lambda_l2 = 1e-2
    lambda_entropy = 5e-3

    # ---------- TRAINING ----------
    for epoch in range(300):
        optimizer.zero_grad()

        # vincolo ≥0 e somma=1
        w_gk = torch.relu(pesi_gk)
        w_out = torch.relu(pesi_out)
        w_gk = w_gk / (w_gk.sum() + 1e-8)
        w_out = w_out / (w_out.sum() + 1e-8)

        # calcola punteggi lineari
        score_gk = X[mask_gk] @ w_gk
        score_out = X[mask_out] @ w_out

        # top-K rigid selection
        k_gk_sel = min(k_gk, len(score_gk))
        k_out_sel = min(k_out, len(score_out))

        top_idx_gk = torch.topk(score_gk, k_gk_sel).indices
        top_idx_out = torch.topk(score_out, k_out_sel).indices

        voti_gk_selezionati = y[mask_gk][top_idx_gk]
        voti_out_selezionati = y[mask_out][top_idx_out]

        # loss: massimizzare somma dei voti reali dei top selezionati
        loss = - (voti_gk_selezionati.sum() + voti_out_selezionati.sum())

        # regolarizzazioni
        l2_reg = lambda_l2 * (torch.sum(w_gk ** 2) + torch.sum(w_out ** 2))
        entropy_reg = -torch.sum(w_gk * torch.log(w_gk + 1e-8)) - \
            torch.sum(w_out * torch.log(w_out + 1e-8))
        loss = loss + l2_reg - lambda_entropy * entropy_reg

        loss.backward()
        optimizer.step()

        # rinormalizzazione in-place
        with torch.no_grad():
            pesi_gk[:] = torch.relu(pesi_gk)
            pesi_out[:] = torch.relu(pesi_out)
            pesi_gk[:] = pesi_gk / (pesi_gk.sum() + 1e-8)
            pesi_out[:] = pesi_out / (pesi_out.sum() + 1e-8)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Top GK voto: {voti_gk_selezionati.sum().item():.2f} | Top OUT voto: {voti_out_selezionati.sum().item():.2f}")

    # ---------- PESI FINALIZZATI ----------
    with torch.no_grad():
        w_gk = torch.relu(pesi_gk) / (torch.relu(pesi_gk).sum() + 1e-8)
        w_out = torch.relu(pesi_out) / (torch.relu(pesi_out).sum() + 1e-8)
        print("\n✅ Pesi finali normalizzati e ≥0:")
        print("OUT:", w_out.detach().numpy(), "→ somma:", w_out.sum().item())
        print("GK :", w_gk.detach().numpy(), "→ somma:", w_gk.sum().item())

    return w_gk.detach().numpy(), w_out.detach().numpy(), df_all


def test_training_set(df_all, w_gk, w_out, feat_cols):
    mask_gk = df_all["Pos"].str.startswith("G")
    mask_out = ~mask_gk

    # punteggi con pesi allenati
    X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)

    score = torch.zeros(len(df_all))
    score[mask_gk] = X[mask_gk] @ torch.tensor(w_gk, dtype=torch.float32)
    score[mask_out] = X[mask_out] @ torch.tensor(w_out, dtype=torch.float32)

    df_all["Punteggio"] = score.numpy()

    giornate = df_all["Giornata"].unique()
    results = []

    for g in giornate:
        df_giornata = df_all[df_all["Giornata"] == g].copy()
        best11 = best_eleven(df_giornata)

        voti_reali = best11["Voto"].sum()
        punteggio_model = best11["Punteggio"].sum()

        results.append({
            "Giornata": g,
            "Voti_reali_best11": voti_reali,
            "Punteggio_model_best11": punteggio_model
        })

    df_results = pd.DataFrame(results)
    print("\n📊 Risultati in-sample per ogni giornata:")
    print(df_results)

    # metriche aggregate
    print("\n🔹 Media voti reali vs punteggio modello:")
    print(df_results[["Voti_reali_best11", "Punteggio_model_best11"]].mean())

    return df_results


def model_formazioni_per_giornata(df_all, w_gk, w_out, feat_cols):
    mask_gk = df_all["Pos"].str.startswith("G")
    mask_out = ~mask_gk

    X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)

    # calcola punteggi
    score = torch.zeros(len(df_all))
    score[mask_gk] = X[mask_gk] @ torch.tensor(w_gk, dtype=torch.float32)
    score[mask_out] = X[mask_out] @ torch.tensor(w_out, dtype=torch.float32)

    df_all["Punteggio"] = score.numpy()

    giornate = df_all["Giornata"].unique()
    tutte_formazioni = []

    for g in giornate:
        df_giornata = df_all[df_all["Giornata"] == g].copy()
        best11 = best_eleven(df_giornata)
        best11["Giornata"] = g
        tutte_formazioni.append(
            best11[["Giornata", "Player", "Pos", "Squad", "Voto", "Punteggio"]])

    df_formazioni = pd.concat(tutte_formazioni, ignore_index=True)
    return df_formazioni


def train_all_weeks_sum_votes_fixed(weeks=range(1, 39), feat_cols=None, k_gk=1, k_out=10):
    all_records = []

    # ---------- RACCOLTA DATI ----------
    for matchweek in weeks:
        try:
            df_keepers = pd.read_parquet("./train/fbref/portieri.parquet")
            df_players = pd.read_parquet("./train/fbref/giocatori.parquet")
            df_teams = pd.read_parquet("./train/fbref/squads.parquet")
            calendario = pd.read_parquet(
                f"./train/fbref/schedule_{matchweek}.parquet")

            df_votes = load_fantacalcio_votes(
                f"./train/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
            )
            df_votes.rename(columns={"Nome": "Player",
                            "Ruolo": "Pos"}, inplace=True)
            df_votes = crea_chiave_giocatore(df_votes)

            prob_set = pd.DataFrame({
                "in_campo_casa": df_votes["chiave"].values,
                "in_campo_ospite": ['']*len(df_votes)
            })

            # compute_scores senza pesi → solo features
            df_scores = compute_scores(
                df_players=df_players,
                df_keepers=df_keepers,
                df_teams=df_teams,
                prob_set=prob_set,
                calendario=calendario,
                # pesi uniformi, non importanti
                pesi_gk={f: 1.0 for f in feat_cols},
                pesi_out={f: 1.0 for f in feat_cols}
            )

            df_scores = crea_chiave_giocatore(df_scores)
            df_scores = df_scores.merge(
                df_votes[["chiave", "Voto"]], on="chiave", how="left")
            df_scores["Giornata"] = matchweek

            # sostituire NaN con 0
            df_scores["Voto"] = df_scores["Voto"].fillna(0.0)

            all_records.append(df_scores)
            print(
                f"✅ Giornata {matchweek} aggiunta con {len(df_scores)} record.")

        except Exception as e:
            print(f"⚠️ Errore giornata {matchweek}: {e}")

    # unisci tutto
    df_all = pd.concat(all_records, ignore_index=True)
    print(f"Totale righe per training: {len(df_all)}")

    # ---------- PREPARAZIONE TENSOR ----------
    mask_gk = df_all["Pos"].str.startswith("G")
    mask_out = ~mask_gk
    X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df_all["Voto"].values, dtype=torch.float32)

    # inizializzazione pesi
    torch.manual_seed(42)
    pesi_gk = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)
    pesi_out = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)

    optimizer = optim.Adam([pesi_gk, pesi_out], lr=0.02)
    lambda_l2 = 1e-2
    lambda_entropy = 5e-3

    # ---------- TRAINING ----------
    for epoch in range(300):
        optimizer.zero_grad()

        # vincolo ≥0 e somma=1
        w_gk = torch.relu(pesi_gk)
        w_out = torch.relu(pesi_out)
        w_gk = w_gk / (w_gk.sum() + 1e-8)
        w_out = w_out / (w_out.sum() + 1e-8)

        # calcola punteggi lineari
        score_gk = X[mask_gk] @ w_gk
        score_out = X[mask_out] @ w_out

        # top-K rigid selection
        k_gk_sel = min(k_gk, len(score_gk))
        k_out_sel = min(k_out, len(score_out))
        top_idx_gk = torch.topk(score_gk, k_gk_sel).indices
        top_idx_out = torch.topk(score_out, k_out_sel).indices

        voti_gk_selezionati = y[mask_gk][top_idx_gk]
        voti_out_selezionati = y[mask_out][top_idx_out]

        # loss: massimizzare somma dei voti reali dei top selezionati
        loss = - (voti_gk_selezionati.sum() + voti_out_selezionati.sum())

        # regolarizzazioni
        l2_reg = lambda_l2 * (torch.sum(w_gk ** 2) + torch.sum(w_out ** 2))
        entropy_reg = -torch.sum(w_gk * torch.log(w_gk + 1e-8)) - \
            torch.sum(w_out * torch.log(w_out + 1e-8))
        loss = loss + l2_reg - lambda_entropy * entropy_reg

        loss.backward()
        optimizer.step()

        # rinormalizzazione in-place
        with torch.no_grad():
            pesi_gk[:] = torch.relu(pesi_gk)
            pesi_out[:] = torch.relu(pesi_out)
            pesi_gk[:] = pesi_gk / (pesi_gk.sum() + 1e-8)
            pesi_out[:] = pesi_out / (pesi_out.sum() + 1e-8)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Top GK voto: {voti_gk_selezionati.sum().item():.2f} | Top OUT voto: {voti_out_selezionati.sum().item():.2f}")

    # ---------- PESI FINALIZZATI ----------
    with torch.no_grad():
        w_gk = torch.relu(pesi_gk) / (torch.relu(pesi_gk).sum() + 1e-8)
        w_out = torch.relu(pesi_out) / (torch.relu(pesi_out).sum() + 1e-8)
        print("\n✅ Pesi finali normalizzati e ≥0:")
        print("OUT:", w_out.detach().numpy(), "→ somma:", w_out.sum().item())
        print("GK :", w_gk.detach().numpy(), "→ somma:", w_gk.sum().item())

        # --- Applica pesi finali a df_all ---
        df_all["Punteggio"] = 0.0
        X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)
        df_all.loc[mask_gk, "Punteggio"] = (X[mask_gk] @ w_gk).numpy()
        df_all.loc[mask_out, "Punteggio"] = (X[mask_out] @ w_out).numpy()

    return w_gk.detach().numpy(), w_out.detach().numpy(), df_all


def train_best_eleven_all_weeks(df_all, feat_cols, k_gk=1, k_out=10, lr=0.02, epochs=300):
    """
    df_all: DataFrame con tutte le giornate e colonne 'Voto', 'Pos', 'Squad', 'Player'
    feat_cols: feature numeriche su cui ottimizzare i pesi
    k_gk, k_out: numero top GK e OUT da considerare (opzionale, per soft selezione futura)
    """
    mask_gk = df_all["Pos"].str.startswith("G")
    mask_out = ~mask_gk

    # Converti le feature in tensori
    X = torch.tensor(df_all[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df_all["Voto"].values, dtype=torch.float32)

    # Inizializzazione pesi
    torch.manual_seed(42)
    pesi_gk = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)
    pesi_out = torch.nn.Parameter(torch.abs(torch.randn(len(feat_cols))) + 0.1)
    optimizer = optim.Adam([pesi_gk, pesi_out], lr=lr)

    lambda_l2 = 1e-2
    lambda_entropy = 5e-3

    for epoch in range(epochs):
        optimizer.zero_grad()

        # vincolo ≥0 e somma=1
        w_gk = torch.relu(pesi_gk)
        w_out = torch.relu(pesi_out)
        w_gk = w_gk / (w_gk.sum() + 1e-8)
        w_out = w_out / (w_out.sum() + 1e-8)

        # calcola punteggi lineari
        score_gk = X[mask_gk] @ w_gk
        score_out = X[mask_out] @ w_out
        df_all["Score"] = 0.0
        df_all.loc[mask_gk, "Score"] = score_gk.detach().numpy()
        df_all.loc[mask_out, "Score"] = score_out.detach().numpy()

        # loss totale: somma negativa dei voti reali dei giocatori selezionati da best_eleven
        total_loss = 0.0
        for g in df_all["Giornata"].unique():
            df_g = df_all[df_all["Giornata"] == g].copy()
            best11 = best_eleven(df_g)
            selected_idx = df_g.index.isin(best11.index)
            total_loss += - \
                torch.tensor(
                    df_g.loc[selected_idx, "Voto"].sum(), dtype=torch.float32)

        # regolarizzazione L2 e entropica
        total_loss = total_loss + lambda_l2 * \
            (torch.sum(w_gk**2) + torch.sum(w_out**2))
        total_loss = total_loss - lambda_entropy * (torch.sum(w_gk*torch.log(w_gk + 1e-8)) +
                                                    torch.sum(w_out*torch.log(w_out + 1e-8)))

        total_loss.backward()
        optimizer.step()

        # rinormalizzazione in-place
        with torch.no_grad():
            pesi_gk[:] = torch.relu(pesi_gk)
            pesi_out[:] = torch.relu(pesi_out)
            pesi_gk[:] = pesi_gk / (pesi_gk.sum() + 1e-8)
            pesi_out[:] = pesi_out / (pesi_out.sum() + 1e-8)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f}")

    # pesi finali
    with torch.no_grad():
        w_gk = torch.relu(pesi_gk) / (torch.relu(pesi_gk).sum() + 1e-8)
        w_out = torch.relu(pesi_out) / (torch.relu(pesi_out).sum() + 1e-8)

    # calcolo delle formazioni finali
    tutte_formazioni = []
    for g in df_all["Giornata"].unique():
        df_g = df_all[df_all["Giornata"] == g].copy()
        score_gk = torch.tensor(
            df_g.loc[df_g["Pos"].str.startswith("G"), feat_cols].values) @ w_gk
        score_out = torch.tensor(
            df_g.loc[~df_g["Pos"].str.startswith("G"), feat_cols].values) @ w_out
        df_g["Score"] = 0.0
        df_g.loc[df_g["Pos"].str.startswith("G"), "Score"] = score_gk.numpy()
        df_g.loc[~df_g["Pos"].str.startswith("G"), "Score"] = score_out.numpy()
        best11 = best_eleven(df_g)
        best11["Giornata"] = g
        tutte_formazioni.append(
            best11[["Giornata", "Player", "Pos", "Squad", "Voto", "Score"]])

    df_formazioni = pd.concat(tutte_formazioni, ignore_index=True)

    print("\n✅ Pesi finali ottimizzati:")
    print("GK :", w_gk.numpy(), "→ somma:", w_gk.sum().item())
    print("OUT:", w_out.numpy(), "→ somma:", w_out.sum().item())

    return w_gk.numpy(), w_out.numpy(), df_formazioni


if __name__ == "__main__":
    w_gk, w_out, df_all = train_all_weeks_sum_votes_fixed(
        range(1, 2), feat_cols=feat_cols)
    w_gk, w_out, df_all = train_best_eleven_all_weeks(
        df_all=df_all, feat_cols=feat_cols)
    # test_training_set(df_all=df_all, w_gk=w_gk,
    #                   w_out=w_out, feat_cols=feat_cols)
    # print(model_formazioni_per_giornata(df_all=df_all, w_gk=w_gk,
    #                                     w_out=w_out, feat_cols=feat_cols))
