from scraping.scraping_fbref_players_gemini import fetch_giocatori, fetch_portieri, fetch_portieri_adv, giocatori, portieri
from scraping.scraping_fbref_squads_gemini import fetch_schedule, fetch_squads, get_squads, get_schedule
from scraping.scraping_sosfanta import get_probabili
from scraping.calendario import get_next_matches
from scoring.scoring import compute_scores
from optimizer.best_eleven import best_eleven, best_eleven
import pandas as pd


def main(matchweek):
    print("🔄 Recupero dati da FBref...")
    df_players = giocatori(fetch_giocatori())
    df_keepers = portieri(fetch_portieri(), fetch_portieri_adv())
    df_teams = get_squads(fetch_squads())

    print("🔄 Recupero probabili formazioni da SosFanta...")
    prob_set = get_probabili()

    print("🔄 Recupero calendario prossima giornata...")
    calendario = get_schedule(fetch_schedule(), matchweek)

    print("⚙️ Calcolo punteggi...")
    df_scored = compute_scores(df_players=df_players, df_keepers=df_keepers,
                               df_teams=df_teams, prob_set=prob_set, calendario=calendario, pesi_out={
                                   "Titolarità": 0.54,
                                   "Forma": 0.0,
                                   "BonusPot": 0.25,
                                   "Affidabilità": 0.03,
                                   "Calendario": 0.19,
                                   "Penalità": -0.0
                               }, pesi_gk={
                                   "Titolarità": 0.,
                                   "Forma": 0.07,
                                   "BonusPot": 0.63,
                                   "Affidabilità": 0.,
                                   "Calendario": 0.,
                                   "Penalità": 0.30})

    print("✅ Ottimizzazione formazione (11 squadre diverse)...")
    formazione = best_eleven(df_scored)

    print("\nFormazione ottimizzata:")
    formazione["Pos"] = pd.Categorical(
        formazione["Pos"], ["GK", "DF", "MF", "FW"])
    print(
        formazione.sort_values("Pos"))


if __name__ == "__main__":
    main(6)
