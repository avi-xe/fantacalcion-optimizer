from scraping.scraping_fbref_players_gemini import fetch_giocatori, fetch_portieri, giocatori, portieri
from scraping.scraping_fbref_squads_gemini import get_squads, get_schedule
from scraping.scraping_sosfanta import get_probabili
from scraping.calendario import get_next_matches
from scoring.scoring import compute_scores
from optimizer.optimizer import best_eleven, best_eleven_343
import pandas as pd


def main(matchweek):
    print("🔄 Recupero dati da FBref...")
    df_players = giocatori(fetch_giocatori())
    df_keepers = portieri(fetch_portieri())
    df_teams = get_squads()

    print("🔄 Recupero probabili formazioni da SosFanta...")
    prob_set = get_probabili()

    print("🔄 Recupero calendario prossima giornata...")
    calendario = get_schedule(matchweek)

    print("⚙️ Calcolo punteggi...")
    df_scored = compute_scores(df_players=df_players, df_keepers=df_keepers,
                               df_teams=df_teams, prob_set=prob_set, calendario=calendario)

    print("✅ Ottimizzazione formazione (11 squadre diverse)...")
    formazione = best_eleven_343(df_scored)

    print("\nFormazione ottimizzata:")
    formazione["Pos"] = pd.Categorical(
        formazione["Pos"], ["GK", "DF", "MF", "FW"])
    print(
        formazione.sort_values("Pos"))


if __name__ == "__main__":
    main(6)
