from scraping_fbref_players_gemini import giocatori, portieri
from scraping_fbref_squads_gemini import 
from scraping_fantacalcio import get_probabili_formazioni
from calendario import get_next_matches
from scoring import compute_scores
from optimizer import best_eleven

def main():
    print("🔄 Recupero dati da FBref...")
    df_players = giocatori()
    df_keepers = portieri()
    compute_scores(df_players, df_keepers, None, None)
    # df_players = get_player_stats_from_page()
    # df_teams = get_team_stats_from_page()

    # print("🔄 Recupero probabili formazioni da Fantacalcio.it...")
    # prob_set = get_probabili_formazioni()

    # print("🔄 Recupero calendario prossima giornata...")
    # calendario = get_next_matches()

    # print("⚙️ Calcolo punteggi...")
    # df_scored = compute_scores(df_players, df_teams, prob_set, calendario)

    # print("✅ Ottimizzazione formazione (11 squadre diverse)...")
    # formazione = best_eleven(df_scored)

    # print("\nFormazione ottimizzata:")
    # print(formazione[["Nome", "Squadra", "Punteggio"]])

if __name__ == "__main__":
    main()
