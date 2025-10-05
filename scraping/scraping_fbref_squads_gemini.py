import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
import cloudscraper

scraper = cloudscraper.create_scraper()


def scrape_fbref_table(url, table_id, header_rename, header_levels=[0, 1]):
    """
    Esegue lo scraping di una tabella FBref specifica, gestendo commenti HTML nascosti 
    e multi-livello di intestazione.

    :param url: L'URL della pagina FBref.
    :param table_id: L'ID della tabella da estrarre (es. 'stats_squads_standard').
    :param header_levels: Lista degli indici di riga da usare come header (es. [0, 1]).
    :return: DataFrame di pandas o None, e un messaggio di errore.
    """

    # 1. Recupero HTML
    try:
        response = scraper.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, f"Errore di rete durante il recupero di {url}: {e}"

    soup = BeautifulSoup(response.content, 'html.parser')
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table_html = None
    target_div_id = f'div_{table_id}'

    # 2. Estrazione della tabella dai commenti (comune per le tabelle di statistiche)
    for comment in comments:
        if target_div_id in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table_tag = comment_soup.find('table', id=table_id)
            if table_tag:
                table_html = str(table_tag)
                break

    try:
        # 3. Lettura del DataFrame
        if table_html:
            # Tabella trovata in un commento
            df = pd.read_html(StringIO(table_html), header=header_levels)[0]
        else:
            # Tabella trovata direttamente nell'HTML (comune per il calendario)
            df = pd.read_html(response.text, attrs={
                              'id': table_id}, header=header_levels)[0]

        # 4. Gestione MultiIndex: appiattisce le colonne al livello inferiore (se richiesto)
        if len(header_levels) > 1:
            new_columns = []
            for top, bottom in df.columns:
                if top in header_rename:
                    new_columns.append(f'{header_rename[top]}_{bottom}')
                else:
                    new_columns.append(bottom)
            df.columns = new_columns
            df.columns.name = None  # Pulisce il nome dell'indice

        # 5. Pulizia (rimuove le righe di intestazione duplicate nel corpo della tabella)
        if 'Rk' in df.columns:
            df = df[df['Rk'] != 'Rk']

        return df, None

    except ValueError:
        return None, f"Tabella con ID '{table_id}' non trovata nell'HTML analizzato."
    except Exception as e:
        return None, f"Errore durante l'analisi della tabella: {e}"


def fetch_schedule(url_schedule='https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures', id_schedule='sched_2025-2026_11_1'):
    df_schedule, error_schedule = scrape_fbref_table(
        url_schedule, id_schedule, {}, header_levels=[0])
    return df_schedule


def get_schedule(df_schedule, matchweek):
    # --- B. Scarica il Calendario (Singolo Header) ---
    # Usiamo [0] perché il calendario ha un solo livello di intestazione

    # --- Output ---
    print("="*70)

    # Gestione e presentazione del calendario della Sesta Giornata
    if df_schedule is not None:
        # 'Wk' (Week) è la colonna della giornata di campionato
        matchweek_column = 'Wk'

        # Filtra la sesta giornata. Usiamo '6' come stringa per robustezza.
        re = rf'^{matchweek}(\.0)?$'
        df_matchweek = df_schedule[df_schedule[matchweek_column].astype(
            str).str.contains(re, na=False)]

        # Pulizia e rinomina colonne per chiarezza
        df_matchweek = df_matchweek.rename(columns={
                                           'Wk': 'Giornata', 'Day': 'Giorno', 'Home': 'Casa', 'Away': 'Ospite', 'Match Report': 'Dettagli', 'Score': 'Gol'})

        cols_to_keep = ['Giornata', 'Date', 'Time', 'Casa',
                        'Gol', 'Ospite', 'Dettagli', 'Attendance', 'Referee']
        final_cols_schedule = [
            col for col in cols_to_keep if col in df_matchweek.columns]

        df_matchweek = df_matchweek[final_cols_schedule]

        print("## 🗓️ Calendario: Sesta Giornata\n")
        print("La tabella seguente mostra tutte le partite relative alla sesta giornata di campionato:")
        # Visualizza il DataFrame del calendario
        print(df_matchweek.to_markdown(index=False))

    else:
        print(f"## ❌ Errore nel recupero del Calendario\n{error_schedule}")

    return df_matchweek


def fetch_squads(url_team_stats='https://fbref.com/en/comps/11/Serie-A-Stats', id_team_stats='results2025-2026111_home_away'):
    # --- A. Scarica le Statistiche di Squadra (Doppio Header) ---
    # Usiamo [0, 1] per ottenere l'header a due livelli e droplevel(0) nel codice della funzione
    df_stats, error_stats = scrape_fbref_table(url_team_stats, id_team_stats, {
                                               "Home": "H", "Away": "A"}, header_levels=[0, 1])

    return df_stats


def get_squads(df_stats):
    # Gestione e presentazione delle statistiche di squadra
    if df_stats is not None:
        # Rinomina e seleziona le colonne principali per la visualizzazione
        df_stats = df_stats.rename(columns={'Squad': 'Squadra'})
        df_stats_clean = df_stats.set_index('Squadra')

        print("## 📊 Statistiche Standard di Squadra (Intestazione Semplificata)\n")
        print("Questa tabella contiene le statistiche generali per ogni squadra, utilizzando solo il secondo livello dell'intestazione.")
        # Visualizza il DataFrame delle statistiche
        print(df_stats_clean.to_markdown())
    else:
        print(
            f"## ❌ Errore nel recupero delle Statistiche di Squadra")

    return df_stats


def main(matchweeks=[1]):
    df_schedule_all = fetch_schedule(url_schedule="https://fbref.com/en/comps/11/2024-2025/schedule/2024-2025-Serie-A-Scores-and-Fixtures",
                                     id_schedule="sched_2024-2025_11_1")
    df_squads = fetch_squads(url_team_stats="https://fbref.com/en/comps/11/2024-2025/2024-2025-Serie-A-Stats",
                             id_team_stats="results2024-2025111_home_away")
    df_squads = get_squads(df_stats=df_squads)
    df_squads.to_parquet("squads.parquet")
    for matchweek in matchweeks:
        df_schedule = get_schedule(
            df_schedule=df_schedule_all, matchweek=matchweek)
        df_schedule.to_parquet(f"schedule_{matchweek}.parquet")


if __name__ == '__main__':
    main(range(1, 39))
