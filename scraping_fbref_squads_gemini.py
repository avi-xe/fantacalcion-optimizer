import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO

def scrape_fbref_table(url, table_id, header_levels=[0, 1]):
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
        response = requests.get(url)
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
            df = pd.read_html(response.text, attrs={'id': table_id}, header=header_levels)[0]

        # 4. Gestione MultiIndex: appiattisce le colonne al livello inferiore (se richiesto)
        if len(header_levels) > 1:
            df.columns = df.columns.droplevel(0)
            df.columns.name = None # Pulisce il nome dell'indice
        
        # 5. Pulizia (rimuove le righe di intestazione duplicate nel corpo della tabella)
        if 'Rk' in df.columns:
            df = df[df['Rk'] != 'Rk']
            
        return df, None
        
    except ValueError:
        return None, f"Tabella con ID '{table_id}' non trovata nell'HTML analizzato."
    except Exception as e:
        return None, f"Errore durante l'analisi della tabella: {e}"

# --- Parametri ---
url_schedule = 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures'
id_schedule = 'sched_2025-2026_11_1' # ID per il Calendario

url_team_stats = 'https://fbref.com/en/comps/11/Serie-A-Stats'
id_team_stats = 'stats_squads_standard_for' # ID per le Statistiche di Squadra Standard

# --- A. Scarica le Statistiche di Squadra (Doppio Header) ---
# Usiamo [0, 1] per ottenere l'header a due livelli e droplevel(0) nel codice della funzione
df_stats, error_stats = scrape_fbref_table(url_team_stats, id_team_stats, header_levels=[0, 1])

# --- B. Scarica il Calendario (Singolo Header) ---
# Usiamo [0] perché il calendario ha un solo livello di intestazione
df_schedule, error_schedule = scrape_fbref_table(url_schedule, id_schedule, header_levels=[0])

# --- Output ---
print("="*70)

# Gestione e presentazione del calendario della Sesta Giornata
if df_schedule is not None:
    # 'Wk' (Week) è la colonna della giornata di campionato
    matchweek_column = 'Wk'
    
    # Filtra la sesta giornata. Usiamo '6' come stringa per robustezza.
    df_matchweek_6 = df_schedule[df_schedule[matchweek_column].astype(str).str.contains(r'^6(\.0)?$', na=False)]
    
    # Pulizia e rinomina colonne per chiarezza
    df_matchweek_6 = df_matchweek_6.rename(columns={'Wk': 'Giornata', 'Day': 'Giorno', 'Home': 'Casa', 'Away': 'Ospite', 'Match Report': 'Dettagli', 'Score': 'Gol'})
    
    cols_to_keep = ['Giornata', 'Date', 'Time', 'Casa', 'Gol', 'Ospite', 'Dettagli', 'Attendance', 'Referee']
    final_cols_schedule = [col for col in cols_to_keep if col in df_matchweek_6.columns]
    
    df_matchweek_6 = df_matchweek_6[final_cols_schedule]
    
    print("## 🗓️ Calendario: Sesta Giornata\n")
    print("La tabella seguente mostra tutte le partite relative alla sesta giornata di campionato:")
    print(df_matchweek_6.to_markdown(index=False)) # Visualizza il DataFrame del calendario

else:
    print(f"## ❌ Errore nel recupero del Calendario\n{error_schedule}")


print("\n" + "="*70 + "\n")


# Gestione e presentazione delle statistiche di squadra
if df_stats is not None:
    # Rinomina e seleziona le colonne principali per la visualizzazione
    df_stats = df_stats.rename(columns={'Squad': 'Squadra'})
    
    stats_cols_to_keep = ['Squadra', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'xG', 'xGA']
    final_cols_stats = [col for col in stats_cols_to_keep if col in df_stats.columns]

    df_stats_clean = df_stats[final_cols_stats].set_index('Squadra')
    
    print("## 📊 Statistiche Standard di Squadra (Intestazione Semplificata)\n")
    print("Questa tabella contiene le statistiche generali per ogni squadra, utilizzando solo il secondo livello dell'intestazione.")
    print(df_stats_clean.to_markdown()) # Visualizza il DataFrame delle statistiche

else:
    print(f"## ❌ Errore nel recupero delle Statistiche di Squadra\n{error_stats}")