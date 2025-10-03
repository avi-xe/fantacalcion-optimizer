import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
import cloudscraper

scraper = cloudscraper.create_scraper()

def scrape_fbref_second_level_header(url, table_id):
    """
    Esegue lo scraping di una tabella FBref e utilizza il secondo livello 
    del MultiIndex come intestazione unica del DataFrame.
    
    :param url: L'URL della pagina FBref.
    :param table_id: L'ID della tabella da estrarre (es. 'stats_standard').
    :return: DataFrame di pandas con intestazioni a livello singolo o None.
    """
    print(f"Sto recuperando i dati da: {url} e cercando la tabella '{table_id}'...")
    
    try:
        response = scraper.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        
        table_html = None
        target_div_id = f'div_{table_id}'
        
        # 1. Estrazione del tag <table> dal commento
        for comment in comments:
            if target_div_id in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table_tag = comment_soup.find('table', id=table_id)
                if table_tag:
                    table_html = str(table_tag)
                    break
        
        if not table_html:
            print(f"Tabella con ID '{table_id}' non trovata.")
            return None
        
        # 2. Lettura del DataFrame, mantenendo l'intestazione a due livelli
        # Istruisce pandas a usare sia la riga 0 che la riga 1 come header
        df = pd.read_html(StringIO(table_html), header=[0, 1])[0] 
        
        # 3. Appiattimento delle Intestazioni (La Soluzione)
        # Prendi l'indice delle colonne e 'sgancia' il primo livello (livello 0)
        new_columns = []
        for top, bottom in df.columns:
            if top == "Per 90 Minutes":
                new_columns.append(f"%{bottom}")
            elif top == "Penalty Kicks":
                new_columns.append(f"PK{bottom}")
            else:
                new_columns.append(bottom)
        df.columns = new_columns
        
        # 4. Pulizia Finale del DataFrame
        # Rimuove le righe di intestazione ripetute che si trovano nel corpo della tabella
        df_pulito = df[df['Rk'] != 'Rk']
        
        # Resetta l'indice per avere una colonna 'Player' pulita
        df_pulito.columns.name = None
        
        return df_pulito

    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return None
    except Exception as e:
        print(f"Si è verificato un errore inatteso: {e}")
        return None
    
def giocatori():
    # Dati per la Serie A
    url_serie_a = 'https://fbref.com/en/comps/11/stats/Serie-A-Stats'
    id_statistiche_giocatori = 'stats_standard' 

    # Esegui lo scraping
    df_giocatori = scrape_fbref_second_level_header(url_serie_a, id_statistiche_giocatori)

    if df_giocatori is not None:
        print("\n--- ✅ DataFrame Statistiche Giocatori (Testa) con solo la seconda intestazione ---")
        print(df_giocatori.head())
        
        print("\n--- Nomi delle Colonne Semplificati ---")
        # Nota: alcune colonne che non avevano un header superiore saranno ancora presenti.
        print(df_giocatori.columns.tolist())
    return df_giocatori
    
def portieri():
    # Dati per la Serie A
    url_serie_a = 'https://fbref.com/en/comps/11/keepersadv/Serie-A-Stats'
    id_statistiche_giocatori = 'stats_keeper_adv' 

    # Esegui lo scraping
    df_giocatori_adv = scrape_fbref_second_level_header(url_serie_a, id_statistiche_giocatori)

    if df_giocatori_adv is not None:
        print("\n--- ✅ DataFrame Statistiche Giocatori (Testa) con solo la seconda intestazione ---")
        print(df_giocatori_adv.head())
        
        print("\n--- Nomi delle Colonne Semplificati ---")
        # Nota: alcune colonne che non avevano un header superiore saranno ancora presenti.
        print(df_giocatori_adv.columns.tolist())

    # Dati per la Serie A
    url_serie_a = 'https://fbref.com/en/comps/11/keepers/Serie-A-Stats'
    id_statistiche_giocatori = 'stats_keeper' 

    # Esegui lo scraping
    df_giocatori = scrape_fbref_second_level_header(url_serie_a, id_statistiche_giocatori)
    merged = df_giocatori_adv.merge(df_giocatori, left_on='Player', right_on='Player', suffixes=('', '_y'))
    merged.drop(merged.filter(regex='_y$').columns, axis=1, inplace=True)
    return merged

if __name__ == '__main__':
    giocatori()
    portieri()