import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
import cloudscraper

scraper = cloudscraper.create_scraper()


def scrape_fbref_second_level_header(url, table_id) -> pd.DataFrame:
    """
    Esegue lo scraping di una tabella FBref e utilizza il secondo livello 
    del MultiIndex come intestazione unica del DataFrame.

    :param url: L'URL della pagina FBref.
    :param table_id: L'ID della tabella da estrarre (es. 'stats_standard').
    :return: DataFrame di pandas con intestazioni a livello singolo o None.
    """
    print(
        f"Sto recuperando i dati da: {url} e cercando la tabella '{table_id}'...")

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
                new_columns.append(f"{bottom}_per90")
            elif top == "Penalty Kicks":
                new_columns.append(f"{bottom}_PK")
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


PLACEHOLDERS = ['stats', 'shooting', 'passing', 'passing_types',
                'gca', 'defense', 'possession', 'playingtime', 'misc']
URLS = [
    f'https://fbref.com/en/comps/11/{placeholder}/Serie-A-Stats' for placeholder in PLACEHOLDERS]


def fetch_giocatori() -> pd.DataFrame:
    # Esegui lo scraping
    df = scrape_fbref_second_level_header(URLS[0], 'stats_standard')
    df2 = scrape_fbref_second_level_header(URLS[1], 'stats_shooting')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[2], 'stats_passing')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')

    df2 = scrape_fbref_second_level_header(URLS[3], 'stats_passing_types')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[4], 'stats_gca')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[5], 'stats_defense')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[6], 'stats_possession')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[7], 'stats_playing_time')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    df2 = scrape_fbref_second_level_header(URLS[8], 'stats_misc')
    different_columns = df2.columns.difference(df.columns)
    df2 = df2[different_columns]
    df = pd.merge(df, df2, left_index=True,
                  right_index=True, how='inner')
    return df


def giocatori(df_giocatori) -> pd.DataFrame:
    if df_giocatori is not None:
        print("\n--- ✅ DataFrame Statistiche Giocatori (Testa) con solo la seconda intestazione ---")
        print(df_giocatori.head())

        print("\n--- Nomi delle Colonne Semplificati ---")
        # Nota: alcune colonne che non avevano un header superiore saranno ancora presenti.
        print(df_giocatori.columns.tolist())
    df_giocatori.Gls = pd.to_numeric(df_giocatori.Gls, errors='coerce')
    df_giocatori.Ast = pd.to_numeric(df_giocatori.Ast, errors='coerce')
    df_giocatori.Min = pd.to_numeric(df_giocatori.Min, errors='coerce')
    return df_giocatori


def fetch_portieri_adv(url_serie_a='https://fbref.com/en/comps/11/keepersadv/Serie-A-Stats') -> pd.DataFrame:
    # Dati per la Serie A
    id_statistiche_giocatori = 'stats_keeper_adv'
    # Esegui lo scraping
    return scrape_fbref_second_level_header(url_serie_a, id_statistiche_giocatori)


def fetch_portieri(url_serie_a='https://fbref.com/en/comps/11/keepers/Serie-A-Stats'):
    # Dati per la Serie A
    id_statistiche_giocatori = 'stats_keeper'
    # Esegui lo scraping
    return scrape_fbref_second_level_header(url_serie_a, id_statistiche_giocatori)


def portieri(df_giocatori, df_giocatori_adv) -> pd.DataFrame:
    if df_giocatori_adv is not None:
        print("\n--- ✅ DataFrame Statistiche Giocatori (Testa) con solo la seconda intestazione ---")
        print(df_giocatori_adv.head())

        print("\n--- Nomi delle Colonne Semplificati ---")
        # Nota: alcune colonne che non avevano un header superiore saranno ancora presenti.
        print(df_giocatori_adv.columns.tolist())

    merged = df_giocatori_adv.merge(
        df_giocatori, left_on='Rk', right_on='Rk', suffixes=('', '_y'))
    merged.drop(merged.filter(regex='_y$').columns, axis=1, inplace=True)
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return merged


if __name__ == '__main__':
    df_giocatori = fetch_giocatori(
        url_serie_a="https://fbref.com/en/comps/11/2024-2025/stats/2024-2025-Serie-A-Stats")
    df_giocatori = giocatori(df_giocatori=df_giocatori)
    df_giocatori.to_parquet("giocatori.parquet")
    df_portieri = fetch_portieri(
        url_serie_a="https://fbref.com/en/comps/11/2024-2025/keepers/2024-2025-Serie-A-Stats")
    df_portieri_adv = fetch_portieri_adv(
        url_serie_a="https://fbref.com/en/comps/11/2024-2025/keepersadv/2024-2025-Serie-A-Stats")
    df_portieri = portieri(
        df_giocatori_adv=df_portieri_adv, df_giocatori=df_portieri)
    df_portieri.to_parquet("portieri.parquet")
