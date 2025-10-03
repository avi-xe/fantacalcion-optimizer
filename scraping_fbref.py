"""
scraping_fbref.py
Playwright-based FBref scraper that returns pandas DataFrames for players and teams.

Usage:
    pip install playwright pandas beautifulsoup4 lxml
    python -m playwright install chromium
    python scraping_fbref.py
"""
from typing import Tuple, Dict, List, Optional
import time
import io
import re
import pandas as pd
from bs4 import BeautifulSoup, Comment

# Playwright import inside functions to avoid requiring it on import-time
FBREF_STATS_URL = "https://fbref.com/en/comps/11/stats/Serie-A-Stats"  # English path tends to be stable


def _fetch_page_content_with_playwright(url: str, headless: bool = True, timeout: int = 30000) -> str:
    """
    Uses Playwright sync API to load the page and return the fully rendered HTML content.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError("Playwright non installato. Esegui: pip install playwright && python -m playwright install chromium") from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/129.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        # navigate and wait for network idle
        page.goto(url, wait_until="networkidle", timeout=timeout)
        # small sleep to let scripts run (FBref fills content fast but safety)
        time.sleep(0.5)
        content = page.content()
        context.close()
        browser.close()
        return content


def _extract_big_comment_block(html: str, tag: str) -> Optional[str]:
    """
    Cerca il commento principale che contiene le table (FBref spesso mette le tabelle dentro un unico commento).
    Se non trova commenti con <table>, ritorna None.
    """
    soup = BeautifulSoup(html, "lxml")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    # preferiamo il primo commento che contiene "<table"
    tables = []
    for c in comments:
        if isinstance(c, str) and "<table" in c and tag in c:
            tables.append(c)
    return tables


def _parse_tables_from_html_block(html_block: str) -> List[pd.DataFrame]:
    """
    Data una stringa HTML (o il commento che contiene <table>), ritorna la lista di DataFrame.
    Tenta header multi-livello e poi pulisce ogni df.
    """
    soup2 = BeautifulSoup(html_block, "lxml")
    tables = soup2.find_all("table")
    dfs: List[pd.DataFrame] = []

    for t in tables:
        t_html = str(t)
        df = None
        # proviamo a leggere con header a 2 livelli, poi 1 livello, poi senza header
        for header in ([0, 1], [0], None):
            try:
                if header is None:
                    tmp = pd.read_html(t_html, header=None)[0]
                else:
                    tmp = pd.read_html(t_html, header=header)[0]
                df = tmp.copy()
                break
            except Exception:
                continue

        if df is None:
            continue

        # Appiattisco colonne multiindex
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_cols.append("_".join([str(x).strip() for x in col if str(x).strip() != ""]))
            else:
                new_cols.append(str(col).strip())
        df.columns = new_cols

        # Pulizia: rimuovo righe duplicate che contengono header ripetuti nel corpo
        # Identifico valori header set e rimuovo righe che hanno molti valori che coincidono con header names
        header_set = set([hc for hc in new_cols if hc])
        def is_header_row(row):
            # conta quante celle della riga corrispondono a header_set
            count = sum(1 for val in row if str(val).strip() in header_set)
            # se almeno metà delle colonne coincidono è molto probabile che sia una riga header ripetuta
            return count >= max(1, len(header_set) // 2)
        try:
            mask = df.apply(lambda r: not is_header_row(r), axis=1)
            df = df[mask]
        except Exception:
            # fallback se qualcosa va storto nella detection
            pass

        # drop rows completamente vuote
        df = df.dropna(how="all").reset_index(drop=True)
        dfs.append(df)

    return dfs


def _try_parse_csv_links_from_page(html: str) -> Dict[str, str]:
    """
    Cerca link a .csv nella pagina (es. link 'Get table as CSV').
    Ritorna mapping text -> href (assoluto o relativo).
    """
    soup = BeautifulSoup(html, "lxml")
    csv_links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".csv" in href.lower():
            txt = (a.get_text(strip=True) or href)
            csv_links[txt] = href
    return csv_links


def _choose_players_df(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Sceglie la tabella giocatori da una lista di DataFrame usando euristiche
    (colonna Player / Giocatore / Squad / Nome ecc.).
    """
    for df in dfs:
        cols = [c.lower() for c in df.columns]
        if any(c in cols for c in ("player", "giocatore", "nome", "player_")):
            return df
        # fallback: ha sia Squad e Gls
        if "squad" in cols and any(c in cols for c in ("gls", "g")):
            return df
    # se non abbiamo trovato nulla, prova il primo che contiene molte colonne
    if dfs:
        return max(dfs, key=lambda x: x.shape[1])
    return None


def _choose_team_tables(dfs: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Separa e ritorna possibili tabelle squadra: usa euristiche su colonne (Squad/Team/Gls/GA).
    Ritorna dict con chiavi descriptive.
    """
    team_tables = {}
    for df in dfs:
        cols = [c.lower() for c in df.columns]
        if "squad" in cols or "team" in cols or "squadra" in cols:
            # tentiamo di classificare il tipo guardando alcune colonne chiave
            key = "team_unknown"
            if any(x in cols for x in ("ga", "g_a", "ga_")) or "ga" in cols:
                key = "defense"  # contiene goals against
            if any(x in cols for x in ("shots", "sh", "sh_xg")) or "shots" in cols:
                key = "shooting"
            if any(x in cols for x in ("pass", "passes")):
                key = "passing"
            # evitare sovrascrittura: aggiungiamo indice se necessario
            i = 1
            base = key
            while f"{base}_{i}" in team_tables:
                i += 1
            team_tables[f"{base}_{i}"] = df
    return team_tables


# Public API functions ------------------------------------------------------

def get_player_stats_from_page(url: str = FBREF_STATS_URL, headless: bool = True) -> pd.DataFrame:
    """
    Carica la pagina con Playwright, estrae il commento principale e ritorna il DataFrame giocatori.
    """
    print(f"[scraper] loading page: {url}")
    html = _fetch_page_content_with_playwright(url, headless=headless)
    # 1) prova a trovare il commento grande
    comment_block = _extract_big_comment_block(html, "stats_standard")
    block_to_parse = comment_block if comment_block else html

    dfs = _parse_tables_from_html_block(block_to_parse[0])
    players_df = _choose_players_df(dfs)
    if players_df is not None:
        # rename colonne italiane/english comuni
        rename_map = {
            "Unnamed: 1_level_0_Player": "Nome",
            "Unnamed: 4_level_0_Squad": "Squadra",
            "Playing Time_90s": "Nineties", "Playing Time_90s": "Partite", "Playing Time_Min": "Min"
        }
        cols_lower = {c: (rename_map.get(c, rename_map.get(c.capitalize(), c))) for c in players_df.columns}
        players_df = players_df.rename(columns=cols_lower)
        return players_df.reset_index(drop=True)

    # fallback: prova a trovare link csv e caricarlo con pandas
    csv_links = _try_parse_csv_links_from_page(html)
    for txt, href in csv_links.items():
        if "player" in txt.lower() or "stats" in txt.lower():
            try:
                csv_url = href if href.startswith("http") else ("https://fbref.com" + href)
                df = pd.read_csv(csv_url, skiprows=1)
                return df
            except Exception:
                continue

    raise RuntimeError("Impossibile trovare la tabella giocatori dalla pagina FBref.")

def get_team_stats_from_page(url: str = FBREF_STATS_URL, headless: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Restituisce un dict di DataFrame relativi alle tabelle squadra trovate.
    """
    print(f"[scraper] loading page (teams): {url}")
    html = _fetch_page_content_with_playwright(url, headless=headless)
    soup = BeautifulSoup(html, "lxml")

    # 1) Trova tabelle già nel body (senza commento)
    tables = soup.find_all("table")
    dfs = []
    for t in tables:
        try:
            df = pd.read_html(str(t))[0]
            dfs.append(df)
        except Exception:
            continue

    # 2) Se non trovi nulla, prova anche nel commento (fallback)
    if not dfs:
        comment_block = _extract_big_comment_block(html)
        if comment_block:
            dfs = _parse_tables_from_html_block(comment_block)

    # 3) Classifica tabelle per tipo (standard/shooting/passing/defense)
    # team_tables = _choose_team_tables(dfs)
    return pd.concat([dfs[9], dfs[10]])



# Test / CLI ---------------------------------------------------------------

if __name__ == "__main__":
    print("Eseguo test rapido scraper FBref (Playwright).")
    try:
        players = get_player_stats_from_page(headless=True)
        print("-> Players table shape:", players.shape)
        print(players.head(6))
    except Exception as e:
        print("Errore nel recupero giocatori:", e)

    try:
        teams = get_team_stats_from_page(headless=True)
        print("\n-> Team tables trovate:", list(teams.keys()))
        # stampo dimensione e prime righe di ciascuna
        for k, df in teams.items():
            print(f"\n--- {k} ({df.shape}) ---")
            print(df.head(4))
    except Exception as e:
        print("Errore nel recupero squadre:", e)
