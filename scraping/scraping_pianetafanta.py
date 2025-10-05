import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_pianetafanta():
    url = 'https://www.pianetafanta.it/voti-ufficiosi.asp'
    r = requests.get(url, timeout=15)
    tables = pd.read_html(r.text)
    # Qui puoi filtrare la tabella giusta manualmente o con indice
    df_votes = tables[0]  # esempio, modificare se necessario
    # Aggiungi colonna 'Giornata' se mancante
    df_votes['Giornata'] = 1  # esempio placeholder
    df_votes.rename(columns={'Nome':'Player','Voto':'Voto'}, inplace=True)
    return df_votes