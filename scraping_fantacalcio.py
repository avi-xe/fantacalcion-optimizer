import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.pianetafanta.it/Probabili-Formazioni-Complete-Serie-A-Live.asp"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_probabili_formazioni(url=URL):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    probabilis = soup.find_all("div", class_="tabella-probabili-titolari")  # esempio di classe, da verificare
    result = []

    for probabili in probabilis:
        titolari = [titolare.get_text(strip=True) for titolare in probabili.find_all("a")]
        [result.append(titolare) for titolare in titolari]

    df = pd.DataFrame(result)
    return df

if __name__ == "__main__":
    df_formazioni = fetch_formazioni()
    print(df_formazioni.head(10))
