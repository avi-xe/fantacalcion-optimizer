import requests
from bs4 import BeautifulSoup

URL_FC = "https://www.fantacalcio.it/probabili-formazioni"

def get_probabili_formazioni():
    html = requests.get(URL_FC).text
    soup = BeautifulSoup(html, "html.parser")
    players = [p.get_text(strip=True) for p in soup.select(".player-name")]
    return set(players)

if __name__ == "__main__":
    print(list(get_probabili_formazioni())[:20])
