import requests
from bs4 import BeautifulSoup

URL_FC = "https://www.fantacalcio.it/probabili-formazioni"

def get_next_matches():
    html = requests.get(URL_FC).text
    soup = BeautifulSoup(html, "html.parser")
    teams = [t.get_text(strip=True) for t in soup.select(".match-team-name")]
    
    # Trasformo in coppie (squadra casa, squadra trasferta)
    matches = [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]
    return matches

if __name__ == "__main__":
    print(get_next_matches())
