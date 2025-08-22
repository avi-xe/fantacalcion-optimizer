import requests
from bs4 import BeautifulSoup
import json

URL = "https://www.pianetafanta.it/probabili-formazioni-complete-serie-a-live.asp"


def fetch_page():
    headers = {"User-Agent": "Mozilla/5.0"}
    return requests.get(URL, headers=headers).text


def parse_table(table):
    probabili = {"casa": [], "trasferta": []}
    for tr in table.select("tbody tr"):
        tds = tr.select("td")
        if len(tds) < 6:
            continue
        _, name_home, role_home, role_away, name_away, _ = tds
        if role_home.text.strip():
            probabili["casa"].append(
                {
                    "nome": extract_text(name_home),
                    "ruolo": extract_text(role_home),
                    "valutazione": 0,
                }
            )
        if role_away.text.strip():
            probabili["trasferta"].append(
                {
                    "nome": extract_text(name_away),
                    "ruolo": extract_text(role_away),
                    "valutazione": 0,
                }
            )
    return probabili


def extract_text(el):
    if el is None:
        return ""
    return " ".join(el.get_text(strip=True).split())


def parse_panel(panel):
    teams = [t.get_text(strip=True) for t in panel.select(".TeamNome")]
    titolari = parse_table(panel.select_one(".tabella-probabili-titolari"))
    panchina = parse_table(panel.select_one(".tabella-probabili-panchina"))
    return {
        "casa": {
            "squadra": teams[0],
            "titolari": titolari['casa'],
            "panchina": panchina['casa']
        },
        "trasferta": {
            "squadra": teams[1],
            "titolari": titolari['trasferta'],
            "panchina": panchina['trasferta']
        },
    }


def main():
    with open("pianeta-fanta.html", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    matches = [parse_panel(panel) for panel in soup.select(".panel")]

    with open("pianeta-fanta.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
