"""
Scraper per: https://www.sosfanta.com/lista-formazioni/probabili-formazioni-serie-a/
Versione: script Python (requests + BeautifulSoup)
Uso: python scraper_sosfanta.py
Output: stampa a schermo e salva `probabili_formazioni.json` e `probabili_formazioni.csv`

Note:
- Il sito sembra server-rendered (contenuto HTML disponibile via requests). Questo script fa una best-effort per estrarre:
  data, giornata, orario, squadra_casa, squadra_ospite, modulo_casa, modulo_ospite, allenatore_casa, allenatore_ospite,
  lista_in_campo (per entrambe le squadre), ballottaggi, squalificati, indisponibili, diffidati, altri
- I selettori sono resilienti ma potrebbero richiedere piccoli aggiustamenti se il layout cambia. I commenti indicano dove modificare.

Requisiti:
- Python 3.9+
- pip install requests beautifulsoup4 pandas

"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd

URL = "https://www.sosfanta.com/lista-formazioni/probabili-formazioni-serie-a/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def text_of(tag: Optional[Tag]) -> str:
    return tag.get_text(separator=" ", strip=True) if tag else ""


def chunk_texts_between(start_tag: Tag, stop_patterns: List[re.Pattern], side: str) -> List[Dict[str, str]]:
    """Raccoglie testi dei nodi successivi a start_tag fino a incontrare uno stop_patterns,
    marcando i record con 'in_casa' o 'fuori_casa'."""
    texts = []
    for sib in start_tag.next_siblings:
        if isinstance(sib, Tag):
            t = sib.get_text(separator=" ", strip=True)
            if any(p.search(t) for p in stop_patterns):
                break
            if t:
                for ln in t.splitlines():
                    ln = ln.strip()
                    if ln:
                        texts.append({"side": side, "player": ln})
    return texts


def parse_match_container(container: Tag) -> Dict:
    raw = container.get_text(separator="\n", strip=True)
    data = None
    giornata = None
    orario = None
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    for i, line in enumerate(lines[:8]):
        if re.search(r"\b\d{1,2}\s+[a-zàéò]+\b", line, re.I):
            data = line
        if re.search(r"\bgiornata\b|\b° giornata\b", line, re.I):
            giornata = line
        if re.search(r"ore\s*\d{1,2}:?\d{0,2}", line, re.I):
            orario = re.search(r"ore\s*(\d{1,2}:?\d{0,2})", line, re.I).group(1)

    teams = [l for l in lines if re.fullmatch(r"[A-ZÀÈÉ0-9\s'\-]{2,}", l)]
    squadra_casa = teams[0] if len(teams) >= 1 else None
    squadra_ospite = teams[1] if len(teams) >= 2 else None

    modulos = re.findall(r"(\d-\d-\d(?:-\d)?)", raw)
    modulo_casa = modulos[0] if len(modulos) >= 1 else None
    modulo_ospite = modulos[1] if len(modulos) >= 2 else None

    allenatori = re.findall(r"Allenatore\s*\n?\s*([A-Za-zÀ-ÖØ-öø-ÿ'\.\- ]{2,60})", container.get_text(separator="\n"))
    allenatore_casa = allenatori[0] if len(allenatori) >= 1 else None
    allenatore_ospite = allenatori[1] if len(allenatori) >= 2 else None

    in_campo_blocks = {"casa": [], "ospite": []}
    for side_div in container.find_all("div", class_="is-6"):
        side = "casa" if "column-left" in side_div.get("class", []) else "ospite"
        for h in side_div.find_all(lambda tag: tag.name in ['h2','h3','strong','b','p']):
            players = h.get_text()
            if players:
                in_campo_blocks[side].append(players)

    def extract_after(keyword):
        m = container.find(string=re.compile(keyword, re.I))
        if m:
            texts = chunk_texts_between(m.parent if isinstance(m.parent, Tag) else m, [re.compile(r'###|Altri|Ballottaggi', re.I)], "")
            return texts
        return []

    ballottaggi = extract_after('Ballottaggi')
    squalificati = extract_after('Squalificati')
    indisponibili = extract_after('Indisponibili')
    diffidati = extract_after('Diffidati')
    altri = extract_after('Altri')

    return {
        'data_raw': data,
        'giornata_raw': giornata,
        'orario': orario,
        'squadra_casa': squadra_casa,
        'squadra_ospite': squadra_ospite,
        'modulo_casa': modulo_casa,
        'modulo_ospite': modulo_ospite,
        'allenatore_casa': allenatore_casa,
        'allenatore_ospite': allenatore_ospite,
        'in_campo_casa': in_campo_blocks["casa"],
        'in_campo_ospite': in_campo_blocks["ospite"],
        'ballottaggi': ballottaggi,
        'squalificati': squalificati,
        'indisponibili': indisponibili,
        'diffidati': diffidati,
        'altri': altri,
        'raw_lines': lines[:40]
    }


def scrape(url=URL) -> List[Dict]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    matches = []
    for tag in soup.find_all(string=re.compile(r'In campo', re.I)):
        parent = tag.parent
        container = parent
        for _ in range(6):
            if container is None:
                break
            text = container.get_text(separator='\n', strip=True)
            if 'Modulo' in text and text.count('\n') > 6:
                parsed = parse_match_container(container)
                matches.append(parsed)
                break
            container = container.parent

    unique = []
    seen = set()
    for m in matches:
        key = (m.get('squadra_casa'), m.get('squadra_ospite'), m.get('data_raw'))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique

def get_probabili():
    print("Richiedo la pagina...")
    data = scrape()
    print(f"Trovati {len(data)} match parsati (best-effort).\nEsempio primo match:\n")
    if data:
        print(json.dumps(data[0], ensure_ascii=False, indent=2))

    with open('probabili_formazioni.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    rows = []
    for m in data:
        rows.append({
            'data': m.get('data_raw'),
            'giornata': m.get('giornata_raw'),
            'orario': m.get('orario'),
            'casa': m.get('squadra_casa'),
            'ospite': m.get('squadra_ospite'),
            'modulo_casa': m.get('modulo_casa'),
            'modulo_ospite': m.get('modulo_ospite'),
            'in_campo_casa': m.get('in_campo_casa'),
            'in_campo_ospite': m.get('in_campo_ospite')
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = get_probabili()
    df.to_csv('probabili_formazioni.csv', index=False)
    print('\nSalvati: probabili_formazioni.json e probabili_formazioni.csv')

    print('\nNOTE:\n- Ora i giocatori “In campo” sono contrassegnati con attributo side="casa" o "ospite" in base al container <div>.')
