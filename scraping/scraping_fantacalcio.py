import requests
from bs4 import BeautifulSoup
import pandas as pd


def crea_chiave_giocatore(df, col_nome="Player"):
    """
    Crea chiave univoca per i giocatori:
      - Cognome + abbreviazione nome se ci sono più nomi diversi con lo stesso cognome
      - Solo cognome se il cognome è unico
    L'abbreviazione prende le lettere necessarie per distinguere i nomi all'interno dello stesso cognome.
    """
    preposizioni = {"de", "del", "della", "dei",
                    "di", "da", "dal", "dalla", "lo", "la"}

    df = df.copy()
    df["tokens"] = df[col_nome].str.split()

    # Estrae NomeProprio e Cognome
    def estrai_nome_cognome(tokens):
        if len(tokens) == 1:
            return "", tokens[0]
        cognome = [tokens[-1]]
        for t in reversed(tokens[:-1]):
            if t.lower() in preposizioni:
                cognome.insert(0, t)
            else:
                break
        nome = tokens[: len(tokens) - len(cognome)]
        return " ".join(nome), " ".join(cognome)

    df[["NomeProprio", "Cognome"]] = df["tokens"].apply(
        lambda x: pd.Series(estrai_nome_cognome(x)))

    # Raggruppa per cognome e ottieni tutti i nomi unici per quel cognome
    gruppi = df.groupby("Cognome")["NomeProprio"].unique().to_dict()

    chiavi = {}

    for cognome, nomi in gruppi.items():
        # Se c'è un solo nome per quel cognome → chiave = cognome
        if len(nomi) == 1:
            chiavi.update({(cognome, nomi[0]): cognome})
        else:
            # Altrimenti genera abbreviazioni uniche tra i nomi
            abbrev_usate = {}
            for nome in nomi:
                # Determina la lunghezza minima per distinguere questo nome dagli altri
                l = 1
                while True:
                    abbrev = nome[:l]
                    # controlla se abbrev è unico tra gli altri nomi
                    collision = any(abbrev == abbrev_usate.get(n, "")
                                    for n in abbrev_usate)
                    if not collision:
                        break
                    l += 1
                abbrev_usate[nome] = abbrev
                chiavi[(cognome, nome)] = f"{cognome} {abbrev}."

    # Applica le chiavi al dataframe
    df["chiave"] = df.apply(
        lambda row: chiavi[(row["Cognome"], row["NomeProprio"])], axis=1)
    return df


def calcola_fanta_voto(x):
    score = float(''.join([c for c in str(x.Voto) if c != '*'])) + x.Gf * 3 - x.Gs + x.Rp*3 - \
        x.Rs*3 + x.Rf*2 - x.Amm*.5 - x.Esp + x.Ass
    return score


def load_fantacalcio_votes(xlsx_path):
    # Leggi il foglio saltando le prime 4 righe (così la riga 5 è la prima squadra)
    df_raw = pd.read_excel(
        xlsx_path, sheet_name="Fantacalcio", skiprows=4, header=None)

    teams_data = []
    current_team = None
    current_block = []

    for _, row in df_raw.iterrows():
        # Rimuove valori NaN e stringhe vuote
        non_nulls = [str(x).strip()
                     for x in row if pd.notna(x) and str(x).strip() != ""]

        # Caso 1 → Riga con solo un valore = nome squadra
        if len(non_nulls) == 1:
            # Se c'era un blocco precedente, salvalo
            if current_team and current_block:
                teams_data.append((current_team, pd.DataFrame(current_block)))
            # Nuova squadra
            current_team = non_nulls[0]
            current_block = []
            continue

        # Caso 2 → Riga giocatore o allenatore
        if len(non_nulls) > 2:
            current_block.append(row.values.tolist())

        # Caso 3 → Riga ALL (fine squadra)
        if len(non_nulls) > 1 and non_nulls[1] == "ALL":
            # Salva il blocco squadra (senza includere ALL)
            if current_team and current_block:
                df_team = pd.DataFrame(current_block)
                teams_data.append((current_team, df_team))
            current_team = None
            current_block = []

    # Combina tutti i blocchi in un unico DataFrame
    all_players = []
    for team_name, df_team in teams_data:
        df_team = df_team.reset_index(drop=True)

        # 🔹 Usa la prima riga come intestazione
        df_team.columns = df_team.iloc[0]
        df_team = df_team[1:].reset_index(drop=True)
        df_team["FantaVoto"] = df_team.apply(
            lambda x: calcola_fanta_voto(x), axis=1)

        # Se ha le colonne che ci servono
        if {'Nome', 'Ruolo', 'Voto', 'Gf', 'Gs', 'Rp', 'Rs', 'Rf', 'Au', 'Amm', 'Esp', 'Ass', 'FantaVoto'}.issubset(df_team.columns):
            temp = df_team[['Nome', 'Ruolo']].copy()
            temp['Voto'] = df_team["FantaVoto"]
            temp['Squad'] = team_name
            all_players.append(temp)

    df_final = pd.concat(all_players, ignore_index=True)
    df_final.rename(columns={'Nome': 'Player', 'Ruolo': 'Pos'}, inplace=True)
    df_final['Voto'] = pd.to_numeric(df_final['Voto'], errors='coerce')
    df_final.dropna(subset=['Voto'], inplace=True)

    return df_final.reset_index(drop=True)


URL = "https://www.pianetafanta.it/Probabili-Formazioni-Complete-Serie-A-Live.asp"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_probabili_formazioni(url=URL):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # esempio di classe, da verificare
    probabilis = soup.find_all("div", class_="tabella-probabili-titolari")
    result = []

    for probabili in probabilis:
        titolari = [titolare.get_text(strip=True)
                    for titolare in probabili.find_all("a")]
        [result.append(titolare) for titolare in titolari]

    df = pd.DataFrame(result)
    return df


if __name__ == "__main__":
    df_formazioni = fetch_formazioni()
    print(df_formazioni.head(10))
