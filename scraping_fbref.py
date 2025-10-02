import pandas as pd

URL_PLAYERS = "https://fbref.com/it/comps/11/stats/Serie-A-Stats"
URL_TEAMS = "https://fbref.com/it/comps/11/Serie-A-Stats"

def get_player_stats():
    tables = pd.read_html(URL_PLAYERS)
    df = tables[0]

    df = df[df["Giocatore"].notna()]
    df = df.rename(columns={
        "Giocatore": "Nome",
        "Squadra": "Squadra",
        "G": "Gol",
        "A": "Assist",
        "90s": "Nineties",
        "Min": "Min"
    })
    return df

def get_team_stats():
    tables = pd.read_html(URL_TEAMS)
    df = tables[0]
    df = df.rename(columns={
        "Squadra": "Team",
        "GA": "GolSubiti"
    })
    return df

if __name__ == "__main__":
    print(get_player_stats().head())
    print(get_team_stats().head())
