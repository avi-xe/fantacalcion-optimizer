import sqlite3
from json import load, dump


def init_db():
    conn = sqlite3.connect("pianeta_fanta.db")
    cur = conn.cursor()

    # Create tables
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        casa TEXT NOT NULL,
        trasferta TEXT NOT NULL
    )
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        squadra TEXT,
        sezione TEXT,   -- titolari / panchina
        nome TEXT,
        ruolo TEXT,
        valutazione REAL DEFAULT 0,
        FOREIGN KEY(match_id) REFERENCES matches(id)
    )
    """
    )

    conn.commit()
    return conn


def save_matches(conn, matches):
    cur = conn.cursor()

    for match in matches:
        # Insert the match
        cur.execute(
            "INSERT INTO matches (casa, trasferta) VALUES (?, ?)",
            (match["casa"]["squadra"], match["trasferta"]["squadra"]),
        )
        match_id = cur.lastrowid

        # Insert players
        for side in ["casa", "trasferta"]:
            for sezione in ["titolari", "panchina"]:
                for player in match[side][sezione]:
                    cur.execute(
                        """
                        INSERT INTO players (match_id, squadra, sezione, nome, ruolo, valutazione)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            match_id,
                            match[side]["squadra"],
                            sezione,
                            player["nome"],
                            player["ruolo"],
                            player["valutazione"],
                        ),
                    )

    conn.commit()


def load_from_db(conn):
    cur = conn.cursor()

    # get all matches
    cur.execute("SELECT id, casa, trasferta FROM matches")
    matches = []

    for match_id, casa, trasferta in cur.fetchall():
        # init structure
        match_data = {
            "casa": {"squadra": casa, "titolari": [], "panchina": []},
            "trasferta": {"squadra": trasferta, "titolari": [], "panchina": []},
        }

        # get players for this match
        cur.execute("""
            SELECT squadra, sezione, nome, ruolo, valutazione
            FROM players
            WHERE match_id = ?
        """, (match_id,))

        for squadra, sezione, nome, ruolo, valutazione in cur.fetchall():
            player = {"nome": nome, "ruolo": ruolo, "valutazione": valutazione}
            if squadra == casa:
                match_data["casa"][sezione].append(player)
            else:
                match_data["trasferta"][sezione].append(player)

        matches.append(match_data)

    return matches

def export_json(conn, filename="pianeta-fanta.json"):
    matches = load_from_db(conn)
    with open(filename, "w", encoding="utf-8") as f:
        dump(matches, f, ensure_ascii=False, indent=2)


conn = init_db()
export_json(conn, filename="pianeta-fanta-updated.json")
