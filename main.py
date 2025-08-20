from fantacalcion.ottimizza_formazione import ottimizza_formazione

giocatori = [
    {"nome": "Maignan", "ruolo": "P", "squadra": "Milan", "punteggio": 7.5},
    {"nome": "Bremer", "ruolo": "D", "squadra": "Juventus", "punteggio": 6.8},
    {"nome": "Dimarco", "ruolo": "D", "squadra": "Inter", "punteggio": 7.2},
    {"nome": "Biraghi", "ruolo": "D", "squadra": "Fiorentina", "punteggio": 6.6},
    {"nome": "Buongiorno", "ruolo": "D", "squadra": "Torino", "punteggio": 6.9},
    {"nome": "Barella", "ruolo": "C", "squadra": "Inter", "punteggio": 7.3},  # stessa squadra, verrà filtrato
    {"nome": "Koopmeiners", "ruolo": "C", "squadra": "Atalanta", "punteggio": 7.4},
    {"nome": "Pellegrini", "ruolo": "C", "squadra": "Roma", "punteggio": 7.0},
    {"nome": "Lobotka", "ruolo": "C", "squadra": "Napoli", "punteggio": 6.8},
    {"nome": "Osimhen", "ruolo": "A", "squadra": "Napoli", "punteggio": 8.0},
    {"nome": "Gudmundsson", "ruolo": "A", "squadra": "Genoa", "punteggio": 7.5},
    {"nome": "Leao", "ruolo": "A", "squadra": "Milan", "punteggio": 7.7},
]

if __name__ == "__main__":
    titolari, panchina, modulo = ottimizza_formazione(giocatori)

    print("Modulo scelto:", modulo)
    print("\n--- Titolari ---")
    for g in titolari:
        print(f"{g['ruolo']} - {g['nome']} ({g['squadra']}) [{g['punteggio']}]")

    print("\n--- Panchina ---")
    for g in panchina:
        print(f"{g['ruolo']} - {g['nome']} ({g['squadra']}) [{g['punteggio']}]")
