from fantacalcion.optimization import (
    MODULI_AMMESSI,
    applica_calendario,
    ottimizza_formazione,
    check_input_validity,
)
from json import load

if __name__ == "__main__":
    with open("giocatori.json") as f:
        giocatori = load(f)

    with open("partite.json") as f:
        partite = load(f)

    # Applico aggiustamenti di calendario
    giocatori = applica_calendario(giocatori, partite)
    valid = check_input_validity(giocatori, MODULI_AMMESSI, verbose=True)
    sol = ottimizza_formazione(
        giocatori, MODULI_AMMESSI, soglie_bonus=[(6.5, 1), (7.0, 3), (7.5, 4)]
    )

    print("\n=== RISULTATO ===")
    print(
        "Modulo scelto:",
        sol["modulo"],
        "| Bonus difesa:",
        sol["bonus"],
        "| Totale titolari atteso:",
        round(sol["total_expected"], 2),
    )
    print("\n-- Titolari --")
    for g in sol["titolari"]:
        print(f"{g['ruolo']} - {g['nome']} ({g['squadra']}) [{g['punteggio']}]")
    print("\n-- Panchina --")
    for g in sol["panchina"]:
        print(f"{g['ruolo']} - {g['nome']} ({g['squadra']}) [{g['punteggio']}]")
