from json import load

from optimization.calendar import applica_calendario
from optimization.const import MODULI_AMMESSI
from optimization.optimization import ottimizza_formazione
from optimization.utils import check_input_validity


def from_sqlite():
    with open("pianeta-fanta-updated.json") as f:
        pianeta_fanta_updated = load(f)

    giocatori = []
    for matches in pianeta_fanta_updated:
        casa = matches["casa"]
        squadra_casa = casa["squadra"]
        for titolare in casa["titolari"]:
            giocatori.append(
                {
                    "nome": titolare["nome"],
                    "ruolo": titolare["ruolo"],
                    "squadra": squadra_casa,
                    "punteggio": titolare["valutazione"],
                }
            )
        for panchinaro in casa["panchina"]:
            giocatori.append(
                {
                    "nome": panchinaro["nome"],
                    "ruolo": panchinaro["ruolo"],
                    "squadra": squadra_casa,
                    "punteggio": panchinaro["valutazione"],
                }
            )

        trasferta = matches["trasferta"]
        squadra_trasferta = trasferta["squadra"]
        for titolare in trasferta["titolari"]:
            giocatori.append(
                {
                    "nome": titolare["nome"],
                    "ruolo": titolare["ruolo"],
                    "squadra": squadra_trasferta,
                    "punteggio": titolare["valutazione"],
                }
            )
        for panchinaro in trasferta["panchina"]:
            giocatori.append(
                {
                    "nome": panchinaro["nome"],
                    "ruolo": panchinaro["ruolo"],
                    "squadra": squadra_trasferta,
                    "punteggio": panchinaro["valutazione"],
                }
            )

    return giocatori


if __name__ == "__main__":
    # with open("giocatori.json") as f:
    #     giocatori = load(f)

    giocatori = from_sqlite()

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
