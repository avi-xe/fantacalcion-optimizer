from heapq import nlargest
from typing import Dict, List, Tuple

from fantacalcion.const import MODULI_AMMESSI


def check_input_validity(giocatori: List[Dict], moduli=MODULI_AMMESSI, verbose=True):
    """
    Controlla se l'input consente almeno un modulo considerando ANCHE la panchina:
    panchina = 2D + 2C + 2A + 1 jolly (no portieri).
    """
    ruoli = {"P": 0, "D": 0, "C": 0, "A": 0}
    squads = set()
    for g in giocatori:
        ruoli[g["ruolo"]] += 1
        squads.add(g["squadra"])

    if verbose:
        print("---- CHECK INPUT ----")
        print("Totale giocatori:", len(giocatori))
        print("Squadre distinte:", len(squads))
        print("Distribuzione ruoli:", ruoli)

    valid = []
    for D, C, A in moduli:
        # minimi per panchina
        need_D = D + 2
        need_C = C + 2
        need_A = A + 2
        # +1 jolly da D/C/A
        # verifica per-ruolo + totale outfield
        ok_roles = (
            ruoli["P"] >= 1
            and ruoli["D"] >= need_D
            and ruoli["C"] >= need_C
            and ruoli["A"] >= need_A
        )
        ok_total_out = (ruoli["D"] + ruoli["C"] + ruoli["A"]) >= (D + C + A + 7)
        if ok_roles and ok_total_out:
            valid.append((D, C, A))

    if verbose:
        if valid:
            print("Moduli realizzabili considerando anche la panchina:", valid)
            if len(squads) < 18:
                print(
                    "⚠️ Attenzione: servono 18 squadre distinte (11+7) per rispettare il vincolo."
                )
        else:
            print(
                "❌ Nessun modulo realizzabile dato l'insieme attuale (considerando anche la panchina)."
            )

    return valid


def _bonus_modificatore_difesa(
    titolari: List[Dict],
    modulo: Tuple[int, int, int],
    soglie_bonus: List[Tuple[float, int]],
) -> int:
    D, _, _ = modulo
    if D not in (4, 5):
        return 0
    dif = [float(g["punteggio"]) for g in titolari if g["ruolo"] == "D"]
    if not dif:
        return 0
    elements = [float(g["punteggio"]) for g in titolari if g["ruolo"] == "P"]
    if not elements:
        return 0
    elif len(elements) != 1:
        return 0
    elements.extend(nlargest(3, dif))
    media = sum(elements) / 4
    bonus = 0
    for soglia, val in sorted(soglie_bonus):
        if media >= soglia:
            bonus = val
    return bonus
