# ottimizza_formazione.py
from typing import List, Dict, Tuple, Optional
import pulp

from optimization.const import MODULI_AMMESSI
from optimization.utils import _bonus_modificatore_difesa, check_input_validity


def ottimizza_formazione(
    giocatori: List[Dict],
    moduli=MODULI_AMMESSI,
    soglie_bonus: List[Tuple[float, int]] = [(6.5, 1), (7.0, 3), (7.5, 4)],
    weight_bench: float = 0.1,
):
    # Pre-check fattibilità (considerando panchina)
    valid_moduli = check_input_validity(giocatori, moduli, verbose=False)
    if not valid_moduli:
        raise ValueError(
            "Nessun modulo realizzabile con i vincoli (inclusa la panchina). "
            "Aggiungi giocatori, in particolare nei ruoli carenti."
        )

    best = None
    best_total = float("-inf")

    for modulo in valid_moduli:
        sol = _solve_for_module(giocatori, modulo, weight_bench=weight_bench)
        if sol is None:
            continue
        bonus = _bonus_modificatore_difesa(sol["titolari"], sol["modulo"], soglie_bonus)
        total = (
            sol["score_titolari"] + bonus
        )  # panchina non conta nel totale fantavoto; serve solo come tie-break
        if total > best_total:
            best_total = total
            best = {**sol, "bonus": bonus, "total_expected": total}

    if best is None:
        raise ValueError(
            "Impossibile trovare una soluzione ottima per i moduli validati. "
            "Probabile conflitto squadre o ruoli."
        )

    return best


def _solve_for_module(
    giocatori: List[Dict],
    modulo: Tuple[int, int, int],
    weight_bench: float = 0.1,
) -> Optional[Dict]:
    """
    Risolve l'IP per un modulo fissato (D,C,A).
    Panchina: no portiere; almeno 2D, 2C, 2A; totale panca 7 (il jolly è libero su D/C/A).
    Vincolo squadre: titolari+panca max 1 per squadra.
    Vincolo giocatore: non può essere sia titolare sia panchinaro.
    """
    D, C, A = modulo
    n = len(giocatori)

    # Variabili binarie
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")  # titolari
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")  # panchina

    prob = pulp.LpProblem(f"Formazione_{D}_{C}_{A}", pulp.LpMaximize)

    # --- Titolari ---
    prob += pulp.lpSum(x[i] for i in range(n)) == 11
    prob += pulp.lpSum(x[i] for i, g in enumerate(giocatori) if g["ruolo"] == "P") == 1
    prob += pulp.lpSum(x[i] for i, g in enumerate(giocatori) if g["ruolo"] == "D") == D
    prob += pulp.lpSum(x[i] for i, g in enumerate(giocatori) if g["ruolo"] == "C") == C
    prob += pulp.lpSum(x[i] for i, g in enumerate(giocatori) if g["ruolo"] == "A") == A

    # --- Panchina ---
    prob += pulp.lpSum(y[i] for i in range(n)) == 7
    prob += pulp.lpSum(y[i] for i, g in enumerate(giocatori) if g["ruolo"] == "P") == 0
    prob += pulp.lpSum(y[i] for i, g in enumerate(giocatori) if g["ruolo"] == "D") >= 2
    prob += pulp.lpSum(y[i] for i, g in enumerate(giocatori) if g["ruolo"] == "C") >= 2
    prob += pulp.lpSum(y[i] for i, g in enumerate(giocatori) if g["ruolo"] == "A") >= 2

    # --- Un giocatore non può essere sia titolare che in panchina ---
    for i in range(n):
        prob += x[i] + y[i] <= 1

    # --- Squadre diverse su Titolari+Panchina ---
    squads = set(g["squadra"] for g in giocatori)
    for s in squads:
        prob += (
            pulp.lpSum(
                x[i] + y[i] for i, g in enumerate(giocatori) if g["squadra"] == s
            )
            <= 1
        )

    # --- Obiettivo ---
    score_titolari = pulp.lpSum(
        x[i] * float(g["punteggio"]) for i, g in enumerate(giocatori)
    )
    score_panchina = pulp.lpSum(
        y[i] * weight_bench * float(g["punteggio"]) for i, g in enumerate(giocatori)
    )
    prob += score_titolari + score_panchina

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None

    titolari_idx = [i for i in range(n) if x[i].value() == 1]
    panchina_idx = [i for i in range(n) if y[i].value() == 1]

    return {
        "titolari": [giocatori[i] for i in titolari_idx],
        "panchina": [giocatori[i] for i in panchina_idx],
        "modulo": modulo,
        "score_titolari": sum(float(giocatori[i]["punteggio"]) for i in titolari_idx),
        "score_panchina": sum(
            weight_bench * float(giocatori[i]["punteggio"]) for i in panchina_idx
        ),
    }
