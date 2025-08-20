# ottimizza_formazione.py
from typing import List, Dict, Tuple, Optional
from heapq import nlargest
import pulp

# Moduli ammessi dalla tua lega
MODULI_AMMESSI = [
    (3, 4, 3),
    (4, 3, 3),
    (4, 4, 2),
    (3, 5, 2),
    (5, 3, 2),
    (5, 4, 1),
    (4, 5, 1),
]


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

def applica_calendario(giocatori, partite,
                       bonus_segno={"att_fav":0.3,"dif_fav":0.2,"att_sfav":-0.3,"dif_sfav":-0.2},
                       bonus_ou={"over_att":0.2,"under_dif":0.2}):
    """
    Aggiorna il punteggio dei giocatori in base al calendario (1X2, Over/Under).
    - bonus_segno: premi/sfavi in base a favorito/sfavorito
    - bonus_ou: premi in base a Over/Under
    """
    squad_to_match = {}
    for m in partite:
        squad_to_match[m["casa"]] = {"side":"home","segno":m["segno"],"ou":m["ou"],"opp":m["trasferta"]}
        squad_to_match[m["trasferta"]] = {"side":"away","segno":m["segno"],"ou":m["ou"],"opp":m["casa"]}

    giocatori_adj = []
    for g in giocatori:
        new_g = g.copy()
        if g["squadra"] not in squad_to_match:
            giocatori_adj.append(new_g)
            continue

        info = squad_to_match[g["squadra"]]

        # --- 1X2 adjustments ---
        if (info["side"]=="home" and info["segno"]=="1") or (info["side"]=="away" and info["segno"]=="2"):
            # squadra favorita
            if g["ruolo"]=="A":
                new_g["punteggio"] += bonus_segno["att_fav"]
            elif g["ruolo"] in ("D","P"):
                new_g["punteggio"] += bonus_segno["dif_fav"]
        elif (info["side"]=="home" and info["segno"]=="2") or (info["side"]=="away" and info["segno"]=="1"):
            # squadra sfavorita
            if g["ruolo"]=="A":
                new_g["punteggio"] += bonus_segno["att_sfav"]
            elif g["ruolo"] in ("D","P"):
                new_g["punteggio"] += bonus_segno["dif_sfav"]
        # segno X non dà variazioni

        # --- Over/Under adjustments ---
        if info["ou"]=="Over" and g["ruolo"]=="A":
            new_g["punteggio"] += bonus_ou["over_att"]
        if info["ou"]=="Under" and g["ruolo"] in ("D","P"):
            new_g["punteggio"] += bonus_ou["under_dif"]

        giocatori_adj.append(new_g)

    return giocatori_adj
