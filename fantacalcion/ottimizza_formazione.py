import pulp

def ottimizza_formazione(giocatori, moduli=[(3,4,3), (4,4,2), (4,3,3), (3,5,2), (5,3,2), (5,4,1), (4,5,1)],
                         bonus_modificatore={4:3, 5:6}):
    """
    giocatori: lista di dict con chiavi:
        - nome (str)
        - ruolo (str: 'P','D','C','A')
        - squadra (str)
        - punteggio (float)
    moduli: lista di tuple (D,C,A) possibili
    bonus_modificatore: dict che mappa n_difensori -> bonus
    """

    prob = pulp.LpProblem("Formazione", pulp.LpMaximize)

    n = len(giocatori)

    # Variabili: 1 se scelto in titolari
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")

    # Variabili: 1 se scelto in panchina
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")

    # Variabili: 1 se modulo scelto
    m = pulp.LpVariable.dicts("m", range(len(moduli)), cat="Binary")

    # --- Vincoli titolari ---
    prob += pulp.lpSum(x[i] for i in range(n)) == 11
    prob += pulp.lpSum(x[i] for i,g in enumerate(giocatori) if g['ruolo']=='P') == 1

    # --- Vincoli panchina ---
    prob += pulp.lpSum(y[i] for i in range(n)) == 7
    prob += pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='P') == 0  # no portieri in panchina
    prob += pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='D') == 2
    prob += pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='C') == 2
    prob += pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='A') == 2
    # 1 jolly qualsiasi ruolo
    prob += pulp.lpSum(y[i] for i in range(n)) - (
        pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='D') +
        pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='C') +
        pulp.lpSum(y[i] for i,g in enumerate(giocatori) if g['ruolo']=='A')
    ) == 1

    # --- Esclusione doppioni squadra ---
    squadre = set(g['squadra'] for g in giocatori)
    for s in squadre:
        prob += pulp.lpSum(x[i] + y[i] for i,g in enumerate(giocatori) if g['squadra']==s) <= 1

    # --- Vincoli modulo scelto ---
    prob += pulp.lpSum(m[j] for j in range(len(moduli))) == 1
    prob += pulp.lpSum(x[i] for i,g in enumerate(giocatori) if g['ruolo']=='D') == \
            pulp.lpSum(moduli[j][0] * m[j] for j in range(len(moduli)))
    prob += pulp.lpSum(x[i] for i,g in enumerate(giocatori) if g['ruolo']=='C') == \
            pulp.lpSum(moduli[j][1] * m[j] for j in range(len(moduli)))
    prob += pulp.lpSum(x[i] for i,g in enumerate(giocatori) if g['ruolo']=='A') == \
            pulp.lpSum(moduli[j][2] * m[j] for j in range(len(moduli)))

    # --- Obiettivo: massimizzare punteggi + bonus difesa ---
    score_titolari = pulp.lpSum(x[i] * g['punteggio'] for i,g in enumerate(giocatori))
    score_panchina = pulp.lpSum(y[i] * (0.2 * g['punteggio']) for i,g in enumerate(giocatori))
    # (panchina pesa poco, giusto per rompere i pareggi; coefficiente 0.2 modificabile)

    # bonus difesa in funzione del modulo
    bonus = pulp.lpSum(bonus_modificatore.get(moduli[j][0], 0) * m[j] for j in range(len(moduli)))

    prob += score_titolari + score_panchina + bonus

    # Risolvi
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output
    titolari = [giocatori[i] for i in range(n) if x[i].value() == 1]
    panchina = [giocatori[i] for i in range(n) if y[i].value() == 1]
    modulo_scelto = moduli[[j for j in range(len(moduli)) if m[j].value() == 1][0]]

    return titolari, panchina, modulo_scelto
