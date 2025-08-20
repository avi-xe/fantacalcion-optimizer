def applica_calendario(
    giocatori,
    partite,
    bonus_segno={"att_fav": 0.3, "dif_fav": 0.2, "att_sfav": -0.3, "dif_sfav": -0.2},
    bonus_ou={"over_att": 0.2, "under_dif": 0.2},
):
    """
    Aggiorna il punteggio dei giocatori in base al calendario (1X2, Over/Under).
    - bonus_segno: premi/sfavi in base a favorito/sfavorito
    - bonus_ou: premi in base a Over/Under
    """
    squad_to_match = {}
    for m in partite:
        squad_to_match[m["casa"]] = {
            "side": "home",
            "segno": m["segno"],
            "ou": m["ou"],
            "opp": m["trasferta"],
        }
        squad_to_match[m["trasferta"]] = {
            "side": "away",
            "segno": m["segno"],
            "ou": m["ou"],
            "opp": m["casa"],
        }

    giocatori_adj = []
    for g in giocatori:
        new_g = g.copy()
        if g["squadra"] not in squad_to_match:
            giocatori_adj.append(new_g)
            continue

        info = squad_to_match[g["squadra"]]

        # --- 1X2 adjustments ---
        if (info["side"] == "home" and info["segno"] == "1") or (
            info["side"] == "away" and info["segno"] == "2"
        ):
            # squadra favorita
            if g["ruolo"] == "A":
                new_g["punteggio"] += bonus_segno["att_fav"]
            elif g["ruolo"] in ("D", "P"):
                new_g["punteggio"] += bonus_segno["dif_fav"]
        elif (info["side"] == "home" and info["segno"] == "2") or (
            info["side"] == "away" and info["segno"] == "1"
        ):
            # squadra sfavorita
            if g["ruolo"] == "A":
                new_g["punteggio"] += bonus_segno["att_sfav"]
            elif g["ruolo"] in ("D", "P"):
                new_g["punteggio"] += bonus_segno["dif_sfav"]
        # segno X non dà variazioni

        # --- Over/Under adjustments ---
        if info["ou"] == "Over" and g["ruolo"] == "A":
            new_g["punteggio"] += bonus_ou["over_att"]
        if info["ou"] == "Under" and g["ruolo"] in ("D", "P"):
            new_g["punteggio"] += bonus_ou["under_dif"]

        giocatori_adj.append(new_g)

    return giocatori_adj
