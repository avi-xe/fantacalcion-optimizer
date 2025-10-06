from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import logging
from optimizer.best_eleven import best_eleven

from scoring.scoring import compute_scores
from scraping.scraping_fantacalcio import crea_chiave_giocatore
import numpy as np


def normalize_weights(weights):
    """
    Normalize weights so that their absolute sum is 1.
    Separates penalty weight (last) from positive weights.
    """
    weights = np.asarray(weights, dtype=float)
    pos_weights = weights[:5]
    pen_weight = weights[5]  # keep sign
    pos_weights = pos_weights / (np.sum(np.abs(pos_weights)) + 1e-8)
    weights[:5] = pos_weights
    weights[5] = pen_weight
    return weights


def make_weights_dict(weights):
    keys = ["Titolarità", "Forma", "BonusPot",
            "Affidabilità", "Calendario", "Penalità"]
    return dict(zip(keys, weights))


def evaluate_weights(weights, dataset, n_perturb=3, epsilon=1e-3):
    """
    Evaluate a weight vector on a single dataset with optional perturbation smoothing.
    """
    df_players, df_keepers, df_teams, prob_set, calendario, df_votes = dataset

    # Base score
    df_scores = compute_scores(
        df_players, df_keepers, df_teams, prob_set, calendario,
        pesi_out=make_weights_dict(weights[:6]),
        pesi_gk=make_weights_dict(weights[6:])
    )
    best11 = best_eleven(df_scores)
    base_score = df_votes.loc[
        df_votes.Player.isin(crea_chiave_giocatore(best11).chiave.values),
        "Voto"
    ].sum()

    # Smooth by small random perturbations
    perturbed_scores = []
    for _ in range(n_perturb):
        perturb = weights + epsilon * np.random.randn(len(weights))
        df_scores_p = compute_scores(
            df_players, df_keepers, df_teams, prob_set, calendario,
            pesi_out=make_weights_dict(perturb[:6]),
            pesi_gk=make_weights_dict(perturb[6:])
        )
        best11_p = best_eleven(df_scores_p)
        score_p = df_votes.loc[
            df_votes.Player.isin(
                crea_chiave_giocatore(best11_p).chiave.values),
            "Voto"
        ].sum()
        perturbed_scores.append(score_p)

    smoothed_score = (base_score + np.mean(perturbed_scores)) / 2
    return smoothed_score


def objective(weights, dataset, history=None):
    """
    DE-friendly objective: maximize total score (we return negative for minimization)
    """
    weights = normalize_weights(weights)
    score = evaluate_weights(weights, dataset)

    if history is not None:
        record = {"weights": weights.copy(), "score": score}
        print(record)
        history.append(record)

    return -score
