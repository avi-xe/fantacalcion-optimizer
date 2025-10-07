from deap import base, creator, tools, algorithms
import numpy as np
import random
import pandas as pd
import logging
from optimizer.best_eleven import best_eleven
from scoring.scoring import compute_scores
from scraping.scraping_fantacalcio import crea_chiave_giocatore, load_fantacalcio_votes

matchweek = 1
df_keepers = pd.read_parquet("./static/fbref/portieri.parquet")
df_players = pd.read_parquet("./static/fbref/giocatori.parquet")
df_teams = pd.read_parquet("./static/fbref/squads.parquet")
calendario = pd.read_parquet(
    f"./static/fbref/schedule_{matchweek}.parquet")

df_votes = load_fantacalcio_votes(
    f"./static/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
)
df_votes.rename(columns={"Nome": "Player",
                "Ruolo": "Pos"}, inplace=True)
df_votes = crea_chiave_giocatore(df_votes)

# prob_set solo per giocatori a voto
prob_set = pd.DataFrame({"in_campo_casa": df_votes["chiave"].values,
                         "in_campo_ospite": ['']*len(df_votes)})


bounds = [(0, 1)] * 5 + [(-1, 0)] + [(0, 1)] * 5 + [(-1, 0)]


logging.info("Starting Differential Evolution optimization...")

history = []

df_players = crea_chiave_giocatore(df_players)
df_keepers = crea_chiave_giocatore(df_keepers)


# ---------- Fitness and individual ----------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# ---------- Parameters ----------
n_outfield = 6
n_gk = 6
n_weights = n_outfield + n_gk  # total 12 weights
epsilon = 1e-8

# ---------- Custom functions to maintain constraints ----------


def init_individual():
    """Initialize individual with normalized positive weights and penalty weight ≤ 0"""
    # Outfield weights: 5 positive + 1 penalty
    pos_out = np.random.rand(n_outfield - 1)
    pos_out = pos_out / (np.sum(np.abs(pos_out)) + epsilon)
    pen_out = random.uniform(-1, 0)
    outfield = np.concatenate([pos_out, [pen_out]])

    # GK weights: 5 positive + 1 penalty
    pos_gk = np.random.rand(n_gk - 1)
    pos_gk = pos_gk / (np.sum(np.abs(pos_gk)) + epsilon)
    pen_gk = random.uniform(-1, 0)
    gk = np.concatenate([pos_gk, [pen_gk]])

    return creator.Individual(np.concatenate([outfield, gk]))


def normalize_individual(ind):
    """Ensure constraints are satisfied after mutation/crossover"""
    ind = np.array(ind)
    # Outfield
    ind[:n_outfield-1] = ind[:n_outfield-1] / \
        (np.sum(np.abs(ind[:n_outfield-1])) + epsilon)
    ind[n_outfield-1] = min(ind[n_outfield-1], 0)
    # GK
    ind[n_outfield:-1] = ind[n_outfield:-1] / \
        (np.sum(np.abs(ind[n_outfield:-1])) + epsilon)
    ind[-1] = min(ind[-1], 0)
    return ind

# ---------- Fitness function ----------


def fitness_function(individual, dataset):
    individual = normalize_individual(individual)
    df_players, df_keepers, df_teams, prob_set, calendario, df_votes = dataset

    w_out = individual[:n_outfield]
    w_gk = individual[n_outfield:]

    df_scores = compute_scores(
        df_players, df_keepers, df_teams, prob_set, calendario,
        pesi_out=dict(zip(["Titolarità", "Forma", "BonusPot",
                      "Affidabilità", "Calendario", "Penalità"], w_out)),
        pesi_gk=dict(zip(["Titolarità", "Forma", "BonusPot",
                     "Affidabilità", "Calendario", "Penalità"], w_gk))
    )

    best11 = best_eleven(df_scores)
    score = df_votes.loc[
        df_votes.Player.isin(crea_chiave_giocatore(best11).chiave.values),
        "Voto"
    ].sum()

    return (score,)


# ---------- GA Setup ----------
dataset = [df_players,
           df_keepers,
           df_teams,
           prob_set,
           calendario, df_votes]
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function, dataset=dataset)

# Crossover and mutation


def normalize_individual(ind):
    """Normalize weights in-place and keep it an Individual"""
    # Outfield
    ind[:n_outfield-1] = ind[:n_outfield-1] / \
        (np.sum(np.abs(ind[:n_outfield-1])) + 1e-8)
    ind[n_outfield-1] = min(ind[n_outfield-1], 0)
    # GK
    ind[n_outfield:-1] = ind[n_outfield:-1] / \
        (np.sum(np.abs(ind[n_outfield:-1])) + 1e-8)
    ind[-1] = min(ind[-1], 0)
    return ind  # still a DEAP Individual


def cxBlend_constrained(ind1, ind2, alpha=0.5):
    tools.cxBlend(ind1, ind2, alpha)
    normalize_individual(ind1)
    normalize_individual(ind2)
    return ind1, ind2


def mutGaussian_constrained(individual, mu=0, sigma=0.1, indpb=0.2):
    tools.mutGaussian(individual, mu, sigma, indpb)
    normalize_individual(individual)
    return individual,


toolbox.register("mate", cxBlend_constrained, alpha=0.5)
toolbox.register("mutate", mutGaussian_constrained, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------- Run GA ----------
population = toolbox.population(n=50)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    # Evaluate fitness
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    # Select next generation
    population = toolbox.select(offspring, k=len(population))
    # Optional logging
    best = tools.selBest(population, k=1)[0]
    print(f"Gen {gen+1}: Best score = {best.fitness.values[0]:.2f}")

# ---------- Best solution ----------
best_ind = tools.selBest(population, k=1)[0]
print("Best weights:", best_ind)
print("Best score:", best_ind.fitness.values[0])