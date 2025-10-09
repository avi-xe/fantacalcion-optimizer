import numpy as np
import random
import numba
from deap import base, creator, tools, algorithms
import pandas as pd

from scoring.scoring import compute_scores
from scraping.scraping_fantacalcio import crea_chiave_giocatore, load_fantacalcio_votes

# ---------- GA setup ----------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

n_outfield = 6
n_gk = 6
n_weights = n_outfield + n_gk
epsilon = 1e-8

# ---------- Precompute features ----------
# df_players_feat: n_players x 6 matrix
# df_keepers_feat: n_keepers x 6 matrix
# positions_players: n_players array of roles ("GK","DF","MF","FW")
# positions_keepers: n_keepers array of roles
# teams_players / teams_keepers: array of team IDs or indices
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

df_players = crea_chiave_giocatore(df_players)
df_keepers = crea_chiave_giocatore(df_keepers)

df_scores = compute_scores(df_players, df_keepers,
                           df_teams, prob_set, calendario)

# Players
player_features = df_scores[df_scores['Pos'] != 'GK'][[
    "Titolarità", "Forma", "BonusPot", "Affidabilità", "Calendario", "Penalità"]].values
positions_players = df_scores[df_scores['Pos']
                              != 'GK']["Pos"].values  # DF, MF, FW
teams_players = df_scores[df_scores['Pos'] != 'GK']["Squad"].values
votes_players = df_votes[df_votes["chiave"].isin(df_scores[df_scores['Pos'] != 'GK']["chiave"])].set_index(
    "Player").loc[df_scores[df_scores['Pos'] != 'GK']["chiave"]]["Voto"].values

# Keepers
gk_features = df_scores[df_scores['Pos'] == 'GK'][[
    "Titolarità", "Forma", "BonusPot", "Affidabilità", "Calendario", "Penalità"]].values
positions_keepers = df_scores[df_scores['Pos'] == 'GK']["Pos"].values
teams_keepers = df_scores[df_scores['Pos'] == 'GK']["Squad"].values
votes_keepers = df_votes[df_votes["Player"].isin(df_scores[df_scores['Pos'] == 'GK']["Player"])].set_index(
    "Player").loc[df_scores[df_scores['Pos'] == 'GK']["Player"]]["Voto"].values


# ---------- Constraint-aware initialization ----------
def init_individual():
    pos_out = np.random.rand(n_outfield-1)
    pos_out = pos_out / (np.sum(np.abs(pos_out)) + epsilon)
    pen_out = random.uniform(-1, 0)
    pos_gk = np.random.rand(n_gk-1)
    pos_gk = pos_gk / (np.sum(np.abs(pos_gk)) + epsilon)
    pen_gk = random.uniform(-1, 0)
    return creator.Individual(np.concatenate([pos_out, [pen_out], pos_gk, [pen_gk]]))


def normalize_individual(ind):
    ind[:n_outfield-1] = ind[:n_outfield-1] / \
        (np.sum(np.abs(ind[:n_outfield-1])) + epsilon)
    ind[n_outfield-1] = min(ind[n_outfield-1], 0)
    ind[n_outfield:-1] = ind[n_outfield:-1] / \
        (np.sum(np.abs(ind[n_outfield:-1])) + epsilon)
    ind[-1] = min(ind[-1], 0)
    return ind

# ---------- Numba-accelerated best_eleven ----------


@numba.njit
def best_eleven_numba(scores, positions, teams):
    """
    Select top 11 players in 3-4-3 respecting:
    - 1 GK, 3 DF, 4 MF, 3 FW
    - Only one player per team
    Returns indices of selected players
    """
    n = len(scores)
    idx_sorted = np.argsort(-scores)  # descending
    selected_idx = np.empty(11, dtype=np.int32)
    role_counts = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    team_set = set()
    count = 0

    for i in idx_sorted:
        role = positions[i]
        if role_counts[role] < {"GK": 1, "DF": 3, "MF": 4, "FW": 3}[role] and teams[i] not in team_set:
            selected_idx[count] = i
            role_counts[role] += 1
            team_set.add(teams[i])
            count += 1
            if count >= 11:
                break
    return selected_idx

# ---------- Fitness function ----------


df_players_feat = player_features
df_keepers_feat = gk_features


def fitness_function(individual):
    individual = normalize_individual(individual)
    w_out = individual[:n_outfield]
    w_gk = individual[n_outfield:]

    # Vectorized weighted scores
    scores_players = df_players_feat @ w_out
    scores_keepers = df_keepers_feat @ w_gk
    scores_all = np.concatenate([scores_players, scores_keepers])
    positions_all = np.concatenate([positions_players, positions_keepers])
    teams_all = np.concatenate([teams_players, teams_keepers])
    votes_all = np.concatenate([votes_players, votes_keepers])

    # Select best 11
    selected_idx = best_eleven_numba(scores_all, positions_all, teams_all)
    total_score = np.sum(votes_all[selected_idx])
    return (total_score,)


# ---------- GA operators ----------
toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)


def cxBlend_constrained(ind1, ind2, alpha=0.5):
    tools.cxBlend(ind1, ind2, alpha)
    normalize_individual(ind1)
    normalize_individual(ind2)
    return ind1, ind2


def mutGaussian_constrained(ind, mu=0, sigma=0.1, indpb=0.2):
    tools.mutGaussian(ind, mu, sigma, indpb)
    normalize_individual(ind)
    return ind,


toolbox.register("mate", cxBlend_constrained)
toolbox.register("mutate", mutGaussian_constrained)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------- Run GA ----------
population = toolbox.population(n=50)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    best = tools.selBest(population, 1)[0]
    print(f"Gen {gen+1}: Best score = {best.fitness.values[0]:.2f}")

best_ind = tools.selBest(population, 1)[0]
print("Best weights:", best_ind)
print("Best score:", best_ind.fitness.values[0])
