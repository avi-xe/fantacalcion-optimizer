from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from numba import njit, prange
from optimizer.best_eleven import best_eleven
from scraping.scraping_fantacalcio import crea_chiave_giocatore, load_fantacalcio_votes

# ---------------- Load Data ----------------
df_keepers = pd.read_parquet("./static/fbref/portieri.parquet")
df_players = pd.read_parquet("./static/fbref/giocatori.parquet")
df_teams = pd.read_parquet("./static/fbref/squads.parquet")
# calendario = pd.read_parquet(f"./static/fbref/schedule_{matchweek}.parquet")

# df_votes = load_fantacalcio_votes(
#     f"./static/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
# )
# df_votes.rename(columns={"Nome": "Player", "Ruolo": "Pos"}, inplace=True)
# df_votes = crea_chiave_giocatore(df_votes)

# prob_set = pd.DataFrame({"in_campo_casa": df_votes["chiave"].values,
#                          "in_campo_ospite": ['']*len(df_votes)})

# ---------------- DEAP setup ----------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_individual():
    pos_out = np.random.rand(n_outfield-1)
    pos_out = pos_out / (np.sum(np.abs(pos_out)) + epsilon)
    pen_out = random.uniform(-1,0)
    outfield = np.concatenate([pos_out,[pen_out]])

    pos_gk = np.random.rand(n_gk-1)
    pos_gk = pos_gk / (np.sum(np.abs(pos_gk)) + epsilon)
    pen_gk = random.uniform(-1,0)
    gk = np.concatenate([pos_gk,[pen_gk]])

    return creator.Individual(np.concatenate([outfield,gk]))

def normalize_individual(ind):
    ind = np.array(ind)
    ind[:n_outfield-1] = ind[:n_outfield-1]/(np.sum(np.abs(ind[:n_outfield-1]))+epsilon)
    ind[n_outfield-1] = min(ind[n_outfield-1],0)
    ind[n_outfield:-1] = ind[n_outfield:-1]/(np.sum(np.abs(ind[n_outfield:-1]))+epsilon)
    ind[-1] = min(ind[-1],0)
    return ind

toolbox = base.Toolbox()
toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------- Numba helpers ----------------

@njit(parallel=True)
def compute_scores_numba(features, weights):
    n_players, n_features = features.shape
    n_individuals = weights.shape[0]
    out = np.zeros((n_players, n_individuals), dtype=np.float32)
    for i in prange(n_players):
        for j in prange(n_individuals):
            s = 0.0
            for k in range(n_features):
                s += features[i,k]*weights[j,k]
            out[i,j] = s
    return out

@njit(fastmath=True, parallel=True)
def normalize_rows(A):
    """
    Normalize each row of A so it sums to 1.
    Rows that sum to 0 remain all zeros.
    """
    A = np.asarray(A, dtype=float)  # ensure float type for division
    row_sums = A.sum(axis=1, keepdims=True)
    
    # Safe elementwise division — avoids NaNs for zero rows
    return np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)

# ---------------- Evaluate population ----------------
def evaluate_population_with_best11(individuals, player_features, keeper_features, df_players, df_keepers, df_votes):
    results = []
    for ind in individuals:
        w_out = np.array(ind[:n_outfield],dtype=np.float32).reshape(1,-1)
        w_gk  = np.array(ind[n_outfield:],dtype=np.float32).reshape(1,-1)

        scores_out = compute_scores_numba(player_features,w_out).flatten()
        scores_gk  = compute_scores_numba(keeper_features,w_gk).flatten()

        df_scores_players = df_players.copy()
        df_scores_players["Punteggio"] = scores_out
        df_scores_keepers = df_keepers.copy()
        df_scores_keepers["Punteggio"] = scores_gk

        df_best11 = best_eleven(pd.concat([df_scores_players,df_scores_keepers],ignore_index=True))
        df_best11 = crea_chiave_giocatore(df_best11)
        keys_best11 = df_best11["chiave"].values
        score = df_votes.loc[df_votes["chiave"].isin(keys_best11),"Voto"].sum()
        results.append(score)
    return np.array(results,dtype=np.float32)
import os

# ---------------- Multi-matchweek optimization ----------------
n_outfield = 6
n_gk = 6
population_size = 50
NGEN = 40
epsilon = 1e-8
all_matchweeks = range(1, 39)  # Customize as needed
best_weights_list = []
results_log = []

output_dir = "./optimization_results"
os.makedirs(output_dir, exist_ok=True)

for matchweek in all_matchweeks:
    print(f"\n==============================")
    print(f"⚽ Optimizing weights for Matchweek {matchweek}")
    print(f"==============================")

    # Reload data for this matchweek
    calendario = pd.read_parquet(f"./static/fbref/schedule_{matchweek}.parquet")
    df_votes = load_fantacalcio_votes(
        f"./static/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
    )
    df_votes.rename(columns={"Nome": "Player", "Ruolo": "Pos"}, inplace=True)
    df_votes = crea_chiave_giocatore(df_votes)

    prob_set = pd.DataFrame({
        "in_campo_casa": df_votes["chiave"].values,
        "in_campo_ospite": [''] * len(df_votes)
    })

    # Preprocess
    player_features, keeper_features, df_players, df_keepers = preprocess_features(
        df_players, df_keepers, df_teams, prob_set, calendario
    )

    # Initialize population
    population = [toolbox.individual() for _ in range(population_size)]

    # Run GA
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = evaluate_population_with_best11(
            offspring, player_features, keeper_features, df_players, df_keepers, df_votes
        )
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = (fit,)
        population = toolbox.select(offspring, k=population_size)

        best_ind = tools.selBest(population, k=1)[0]
        print(f"Gen {gen+1}/{NGEN} | Best score: {best_ind.fitness.values[0]:.2f}")

    # Store best weights for this week
    best_ind = tools.selBest(population, k=1)[0]
    best_weights = np.array(best_ind)
    best_score = best_ind.fitness.values[0]

    best_weights_list.append(best_weights)
    results_log.append({
        "Matchweek": matchweek,
        "BestScore": best_score,
        **{f"W{i+1}": w for i, w in enumerate(best_weights)}
    })

    # Save this matchweek’s weights immediately
    pd.DataFrame([results_log[-1]]).to_csv(
        os.path.join(output_dir, f"best_weights_week_{matchweek}.csv"),
        index=False
    )

# ---------------- Average weights across weeks ----------------
avg_weights = np.mean(np.vstack(best_weights_list), axis=0)

print("\n==============================")
print("🏁 FINAL AVERAGED WEIGHTS")
print("==============================")
print(avg_weights)

# Save all results
df_all = pd.DataFrame(results_log)
df_all.to_csv(os.path.join(output_dir, "best_weights_per_week.csv"), index=False)

pd.DataFrame({
    "WeightIndex": range(1, len(avg_weights)+1),
    "AvgWeight": avg_weights
}).to_csv(os.path.join(output_dir, "final_average_weights.csv"), index=False)

print(f"\n💾 Results saved to: {output_dir}")
