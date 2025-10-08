from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from numba import njit, prange
from optimizer.best_eleven import best_eleven
from scraping.scraping_fantacalcio import crea_chiave_giocatore, load_fantacalcio_votes

# ---------------- Parameters ----------------
matchweek = 15
n_outfield = 6
n_gk = 6
population_size = 50
NGEN = 40
epsilon = 1e-8

# ---------------- Load Data ----------------
df_keepers = pd.read_parquet("./static/fbref/portieri.parquet")
df_players = pd.read_parquet("./static/fbref/giocatori.parquet")
df_teams = pd.read_parquet("./static/fbref/squads.parquet")
calendario = pd.read_parquet(f"./static/fbref/schedule_{matchweek}.parquet")

df_votes = load_fantacalcio_votes(
    f"./static/voti_fantacalcio/Voti_Fantacalcio_Stagione_2024_25_Giornata_{matchweek}.xlsx"
)
df_votes.rename(columns={"Nome": "Player", "Ruolo": "Pos"}, inplace=True)
df_votes = crea_chiave_giocatore(df_votes)

prob_set = pd.DataFrame({"in_campo_casa": df_votes["chiave"].values,
                         "in_campo_ospite": ['']*len(df_votes)})

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
@njit(fastmath=True, parallel=True)
def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    diff = max_val - min_val if max_val != min_val else 1.0
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        out[i] = (arr[i]-min_val)/diff
    return out

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

# ---------------- Preprocessing ----------------
def preprocess_features(df_players, df_keepers, df_teams, prob_set, calendario):
    # Convert numeric
    numeric_cols_players = ["%xG","%xAG","%G+A","Starts","MP","Min","CrdY","CrdR"]
    numeric_cols_keepers = ["Save%","PSxG+/-","CS%","Starts","MP","GA90"]
    df_players[numeric_cols_players] = df_players[numeric_cols_players].fillna(0).astype(np.float32)
    df_keepers[numeric_cols_keepers] = df_keepers[numeric_cols_keepers].fillna(0).astype(np.float32)

    # Titolarità
    titolari = set(prob_set["in_campo_casa"].explode()).union(set(prob_set["in_campo_ospite"].explode()))
    df_players = crea_chiave_giocatore(df_players)
    df_keepers = crea_chiave_giocatore(df_keepers)
    df_players["Titolarità"] = np.where(df_players["chiave"].isin(titolari),1,0).astype(np.float32)
    df_keepers["Titolarità"] = np.where(df_keepers["chiave"].isin(titolari),1,0).astype(np.float32)

    # Forma
    df_players["Forma"] = normalize_array((df_players["%xG"].values + df_players["%xAG"].values).astype(np.float32))
    df_keepers["Forma"] = normalize_array(df_keepers[["Save%","PSxG+/-"]].mean(axis=1).values.astype(np.float32))

    # BonusPot
    df_players["BonusPot"] = normalize_array(df_players["%G+A"].values.astype(np.float32))
    df_keepers["BonusPot"] = normalize_array(df_keepers["CS%"].values.astype(np.float32))

    # Affidabilità
    aff_players = (df_players["Starts"].values / np.where(df_players["MP"].values==0,1,df_players["MP"].values)) * \
                  (df_players["Min"].values / (np.where(df_players["MP"].values==0,1,df_players["MP"].values)*90))
    df_players["Affidabilità"] = normalize_array(aff_players.astype(np.float32))

    aff_keepers = df_keepers["Starts"].values / np.where(df_keepers["MP"].values==0,1,df_keepers["MP"].values)
    df_keepers["Affidabilità"] = normalize_array(aff_keepers.astype(np.float32))

    # Penalità
    df_players["Penalità"] = normalize_array((df_players["CrdY"].values*0.05 + df_players["CrdR"].values*0.2).astype(np.float32))
    df_keepers["Penalità"] = normalize_array(df_keepers["GA90"].values.astype(np.float32))

    # Calendario (simple loop)
    team_stats = df_teams.set_index("Squadra")
    def compute_cal(df,xga_col,xg_col):
        cal_scores = np.ones(len(df),dtype=np.float32)
        for i in range(len(df)):
            squadra = df.iloc[i]["Squad"]
            pos = df.iloc[i]["Pos"]
            match = calendario[(calendario["Casa"]==squadra)|(calendario["Ospite"]==squadra)].head(1)
            if match.empty:
                cal_scores[i] = 1.0
            else:
                in_casa = match["Casa"].values[0]==squadra
                avv = match["Ospite"].values[0] if in_casa else match["Casa"].values[0]
                if avv not in team_stats.index:
                    cal_scores[i] = 1.0
                else:
                    if pos!="P":
                        cal_scores[i] = team_stats.loc[avv,xga_col] if in_casa else team_stats.loc[avv,"H_xGA"]
                    else:
                        cal_scores[i] = team_stats.loc[avv,xg_col] if in_casa else team_stats.loc[avv,"H_xG"]
        return normalize_array(cal_scores)

    df_players["Calendario"] = compute_cal(df_players,"A_xGA","A_xG")
    df_keepers["Calendario"] = compute_cal(df_keepers,"A_xGA","A_xG")

    player_features = df_players[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)
    keeper_features = df_keepers[["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]].values.astype(np.float32)

    return player_features, keeper_features, df_players, df_keepers

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

# ---------------- Main GA ----------------
player_features, keeper_features, df_players, df_keepers = preprocess_features(df_players, df_keepers, df_teams, prob_set, calendario)
population = [toolbox.individual() for _ in range(population_size)]

for gen in range(NGEN):
    # Generate offspring
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)

    # Evaluate fitness
    fits = evaluate_population_with_best11(offspring, player_features, keeper_features, df_players, df_keepers, df_votes)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = (fit,)

    # Select next generation
    population = toolbox.select(offspring, k=population_size)

    # Get best individual and best 11
    best_ind = tools.selBest(population, k=1)[0]

    # Compute scores for logging best 11
    w_out = np.array(best_ind[:n_outfield], dtype=np.float32).reshape(1, -1)
    w_gk  = np.array(best_ind[n_outfield:], dtype=np.float32).reshape(1, -1)

    scores_out = compute_scores_numba(player_features, w_out).flatten()
    scores_gk  = compute_scores_numba(keeper_features, w_gk).flatten()

    df_scores_players = df_players.copy()
    df_scores_players["Punteggio"] = scores_out
    df_scores_keepers = df_keepers.copy()
    df_scores_keepers["Punteggio"] = scores_gk

    df_best11 = best_eleven(pd.concat([df_scores_players, df_scores_keepers], ignore_index=True))
    df_best11 = crea_chiave_giocatore(df_best11)
    
    # Log generation info
    print(f"\nGeneration {gen+1}")
    print(f"Best score: {best_ind.fitness.values[0]:.2f}")
    print("Best weights (outfield + GK):", best_ind)
    print("Best 11 players:")
    i = 0
    for idx, row in df_best11.iterrows():
        voto = df_votes[df_votes.chiave == row.chiave].Voto.values
        if voto is None:
            voto = 0
        print(f"{row['Player']} ({row['Pos']}) - {row['Punteggio']:.2f} - {voto}")
        i = i+1


best_ind = tools.selBest(population,k=1)[0]
print("Best weights:", best_ind)
print("Best score:", best_ind.fitness.values[0])
