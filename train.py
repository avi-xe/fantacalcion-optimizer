# requisiti: pandas, numpy, torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# Helper: mappatura pos -> slots (esempio per 3-4-3; adattare)
ROLE_SLOTS = {
    "GK": 1,
    "DF": 3,   # es. difensori in campo
    "MF": 4,
    "FW": 3
}
# Se usi formazioni dinamiche, passa i slot per giornata/utente
# ---------------------------------------------------------

# df_features: columns include ['Giornata','Player','Squad','Pos', feat_1, feat_2, ...]
# df_votes: columns include ['Giornata','Player','Voto']

def prepare_tensors_for_day(df_features, df_votes, giornata, feat_cols):
    """
    Ritorna:
     - X: torch tensor (N x F) features normalizzate per giornata
     - y: torch tensor (N,) voti reali (se mancanti -> 0)
     - pos_idx: dict(role -> mask indices for players of that role)
     - players: list of player names (index -> player)
    """
    df_day = df_features[df_features["Giornata"] == giornata].copy()
    if df_day.empty:
        return None

    # allinea nomi e unisci voti
    votes_day = df_votes[df_votes["Giornata"] == giornata][["Player","Voto"]]
    df_day = df_day.merge(votes_day, on="Player", how="left")
    df_day["Voto"] = df_day["Voto"].fillna(0.0)

    players = df_day["Player"].tolist()
    pos = df_day["Pos"].tolist()
    X = torch.tensor(df_day[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df_day["Voto"].values, dtype=torch.float32)

    # normalizza features (per giornata)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)

    # pos masks
    pos_idx = {}
    for r in ROLE_SLOTS.keys():
        pos_idx[r] = torch.tensor([i for i,p in enumerate(pos) if p.startswith(r)], dtype=torch.long)

    return X, y, pos_idx, players

# ---------------------------------------------------------
# Modello semplice: logit = X @ w + b
class LinearScorer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features) * 0.01)
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        # X: N x F
        return X.matmul(self.w) + self.b  # shape N

# ---------------------------------------------------------
# Soft selection per ruolo (differenziabile)
def expected_formation_vote(logits, y, pos_idx, role_slots, temp=0.1):
    """
    logits: tensor (N,)
    y: tensor (N,) voti reali
    pos_idx: dict role-> indices (LongTensor)
    role_slots: dict role-> number of players required
    temp: temperature for softmax (basso -> più "hard" selezione)
    Returns expected total vote (scalar differentiable)
    """
    device = logits.device
    total_expected = torch.tensor(0.0, device=device)
    for role, k in role_slots.items():
        idx = pos_idx.get(role)
        if idx is None or idx.numel() == 0:
            continue
        logits_role = logits[idx]  # m
        # softmax to get selection probabilities; to allow selecting k slots, we scale
        # Use softmax then multiply by k (expected #selected = k)
        probs = torch.softmax(logits_role / temp, dim=0)  # sum=1
        # scale to expected k picks: p_scaled = probs * k
        p_scaled = probs * k
        # expected vote from this role:
        expected_role_vote = torch.sum(p_scaled * y[idx])
        total_expected = total_expected + expected_role_vote
    return total_expected

# ---------------------------------------------------------
# Training loop over multiple giornate
def train_weights(df_features, df_votes, feat_cols, train_giornate,
                  role_slots=ROLE_SLOTS, n_epochs=200, lr=0.05, temp=0.1, weight_decay=1e-4):
    n_features = len(feat_cols)
    model = LinearScorer(n_features)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        count = 0
        for g in train_giornate:
            prepared = prepare_tensors_for_day(df_features, df_votes, g, feat_cols)
            if prepared is None:
                continue
            X, y, pos_idx, players = prepared
            logits = model(X)  # N
            expected_vote = expected_formation_vote(logits, y, pos_idx, role_slots, temp=temp)
            loss = - expected_vote  # want to maximize expected_vote
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
        if (epoch+1) % 20 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{n_epochs} avg loss {epoch_loss/ max(1,count):.4f}")
    return model

# ---------------------------------------------------------
# Example di uso:
# feat_cols = ["Titolarità","Forma","BonusPot","Affidabilità","Calendario","Penalità"]
# train_giornate = sorted(df_features["Giornata"].unique())[:20]  # esempio primo set
# model = train_weights(df_features, df_votes, feat_cols, train_giornate)
# ---------------------------------------------------------
