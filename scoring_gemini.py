
import pandas as pd
import numpy as np

def scoring_gca(df_giocatori):

    # Assunzione: df_giocatori è il tuo DataFrame delle Statistiche Avanzate Giocatori
    df = df_giocatori.copy()
    min_90s = 5  # Minimo 5 partite complete per essere incluso

    # 1. Pulizia e Conversione Tipi
    # Converte le colonne in numerico, forzando errori a NaN (cruciale per '90s')
    for col in ['Gls', 'xG', 'npxG', 'xAG', '90s']:
        # Sostituisci i valori non validi/stringhe con NaN, poi converti in float
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rimuovi eventuali righe con dati mancanti dopo la conversione
    df = df.dropna(subset=['90s', 'Gls', 'npxG', 'xAG'])

    # 2. Filtro e Calcolo delle Metriche di Base
    df = df[df['Pos'] != 'GK']
    df = df[df['90s'] >= min_90s]

    # Calcolo: Contributo offensivo per 90 minuti (più oggettivo)
    df['Offensive_Score'] = (df['npxG'] + df['xAG']) / df['90s'] 

    # Calcolo: Efficienza (Gol Segnati - Gol Attesi)
    df['Finishing_Efficiency'] = df['Gls'] - df['xG']

    # 3. Normalizzazione (Min-Max Scaling)
    # Normalizza i punteggi su una scala 0-1
    df['Offensive_Score_Norm'] = (df['Offensive_Score'] - df['Offensive_Score'].min()) / \
                                (df['Offensive_Score'].max() - df['Offensive_Score'].min())

    # 4. Punteggio Finale (Assegna Peso)
    # Peso 70% al Contributo Offensivo Normalizzato, 30% all'Efficienza (non normalizzata, ma più intuitiva)
    df['Total_Player_Score'] = (df['Offensive_Score_Norm'] * 0.7) + (df['Finishing_Efficiency'] * 0.3) 

    # Ordina per i migliori
    migliori_giocatori = df.sort_values(by='Total_Player_Score', ascending=False)


def scoring_por(df_giocatori):
    # Assunzione: df_portieri è il tuo DataFrame delle Statistiche Portieri Avanzate
    df_gk = df_portieri.copy()
    min_90s_gk = 10 

    # 1. Pulizia e Conversione Tipi
    # Converte le colonne in numerico, forzando errori a NaN
    for col in ['PSxG', 'GA', 'Save%', '90s']:
        # Sostituisci i valori non validi/stringhe con NaN, poi converti in float
        df_gk[col] = pd.to_numeric(df_gk[col], errors='coerce')
        
    # Rimuovi eventuali righe con dati mancanti dopo la conversione
    df_gk = df_gk.dropna(subset=['90s', 'PSxG', 'GA', 'Save%'])

    # 2. Filtro e Calcolo delle Metriche di Base
    df_gk = df_gk[df_gk['Pos'] == 'GK']
    df_gk = df_gk[df_gk['90s'] >= min_90s_gk]

    # Calcolo: Prestazione Netta (Gol salvati in più o in meno rispetto all'attesa)
    df_gk['Net_Save_Value'] = df_gk['PSxG'] - df_gk['GA'] 

    # 3. Normalizzazione
    # Normalizza Save% (che è in %) per renderla compatibile (da 0-100 a 0-1)
    df_gk['Save%_Norm'] = df_gk['Save%'] / 100 

    # 4. Punteggio Finale (Combinazione)
    # Punteggio: Net_Save_Value (il valore più importante) + un bonus dalla Save% normalizzata. 
    # Si moltiplica Save%_Norm per 2 per dare un peso ragionevole al parametro %parate.
    df_gk['Total_GK_Score'] = df_gk['Net_Save_Value'] + (df_gk['Save%_Norm'] * 2) 

    # Ordina per i migliori
    migliori_portieri = df_gk.sort_values(by='Total_GK_Score', ascending=False)

from scraping_fbref_players_gemini import giocatori

print(scoring_gca(giocatori()))