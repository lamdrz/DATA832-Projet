import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fusionner_datasets(co2, nrj, hdi_melted):
# CO2 + Energie
    df_merged = pd.merge(
        co2, 
        nrj.drop(columns=['country']),
        on=['iso_code', 'year'], 
        how='inner'
    )

    # (CO2 + Energie) + HDI
    final_df = pd.merge(
        df_merged,
        hdi_melted.drop(columns=['country']),
        on=['iso_code', 'year'],
        how='inner'
    )

    final_df = final_df[final_df['year'] >= 1990]

    # Nettoyer les agrégats régionaux/mondiaux en ne gardant que les vrais pays (code ISO de 3 lettres)
    final_df = final_df[final_df['iso_code'].notna()]
    final_df = final_df[final_df['iso_code'].str.len() == 3]

    # Réinitialisation de l'index pour un dataframe propre
    final_df = final_df.reset_index(drop=True)

    # 6. Vérification du résultat
    print("\n--- Pipeline de fusion terminé ---")
    print(f"Dimensions du dataset final : {final_df.shape}")
    print(f"Période couverte : {final_df['year'].min()} - {final_df['year'].max()}")

    return final_df

def normalize_hdi(hdi):

    hdi_cols = [col for col in hdi.columns if col.startswith('hdi_') and col[4:].isdigit()]

    hdi_melted = hdi.melt(
        id_vars=['iso3', 'country', 'hdicode'], 
        value_vars=hdi_cols, 
        var_name='year', 
        value_name='hdi_value'
    )

    # Extraction de l'année au format entier et alignement du nom de la colonne ISO
    hdi_melted['year'] = hdi_melted['year'].str.replace('hdi_', '').astype(int)
    hdi_melted = hdi_melted.rename(columns={'iso3': 'iso_code'})

    return hdi_melted

if __name__ == "__main__":

    path_co2 = "data/raw/owid-co2-data.csv"
    path_nrj = "data/raw/owid-energy-data.csv"
    path_hdi = "data/raw/HDR23-24_Composite_indices_complete_time_series.csv"

    print("Chargement des données en cours (cela peut prendre quelques instants)...")
    co2 = pd.read_csv(path_co2, encoding='latin-1')
    nrj = pd.read_csv(path_nrj, encoding='latin-1')
    hdi = pd.read_csv(path_hdi, encoding='latin-1')

    hdi_melted = normalize_hdi(hdi)
    final_df = fusionner_datasets(co2, nrj, hdi_melted)



    # 1. Définir l'intervalle souhaité (5 ou 10 ans)
    intervalle = 5

    # 2. Créer une colonne pour identifier la période (ex: 1990, 1995, 2000...)
    # Astuce mathématique : la division entière // permet d'arrondir à l'intervalle inférieur
    final_df['periode_debut'] = (final_df['year'] // intervalle) * intervalle

    # Créer un label lisible pour le graphique (ex: "1990-1994")
    final_df['label_periode'] = final_df['periode_debut'].astype(str) + "-" + \
                                (final_df['periode_debut'] + intervalle - 1).astype(str)

    # 3. Calculer le pourcentage de données non nulles
    # .notna() transforme le tableau en True (1) / False (0)
    # Le premier .mean() fait la moyenne par colonne, le second fait la moyenne globale
    remplissage_global = final_df.groupby('label_periode').apply(
        lambda df_periode: df_periode.notna().mean().mean() * 100
    ).reset_index(name='taux_remplissage_pct')

    # 4. Tracer le graphique
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(
        data=remplissage_global, 
        x='label_periode', 
        y='taux_remplissage_pct', 
        palette='Blues'
    )

    # Ajouter un titre et formater les axes
    plt.title(f"Taux global de remplissage des données par tranches de {intervalle} ans", fontsize=14)
    plt.xlabel("Période", fontsize=12)
    plt.ylabel("Données non nulles (%)", fontsize=12)
    plt.ylim(0, 105) # Marge au-dessus de 100% pour l'affichage du texte

    # Ajouter le pourcentage exact au-dessus de chaque barre
    for p in barplot.patches:
        hauteur = p.get_height()
        plt.text(p.get_x() + p.get_width() / 2., hauteur + 1.5,
                 f'{hauteur:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()