import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS_TO_KEEP = (
	'country', 'iso_code', 'year',

	# Pour l'ICA
	'co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2',

	# Pour la NMF (>= 0)
	 'coal_consumption', 'gas_consumption',
	'oil_consumption', 'renewables_consumption', 'nuclear_consumption',

	# Pour l'ACP et le clustering
	'hdi_value', 'hdicode'
)

PCA_FEATURES = (
	'hdi_value',
	'co2_per_capita',
	'coal_co2',
	'oil_co2',
	'gas_co2',
	'cement_co2',
	'coal_consumption',
	'gas_consumption',
	'oil_consumption',
	'renewables_consumption',
	'nuclear_consumption',
)

NMF_FEATURES = (
	'coal_consumption',
	'gas_consumption',
	'oil_consumption',
	'renewables_consumption',
	'nuclear_consumption',
)

ICA_FEATURES = (
	'coal_co2',
	'oil_co2',
	'gas_co2',
	'cement_co2',
	'co2_per_capita',
)


def visualize_data_ACP_ICA_NMF(data):
    final_df = data.copy()

    intervalle = 5
    final_df['periode_debut'] = (final_df['year'] // intervalle) * intervalle
    final_df['label_periode'] = final_df['periode_debut'].astype(str) + "-" + \
                                (final_df['periode_debut'] + intervalle - 1).astype(str)


    
    # On regroupe tout et on s'assure que les colonnes existent bien dans le CSV pour éviter les erreurs
    cols_all = list(COLUMNS_TO_KEEP)
    cols_acp = list(PCA_FEATURES)
    cols_nmf = list(NMF_FEATURES)
    cols_ica = list(ICA_FEATURES)

    # --- 1. Calculs des statistiques ---
    
    # Taux de remplissage pour la Heatmap
    remplissage_detail = final_df.groupby('label_periode')[cols_all].apply(
        lambda x: x.notna().mean() * 100
    )

    # Moyennes annuelles
    stats_annuelles = final_df.groupby('year')[cols_all].mean()
    
    # Transformation en Base 100 pour ACP et ICA (1990 = 100)
    stats_acp_base100 = (stats_annuelles[cols_acp] / stats_annuelles[cols_acp].iloc[0]) * 100 
    stats_ica_base100 = (stats_annuelles[cols_ica] / stats_annuelles[cols_ica].iloc[0]) * 100 
    
    # Conservation des valeurs brutes pour la NMF (car ce sont déjà des parts en %)
    stats_nmf_brut = stats_annuelles[cols_nmf]

    # --- 2. Création des graphiques ---
    sns.set_theme(style="whitegrid")

    # ==========================================
    # GRAPHIQUE 1 : Heatmap globale des données cibles
    # ==========================================
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(remplissage_detail.T, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, 
                cbar_kws={'label': '% de remplissage'})
    plt.title("1. Qualité des données pour ACP, NMF et ICA (%)", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Variables")
    plt.tight_layout()

    # ==========================================
    # GRAPHIQUE 2 : Statistiques ACP
    # ==========================================
    plt.figure(figsize=(10, 4)) 
    sns.lineplot(data=stats_acp_base100, markers=True, dashes=False, linewidth=2)
    plt.title("2. Statistiques ACP : Évolution (Base 100 en 1990)", fontsize=14, fontweight='bold')
    plt.xlabel("Année")
    plt.ylabel("Indice (1990 = 100)")
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Niveau de 1990')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # ==========================================
    # GRAPHIQUE 3 : Statistiques NMF
    # ==========================================
    plt.figure(figsize=(10, 4)) 
    sns.lineplot(data=stats_nmf_brut, markers=True, dashes=False, linewidth=2)
    plt.title("3. Statistiques NMF : Évolution du mix énergétique mondial (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Année")
    plt.ylabel("Part dans le mix énergétique (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # ==========================================
    # GRAPHIQUE 4 : Statistiques ICA
    # ==========================================
    plt.figure(figsize=(10, 4)) 
    sns.lineplot(data=stats_ica_base100, markers=True, dashes=False, linewidth=2)
    plt.title("4. Statistiques ICA : Sources de CO2 par habitant (Base 100 en 1990)", fontsize=14, fontweight='bold')
    plt.xlabel("Année")
    plt.ylabel("Indice (1990 = 100)")
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Niveau de 1990')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Affichage simultané des 4 fenêtres
    plt.show()

def visualize_data_dif_merged_cleaned(data_merged, data_cleaned):
    
    final_df = data_merged.copy()
    cleaned_df = data_cleaned.copy()
    
    # 2. Identification des colonnes communes numériques à comparer
    colonnes_communes = list(COLUMNS_TO_KEEP)

    # ==========================================
    # ANALYSE 1 : Avant/Après des valeurs manquantes
    # ==========================================
    
    # Calcul du % de NaN pour le jeu Brut
    nan_brut = final_df[colonnes_communes].isna().mean() * 100
    # Calcul du % de NaN pour le jeu Nettoyé (devrait être à 0 ou presque)
    nan_clean = cleaned_df[colonnes_communes].isna().mean() * 100
    
    # Création d'un DataFrame comparatif
    df_nan_compare = pd.DataFrame({
        'Variable': colonnes_communes,
        'Avant imputation (Brut)': nan_brut.values,
        'Après imputation KNN (Nettoyé)': nan_clean.values
    })
    
    # Passage au format "long" pour Seaborn
    df_nan_melted = df_nan_compare.melt(id_vars='Variable', var_name='Dataset', value_name='% de valeurs manquantes')

    # ==========================================
    # ANALYSE 2 : Impact sur les distributions (Moyennes)
    # ==========================================
    
    # On calcule la moyenne annuelle pour quelques variables clés
    variables_cles = ['hdi_value', 'co2_per_capita', 'coal_consumption']
    
    moyennes_brut = final_df.groupby('year')[variables_cles].mean().reset_index()
    moyennes_brut['Dataset'] = 'Brut (avec NaN)'
    
    moyennes_clean = cleaned_df.groupby('year')[variables_cles].mean().reset_index()
    moyennes_clean['Dataset'] = 'Nettoyé (Imputé KNN)'
    
    # Fusion pour l'affichage
    df_moyennes = pd.concat([moyennes_brut, moyennes_clean])

    # ==========================================
    # CRÉATION DES GRAPHIQUES
    # ==========================================
    sns.set_theme(style="whitegrid")

    # GRAPHIQUE 1 : Barplot des valeurs manquantes
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_nan_melted, x='Variable', y='% de valeurs manquantes', hue='Dataset', palette=['#e74c3c', '#2ecc71'])
    plt.title("1. Efficacité de l'imputation KNN : % de données manquantes (Avant / Après)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("% de NaN")
    plt.tight_layout()

    # GRAPHIQUES 2, 3, 4 : Comparaison des moyennes temporelles
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("2. Impact de l'imputation KNN sur les moyennes globales temporelles", fontsize=16, fontweight='bold', y=1.05)

    for i, var in enumerate(variables_cles):
        sns.lineplot(data=df_moyennes, x='year', y=var, hue='Dataset', style='Dataset', 
                     markers=True, dashes=False, ax=axes[i], palette=['#e74c3c', '#2ecc71'])
        axes[i].set_title(f"Évolution de : {var}", fontsize=12)
        axes[i].set_xlabel("Année")
        axes[i].set_ylabel("Moyenne")

    plt.tight_layout()
    
    # Affichage
    plt.show()
        
    

if __name__ == "__main__":

    merged_data  = pd.read_csv("data/processed/merged_data.csv", encoding='latin-1')
    cleaned_data = pd.read_csv("data/processed/cleaned_data.csv", encoding='latin-1')
    #visualize_data_ACP_ICA_NMF(merged_data)
    visualize_data_dif_merged_cleaned(merged_data, cleaned_data)