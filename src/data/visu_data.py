import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    final_df = pd.read_csv("data/processed/merged_data.csv", encoding='latin-1')

    intervalle = 5
    final_df['periode_debut'] = (final_df['year'] // intervalle) * intervalle
    final_df['label_periode'] = final_df['periode_debut'].astype(str) + "-" + \
                                (final_df['periode_debut'] + intervalle - 1).astype(str)

    colonnes_cles = ['co2', 'coal_consumption', 'hdi_value', 'gdp_x', 'population_x']

    
    # Graphique 1 : Taux global
    remplissage_global = final_df.groupby('label_periode').apply(
        lambda df_p: df_p.notna().mean().mean() * 100
    ).reset_index(name='taux_remplissage_pct')

    # Graphique 2 : Taux par colonne cible (Format croisé pour Heatmap)
    remplissage_detail = final_df.groupby('label_periode')[colonnes_cles].apply(
        lambda x: x.notna().mean() * 100
    )

    # Graphique 3 : Statistiques (Moyennes annuelles en Base 100)
    stats_annuelles = final_df.groupby('year')[colonnes_cles].mean()
    # On divise chaque ligne par la première année (1990) et on multiplie par 100
    stats_base100 = (stats_annuelles / stats_annuelles.iloc[0]) * 100 

    # --- 3. Création des graphiques séparés ---
    sns.set_theme(style="whitegrid")

    # ==========================================
    # GRAPHIQUE 1 : Taux global (Barplot)
    # ==========================================
    plt.figure(figsize=(10, 5)) 
    barplot = sns.barplot(data=remplissage_global, x='label_periode', y='taux_remplissage_pct', palette='Blues')
    plt.title(f"1. Taux global de remplissage par tranches de {intervalle} ans", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Données non nulles (%)")
    plt.ylim(0, 110)
    
    for p in barplot.patches:
        plt.text(p.get_x() + p.get_width() / 2., p.get_height() + 1.5,
                 f'{p.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()


    # ==========================================
    # GRAPHIQUE 2 : Détail par colonne (Heatmap)
    # ==========================================
    plt.figure(figsize=(10, 4)) 
    sns.heatmap(remplissage_detail.T, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100, 
                cbar_kws={'label': '% de remplissage'})
    plt.title("2. Détail du remplissage pour les variables clés (%)", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Variables")
    plt.tight_layout()


    # ==========================================
    # GRAPHIQUE 3 : Évolution des moyennes
    # ==========================================
    plt.figure(figsize=(10, 5)) 
    sns.lineplot(data=stats_base100, markers=True, dashes=False, linewidth=2.5)
    plt.title("3. Évolution des moyennes mondiales (Base 100 en 1990)", fontsize=14, fontweight='bold')
    plt.xlabel("Année", fontsize=12)
    plt.ylabel("Indice (1990 = 100)", fontsize=12)
    plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Niveau de 1990')
    plt.legend(title="Indicateurs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.show()