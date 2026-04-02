import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config.logger import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

logging = get_logger(__name__)


class ClimateJusticeVisualizer:
    def __init__(self):
        pass

    def create_and_visualize_index(self, df):
        """
        Calcule l'Indice de Justice Climatique par le ratio brut (CO2 par point de HDI).
        """
        logging.info("Création de l'Indice de Justice Climatique (Ratio brut)...")
        df_viz = df.copy()
        
        # 1. Calcul du Ratio : Tonnes de CO2 émises pour 1 point de HDI
        df_viz['justice_index'] = df_viz['co2_per_capita'] / df_viz['hdi_value']
        
        # On fait la moyenne par pays sur toute la période
        df_mean = df_viz.groupby(['country', 'iso_code'])[['justice_index', 'co2_per_capita', 'hdi_value']].mean().reset_index()
        df_mean = df_mean.sort_values(by='justice_index', ascending=False)
        
        # --- VISUALISATION 1 : CLASSEMENT (TOP 10 et FLOP 10) ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Les pires débiteurs (Ceux qui polluent le plus proportionnellement à leur HDI)
        sns.barplot(data=df_mean.head(10), y='country', x='justice_index', palette='Reds_r', ax=ax[0])
        ax[0].set_title("Les 10 'Débiteurs Climatiques'")
        ax[0].set_xlabel("Tonnes de CO2 par point de HDI")
        ax[0].set_ylabel("")
        
        # Les plus vulnérables / sobres
        sns.barplot(data=df_mean.tail(10), y='country', x='justice_index', palette='Greens', ax=ax[1])
        ax[1].set_title("Les 10 plus 'Vulnérables / Sobres'")
        ax[1].set_xlabel("Tonnes de CO2 par point de HDI")
        ax[1].set_ylabel("")
        
        plt.tight_layout()
        plt.show()

        # --- VISUALISATION 2 : CARTE MONDIALE INTERACTIVE ---
        logging.info("Génération de la carte mondiale...")
        
        # Astuce : On coupe l'échelle des couleurs au 95ème centile pour que les "super-pollueurs" 
        # (comme le Qatar ou Trinité-et-Tobago) ne rendent pas le reste du monde tout blanc.
        val_max_color = df_mean['justice_index'].quantile(0.95)
        
        fig_map = px.choropleth(
            df_mean,
            locations="iso_code",
            color="justice_index",
            hover_name="country",
            color_continuous_scale="YlOrRd", # Jaune (Faible) vers Rouge foncé (Fort)
            range_color=[0, val_max_color],  
            title="Carte du Décalage Climatique (CO2 par point de HDI)",
            labels={'justice_index': 'Score de Décalage'}
        )
        fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
        fig_map.show()
        
        return df_viz

    def plot_tsne_comparisons(self, X_scaled, df_viz, labels_cluster):
        """
        Projette les données en 2D avec t-SNE et compare les colorations.
        """
        logging.info("Calcul du t-SNE en 2D (cela peut prendre quelques secondes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        df_viz['tsne_1'] = X_tsne[:, 0]
        df_viz['tsne_2'] = X_tsne[:, 1]
        df_viz['Cluster'] = labels_cluster
        
        # Récupération rapide des continents via Plotly pour la coloration géo
        gap_continents = px.data.gapminder()[['iso_alpha', 'continent']].drop_duplicates()
        df_viz = df_viz.merge(gap_continents, left_on='iso_code', right_on='iso_alpha', how='left')
        df_viz['continent'] = df_viz['continent'].fillna('Autre') # Pour les pays non trouvés
        
        # --- VISUALISATION 3 : COMPARAISON t-SNE ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Comparaison de l'espace latent t-SNE en 2D", fontsize=16, fontweight='bold', y=1.05)
        
        # 1. Coloré par Région (Continent)
        sns.scatterplot(data=df_viz, x='tsne_1', y='tsne_2', hue='continent', palette='Set1', s=30, alpha=0.7, ax=axes[0])
        axes[0].set_title("1. Par Région Géographique")
        axes[0].legend(title="Continent", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Coloré par Cluster (ex: K-Means)
        sns.scatterplot(data=df_viz, x='tsne_1', y='tsne_2', hue='Cluster', palette='tab10', s=30, alpha=0.7, ax=axes[1])
        axes[1].set_title("2. Par Cluster (Algorithme Non-Supervisé)")
        axes[1].legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Coloré par Indice de Justice Climatique
        scatter = axes[2].scatter(df_viz['tsne_1'], df_viz['tsne_2'], c=df_viz['justice_index'], cmap='coolwarm', s=30, alpha=0.7)
        axes[2].set_title("3. Par Indice de Justice Climatique")
        fig.colorbar(scatter, ax=axes[2], label='Indice (CO2 / HDI)')
        
        # Nettoyage des axes pour faire joli (les valeurs de l'axe t-SNE n'ont pas de sens absolu)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            
        plt.tight_layout()
        plt.show()