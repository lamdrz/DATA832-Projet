import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
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
        Calcule un indice COMPOSITE de justice climatique basé sur de multiples caractéristiques.
        """
        logging.info("Création de l'Indice Composite de Justice Climatique...")
        df_viz = df.copy()
        
        # 1. Sélection des variables pour l'indice
        resp_cols = ['co2_per_capita', 'cumulative_co2'] 
        cap_cols = ['hdi_value', 'gdp_per_capita']
        
        # Sécurité : on s'assure qu'elles existent
        all_cols = resp_cols + cap_cols
        for col in all_cols:
            if col not in df_viz.columns:
                logging.error(f"Colonne manquante pour l'indice : {col}")
                return df_viz
                
        # 2. Normalisation Min-Max (tout entre 0 et 1 pour pouvoir les additionner)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_viz[all_cols]), columns=all_cols, index=df_viz.index)
        
        # 3. Calcul des "Super-Scores"
        # Responsabilité : Moyenne des émissions (actuelles et historiques)
        df_viz['Score_Responsabilite'] = df_scaled[resp_cols].mean(axis=1)
        
        # Capacité : Moyenne de la richesse et du développement
        df_viz['Score_Capacite'] = df_scaled[cap_cols].mean(axis=1)
        
        # 4. INDICE FINAL : Le Décalage (Responsabilité - Capacité)
        df_viz['justice_index'] = df_viz['Score_Responsabilite'] - df_viz['Score_Capacite']
        
        # On fait la moyenne par pays pour la visualisation globale
        df_mean = df_viz.groupby(['country', 'iso_code'])[['justice_index', 'Score_Responsabilite', 'Score_Capacite']].mean().reset_index()
        df_mean = df_mean.sort_values(by='justice_index', ascending=False)
        
        # --- VISUALISATION 1 : CLASSEMENT (TOP 10 et FLOP 10) ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.barplot(data=df_mean.head(10), y='country', x='justice_index', palette='Reds_r', ax=ax[0])
        ax[0].set_title("Les 10 pires 'Débiteurs Climatiques'")
        ax[0].set_xlabel("Indice de Décalage (Responsabilité > Capacité)")
        ax[0].set_ylabel("")
        
        sns.barplot(data=df_mean.tail(10), y='country', x='justice_index', palette='Greens_r', ax=ax[1])
        ax[1].set_title("Les 10 plus 'Vulnérables / Sobres'")
        ax[1].set_xlabel("Indice de Décalage (Capacité > Responsabilité)")
        ax[1].set_ylabel("")
        
        plt.tight_layout()
        plt.show()

        # --- VISUALISATION 2 : CARTE MONDIALE INTERACTIVE ---
        logging.info("Génération de la carte mondiale...")
        
        # Pour Plotly, une échelle divergente (Bleu-Blanc-Rouge) est parfaite car notre indice va de -1 à +1
        val_max = abs(df_mean['justice_index']).max()
        
        fig_map = px.choropleth(
            df_mean,
            locations="iso_code",
            color="justice_index",
            hover_name="country",
            color_continuous_scale=px.colors.diverging.RdYlBu_r, # Rouge pour débiteurs, Bleu pour vulnérables
            range_color=[-val_max, val_max], # On centre le blanc sur 0
            color_continuous_midpoint=0,
            title="Carte de l'Indice Composite de Justice Climatique",
            labels={'justice_index': 'Indice (Rouge=Dette, Bleu=Vulnérabilité)'}
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