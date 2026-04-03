import logging
import os
from config.logger import get_logger
from src.models.climateJustice import ClimateJusticeVisualizer
from src.models.clustering import ClusteringAnalyzer
from src.data.loader import DataLoader
from src.data.processor import DataProcessing
from src.models.reduction import DimensionalityReducer

logging = get_logger(__name__)

def run_pipeline():
    logging.info("Démarrage du pipeline")

    # Initialisation des classes
    loader = DataLoader(os.path.join(os.path.dirname(__file__), 'data/raw'))
    processor = DataProcessing()
    reducer = DimensionalityReducer()
    cluster_analyzer = ClusteringAnalyzer()
    
    # Chargement des données
    data = loader.load_and_process_all(processor)
    loader.save_df(data, "merged_data.csv")
    
    # Nettoyage des données
    cleaned_data = processor.clean_merged_data(data)
    loader.save_df(cleaned_data, "cleaned_data.csv")
    
    # Réduction de dimension
    ## ACP
    pca_model, X_pca, pca_features = reducer.run_pca(cleaned_data)
    if pca_model is not None:
        reducer.plot_explained_variance(pca_model)
        reducer.plot_loadings(pca_model, pca_features)
        
    ## NMF
    # reducer.find_optimal_k_nmf(cleaned_data, max_k=8) # Affiche le graphe pour choisir k
    k_choisi = 3 
    nmf_model, W, H, nmf_features = reducer.run_nmf(cleaned_data, k=k_choisi)
    if nmf_model is not None:
        reducer.plot_nmf_profiles(nmf_model, nmf_features)
        
    ## ICA
    ica_model, S_ica, ica_features = reducer.run_ica(cleaned_data, n_components=3)
    if ica_model is not None:
        reducer.plot_ica_components(ica_model, ica_features)

    #On prépare les données (On utilise les variables de l'ACP pour le clustering)
    X_scaled, y_hdi, y_gdp, df_cluster = cluster_analyzer.prepare_labels_and_data(cleaned_data)

    #On trace Coude + Silhouette pour choisir le bon nombre de clusters
    cluster_analyzer.find_optimal_kmeans(X_scaled, max_k=8)
    # Affiche les graphiques BIC/AIC pour GMM
    cluster_analyzer.find_optimal_gmm(X_scaled, max_k=8)

    # Affiche le K-distance plot pour DBSCAN
    cluster_analyzer.plot_k_distance(X_scaled, min_samples=5)

    K_OPTIMAL = 4         # Choisi d'après Coude / Silhouette / BIC / AIC
    EPS_OPTIMAL = 0.8     # Choisi d'après la cassure du K-distance plot
    MIN_SAMPLES = 5

    labels_km, labels_gmm, labels_dbs = cluster_analyzer.run_all_clusterings_and_evaluate(
        X_scaled, y_hdi, y_gdp, 
        k_chosen=K_OPTIMAL, 
        eps_chosen=EPS_OPTIMAL, 
        min_samples_chosen=MIN_SAMPLES
    )

    cluster_analyzer.evaluate_clustering(y_hdi, labels_km, model_name="K-Means", label_name="HDI")
    cluster_analyzer.evaluate_clustering(y_gdp, labels_km, model_name="K-Means", label_name="GDP Class")
    cluster_analyzer.evaluate_clustering(y_hdi, labels_gmm, model_name="GMM", label_name="HDI")
    cluster_analyzer.evaluate_clustering(y_gdp, labels_gmm, model_name="GMM", label_name="GDP Class")
    cluster_analyzer.evaluate_clustering(y_hdi, labels_dbs, model_name="DBSCAN", label_name="HDI")
    cluster_analyzer.evaluate_clustering(y_gdp, labels_dbs, model_name="DBSCAN", label_name="GDP Class")


    viz = ClimateJusticeVisualizer()

    #Créer l'indice, voir le Top 10 et la carte
    # On utilise df_cluster qui contient toutes nos colonnes brutes alignées
    df_with_index = viz.create_and_visualize_index(df_cluster)
    
    # 3. Générer le t-SNE et le comparer avec un de vos clusterings (par exemple K-Means)
    # labels_km correspond à la variable qui contient vos prédictions K-Means de l'étape précédente
    viz.plot_tsne_comparisons(X_scaled, df_with_index, labels_cluster=labels_km)

    viz.plot_tsne_comparisons(X_scaled, df_with_index, labels_cluster=labels_gmm)

    viz.plot_tsne_comparisons(X_scaled, df_with_index, labels_cluster=labels_dbs)


if __name__ == "__main__":
    run_pipeline()