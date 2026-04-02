import logging
import os
from config.logger import get_logger
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


if __name__ == "__main__":
    run_pipeline()