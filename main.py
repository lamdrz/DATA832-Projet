import logging
import os

from config.logger import get_logger
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


if __name__ == "__main__":
    run_pipeline()