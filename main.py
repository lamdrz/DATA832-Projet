import logging
import os

from config.logger import get_logger
from src.data.loader import DataLoader
from src.data.processor import DataProcessing

logging = get_logger(__name__)

def run_pipeline():
    logging.info("Démarrage du pipeline")

    # Initialisation des classes
    loader = DataLoader(os.path.join(os.path.dirname(__file__), 'data/raw'))
    processor = DataProcessing()
    
    # Chargement des données
    data = loader.load_and_process_all(processor)
    loader.save_df(data, "merged_data.csv")
    
    # Nettoyage des données
    cleaned_data = processor.clean_merged_data(data)
    loader.save_df(cleaned_data, "cleaned_data.csv")


if __name__ == "__main__":
    run_pipeline()