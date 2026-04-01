import logging
import os

from config.logger import setup_logging
from src.data.loader import DataLoader

setup_logging(base_dir=os.path.dirname(__file__))

def run_pipeline():
    logging.info("Démarrage du pipeline")

    # Chargement des données
    loader = DataLoader(os.path.join(os.path.dirname(__file__), 'data/raw'))
    data = loader.load_and_process_all()
    loader.save_df(data, "merged_data.csv")

if __name__ == "__main__":
    run_pipeline()