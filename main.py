import logging
import os
from datetime import datetime

from config.settings import DEBUG
from data.loader import DataLoader

logs_folder_path = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(logs_folder_path):
    os.makedirs(logs_folder_path)
    
log_file_path = os.path.join(os.path.dirname(__file__), f"logs/{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    handlers=[
        logging.FileHandler(log_file_path, 'a', 'utf-8'),
        logging.StreamHandler() # Affiche aussi dans la console
    ]
)

def run_pipeline():
    logging.info("Démarrage du pipeline")

    # Initialisation
    loader = DataLoader(os.path.join(os.path.dirname(__file__), 'data/raw'))

if __name__ == "__main__":
    run_pipeline()