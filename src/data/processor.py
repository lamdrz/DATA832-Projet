import pandas as pd
from sklearn.impute import KNNImputer

from config.logger import get_logger
from config.settings import TARGET_YEARS, COLUMNS_TO_KEEP

logging = get_logger(__name__)

class DataProcessing:      
    def __init__(self):
        self.target_years = list(TARGET_YEARS)
        self.columns_to_keep = list(COLUMNS_TO_KEEP)

    def _collapse_duplicate_columns(self, df):
        """
        Fusionne les colonnes dupliquées issues des merge pandas (_x/_y).
        - Si les 2 colonnes sont numériques: moyenne ligne par ligne.
        - Sinon: priorité à la valeur non nulle de _x, puis _y.
        """
        for col in list(df.columns):
            if not col.endswith('_x'):
                continue

            base_col = col[:-2]
            col_x = f"{base_col}_x"
            col_y = f"{base_col}_y"

            if col_y not in df.columns:
                continue

            is_numeric_pair = (
                pd.api.types.is_numeric_dtype(df[col_x])
                and pd.api.types.is_numeric_dtype(df[col_y])
            )

            if is_numeric_pair:
                df[base_col] = df[[col_x, col_y]].mean(axis=1, skipna=True)
            else:
                df[base_col] = df[col_x].combine_first(df[col_y])

            df = df.drop(columns=[col_x, col_y])

        # Renomme les colonnes restantes suffixées si elles n'ont pas de paire.
        df = df.rename(columns=lambda c: c[:-2] if c.endswith('_x') else c)
        df = df.rename(columns=lambda c: c[:-2] if c.endswith('_y') else c)
        return df
      
    def keep_only_countries(self, df, iso_col="iso_code"):
        df = df[df[iso_col].notna()]
        return df[df[iso_col].str.len() == 3]
    
    def normalize_hdi(self, hdi):
        hdi_cols = [col for col in hdi.columns if col.startswith('hdi_') and col[4:].isdigit()]

        hdi_melted = hdi.melt(
            id_vars=['iso3', 'country', 'hdicode'], 
            value_vars=hdi_cols,
            var_name='year',
            value_name='hdi_value'
        )

        # Extraction de l'année au format entier et alignement du nom de la colonne ISO
        hdi_melted['year'] = hdi_melted['year'].str.replace('hdi_', '').astype(int)
        hdi_melted = hdi_melted.rename(columns={'iso3': 'iso_code'})

        return hdi_melted
    
    def merge_datasets(self, co2, nrj, hdi):
        # CO2 + Energie
        df_merged = pd.merge(
            co2, 
            nrj.drop(columns=['country']),
            on=['iso_code', 'year'], 
            how='inner'
        )

        # (CO2 + Energie) + HDI
        final_df = pd.merge(
            df_merged,
            hdi.drop(columns=['country']),
            on=['iso_code', 'year'],
            how='inner'
        )

        final_df = self._collapse_duplicate_columns(final_df)
        
        final_df = final_df.reset_index(drop=True)
        return final_df
    
    def clean_merged_data(self, df):
        """
        Nettoie le dataset fusionné pour le rendre compatible avec scikit-learn.
        """
        logging.info(f"Shape : {df.shape}")
        
        # On garde que la fenetre qui nous intéresse
        df_clean = df[df['year'].isin(self.target_years)].copy()
        
        # On garde que les colonnes qui nous intéressent
        cols_present = [c for c in self.columns_to_keep if c in df_clean.columns]
        df_clean = df_clean[cols_present]
        
        # hdicode est indispensable : c'est le pseudo-label pour le clustering (Low, Medium, High, Very High)
        if 'hdicode' in df_clean.columns: df_clean = df_clean.dropna(subset=['hdicode', 'hdi_value'])
            
        logging.info(f"Shape clean : {df_clean.shape}")
        
        # Imputation des NaN restants
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns # On ne peut pas imputer du texte
        if 'year' in numeric_cols: numeric_cols = numeric_cols.drop('year')
            
        # Utilisation de KNNImputer :
        # Au lieu de mettre bêtement la moyenne mondiale, il regarde les 5 "voisins" 
        # (les pays qui ont des caractéristiques similaires) pour deviner la valeur manquante.
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
        # Vérification finale
        remaining_nans = df_clean.isna().sum().sum()
        logging.info(f"Shape final : {df_clean.shape}")
        logging.info(f"NaN restants : {remaining_nans}")
        
        if remaining_nans > 0:
            logging.warning("Il reste des NaN")
            
        return df_clean