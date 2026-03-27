import pandas as pd

class DataProcessing:        
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
        
        final_df = final_df.reset_index(drop=True)
        return final_df