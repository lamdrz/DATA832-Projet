import pandas as pd

class DataLoader:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir

    def load_co2(self):
        url_co2 = (f"{self.raw_data_dir}/owid-co2-data.csv")
        return pd.read_csv(url_co2)

    def load_energy(self):
        url_nrj = (f"{self.raw_data_dir}/owid-energy-data.csv")
        return pd.read_csv(url_nrj)

    def load_hdi(self):
        url_hdi = (f"{self.raw_data_dir}/HDR23-24_Composite_indices_complete_time_series.csv")
        return pd.read_csv(url_hdi)

    def load_gapminder(self):
        url_gap = (f"{self.raw_data_dir}/gapminder_unfiltered.csv")
        return pd.read_csv(url_gap)
    
    def save_df(self, df, file_name):
        output_path = f"{self.raw_data_dir}/../processed/{file_name}"
        df.to_csv(output_path, index=False)