import pandas as pd

class DataLoader:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        
        self.co2 = None
        self.nrj = None
        self.hdi = None
        self.gap = None

    def load_co2(self):
        if self.co2 is not None:
            return self.co2
        url_co2 = (f"{self.raw_data_dir}/owid-co2-data.csv")
        self.co2 = pd.read_csv(url_co2, encoding='latin-1')
        return self.co2

    def load_energy(self):
        if self.nrj is not None:
            return self.nrj
        url_nrj = (f"{self.raw_data_dir}/owid-energy-data.csv")
        self.nrj = pd.read_csv(url_nrj, encoding='latin-1')
        return self.nrj

    def load_hdi(self):
        if self.hdi is not None:
            return self.hdi
        url_hdi = (f"{self.raw_data_dir}/HDR23-24_Composite_indices_complete_time_series.csv")
        self.hdi = pd.read_csv(url_hdi, encoding='latin-1')
        return self.hdi

    def load_gapminder(self):
        if self.gap is not None:
            return self.gap
        url_gap = (f"{self.raw_data_dir}/gapminder_unfiltered.csv")
        self.gap = pd.read_csv(url_gap, encoding='latin-1')
        return self.gap
    
    def load_all(self):
        return self.load_co2(), self.load_energy(), self.load_hdi(), self.load_gapminder()
    
    def load_and_process_all(self, processor):
        co2, nrj, hdi, _ = self.load_all()
        hdi_normalized = processor.normalize_hdi(hdi)
        merged_data = processor.merge_datasets(co2, nrj, hdi_normalized)
        return merged_data
    
    def save_df(self, df, file_name):
        output_path = f"{self.raw_data_dir}/../processed/{file_name}"
        df.to_csv(output_path, index=False)
        
    def load_df(self, file_name):
        input_path = f"{self.raw_data_dir}/../processed/{file_name}"
        return pd.read_csv(input_path)
        