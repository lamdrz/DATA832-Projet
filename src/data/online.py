import pandas as pd

# 1) Our World in Data -- CO2 & GHG emissions
url_co2 = ("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")
co2 = pd.read_csv(url_co2)
# ~80 colonnes : country, year, iso_code, co2, co2_per_capita,
# coal_co2, oil_co2, gas_co2, cumulative_co2, ghg_per_capita, ...

# 2) Our World in Data -- Energy (colonnes positives : consommation)
url_nrj = ("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")
nrj = pd.read_csv(url_nrj)
# 130 colonnes : coal_consumption, gas_consumption,
# renewables_share_energy, fossil_share_energy, ...

# 3) UNDP HDI
url_hdi = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv")
hdi = pd.read_csv(url_hdi)
# Colonnes : iso3, country, hdicode (Low/Medium/High/Very High), ...

# 4) Gapminder
url_gap = ("https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv")
gap = pd.read_csv(url_gap)
# Colonnes : country, continent, year, lifeExp, pop, gdpPercap