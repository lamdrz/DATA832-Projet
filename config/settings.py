DEBUG = False

TARGET_YEARS = tuple(range(2000, 2022))

COLUMNS_TO_KEEP = (
	'country', 'iso_code', 'year',

	# Pour l'ICA
	'co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2','coal_co2_per_capita', 
    'oil_co2_per_capita', 'gas_co2_per_capita','cement_co2_per_capita'

	# Pour la NMF (>= 0)
	'primary_energy_consumption_x', 'coal_consumption', 'gas_consumption',
	'oil_consumption', 'renewables_consumption', 'nuclear_consumption','coal_share_energy', 
    'gas_share_energy', 'oil_share_energy', 'nuclear_share_energy','renewables_share_energy',

	# Pour l'ACP et le clustering
	'hdi_value', 'hdicode', 'gdp_per_capita', 'population_x', 'energy_per_capita_x', 'cumulative_co2'
)

PCA_FEATURES = (
	'hdi_value',
    'gdp_per_capita',
    'co2_per_capita',
    'energy_per_capita_x',
    'population_x'
)

NMF_FEATURES = (
	'coal_share_energy',
    'gas_share_energy',
    'oil_share_energy',
    'nuclear_share_energy',
    'renewables_share_energy'
)

ICA_FEATURES = (
    'coal_co2_per_capita',
    'oil_co2_per_capita',
    'gas_co2_per_capita',
    'cement_co2_per_capita',
	'co2_per_capita'
)