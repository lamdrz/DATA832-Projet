DEBUG = False

TARGET_YEARS = tuple(range(2010, 2020))

COLUMNS_TO_KEEP = (
	'country', 'iso_code', 'year',

	# Pour l'ICA
	'co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2',

	# Pour la NMF (>= 0)
	'primary_energy_consumption', 'coal_consumption', 'gas_consumption',
	'oil_consumption', 'renewables_consumption', 'nuclear_consumption',

	# Pour l'ACP et le clustering
	'hdi_value', 'hdicode', 'gdpPercap', 'lifeExp', 'pop'
)

PCA_FEATURES = (
	'hdi_value',
	'co2_per_capita',
	'coal_co2',
	'oil_co2',
	'gas_co2',
	'cement_co2',
	'coal_consumption',
	'gas_consumption',
	'oil_consumption',
	'renewables_consumption',
	'nuclear_consumption',
)

NMF_FEATURES = (
	'coal_consumption',
	'gas_consumption',
	'oil_consumption',
	'renewables_consumption',
	'nuclear_consumption',
)

ICA_FEATURES = (
	'coal_co2',
	'oil_co2',
	'gas_co2',
	'cement_co2',
	'co2_per_capita',
)