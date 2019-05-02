import pandas as pd

def load_data():
	df = pd.read_csv('bigml_data.csv')
	df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	