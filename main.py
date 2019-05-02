import pandas as pd
import numpy as np
import collect_data

def load_data():
	df = pd.read_csv('./data/bigml_data.csv')
	df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	return df

def web_scrape():
	imdb = np.zeros((0, 2))
	wiki = np.zeros((0,))
	wild = np.zeros((0,))
	for y in list(range(2000, 2019)):
		imdb = np.concatenate([imdb, collect_data.imdb_feature_film(y)])
		wiki = np.concatenate([wiki, collect_data.wikipedia_in_film(y)])
		wild = np.concatenate([wild, collect_data.wildaboutmovies(y)])
	np.save('./saves/imdb', imdb)
	np.save('./saves/wiki', wiki)
	np.save('./saves/wild', wild)

def main():
	load_data()

if __name__ == '__main__':
	main()
