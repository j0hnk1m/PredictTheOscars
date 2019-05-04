import pandas as pd
import numpy as np
import collect_data
import os
import math
from sklearn.model_selection import train_test_split
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)


def retrieve_movie_list():
	if not os.path.exists('./data/imdb.csv') or not os.path.exists('./data/wiki.csv'):
		imdb = collect_data.imdb_feature_film(2000)
		wiki = collect_data.wikipedia_in_film(2000)
		for y in list(range(2001, 2019)):
			imdb = imdb.append(collect_data.imdb_feature_film(y))
			wiki = wiki.append(collect_data.wikipedia_in_film(y))

		# Removes duplicate movies
		temp = []
		for index, row in wiki.iterrows():
			if row['movie'] not in list(imdb['movie']):
				temp.append([row['movie'], row['year']])
		wiki = pd.DataFrame(temp, columns=['movie', 'year'])

		imdb.to_csv('./data/imdb.csv')
		wiki.to_csv('./data/wiki.csv')
	else:
		imdb = pd.read_csv('./data/imdb.csv')
		wiki = pd.read_csv('./data/wiki.csv')

	combined = pd.concat([imdb, wiki],ignore_index=True, sort=True)
	combined.to_csv('./data/combined.csv')
	return combined


def build_model(type):
	if type == 'svm':
		model = svm.SVC(kernel='linear')
	elif type == 'mlp':
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))


def main():
	df = pd.read_csv('./data/bigml_data.csv')
	df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	combined = retrieve_movie_list()

	# # See what movies from the oscar winners/nominees bigml data are missing in the web-scraped data
	# missing=[]
	# for m in list(df['movie']):
	# 	if m not in list(combined['movie']):
	# 		missing.append(m)

	for index, row in combined.iterrows():
		imdb_id = row['imdb_id']
		if type(imdb_id) is str:

		else:
			print("no imdb")

if __name__ == '__main__':
	main()
