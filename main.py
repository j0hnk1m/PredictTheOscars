import pandas as pd
import numpy as np
import collect_data
import os
from sklearn.model_selection import train_test_split
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
np.random.seed(1)


def retrieve_movie_list():
	if not os.path.exists('./data/imdb.csv'):
		imdb = collect_data.imdb_feature_film(2000)
		for y in list(range(2001, 2019)):
			imdb = imdb.append(collect_data.imdb_feature_film(y))

		# Removes duplicate movies
		df = pd.read_csv('./data/winners_nominees.csv')
		temp = []
		for index, row in imdb.iterrows():
			if row['movie'] not in list(df['movie']):
				temp.append([row['year'], row['movie'], row['movie_id']])
		imdb = pd.DataFrame(temp, columns=['year', 'movie', 'movie_id'])

		imdb.to_csv('./data/imdb.csv')
	else:
		imdb = pd.read_csv('./data/imdb.csv')

	return imdb


def build_model(type):
	if type == 'svm':
		model = svm.SVC(kernel='linear')
	elif type == 'mlp':
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))


def main():
	df = pd.read_csv('./data/winners_nominees.csv')
	df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	imdb = retrieve_movie_list()

	for index, row in imdb.iterrows():
		movie_id = row['movie_id']


if __name__ == '__main__':
	main()
