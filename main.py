import pandas as pd
import numpy as np
import collect_data
import os
from sklearn.model_selection import train_test_split
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
np.random.seed(1)


def extract_movie_data():
	if os.path.exists('./data/imdb.csv'):
		imdb = pd.read_csv('./data/imdb.csv')
	else:
		imdb = collect_data.imdb_feature_film(2000)
		for y in list(range(2001, 2019)):
			imdb = imdb.append(collect_data.imdb_feature_film(y))

		# Removes duplicate movies
		df = pd.read_csv('./data/bigml.csv')
		temp = []
		for index, row in imdb.iterrows():
			if row['movie'] not in list(df['movie']):
				temp.append([row['year'], row['movie'], row['movie_id']])
		imdb = pd.DataFrame(temp, columns=['year', 'movie', 'movie_id'])

		tags = []
		for index, row in imdb.iterrows():
			print(str(index) + '. ' + row['movie'])
			id = row['movie_id']
			extra = collect_data.collect_movie_info(id)

			if extra is not None:
				tags.append([row['year'], row['movie'], row['movie_id']] + extra)

		imdb = pd.DataFrame(tags, columns=['year', 'movie', 'movie_id', 'certificate', 'duration', 'genre', 'rate',
										   'metascore', 'synopsis', 'votes', 'gross', 'user_reviews', 'critic_reviews',
										   'popularity', 'awards_wins', 'awards_nominations'])
		imdb.to_csv('./data/imdb.csv')

	return imdb


def edit_bigml():
	bigml = pd.read_csv('./data/bigml.csv')
	bigml = bigml.fillna(value={'gross': -1, 'popularity': -1})
	bigml.to_csv('./data/bigml.csv')


def build_model(type):
	if type == 'svm':
		model = svm.SVC(kernel='linear')
	elif type == 'mlp':
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))


def main():
	bigml = pd.read_csv('./data/bigml.csv')
	bigml.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	imdb = extract_movie_data()
	del imdb['Unnamed: 0']



if __name__ == '__main__':
	main()
