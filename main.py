import pandas as pd
import numpy as np
import collect_data
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm

from keras.models import Sequential
from keras.layers import Dense
np.random.seed(1)


def extract_movie_data():
	if os.path.exists('./data/imdb.csv'):
		imdb = pd.read_csv('./data/imdb.csv', index_col=0)
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
			extra = collect_data.movie_tags(id)

			if extra is not None:
				tags.append([row['year'], row['movie'], row['movie_id']] + extra)

		imdb = pd.DataFrame(tags, columns=['year', 'movie', 'movie_id', 'certificate', 'duration', 'genre', 'rate',
										   'metascore', 'synopsis', 'votes', 'gross', 'user_reviews', 'critic_reviews',
										   'popularity', 'awards_wins', 'awards_nominations'], index=False)
		imdb.to_csv('./data/imdb.csv')

	return imdb


def edit_bigml():
	bigml = pd.read_csv('./data/bigml.csv', index_col=0)
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
	# bigml = pd.read_csv('./data/bigml.csv', index_col=0)
	# imdb = extract_movie_data()
	# df = bigml.append(imdb, sort=False, ignore_index=True)
	# df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	# df = df.reset_index(drop=True)
	# df.to_csv('./data/combined.csv')
	df = pd.read_csv('./data/combined.csv', index_col=0)

	categories = []
	awards = []
	for y in range(2000, 2019):
		print(y)
		results = collect_data.movie_awards(y)
		categories.append(results[0])
		awards.append(results[1])

	with open('categories', 'wb') as f:
		pickle.dump(categories, f)
	with open('awards', 'wb') as f:
		pickle.dump(awards, f)

	with open('categories', 'rb') as f:
		categories = pickle.load(f)
	with open('awards', 'rb') as f:
		awards = pickle.load(f)

	ggs = [i[0] for i in categories]
	baftas = [i[1] for i in categories]
	sags = [i[2] for i in categories]
	dgs = [i[3] for i in categories]
	pgs = [i[4] for i in categories]
	adgs = [i[5] for i in categories]
	wgs = [i[6] for i in categories]
	cdgs = [i[7] for i in categories]
	oftas = [i[8] for i in categories]
	ofcss = [i[9] for i in categories]
	ccs = [i[10] for i in categories]
	lccfs = [i[11] for i in categories]
	aces = [i[12] for i in categories]
	oscars = [i[13] for i in categories]

if __name__ == '__main__':
	main()
