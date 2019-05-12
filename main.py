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


def build_model(type):
	if type == 'svm':
		model = svm.SVC(kernel='linear')
	elif type == 'mlp':
		model = Sequential()
		model.add(Dense(12, input_dim=8, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))


def add_award_points(dataframe):
	if os.path.exists('./data/categories') and os.path.exists('./data/awards') and os.path.exists('./data/oscar_cs') and os.path.exists('./data/oscar_aw'):
		with open('./data/categories', 'rb') as f:
			categories = pickle.load(f)
		with open('./data/awards', 'rb') as f:
			awards = pickle.load(f)
		with open('./data/oscar_cs', 'rb') as f:
			oscar_cs = pickle.load(f)
		with open('./data/oscar_aw', 'rb') as f:
			oscar_aw = pickle.load(f)
	else:
		categories = []
		awards = []
		oscar_cs = []
		oscar_aw = []
		for y in range(2000, 2019):
			print(y)
			results = collect_data.scrape_movie_awards(y)
			categories.append(results[0])
			awards.append(results[1])
			oscar_cs.append(results[2])
			oscar_aw.append(results[3])
		with open('./data/categories', 'wb') as f:
			pickle.dump(categories, f)
		with open('./data/awards', 'wb') as f:
			pickle.dump(awards, f)
		with open('./data/oscar_cs', 'wb') as f:
			pickle.dump(oscar_cs, f)
		with open('./data/oscar_aw', 'wb') as f:
			pickle.dump(oscar_aw, f)

	# Ensures that all movies' award points start at 0
	for i in dataframe.columns[16:]:
		dataframe[i] = 0

	# Adds points to all of the movies that have won/been nominated for awards in all categories (except Oscar)
	start = dataframe.columns.get_loc('best_picture')
	for i, year in enumerate(categories):
		for j, event in enumerate(year):
			for k, award in enumerate(event):
				for l, movie in enumerate(awards[i][j][k]):
					index = dataframe.index[(dataframe.movie == movie)&((dataframe.year == 2000 + i)|(dataframe.year == 2000 + i + 1)|(dataframe.year == 2000 + i - 1))]
					if len(index) != 0:
						print(str(i) + ", " + str(j) + ", " + str(k) + ", " + str(l))
						print(movie + '\n')
						if l == 0: points = 1
						else: points = 0.5
						dataframe.loc[index[0], dataframe.columns[start + int(award)]] += points

	# Oscar points for data labels
	oscar_start = dataframe.columns.get_loc('oscar_best_picture')
	for i, year in enumerate(oscar_cs):
		for j, award in enumerate(year):
			for l, movie in enumerate(oscar_aw[i][j]):
				index = dataframe.index[(dataframe.movie == movie)&((dataframe.year == 2000 + i)|(dataframe.year == 2000 + i + 1)|(dataframe.year == 2000 + i - 1))]
				if len(index) != 0:
					print(str(i) + ", " + str(j) + ", " + str(l))
					print(movie)
					print(str(dataframe.loc[index[0], dataframe.columns[oscar_start + int(award)]]))
					if l == 0: points = 1
					else: points = 0.5
					dataframe.loc[index[0], dataframe.columns[oscar_start + int(award)]] = points
					print(str(dataframe.loc[index[0], dataframe.columns[oscar_start + int(award)]]) + '\n')

	# Computes average sum by dividing the award points by the number of award ceremonies the movie could have won in
	N = [11, 7, 7, 7, 7, 8, 4, 4, 5, 8, 1, 4, 6, 3, 4, 4, 3, 1, 1, 3, 1, 4, 6, 5]
	for i, col in enumerate(dataframe.columns[16:oscar_start]):
		dataframe[col] /= N[i]

	return dataframe


def main():
	# bigml = pd.read_csv('./data/bigml.csv', index_col=0)
	# bigml = bigml.fillna(value={'gross': -1, 'popularity': -1})
	# imdb = extract_movie_data()
	# df = bigml.append(imdb, sort=False, ignore_index=True)
	# df.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	# df = df.reset_index(drop=True)

	df = pd.read_csv('./data/combined.csv', index_col=0)
	df = add_award_points(df)
	df.to_csv('./data/combined.csv')


if __name__ == '__main__':
	main()
