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

	# categories = []
	# awards = []
	# for y in range(2000, 2019):
	# 	print(y)
	# 	results = collect_data.movie_awards(y)
	# 	categories.append(results[0])
	# 	awards.append(results[1])
	# with open('categories', 'wb') as f:
	# 	pickle.dump(categories, f)
	# with open('awards', 'wb') as f:
	# 	pickle.dump(awards, f)

	with open('categories', 'rb') as f:
		categories = pickle.load(f)
	with open('awards', 'rb') as f:
		awards = pickle.load(f)

	gg_cs = [i[0] for i in categories]
	gg_aw = [i[0] for i in awards]
	bafta_cs = [i[1] for i in categories]
	bafta_aw = [i[1] for i in awards]
	sag_cs = [i[2] for i in categories]
	sag_aw = [i[2] for i in awards]
	dg_cs = [i[3] for i in categories]
	dg_aw = [i[3] for i in awards]
	pg_cs = [i[4] for i in categories]
	pg_aw = [i[4] for i in awards]
	adg_cs = [i[5] for i in categories]
	adg_aw = [i[5] for i in awards]
	wg_cs = [i[6] for i in categories]
	wg_aw = [i[6] for i in awards]
	cdg_cs = [i[7] for i in categories]
	cdg_aw = [i[7] for i in awards]
	ofta_cs = [i[8] for i in categories]
	ofta_aw = [i[8] for i in awards]
	ofcs_cs = [i[9] for i in categories]
	ofcs_aw = [i[9] for i in awards]
	cc_cs = [i[10] for i in categories]
	cc_aw = [i[10] for i in awards]
	lccf_cs = [i[11] for i in categories]
	lccf_aw = [i[11] for i in awards]
	ace_cs = [i[12] for i in categories]
	ace_aw = [i[12] for i in awards]
	oscar_cs = [i[13] for i in categories]
	oscar_aw = [i[13] for i in awards]

	start = df.columns.get_loc('best_picture')
	for i, year in enumerate(categories):
		for j, event in enumerate(year):
			for k, award in enumerate(event):
				

if __name__ == '__main__':
	main()
