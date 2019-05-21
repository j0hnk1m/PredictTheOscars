import pandas as pd
import numpy as np
import collect_data
import os
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
np.random.seed(1)


def extract_movie_data():
	"""
	Extracts the movie titles from years 2000~2018 and their respective tags/details and outputs them in the form of
	a dataframe in similar format to the BIGML dataset.
	:return: dataframe of web-scraped movies and their tags
	"""
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


def combine_datasets():
	"""
	Combines the BIGML dataset and our IMDB-scraped dataset. Also fills the NaN values in the gross and popularity
	columns with -1 for better model training and sorts the data based on year and movie.
	:return: combined dataset
	"""
	bigml = pd.read_csv('./data/bigml.csv', index_col=0)
	imdb = extract_movie_data()
	dataframe = bigml.append(imdb, sort=False, ignore_index=True)
	dataframe.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	dataframe = dataframe.reset_index(drop=True)

	return dataframe


def add_award_points(dataframe):
	"""
	Adds points to movies in categories that it won / was nominated in from all 14 award ceremonies. 1 point for winner,
	1/(number of nominees) points for nominee, and 0 points for neither.
	:param dataframe: final (combined) dataset
	:return: edited dataset with points added in
	"""
	if os.path.exists('./data/categories') and os.path.exists('./data/awards') and os.path.exists('./data/oscar_cs') and os.path.exists('./data/oscar_aw'):
		with open('./data/categories', 'rb') as f:
			categories = pickle.load(f)
		with open('./data/awards', 'rb') as f:
			awards = pickle.load(f)
		with open('./data/oscarCategories', 'rb') as f:
			oscarCategories = pickle.load(f)
		with open('./data/oscarAwards', 'rb') as f:
			oscarAwards = pickle.load(f)
	else:
		categories = []
		awards = []
		oscarCategories = []
		oscarAwards = []
		for y in range(2000, 2019):
			print(y)
			results = collect_data.scrape_movie_awards(y)
			categories.append(results[0])
			awards.append(results[1])
			oscarCategories.append(results[2])
			oscarAwards.append(results[3])
		with open('./data/categories', 'wb') as f:
			pickle.dump(categories, f)
		with open('./data/awards', 'wb') as f:
			pickle.dump(awards, f)
		with open('./data/oscarCategories', 'wb') as f:
			pickle.dump(oscarCategories, f)
		with open('./data/oscarAwards', 'wb') as f:
			pickle.dump(oscarAwards, f)

	# Adds points to all of the movies that have won/been nominated for awards in all categories (except Oscar)
	start = dataframe.columns.get_loc('best_picture')

	# Ensures that all movies' award points start at 0
	for i in dataframe.columns[start:]:
		dataframe[i] = 0

	for i, year in enumerate(categories):
		for j, event in enumerate(year):
			for k, award in enumerate(event):
				for l, movie in enumerate(awards[i][j][k]):
					index = dataframe.index[(dataframe.movie == movie)&((dataframe.year == 2000 + i)|(dataframe.year == 2000 + i + 1)|(dataframe.year == 2000 + i - 1))]
					if len(index) != 0:
						print(str(i) + ", " + str(j) + ", " + str(k) + ", " + str(l))
						print(movie + '\n')
						if l == 0: points = 1
						else: points = 1.0/len(awards[i][j][k])
						dataframe.loc[index[0], dataframe.columns[start + int(award)]] += points

	# Oscar points for data labels
	oscarStart = dataframe.columns.get_loc('oscar_best_picture')
	for i, year in enumerate(oscarCategories):
		for j, award in enumerate(year):
			for l, movie in enumerate(oscarAwards[i][j]):
				index = dataframe.index[(dataframe.movie == movie)&((dataframe.year == 2000 + i)|(dataframe.year == 2000 + i + 1)|(dataframe.year == 2000 + i - 1))]
				if len(index) != 0:
					print(str(i) + ", " + str(j) + ", " + str(l))
					print(movie)
					print(str(dataframe.loc[index[0], dataframe.columns[oscarStart + int(award)]]))
					if l == 0: points = 1
					else: points = 1.0/len(oscarAwards[i][j])
					dataframe.loc[index[0], dataframe.columns[oscarStart + int(award)]] = points
					print(str(dataframe.loc[index[0], dataframe.columns[oscarStart + int(award)]]) + '\n')

	# Computes average sum by dividing the award points by the number of award ceremonies the movie could have won in
	N = [11, 7, 7, 7, 7, 8, 4, 4, 5, 8, 1, 4, 6, 3, 4, 4, 3, 1, 1, 3, 1, 4, 6, 5]
	for i, col in enumerate(dataframe.columns[start:oscarStart]):
		dataframe[col] /= N[i]

	dataframe.to_csv('./data/combined.csv')
	return dataframe


def main():
	# df = combine_datasets()
	df = pd.read_csv('./data/combined.csv', index_col=0)
	# df.fillna(-1, inplace=True)
	# df = add_award_points(df)
	# df = id_genre(df)
	# df = id_certificate(df)

	df = df.drop(['movie', 'movie_id', 'synopsis'], axis=1)
	oscarStart = df.columns.get_loc('oscar_best_picture')
	modelType = 'neuralnetwork'

	# Data preprocessing
	x = df.iloc[:, :oscarStart].values
	y = df.iloc[:, oscarStart:].values
	y[y == 1] = 2
	y[(y > 0) & (y < 1)] = 1
	y = y.astype(int)

	# Label encoding
	df = df.drop(df[~df['certificate'].isin(['G', 'PG', 'PG-13', 'R', 'Not Rated'])].index)
	labelencoder_certificate = LabelEncoder()
	x[:, df.columns.get_loc('certificate')] = labelencoder_certificate.fit_transform(x[:, df.columns.get_loc('certificate')])
	labelencoder_genre = LabelEncoder()
	x[:, df.columns.get_loc('genre')] = labelencoder_genre.fit_transform(x[:, df.columns.get_loc('genre')])

	# Dummy variables
	onehotencoder = OneHotEncoder(categorical_features=[1])
	x = onehotencoder.fit_transform(x).toarray()
	x = x[:, 1:]

	# Splits data into training and testing sets
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=21)

	# Scales inputs to avoid one variable having more weight than another
	sc = StandardScaler()
	xTrain = sc.fit_transform(xTrain)
	xTest = sc.transform(xTest)

	if modelType == 'svm':
		# Because SVM cannot handle multiple outputs, split into 24 SVM models for each category
		for category in df.columns[oscarStart:]:
			print(category)
			y = np.array(df[category])
			y[y == 1] = 2
			y[(y > 0) & (y < 1)] = 1
			y = y.astype(int)
			xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=21)

			model = svm.SVC(decision_function_shape='ovo')
			model.fit(xTrain, yTrain)

	elif modelType == 'decisiontree':
		model = DecisionTreeClassifier(random_state=21)
		model.fit(xTrain, yTrain)
		yPred = model.predict(xTest)
		p = np.where(yPred == 2)
		v = np.where(yTest == 2)

		# x = np.array(df[df.columns[:oscarStart]].values)
		# y = np.array(df[df.columns[oscarStart:]])
		# y[y == 1] = 2
		# y[(y > 0) & (y < 1)] = 1
		# y = y.astype(int)
		# xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.2, random_state=21)
		# model = DecisionTreeClassifier(random_state=21)
		# model.fit(xTrain, yTrain)
		# model.score(xVal, yVal)

	elif modelType == 'randomforest':
		model = RandomForestClassifier(random_state=21)
		model.fit(xTrain, yTrain)
		yPred = model.predict(xTest)
		p = np.where(yPred==2)
		v = np.where(yTest==2)

		# x = np.array(df[df.columns[:oscarStart]].values)
		# y = np.array(df[df.columns[oscarStart:]])
		# y[y == 1] = 2
		# y[(y > 0) & (y < 1)] = 1
		# y = y.astype(int)
		# xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.2, random_state=21)
		# model = DecisionTreeClassifier(random_state=21)
		# model.fit(xTrain, yTrain)
		# model.score(xVal, yVal)

	elif modelType == 'neuralnetwork':
		model = Sequential()
		model.add(Dense(128, input_dim=39))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(24))
		model.add(Activation('sigmoid'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary()
		model.fit(xTrain, yTrain, epochs=10, batch_size=32)


if __name__ == '__main__':
	main()
