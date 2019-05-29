import pandas as pd
import numpy as np
import collect_data as cd
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
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
		imdb = cd.imdb_feature_film(2000)
		for y in list(range(2001, 2019)):
			imdb = imdb.append(cd.imdb_feature_film(y))

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
			extra = cd.movie_tags(id)

			if extra is not None:
				tags.append([row['year'], row['movie'], row['movie_id']] + extra)

		imdb = pd.DataFrame(tags, columns=['year', 'movie', 'movie_id', 'certificate', 'duration', 'genre', 'rate',
										   'metascore', 'synopsis', 'votes', 'gross', 'user_reviews', 'critic_reviews',
										   'popularity', 'awards_wins', 'awards_nominations'])
		imdb.to_csv('./data/imdb.csv')

	return imdb


def combine_datasets():
	"""
	Combines the BIGML dataset and our IMDB-scraped dataset
	:return: combined dataset
	"""
	bigml = pd.read_csv('./data/bigml.csv')
	bigml = bigml.drop(list(bigml.columns[bigml.columns.get_loc('Oscar_Best_Picture_won'):])+['release_date'], axis=1)
	imdb = extract_movie_data()
	dataframe = bigml.append(imdb, sort=False, ignore_index=True)
	dataframe.sort_values(['year', 'movie'], axis=0, ascending=True, inplace=True)
	dataframe = dataframe.reset_index(drop=True)
	dataframe.to_csv('./data/combined.csv')

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
			results = cd.scrape_movie_awards(y)
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

	categoryNames = ['best_picture', 'actor', 'actress', 'supporting_actor', 'supporting_actress', 'animated']
	for category in categoryNames:
		dataframe[category] = np.nan
	for category in categoryNames:
		dataframe['oscar_' + category] = np.nan

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
	N = [10, 7, 7, 7, 7, 8]
	for i, col in enumerate(dataframe.columns[start:oscarStart]):
		dataframe[col] /= N[i]

	dataframe.to_csv('./data/combined.csv')
	return dataframe


def compute_model_accuracies(predCategory, printout, m, x, y, split):
	categoryNames = ['best_picture', 'best_actor', 'best_actress', 'best_supporting_actor', 'best_supporting_actress', 'best_animated']
	df = pd.read_csv('./data/combined.csv', index_col=0)
	df = df.reset_index(drop=True)

	if not predCategory:
		yPred = m.predict_classes(x)

		totalAccuracy = accuracy_score(y, yPred)

		winnerIdx = [i for i, h in enumerate(y) if h == 0]
		winnerTrain = [y[i] for i in winnerIdx]
		winnerPred = [yPred[i] for i in winnerIdx]
		winnerAccuracy = accuracy_score(winnerTrain, winnerPred)

		nomineeIdx = [i for i, h in enumerate(y) if h == 1]
		nomineeTrain = [y[i] for i in nomineeIdx]
		nomineePred = [yPred[i] for i in nomineeIdx]
		nomineeAccuracy = accuracy_score(nomineeTrain, nomineePred)

		loserIdx = [i for i, h in enumerate(y) if h == 2]
		loserTrain = [y[i] for i in loserIdx]
		loserPred = [yPred[i] for i in loserIdx]
		loserAccuracy = accuracy_score(loserTrain, loserPred)

		print(printout + ' Total accuracy: ' + str(totalAccuracy))
		print('   ' + printout + ' Accuracy for predicting winners: ' + str(winnerAccuracy))
		print('   ' + printout + ' Accuracy for predicting nominees: ' + str(nomineeAccuracy))
		print('   ' + printout + ' Accuracy for predicting losers: ' + str(loserAccuracy))

		# Print the names of the predicted winners/nominees
		if printout == '(TESTING)':
			yPred = m.predict(x)
			for i, pred in enumerate(yPred):
				if pred == 2 or pred == 1:
					print(df.iloc[split + i].movie)
	else:
		yPred = m.predict(x)

		totalAccuracy = 0
		winnerAccuracy = 0
		nomineeAccuracy = 0
		loserAccuracy = 0
		for i in range(0, 6):
			true = y[i].argmax(axis=-1)
			pred = yPred[i].argmax(axis=-1)

			totalAccuracy += accuracy_score(true, pred)

			winnerIdx = [a for a, h in enumerate(true) if h == 0]
			winnerTrain = [true[a] for a in winnerIdx]
			winnerPred = [pred[a] for a in winnerIdx]
			winnerAccuracy += accuracy_score(winnerTrain, winnerPred)

			nomineeIdx = [i for i, h in enumerate(true) if h == 1]
			nomineeTrain = [true[a] for a in nomineeIdx]
			nomineePred = [pred[a] for a in nomineeIdx]
			nomineeAccuracy += accuracy_score(nomineeTrain, nomineePred)

			loserIdx = [a for a, h in enumerate(true) if h == 2]
			loserTrain = [true[a] for a in loserIdx]
			loserPred = [pred[a] for a in loserIdx]
			loserAccuracy += accuracy_score(loserTrain, loserPred)

		totalAccuracy /= 6; winnerAccuracy /= 6; nomineeAccuracy /= 6; loserAccuracy /= 6
		print(printout + ' Total accuracy: ' + str(totalAccuracy))
		print('   ' + printout + ' Accuracy for predicting winners: ' + str(winnerAccuracy))
		print('   ' + printout + ' Accuracy for predicting nominees: ' + str(nomineeAccuracy))
		print('   ' + printout + ' Accuracy for predicting losers: ' + str(loserAccuracy))
		print()

		# Print the names of the predicted winners/nominees
		if printout == '(TESTING)':
			yPred = [i.argmax(axis=-1) for i in yPred]
			temp = []
			for s in range(yPred[0].shape[0]):
				sample = []
				[sample.append(i[s]) for i in yPred]
				temp.append(sample)
			yPred = np.array(temp)

			for i, pred in enumerate(yPred):
				movie = df.iloc[split + i].movie
				winnerCategories = [categoryNames[a] for a, b in enumerate(pred) if b == 0]
				nomineeCategories = [categoryNames[a] for a, b in enumerate(pred) if b == 1]

				if winnerCategories and nomineeCategories:
					print(movie + ': Won ' + '|'.join(winnerCategories) + ', Nominated for ' + '|'.join(nomineeCategories))
				elif winnerCategories and not nomineeCategories:
					print(movie + ': Won ' + '|'.join(winnerCategories))
				elif not winnerCategories and nomineeCategories:
					print(movie + ': Nominated for ' + '|'.join(nomineeCategories))


def main():
	# df = combine_datasets()
	df = pd.read_csv('./data/combined.csv', index_col=0)
	# df.fillna(-1, inplace=True)
	# df = df.drop(df[~df['certificate'].isin(['G', 'PG', 'PG-13', 'R', 'Not Rated'])].index)
	# df = add_award_points(df)

	# Data preprocessing/encoding
	df = df.drop(['movie', 'movie_id', 'synopsis', 'genre'], axis=1)
	df['popularity'] = 1/np.array(df['popularity']) * 100
	df = pd.get_dummies(df, columns=['certificate'])
	cols = df.columns.tolist()
	cols = cols[df.columns.get_loc('oscar_animated') + 1:] + cols[:df.columns.get_loc('oscar_animated') + 1]
	df = df[cols]
	df = df.reset_index(drop=True)
	splitIndex = df.index[df['year'] == 2018][0]
	df = df.drop(['year'], axis=1)

	# Splits data into training and testing sets
	oscarStart = df.columns.get_loc('oscar_best_picture')
	x = df.iloc[:, :oscarStart].values
	y = df.iloc[:, oscarStart:].values
	y[(y > 0) & (y < 1)] = 0.5  # winner is 1, nominee is 0.5, nothing is 0
	xTrain, xTest = x[:splitIndex], x[splitIndex:]
	yTrain, yTest = y[:splitIndex], y[splitIndex:]

	# Checks how imbalanced the data is
	unique, counts = np.unique(yTrain, return_counts=True)
	print(dict(zip(unique, counts)))

	# Scales inputs to avoid one variable having more weight than another
	sc = StandardScaler()
	xTrain = sc.fit_transform(xTrain)
	xTest = sc.transform(xTest)

	modelType = 'neuralnetwork'
	predictCategory = True
	if modelType == 'randomforest':
		model = RandomForestClassifier(random_state=21)
		model.fit(xTrain, yTrain)
		yPred = model.predict(xTest)
		p = np.where(yPred==2)
		v = np.where(yTest==2)

	elif modelType == 'neuralnetwork':
		if not predictCategory:
			# One hot encoding for softmax activation function
			trainTargets = []
			for i in yTrain:
				if 1 in i:
					trainTargets.append([1, 0, 0])
				elif 0.5 in i:
					trainTargets.append([0, 1, 0])
				else:
					trainTargets.append([0, 0, 1])
			yTrain = np.array(trainTargets)
			testTargets = []
			for i in yTest:
				if 1 in i:
					testTargets.append([1, 0, 0])
				elif 0.5 in i:
					testTargets.append([0, 1, 0])
				else:
					testTargets.append([0, 0, 1])
			yTest = np.array(testTargets)

			model = Sequential()
			model.add(Dense(256, input_dim=xTrain.shape[1]))
			model.add(Activation('relu'))
			model.add(Dropout(0.2))
			model.add(Dense(3))
			model.add(Activation('softmax'))
			model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['mse'])

			classWeights = {0: counts.sum()/counts[2], 1: counts.sum()/counts[1], 2: counts.sum()/counts[0]}
			model.fit(xTrain, yTrain, epochs=512, batch_size=32, class_weight=classWeights)
		else:
			# One hot encoding for softmax activation function
			trainTargets = [[] for i in range(0, 6)]
			for i in yTrain:
				for idx, j in enumerate(i):
					if j == 1:  # winner
						trainTargets[idx].append([1, 0, 0])
					elif j == 0.5:  # nominee
						trainTargets[idx].append([0, 1, 0])
					else:  # loser/nothing
						trainTargets[idx].append([0, 0, 1])
			yTrain = [np.array(i) for i in trainTargets]
			testTargets = [[] for i in range(0, 6)]
			for i in yTest:
				for idx, j in enumerate(i):
					if j == 1:  # winner
						testTargets[idx].append([1, 0, 0])
					elif j == 0.5:  # nominee
						testTargets[idx].append([0, 1, 0])
					else:  # loser/nothing
						testTargets[idx].append([0, 0, 1])
			yTest = [np.array(i) for i in testTargets]

			input = Input(shape=(xTrain.shape[1],))
			x = Dense(128, activation='relu')(input)
			x = BatchNormalization()(x)
			x = Dropout(0.2)(x)
			output1 = Dense(3, activation='softmax')(x)
			output2 = Dense(3, activation='softmax')(x)
			output3 = Dense(3, activation='softmax')(x)
			output4 = Dense(3, activation='softmax')(x)
			output5 = Dense(3, activation='softmax')(x)
			output6 = Dense(3, activation='softmax')(x)
			model = Model(inputs=input, outputs=[output1, output2, output3, output4, output5, output6])
			model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')

			classWeights = {0: counts.sum()/counts[2], 1: counts.sum()/counts[1], 2: counts.sum()/counts[0]}
			model.fit(xTrain, yTrain, epochs=512, batch_size=32, class_weight=classWeights)
			# model.save('best.h5')
			# model = load_model('best.h5')

		# Training accuracy (put training data back in) and testing accuracy
		compute_model_accuracies(predictCategory, '(TRAINING)', model, xTrain, yTrain, splitIndex)
		compute_model_accuracies(predictCategory, '(TESTING)', model, xTest, yTest, splitIndex)


if __name__ == '__main__':
	main()
