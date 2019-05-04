import numpy as np
import pandas as pd
import requests
import re


def imdb_feature_film(year):
	"""
	Given a specific year (from 2000~2018), returns a numpy array of 250 movies and their respective IMDB IDs.
	Example link where this function scrapes data from: https://www.imdb.com/year/2018/
	"""
	print(year)
	response = requests.get("https://www.imdb.com/year/" + str(year))
	html = response.text

	movies = np.zeros((0, 2))
	for i in range(0, 7):  # 7 pages of 50 movies each = 500 top movies
		movies = np.concatenate([movies, np.flip(np.array(re.findall(r'<a href="/title/([^:?%]+?)/"[\r\n]+> <img alt="([^%]+?)"[\r\n]+', html)))])
		nextLink = "https://www.imdb.com" + re.findall(r'<a href="(/search/title\?title_type=feature&year=(?:.*)&start=(?:.*))"[\r\n]+class="lister-page-next next-page"', html)[0]
		response = requests.get(nextLink)
		html = response.text

	df = pd.DataFrame(movies, columns=['movie', 'imdb_id'])
	df.insert(0, 'year', [year]*movies.shape[0], True)
	return df


def wikipedia_in_film(year):
	"""
	Given a specific year (from 2000~2018), returns a numpy array of movies.
	Example link where this function scrapes data from: https://en.wikipedia.org/wiki/2018_in_film
	"""
	print(year)
	response = requests.get("https://en.wikipedia.org/wiki/" + str(year) + "_in_film")
	html = response.text
	movies = np.unique(re.findall(r'<td><i><a href="/wiki/(?:.*)" title="(.*)">', html))
	movies = np.array([i.replace('&#39;', "'").replace('&amp;', '&').replace(' (film)', '') for i in movies])

	df = pd.DataFrame(movies, columns=['movie'])
	df.insert(0, 'year', [year] * movies.shape[0], True)
	return df


def collect_movie_info(id):
	response = requests.get("https://www.imdb.com/title/" + str(id))
	html = response.text

	certificate = re.findall(r'"contentRating": "PG-13"', html)
	Duration
	genre
	IMDB rating
	Synopsis
	Votes
	Gross
	Metacritic
	Release date
