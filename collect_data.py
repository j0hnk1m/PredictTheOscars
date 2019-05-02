import numpy as np
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
	# totalCount = int(re.findall(r'<span>1-50 of ([^%]+?) titles.</span>[\r\n]+', html)[0].replace(',', ''))

	movies = np.zeros((0, 2))
	for i in range(0, 5):  # 5 pages of 50 movies each = 250 top movies
		movies = np.concatenate([movies, np.array(re.findall(r'<a href="/title/([^:?%]+?)/"[\r\n]+> <img alt="([^%]+?)"[\r\n]+', html))])
		nextLink = "https://www.imdb.com" + re.findall(r'<a href="(/search/title\?title_type=feature&year=(?:.*)&start=(?:.*))"[\r\n]+class="lister-page-next next-page"', html)[0]
		response = requests.get(nextLink)
		html = response.text

	return movies


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
	return movies


def wildaboutmovies(year):
	"""
	Given a specific year (from 2000~2018), returns a numpy array of movies.
	Example link where this function scrapes data from: https://www.wildaboutmovies.com/2018_movies/
	"""
	print(year)
	response = requests.get("https://www.wildaboutmovies.com/" + str(year) + "_movies/")
	html = response.text
	movies = np.array(re.findall(r'<a href="/(?:.*)" alt="(.*)" /><p>', html))
	return movies
