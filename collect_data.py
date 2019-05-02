import numpy as np
import pandas as pd
import requests
import re

def imdb_feature_film(year):
	# ex: https://www.imdb.com/year/2018/
	response = requests.get("https://www.imdb.com/year/" + str(year))
	html = response.text
	# totalCount = int(re.findall(r'<span>1-50 of ([^%]+?) titles.</span>[\r\n]+', html)[0].replace(',', ''))

	movies = np.zeros((50, 2))
	for i in range(0, 5):  # 5 pages of 50 movies each = 250 top movies
		print(i)
		movies = np.concatenate((movies, np.array(re.findall(r'<a href="/title/([^:?%]+?)/"[\r\n]+> <img alt="([^%]+?)"[\r\n]+', html))))
		nextLink = "https://www.imdb.com" + re.findall(r'<a href="(/search/title\?title_type=feature&year=(?:.*)&start=(?:.*))"[\r\n]+class="lister-page-next next-page"', html)[0]
		response = requests.get(nextLink)
		html = response.text

	movies=movies[50:]
	return movies


def wikipedia_in_film(year):
	# ex: https://en.wikipedia.org/wiki/2018_in_film
	response = requests.get("https://en.wikipedia.org/wiki/" + str(year) + "_in_film")
	html = response.text
	movies = np.unique(re.findall(r'<td><i><a href="/wiki/(?:.*)" title="(.*)">', html))
	movies = np.array([i.replace('&#39;', "'").replace('&amp;', '&') for i in movies])
	return movies


def wildaboutmovies(year):
	# ex: https://www.wildaboutmovies.com/2018_movies/
	response = requests.get("https://www.wildaboutmovies.com/" + str(year) + "_movies/")
	html = response.text
	movies = np.array(re.findall(r'<a href="/(?:.*)" alt="(.*)" /><p>', html))
	return movies
