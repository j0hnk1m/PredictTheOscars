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
	html = requests.get("https://www.imdb.com/year/" + str(year)).text

	movies = np.zeros((0, 2))
	for i in range(0, 7):  # 7 pages of 50 movies each = 500 top movies
		movies = np.concatenate([movies, np.flip(np.array(re.findall(r'<a href="/title/([^:?%]+?)/"[\r\n]+> <img alt="([^%]+?)"[\r\n]+', html)))])
		nextLink = "https://www.imdb.com" + re.findall(r'<a href="(/search/title\?title_type=feature&year=(?:.*)&start=(?:.*))"[\r\n]+class="lister-page-next next-page"', html)[0]
		html = requests.get(nextLink).text

	df = pd.DataFrame(movies, columns=['movie', 'movie_id'])
	df.insert(0, 'year', [year]*movies.shape[0], True)
	return df

def collect_movie_info(id):
	# ---------------TAGS---------------
	# certificate
	# duration
	# genre
	# rate
	# metascore
	# synopsis
	# votes
	# gross
	# release_date
	# user_reviews
	# critic_reviews
	# popularity
	# awards_wins
	# awards_nominations
	html = requests.get("https://www.imdb.com/title/" + str(id)).text
	tags = re.findall('"genre": \[([\s\S]+)\],\\n(?:\s*)"contentRating": "(.*)",\\n[\s\S]+<strong title="(.*) based on '
					  '([,0-9]+) user ratings">[\s\S]+<span itemprop="reviewCount">([,0-9]+) user</span>[\s\S]+<span itemprop'
					  '="reviewCount">([,0-9]+) critic</span>[\s\S]+<time datetime="PT(\d+)M">\\n[\s\S]+<div class="summary_text">\\n'
					  '(.*)\\n[\s\S]+<div class="metacriticScore score_favorable titleReviewBarSubItem">\\n<span>(\d{2,3})<'
					  '[\s\S]+ ([,0-9]+)\\n[\s\S]+\(<span class="titleOverviewSprite popularity[\s\S]+<span class='
					  '"awards-blurb">[\s\S]+Another[\s\S]+ (\d+) wins &amp; (\d+) nominations.[\s\S]+<h4 class="inline"'
					  '>Gross USA:</h4> \$([,0-9]+), <span', html)[0]
	tags = [" ".join(i.split()).replace('"', '') for i in tags]
	order = [1, 6, 0, 2, 8, 7, 3, 12, 4, 5, 9, 10, 11]
	tags = [tags[i] for i in order]
	return tags


	# ---------------AWARDS---------------
	# Oscar
	# Golden Globe
	# BAFTA
	# Screen Actors Guild
	# Critics Choice
	# Directors Guild
	# Producers Guild
	# Art Directors Guild
	# Writers Guild
	# Costume Designers Guild
	# Online Film Television Association
	# Online Film Critics Society
	# People Choice
	# London Critics Circle Film
	# American Cinema Editors
	# Hollywood Film
	# Austin Film Critics Association
	# Denver Film Critics_Society
	# Boston Society of Film Critics
	# New York Film Critics Circle
	# Los Angeles Film Critics Association

