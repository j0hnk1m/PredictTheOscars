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

def movie_tags(id):
	html = requests.get("https://www.imdb.com/title/" + str(id)).text
	# ---------------TAGS---------------
	# certificate
	# duration
	# genre
	# rate
	# metascore
	# synopsis
	# votes
	# gross
	# user reviews
	# critic reviews
	# popularity
	# awards wins
	# awards nominations

	genre = re.findall('"genre": ([\s\S]+),\\n[\s\S]+"contentRating":', html)
	certificate = re.findall('"contentRating": "(.*)",\\n[\s\S]+<strong', html)
	rate = re.findall('<strong title="(.*) based on ', html)
	votes = re.findall('based on ([,0-9]+) user ratings">', html)
	user_reviews = re.findall('<span itemprop="reviewCount">([,0-9]+) user</span>', html)
	critic_reviews = re.findall('<span itemprop="reviewCount">([,0-9]+) critic</span>', html)
	duration = re.findall('<time datetime="PT(\d+)M">\\n', html)
	synopsis = re.findall('<div class="summary_text">\\n(.*)\\n', html)[0].strip()
	metascore = re.findall('<div class="metacriticScore score_[\w]+ titleReviewBarSubItem">\\n<span>([0-9]+)<', html)

	if len(genre) == 0 or len(certificate) == 0 or len(rate) == 0 or len(votes) == 0 or len(user_reviews) == 0 or len(critic_reviews) == 0 or len(duration) == 0 or len(metascore) == 0:
		return None
	genre = ' '.join(genre[0].split()).replace('"', '').replace('[ ', '').replace(' ]', '')
	certificate = certificate[0]
	rate = float(rate[0])
	votes = int(votes[0].replace(',', ''))
	user_reviews = int(user_reviews[0].replace(',', ''))
	critic_reviews = int(critic_reviews[0].replace(',', ''))
	duration = int(duration[0].replace(',', ''))
	metascore = int(metascore[0])

	popularity = re.findall('titleReviewBarSubItem">\\n<span>[0-9]+<[\s\S]+ ([,0-9]+)\\n[\s\S]+\(<span class="titleOverviewSprite popularity', html)
	if len(popularity) == 0:
		popularity = -1
	else:
		popularity = int(popularity[0].replace(',', ''))

	awards_wins = re.findall('<span class="awards-blurb">[\s\S]+(\d+) wins', html)
	if len(awards_wins) == 0:
		awards_wins = 0
	else:
		awards_wins = int(awards_wins[0])

	awards_nominations = re.findall('<span class="awards-blurb">[\s\S]+(\d+) nominations', html)
	if len(awards_nominations) == 0:
		awards_nominations = 0
	else:
		awards_nominations = int(awards_nominations[0])

	gross = re.findall('Gross USA:</h4> \$([,0-9]+)', html)
	if len(gross) == 0:
		gross = -1
	else:
		gross = int(gross[0].replace(',', ''))

	tags = [certificate, duration, genre, rate, metascore, synopsis, votes, gross, user_reviews, critic_reviews,
			popularity, awards_wins, awards_nominations]
	return tags


def movie_awards(year):
	html = requests.get("https://www.oscars.org/oscars/ceremonies/" + str(year)).text
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
	#
	# *wins = 1
	# *nominations = 0.5
	# *nothing = 0
	# *refer to ./data/awards_categories.csv for category indices

	oscar = re.findall('', html)