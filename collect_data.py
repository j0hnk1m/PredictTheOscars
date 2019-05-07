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
	keywords = re.findall('<div class="summary_text">\\n(.*)\\n', html)[0].strip()
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

	tags = [certificate, duration, genre, rate, metascore, keywords, votes, gross, user_reviews, critic_reviews,
			popularity, awards_wins, awards_nominations]
	return tags


def movie_awards(id):
	events = ['ev0000292', 'ev0000123', 'ev0000598', 'ev0000212', 'ev0000531', 'ev0000618', 'ev0000710',
			  'ev0000190', 'ev0002704', 'ev0000511', 'ev0000530', 'ev0000403', 'ev0000017', 'ev0000003']

	htmls = []
	for e in events:
		htmls.append(requests.get("https://www.imdb.com/event/" + e + "/" + str(year) + "/1?ref_=ttawd_ev_1").text)
	# ---------------AWARDS---------------
	# Golden Globe
	# BAFTA
	# Screen Actors Guild
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
	# Oscar

	gg = re.findall('"name":"([^"]*)","note":null', htmls[0])
	bafta = re.findall('', htmls[1])
	sag = re.findall('', htmls[2])
	dg = re.findall('', htmls[3])
	pg = re.findall('', htmls[4])
	adg = re.findall('', htmls[5])
	wg = re.findall('', htmls[6])
	cdg = re.findall('', htmls[7])
	ofta = re.findall('', htmls[8])
	ofcs = re.findall('', htmls[9])
	pc = re.findall('', htmls[10])
	lccf = re.findall('', htmls[11])
	ace = re.findall('', htmls[12])
	oscar = re.findall('', htmls[13])


	return [oscar, gg, bafta, sag, cc, dg, pg, adg, wg, cdg, ofta, ofcs, pc, lccf, ace]