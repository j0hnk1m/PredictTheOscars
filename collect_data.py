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
	html = requests.get("https://www.imdb.com/title/" + str(id)).text

	tags = re.findall('"genre": \[([^~]+)\],\\n(?:\s)"contentRating": "(.*)"(?:.* )', html)

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
	#
	# Oscar_Best_Picture_won
	# Oscar_Best_Picture_nominated
	# Oscar_Best_Director_won
	# Oscar_Best_Director_nominated
	# Oscar_Best_Actor_won
	# Oscar_Best_Actor_nominated
	# Oscar_Best_Actress_won
	# Oscar_Best_Actress_nominated
	# Oscar_Best_Supporting_Actor_won
	# Oscar_Best_Supporting_Actor_nominated
	# Oscar_Best_Supporting_Actress_won
	# Oscar_Best_Supporting_Actress_nominated
	# Oscar_Best_AdaScreen_won
	# Oscar_Best_AdaScreen_nominated
	# Oscar_Best_OriScreen_won
	# Oscar_Best_OriScreen_nominated
	# Oscar_nominated
	# Oscar_nominated_categories
	#
	# Golden_Globes_won
	# Golden_Globes_won_categories
	# Golden_Globes_nominated
	# Golden_Globes_nominated_categories
	# BAFTA_won
	# BAFTA_won_categories
	# BAFTA_nominated
	# BAFTA_nominated_categories
	# Screen_Actors_Guild_won
	# Screen_Actors_Guild_won_categories
	# Screen_Actors_Guild_nominated
	# Screen_Actors_Guild_nominated_categories
	# Critics_Choice_won
	# Critics_Choice_won_categories
	# Critics_Choice_nominated
	# Critics_Choice_nominated_categories
	# Directors_Guild_won
	# Directors_Guild_won_categories
	# Directors_Guild_nominated
	# Directors_Guild_nominated_categories
	# Producers_Guild_won	Producers_Guild_won_categories
	# Producers_Guild_nominated
	# Producers_Guild_nominated_categories
	# Art_Directors_Guild_won
	# Art_Directors_Guild_won_categories
	# Art_Directors_Guild_nominated
	# Art_Directors_Guild_nominated_categories
	# Writers_Guild_won
	# Writers_Guild_won_categories
	# Writers_Guild_nominated
	# Writers_Guild_nominated_categories
	# Costume_Designers_Guild_won
	# Costume_Designers_Guild_won_categories
	# Costume_Designers_Guild_nominated
	# Costume_Designers_Guild_nominated_categories
	# Online_Film_Television_Association_won
	# Online_Film_Television_Association_won_categories
	# Online_Film_Television_Association_nominated
	# Online_Film_Television_Association_nominated_categories
	# Online_Film_Critics_Society_won
	# Online_Film_Critics_Society_won_categories
	# Online_Film_Critics_Society_nominated
	# Online_Film_Critics_Society_nominated_categories
	# People_Choice_won
	# People_Choice_won_categories
	# People_Choice_nominated
	# People_Choice_nominated_categories
	# London_Critics_Circle_Film_won
	# London_Critics_Circle_Film_won_categories
	# London_Critics_Circle_Film_nominated
	# London_Critics_Circle_Film_nominated_categories
	# American_Cinema_Editors_won
	# American_Cinema_Editors_won_categories
	# American_Cinema_Editors_nominated
	# American_Cinema_Editors_nominated_categories
	# Hollywood_Film_won
	# Hollywood_Film_won_categories
	# Hollywood_Film_nominated
	# Hollywood_Film_nominated_categories
	# Austin_Film_Critics_Association_won
	# Austin_Film_Critics_Association_won_categories
	# Austin_Film_Critics_Association_nominated
	# Austin_Film_Critics_Association_nominated_categories
	# Denver_Film_Critics_Society_won
	# Denver_Film_Critics_Society_won_categories
	# Denver_Film_Critics_Society_nominated
	# Denver_Film_Critics_Society_nominated_categories
	# Boston_Society_of_Film_Critics_won
	# Boston_Society_of_Film_Critics_won_categories
	# Boston_Society_of_Film_Critics_nominated
	# Boston_Society_of_Film_Critics_nominated_categories
	# New_York_Film_Critics_Circle_won
	# New_York_Film_Critics_Circle_won_categories
	# New_York_Film_Critics_Circle_nominated
	# New_York_Film_Critics_Circle_nominated_categories
	# Los_Angeles_Film_Critics_Association_won
	# Los_Angeles_Film_Critics_Association_won_categories
	# Los_Angeles_Film_Critics_Association_nominated
	# Los_Angeles_Film_Critics_Association_nominated_categories
	# release_date.month
	# release_date.day-of-month

