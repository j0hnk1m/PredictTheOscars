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


def movie_awards(year):
	events = ['ev0000292', 'ev0000123', 'ev0000598', 'ev0000212', 'ev0000531', 'ev0000618', 'ev0000710',
			  'ev0000190', 'ev0002704', 'ev0000511', 'ev0000133', 'ev0000403', 'ev0000017', 'ev0000003']

	htmls = []
	for e in events:
		htmls.append(requests.get("https://www.imdb.com/event/" + e + "/" + str(year + 1) + "/1?ref_=ttawd_ev_1").text)
	# ---------------AWARDS---------------
	# 1. Golden Globe
	# 2. BAFTA
	# 3. Screen Actors Guild
	# 4. Directors Guild
	# 5. Producers Guild
	# 6. Art Directors Guild
	# 7. Writers Guild
	# 8. Costume Designers Guild
	# 9. Online Film Television Association
	# 10. Online Film Critics Society
	# 11. Critics Choice
	# 12. London Critics Circle Film
	# 13. American Cinema Editors
	# 14. Oscar

	gg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[0]) if 'Television' not in i]
	gg = []
	for c in gg_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			gg.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[0])[:-1])
		else:
			gg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[0])[:-1])

	bafta_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[1]) if 'British' not in i][:20]
	bafta = []
	for c in bafta_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			bafta.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[1])[:5])
		else:
			bafta.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[1])[:-1])

	sag_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[2]) if 'Series' not in i and 'Motion Picture' not in i]
	sag = []
	for c in sag_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			sag.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[2])[:-1])
		else:
			sag.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[2])[:-1])

	dg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[3]) if 'Feature Film' in i or 'Documentary' in i]
	dg = []
	for c in dg_categories:
		dg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[3])[:-1])

	pg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[4]) if 'Theatrical Motion Pictures' in i]
	pg = []
	for c in pg_categories:
		pg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[4])[:-1])

	adg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[5]) if 'Film' in i]
	adg = []
	for c in adg_categories:
		adg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[5])[:-1])

	wg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[6]) if 'Original Screenplay'
					 in i or 'Adapted Screenplay' in i]
	wg = []
	for c in wg_categories:
		wg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[6])[:-1])

	cdg_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[7]) if 'Film' in i]
	cdg = []
	for c in cdg_categories:
		cdg.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[7])[:-1])

	ofta_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[8]) if 'Ensemble' not in i
					   and 'Debut' not in i and 'Poster' not in i and 'Trailer' not in i and 'Stunt' not in i and
					   'Sequence' not in i and 'Voice-Over' not in i and 'Youth' not in i and 'Cinematic' not in i]
	ofta = []
	for c in ofta_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			ofta.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[8])[:-1])
		else:
			ofta.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[8])[:-1])

	ofcs_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[9]) if 'Debut' not in i and 'Stunt' not in i]
	ofcs = []
	for c in ofcs_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			ofcs.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[9])[:-1])
		else:
			ofcs.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[9])[:-1])

	cc_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[10]) if 'Series' not in i and 'Young' not in i and 'Ensemble' not in i and 'TV' not in i]
	cc = []
	for c in cc_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			cc.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[10])[:-1])
		else:
			cc.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[10])[:-1])

	lccf_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[11]) if 'British' not in i and 'Technical' not in i]
	lccf = []
	for c in lccf_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			lccf.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[11])[:-1])
		else:
			lccf.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[11])[:-1])

	ace_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[12]) if 'Series' not in i and 'Non-Theatrical' not in i]
	ace = []
	for c in ace_categories:
		if 'Actor' in c or 'Actress' in c or 'Director' in c:
			ace.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[12])[:-1])
		else:
			ace.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[12])[:-1])

	oscar_categories = [i for i in re.findall('"categoryName":"([^"]*)","nominations"', htmls[13])]
	oscar = []
	for c in oscar_categories:
		if c == oscar_categories[-1]:
			if 'Actor' in c or 'Actress' in c or 'Director' in c:
				oscar.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[13]))
			else:
				oscar.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[13]))
		else:
			if 'Actor' in c or 'Actress' in c or 'Director' in c:
				oscar.append(re.findall(re.escape(c) + '",(?:.*?)"secondaryNominees":\[{"name":"([^"]*)","note":null', htmls[13])[:-1])
			else:
				oscar.append(re.findall(re.escape(c) + '",(?:.*?)"primaryNominees":\[{"name":"([^"]*)","note":null', htmls[13])[:-1])

	return [gg_categories, bafta_categories, sag_categories, dg_categories, pg_categories, adg_categories, wg_categories, cdg_categories, ofta_categories, ofcs_categories, cc_categories, lccf_categories, ace_categories, oscar_categories], [gg, bafta, sag, dg, pg, adg, wg, cdg, ofta, ofcs, cc, lccf, ace, oscar]


def order_categories(name, cs):
	if name == 'Golden Globe':
		order = [next((s for s in cs if 'Best Motion Picture' in s and 'Drama' in s), None),
					next((s for s in cs if 'Best Motion Picture' in s and 'Comedy' in s), None),
					next((s for s in cs if 'Actor' in s and 'Drama' in s), None),
					next((s for s in cs if 'Actor' in s and 'Comedy' in s), None),
					next((s for s in cs if 'Actress' in s and 'Drama' in s), None),
					next((s for s in cs if 'Actress' in s and 'Comedy' in s), None),
					next((s for s in cs if 'Actor' in s and 'Supporting' in s), None),
					next((s for s in cs if 'Actress' in s and 'Supporting' in s), None),
					next((s for s in cs if 'Animated' in s and 'Comedy' in s), None),
				   next((s for s in cs if 'Director' in s), None),
					next((s for s in cs if 'Foreign' in s), None),
					next((s for s in cs if 'Original Score' in s), None),
					next((s for s in cs if 'Original Song' in s), None),
					next((s for s in cs if 'Screenplay' in s), None)]
		order = [i for i in order if i is not None]
		return order
