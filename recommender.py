import re
from math import log10

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tmdb_pipeline import filter_movies

nltk.download("vader_lexicon", quiet=True)


class ContentRecommender:
	"""Content-based movie recommender using TF-IDF + cosine similarity."""

	def __init__(self):
		self.vectorizer = TfidfVectorizer(
			max_features=5000,
			stop_words="english",
			ngram_range=(1, 2),
			min_df=2,
		)
		self.tfidf_matrix = None
		self.df = None

	@staticmethod
	def _join_list(value):
		if isinstance(value, list):
			return " ".join(str(item) for item in value)
		return "" if pd.isna(value) else str(value)

	def fit(self, df):
		self.df = df.reset_index(drop=True).copy()

		corpus = []
		for _, row in self.df.iterrows():
			text_features = row.get("text_features", "")
			genres = self._join_list(row.get("genres", []))
			directors = self._join_list(row.get("directors", []))
			cast = self._join_list(row.get("cast", []))
			keywords = self._join_list(row.get("keywords", []))

			text_blob = " ".join(
				[
					str(text_features) if not pd.isna(text_features) else "",
					genres,
					genres,
					directors,
					directors,
					cast,
					keywords,
				]
			).strip()
			corpus.append(text_blob)

		self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
		print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

	def recommend_by_movie(self, movie_id, top_n=10):
		if self.df is None or self.tfidf_matrix is None:
			raise RuntimeError("ContentRecommender must be fitted before recommending.")

		matches = self.df.index[self.df["movie_id"] == movie_id].tolist()
		if not matches:
			raise ValueError(f"movie_id={movie_id} not found in fitted dataset.")

		idx = matches[0]
		similarities = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
		similarities[idx] = 0.0

		result = self.df.copy()
		result["similarity_score"] = similarities
		return result.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)

	def recommend_by_query(self, query_text, top_n=10):
		if self.df is None or self.tfidf_matrix is None:
			raise RuntimeError("ContentRecommender must be fitted before recommending.")

		query_vec = self.vectorizer.transform([query_text])
		similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

		result = self.df.copy()
		result["similarity_score"] = similarities
		return result.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)

	def explain_similarity(self, movie_id_a, movie_id_b):
		if self.df is None or self.tfidf_matrix is None:
			raise RuntimeError("ContentRecommender must be fitted before explaining similarity.")

		idx_a = self.df.index[self.df["movie_id"] == movie_id_a].tolist()
		idx_b = self.df.index[self.df["movie_id"] == movie_id_b].tolist()
		if not idx_a or not idx_b:
			raise ValueError("One or both movie IDs are not present in the fitted dataset.")

		vec_a = self.tfidf_matrix[idx_a[0]].toarray().ravel()
		vec_b = self.tfidf_matrix[idx_b[0]].toarray().ravel()
		overlap = np.minimum(vec_a, vec_b)

		feature_names = np.array(self.vectorizer.get_feature_names_out())
		non_zero_idx = np.where(overlap > 0)[0]
		if non_zero_idx.size == 0:
			return "Both movies share themes of: no strong overlapping themes"

		top_idx = non_zero_idx[np.argsort(overlap[non_zero_idx])[::-1][:5]]
		terms = feature_names[top_idx].tolist()
		return "Both movies share themes of: " + ", ".join(terms)


class SentimentRecommender:
	"""Sentiment-based movie recommender using NLTK VADER."""

	def __init__(self):
		self.sia = SentimentIntensityAnalyzer()
		self.df = None

	def fit(self, df):
		self.df = df.reset_index(drop=True).copy()
		scores = self.df["overview"].fillna("").apply(self.sia.polarity_scores)
		self.df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
		self.df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
		self.df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
		self.df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

	def get_sentiment_label(self, compound_score):
		if compound_score < -0.2:
			return "Dark & Intense"
		if compound_score > 0.2:
			return "Light & Uplifting"
		return "Balanced"

	def recommend_by_mood(self, mood, top_n=10):
		if self.df is None:
			raise RuntimeError("SentimentRecommender must be fitted before recommending.")

		scored = self.df.copy()
		scored["mood_label"] = scored["sentiment_compound"].apply(self.get_sentiment_label)
		results = scored[scored["mood_label"] == mood]
		return results.sort_values("vote_average", ascending=False).head(top_n).reset_index(drop=True)


class HybridRecommender:
	"""Hybrid recommender combining content similarity and quality score."""

	def __init__(self, content_weight=0.6, quality_weight=0.4):
		self.content_weight = content_weight
		self.quality_weight = quality_weight
		self.content = ContentRecommender()
		self.sentiment = SentimentRecommender()
		self.df = None

	def fit(self, df):
		self.content.fit(df)
		self.sentiment.fit(df)

		enriched = self.sentiment.df.copy()
		quality_score = (enriched["vote_average"].fillna(0) / 10.0) * (
			enriched["vote_count"].fillna(0).apply(lambda x: log10(x + 1))
		)

		q_min = float(quality_score.min()) if len(quality_score) else 0.0
		q_max = float(quality_score.max()) if len(quality_score) else 0.0
		if q_max > q_min:
			enriched["quality_score"] = (quality_score - q_min) / (q_max - q_min)
		else:
			enriched["quality_score"] = 0.0

		self.df = enriched.reset_index(drop=True)

	def recommend(self, query_text=None, movie_id=None, filters=None, top_n=10):
		if self.df is None:
			raise RuntimeError("HybridRecommender must be fitted before recommending.")
		if movie_id is None and query_text is None:
			raise ValueError("Provide either movie_id or query_text.")

		if movie_id is not None:
			scored = self.content.recommend_by_movie(movie_id=movie_id, top_n=len(self.df))
		else:
			scored = self.content.recommend_by_query(query_text=query_text, top_n=len(self.df))

		scored = scored.merge(
			self.df[["movie_id", "quality_score", "sentiment_compound"]],
			on="movie_id",
			how="left",
		)

		mood = None
		if filters:
			filters_copy = dict(filters)
			mood = filters_copy.pop("mood", None)
			filters_copy.pop("raw_query", None)
			if filters_copy:
				scored = filter_movies(scored, **filters_copy)

		if mood:
			scored["mood_label"] = scored["sentiment_compound"].apply(
				self.sentiment.get_sentiment_label
			)
			scored = scored[scored["mood_label"] == mood]

		scored["similarity_score"] = scored["similarity_score"].fillna(0)
		scored["quality_score"] = scored["quality_score"].fillna(0)
		scored["final_score"] = (
			self.content_weight * scored["similarity_score"]
			+ self.quality_weight * scored["quality_score"]
		)

		scored = scored.sort_values("final_score", ascending=False)
		keep_cols = [
			"movie_id",
			"title",
			"year",
			"genres",
			"vote_average",
			"vote_count",
			"certification",
			"runtime",
			"overview",
			"similarity_score",
			"final_score",
		]
		for col in keep_cols:
			if col not in scored.columns:
				scored[col] = None
		return scored[keep_cols].head(top_n).reset_index(drop=True)


class NaturalLanguageParser:
	"""Parses plain-English requests into structured filter parameters."""

	def __init__(self):
		self.genre_map = {
			"Horror": ["scary", "horror"],
			"Comedy": ["funny", "comedy"],
			"Action": ["action", "exciting"],
			"Romance": ["romantic", "romance"],
			"Animation": ["animated", "cartoon"],
			"Science Fiction": ["sci-fi", "space", "future"],
			"Crime": ["crime", "detective"],
			"Thriller": ["thriller", "suspense"],
			"Family": ["family", "kids"],
			"Drama": ["drama"],
		}

		self.decade_map = {
			(1980, 1989): ["80s", "1980s"],
			(1990, 1999): ["90s", "1990s"],
			(2000, 2009): ["2000s"],
			(2010, 2019): ["2010s"],
			(2020, 2029): ["2020s"],
			(1900, 1979): ["classic", "old"],
			(2020, 2030): ["recent", "new", "latest"],
		}

		self.mood_map = {
			"Dark & Intense": ["dark", "intense", "serious"],
			"Light & Uplifting": ["happy", "uplifting", "feel-good"],
		}

	def parse(self, query):
		q = (query or "").lower()

		genres = []
		for genre, keywords in self.genre_map.items():
			if any(keyword in q for keyword in keywords):
				genres.append(genre)

		year_min = None
		year_max = None
		for (start, end), keywords in self.decade_map.items():
			if any(keyword in q for keyword in keywords):
				year_min, year_max = start, end
				break

		mood = None
		for label, keywords in self.mood_map.items():
			if any(keyword in q for keyword in keywords):
				mood = label
				break

		min_rating = None
		if re.search(r"highly rated|\bgood\b|\bgreat\b|top rated", q):
			min_rating = 7.0

		runtime_min = None
		runtime_max = None
		if "short" in q:
			runtime_max = 100
		if "long" in q:
			runtime_min = 120

		result = {"raw_query": query}
		if genres:
			result["genres"] = genres
		if year_min is not None:
			result["year_min"] = year_min
		if year_max is not None:
			result["year_max"] = year_max
		if mood is not None:
			result["mood"] = mood
		if min_rating is not None:
			result["min_rating"] = min_rating
		if runtime_min is not None:
			result["runtime_min"] = runtime_min
		if runtime_max is not None:
			result["runtime_max"] = runtime_max
		return result


class CineMatchRecommender:
	"""Convenience wrapper that combines parser + hybrid recommender."""

	def __init__(self, content_weight=0.6, quality_weight=0.4):
		self.hybrid = HybridRecommender(content_weight=content_weight, quality_weight=quality_weight)
		self.parser = NaturalLanguageParser()

	def fit(self, df):
		self.hybrid.fit(df)

	def recommend(self, query, movie_id=None, top_n=10):
		filters = self.parser.parse(query) if query is not None else None
		return self.hybrid.recommend(
			query_text=query,
			movie_id=movie_id,
			filters=filters,
			top_n=top_n,
		)


class Watchlist:
	"""Stores saved movies and supports watchlist-based recommendations."""

	def __init__(self):
		self._movies = {}

	def __len__(self):
		return len(self._movies)

	def add(self, movie):
		movie_id = movie.get("movie_id")
		if movie_id is None:
			return False
		self._movies[movie_id] = dict(movie)
		return True

	def remove(self, movie_id):
		return self._movies.pop(movie_id, None) is not None

	def contains(self, movie_id):
		return movie_id in self._movies

	def items(self):
		return list(self._movies.values())

	def recommend_from_watchlist(self, recommender, top_n=10):
		if not self._movies:
			return pd.DataFrame()

		recommendations = []
		for movie_id in self._movies:
			try:
				rec = recommender.recommend(movie_id=movie_id, top_n=top_n)
			except Exception:
				continue
			recommendations.append(rec)

		if not recommendations:
			return pd.DataFrame()

		combined = pd.concat(recommendations, ignore_index=True)
		if "movie_id" in combined.columns:
			combined = combined[~combined["movie_id"].isin(self._movies.keys())]

		if "title" in combined.columns:
			combined = combined.drop_duplicates(subset=["title"])

		sort_col = "final_score" if "final_score" in combined.columns else "vote_average"
		return combined.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
