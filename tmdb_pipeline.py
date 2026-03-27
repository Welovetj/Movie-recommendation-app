import os
import json
import time
import hashlib
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
CACHE_DIR = Path("cache")
REQUEST_DELAY = 0.26


def tmdb_get(endpoint, params=None):
    """Makes a GET request to BASE_URL + endpoint with file caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY environment variable is not set.")

    params = dict(params or {})
    cache_key_data = {
        "endpoint": endpoint,
        "params": sorted(params.items(), key=lambda item: item[0]),
    }
    cache_key = json.dumps(cache_key_data, separators=(",", ":"), ensure_ascii=True)
    cache_name = hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".json"
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    params["api_key"] = api_key
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    time.sleep(REQUEST_DELAY)
    return data


def fetch_movie_details(movie_id):
    """Calls /movie/{movie_id} and returns full JSON."""
    return tmdb_get(f"/movie/{movie_id}")


def fetch_movie_credits(movie_id):
    """Calls /movie/{movie_id}/credits and returns cast and crew JSON."""
    return tmdb_get(f"/movie/{movie_id}/credits")


def fetch_movie_keywords(movie_id):
    """Calls /movie/{movie_id}/keywords and returns keywords JSON."""
    return tmdb_get(f"/movie/{movie_id}/keywords")


def fetch_movie_release_info(movie_id):
    """Calls /movie/{movie_id}/release_dates for US certification extraction."""
    return tmdb_get(f"/movie/{movie_id}/release_dates")


def fetch_popular_movies(pages=5):
    """Calls /movie/popular for each page and returns one flat list."""
    movies = []
    for page in range(1, pages + 1):
        data = tmdb_get("/movie/popular", {"page": page})
        movies.extend(data.get("results", []))
    return movies


def fetch_top_rated_movies(pages=5):
    """Calls /movie/top_rated for each page and returns one flat list."""
    movies = []
    for page in range(1, pages + 1):
        data = tmdb_get("/movie/top_rated", {"page": page})
        movies.extend(data.get("results", []))
    return movies


def search_movies(query, pages=1):
    """Calls /search/movie with query and returns one flat list."""
    movies = []
    for page in range(1, pages + 1):
        data = tmdb_get("/search/movie", {"query": query, "page": page})
        movies.extend(data.get("results", []))
    return movies


def enrich_movie(movie_id):
    """Calls all fetch helpers and returns one flattened movie dictionary."""
    try:
        details = fetch_movie_details(movie_id)
        credits = fetch_movie_credits(movie_id)
        keywords_payload = fetch_movie_keywords(movie_id)
        release_info = fetch_movie_release_info(movie_id)
    except requests.HTTPError as exc:
        print(f"Warning: HTTP error for movie_id={movie_id}: {exc}")
        return None

    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
    keywords = [k.get("name") for k in keywords_payload.get("keywords", []) if k.get("name")]
    cast = [c.get("name") for c in credits.get("cast", []) if c.get("name")][:5]
    directors = [
        c.get("name")
        for c in credits.get("crew", [])
        if c.get("job") == "Director" and c.get("name")
    ]
    production_companies = [
        c.get("name") for c in details.get("production_companies", []) if c.get("name")
    ]
    spoken_languages = [
        l.get("english_name") or l.get("name")
        for l in details.get("spoken_languages", [])
        if (l.get("english_name") or l.get("name"))
    ]

    certification = "NR"
    for country in release_info.get("results", []):
        if country.get("iso_3166_1") != "US":
            continue
        for rel in country.get("release_dates", []):
            cert = (rel.get("certification") or "").strip()
            if cert:
                certification = cert
                break
        if certification != "NR":
            break

    release_date = details.get("release_date")
    year = None
    if isinstance(release_date, str) and len(release_date) >= 4 and release_date[:4].isdigit():
        year = int(release_date[:4])

    text_parts = [
        details.get("overview") or "",
        details.get("tagline") or "",
        " ".join(keywords),
        " ".join(genres),
    ]
    text_features = " ".join(part.strip() for part in text_parts if part and part.strip())

    return {
        "movie_id": movie_id,
        "imdb_id": details.get("imdb_id"),
        "title": details.get("title"),
        "original_title": details.get("original_title"),
        "release_date": release_date,
        "year": year,
        "runtime": details.get("runtime"),
        "status": details.get("status"),
        "overview": details.get("overview", ""),
        "tagline": details.get("tagline", ""),
        "genres": genres,
        "keywords": keywords,
        "original_language": details.get("original_language"),
        "spoken_languages": spoken_languages,
        "cast": cast,
        "directors": directors,
        "production_companies": production_companies,
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),
        "budget": details.get("budget"),
        "revenue": details.get("revenue"),
        "certification": certification,
        "text_features": text_features,
    }


def build_dataset(popular_pages=5, top_rated_pages=5):
    """Fetches movie lists, deduplicates IDs, enriches rows, and returns a DataFrame."""
    popular = fetch_popular_movies(pages=popular_pages)
    top_rated = fetch_top_rated_movies(pages=top_rated_pages)

    unique_ids = []
    seen_ids = set()
    for movie in popular + top_rated:
        movie_id = movie.get("id")
        if movie_id is None or movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)
        unique_ids.append(movie_id)

    rows = []
    for movie_id in unique_ids:
        enriched = enrich_movie(movie_id)
        if enriched is not None:
            rows.append(enriched)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["title", "overview"])
    df = df[df["overview"].astype(str).str.len() >= 20]
    return df.reset_index(drop=True)


def save_dataset(df, path="movies.csv"):
    """Saves DataFrame to CSV, joining list columns with pipes."""
    df_out = df.copy()
    for col in df_out.columns:
        df_out[col] = df_out[col].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    df_out.to_csv(path, index=False, encoding="utf-8")


def load_dataset(path="movies.csv"):
    """Loads CSV and converts known pipe-separated columns back to lists."""
    df = pd.read_csv(path, encoding="utf-8")
    list_cols = ["genres", "keywords", "spoken_languages", "cast", "directors", "production_companies"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.split("|") if isinstance(x, str) and x.strip() else [])
    return df


def filter_movies(df, year_min=None, year_max=None, min_rating=None,
                  min_votes=None, runtime_min=None, runtime_max=None,
                  certifications=None, genres=None, language=None,
                  cast_member=None, director=None):
    """Applies optional filters and returns filtered DataFrame with reset index."""
    mask = pd.Series(True, index=df.index)

    if year_min is not None:
        mask &= df["year"] >= year_min
    if year_max is not None:
        mask &= df["year"] <= year_max
    if min_rating is not None:
        mask &= df["vote_average"] >= min_rating
    if min_votes is not None:
        mask &= df["vote_count"] >= min_votes
    if runtime_min is not None:
        mask &= df["runtime"] >= runtime_min
    if runtime_max is not None:
        mask &= df["runtime"] <= runtime_max
    if certifications is not None:
        mask &= df["certification"].isin(certifications)
    if language is not None:
        mask &= df["original_language"] == language

    if genres is not None:
        for genre in genres:
            mask &= df["genres"].apply(
                lambda gs: isinstance(gs, list) and genre in gs
            )

    if cast_member is not None:
        q = cast_member.lower()
        mask &= df["cast"].apply(
            lambda names: isinstance(names, list)
            and any(q in str(name).lower() for name in names)
        )

    if director is not None:
        q = director.lower()
        mask &= df["directors"].apply(
            lambda names: isinstance(names, list)
            and any(q in str(name).lower() for name in names)
        )

    return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    save_dataset(df)
    print(df[["title", "year", "vote_average", "genres"]].head(10))
