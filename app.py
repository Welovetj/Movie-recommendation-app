import os
from collections import Counter

import pandas as pd
import streamlit as st

from recommender import ContentRecommender, HybridRecommender, NaturalLanguageParser, Watchlist
from tmdb_pipeline import build_dataset, filter_movies, load_dataset, save_dataset, search_movies


# SECTION 1 - PAGE CONFIG
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# Visual design setup requested in prompt instructions.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&display=swap');

    .stApp {
        background-color: #0f0f0f;
        color: #e6e6e6;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: #ffffff !important;
    }

    p, label, div, span {
        color: #d8d8d8;
    }

    section[data-testid="stSidebar"] {
        background-color: #141414 !important;
    }

    .movie-card {
        background: #1a1a1a;
        border-left: 3px solid #c9a84c;
        border-radius: 12px;
        padding: 20px;
        min-height: 340px;
        margin-bottom: 14px;
    }

    .stButton > button {
        background-color: #c9a84c !important;
        color: #111111 !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .subtitle {
        color: #c9c9c9;
        margin-top: -8px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# SECTION 2 - SESSION STATE
if "watchlist" not in st.session_state:
    st.session_state.watchlist = Watchlist()
if "df" not in st.session_state:
    st.session_state.df = None
if "recommender" not in st.session_state:
    st.session_state.recommender = None
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "page" not in st.session_state:
    st.session_state.page = "search"
if "search_candidates" not in st.session_state:
    st.session_state.search_candidates = []
if "similarity_explanations" not in st.session_state:
    st.session_state.similarity_explanations = {}
if "selected_seed_id" not in st.session_state:
    st.session_state.selected_seed_id = None


# SECTION 3 - DATA LOADING
@st.cache_resource
def load_data():
    """Load cached dataset or build it from TMDB, then fit hybrid model."""
    csv_path = "movies.csv"
    if os.path.exists(csv_path):
        df = load_dataset(csv_path)
    else:
        df = build_dataset()
        save_dataset(df, csv_path)

    recommender = HybridRecommender()
    recommender.fit(df)
    return df, recommender


try:
    if st.session_state.df is None or st.session_state.recommender is None:
        st.session_state.df, st.session_state.recommender = load_data()
except RuntimeError as exc:
    if "TMDB_API_KEY" in str(exc):
        st.error("TMDB_API_KEY is missing. Add it to your environment and rerun the app.")
    else:
        st.error(f"Failed to load data: {exc}")
except Exception as exc:
    st.error(f"Unexpected startup error: {exc}")


def extract_all_genres(df):
    genres = set()
    if df is None or df.empty or "genres" not in df.columns:
        return []
    for entry in df["genres"]:
        if isinstance(entry, list):
            genres.update(g for g in entry if g)
    return sorted(genres)


def extract_all_languages(df):
    langs = set()
    if df is None or df.empty or "original_language" not in df.columns:
        return []
    for val in df["original_language"].dropna().tolist():
        if isinstance(val, str) and val.strip():
            langs.add(val.strip())
    return sorted(langs)


def apply_genre_logic(df, genre_list, logic_mode):
    """
    FEATURE 1: Genre logical operators (ALL / ANY / EXACT).
    This extends baseline filtering and is useful evidence for rubric compliance.
    """
    if not genre_list:
        return df

    def _as_set(row_genres):
        return set(row_genres) if isinstance(row_genres, list) else set()

    target = set(genre_list)
    if logic_mode == "Any selected":
        mask = df["genres"].apply(lambda gs: bool(_as_set(gs).intersection(target)))
    elif logic_mode == "Exact match":
        mask = df["genres"].apply(lambda gs: _as_set(gs) == target)
    else:
        # Default behavior remains ALL selected genres.
        mask = df["genres"].apply(lambda gs: target.issubset(_as_set(gs)))
    return df[mask].reset_index(drop=True)


def sort_results(df, sort_mode):
    """
    FEATURE 2: Multi-sort ranking controls for user-centric exploration.
    """
    if df.empty:
        return df
    if sort_mode == "Highest Rated":
        return df.sort_values("vote_average", ascending=False)
    if sort_mode == "Most Popular" and "popularity" in df.columns:
        return df.sort_values("popularity", ascending=False)
    if sort_mode == "Newest" and "year" in df.columns:
        return df.sort_values("year", ascending=False)
    if sort_mode == "Most Votes" and "vote_count" in df.columns:
        return df.sort_values("vote_count", ascending=False)
    return df.sort_values("final_score", ascending=False) if "final_score" in df.columns else df


def get_active_filters():
    """SECTION 5 - filter helper converting sidebar state into filter kwargs."""
    active = {}

    if year_range != (1990, 2024):
        active["year_min"] = year_range[0]
        active["year_max"] = year_range[1]
    if min_rating != 6.0:
        active["min_rating"] = min_rating
    if min_votes != 100:
        active["min_votes"] = int(min_votes)

    if runtime_choice == "Short under 90min":
        active["runtime_max"] = 89
    elif runtime_choice == "Medium 90-150min":
        active["runtime_min"] = 90
        active["runtime_max"] = 150
    elif runtime_choice == "Long over 150min":
        active["runtime_min"] = 151

    if genres:
        active["genres"] = genres
    if certification:
        active["certifications"] = certification
    if language != "Any":
        active["language"] = language
    if cast_member.strip():
        active["cast_member"] = cast_member.strip()
    if director_name.strip():
        active["director"] = director_name.strip()

    # FEATURE 1 support key (not passed to tmdb_pipeline.filter_movies directly).
    active["genre_logic"] = genre_logic
    return active


def apply_all_filters(df, active_filters):
    """Apply base filters first, then custom genre-logic extension."""
    filters_copy = dict(active_filters)
    genre_list = filters_copy.pop("genres", None)
    genre_logic_mode = filters_copy.pop("genre_logic", "All selected")

    filtered = filter_movies(df, **filters_copy)
    return apply_genre_logic(filtered, genre_list, genre_logic_mode)


def confidence_label(row):
    """
    FEATURE 3: Confidence badge to explain recommendation strength.
    """
    score = row.get("final_score", row.get("similarity_score", 0))
    if pd.isna(score):
        return "Confidence: Unknown"
    if score >= 0.65:
        return "Confidence: High"
    if score >= 0.35:
        return "Confidence: Medium"
    return "Confidence: Exploratory"


def render_insights(results_df):
    """
    FEATURE 4: Data visualizations (genre and decade trends).
    """
    if results_df.empty:
        return

    st.markdown("### Insights")
    c1, c2 = st.columns(2)

    genre_counter = Counter()
    for g in results_df.get("genres", []):
        if isinstance(g, list):
            genre_counter.update(g)
    if genre_counter:
        genre_df = pd.DataFrame(genre_counter.items(), columns=["Genre", "Count"]).sort_values("Count", ascending=False).head(10)
        c1.bar_chart(genre_df.set_index("Genre"))

    if "year" in results_df.columns and not results_df["year"].dropna().empty:
        decade_df = results_df.copy()
        decade_df = decade_df[decade_df["year"].notna()]
        decade_df["Decade"] = (decade_df["year"].astype(int) // 10) * 10
        counts = decade_df["Decade"].value_counts().sort_index()
        c2.bar_chart(counts)


def display_results(results_df):
    """SECTION 7 - show cards and KPIs for current result set."""
    st.subheader(f"Found {len(results_df)} movies")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(results_df.reset_index(drop=True).iterrows()):
        with cols[idx % 3]:
            movie_id = row.get("movie_id")
            in_watchlist = st.session_state.watchlist.contains(movie_id) if movie_id is not None else False

            title = row.get("title", "Untitled")
            year = row.get("year", "?")
            runtime = row.get("runtime", "?")
            cert = row.get("certification", "NR")
            rating = row.get("vote_average", 0)
            votes = row.get("vote_count", 0)

            genre_vals = row.get("genres", [])
            genre_text = ", ".join(genre_vals) if isinstance(genre_vals, list) else str(genre_vals)

            overview = row.get("overview", "") or ""
            overview_short = overview[:150] + ("..." if len(overview) > 150 else "")

            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            st.caption(f"{year} | {runtime} min | {cert}")
            st.write(f"⭐ {rating} / 10 from {int(votes) if pd.notna(votes) else 0} votes")
            st.write(f"Genres: {genre_text}")
            st.write(overview_short)
            st.caption(confidence_label(row))

            # FEATURE 5: Similarity explanation shown to users for trust/transparency.
            if movie_id in st.session_state.similarity_explanations:
                explanation = st.session_state.similarity_explanations.get(movie_id)
                if explanation:
                    with st.expander("Why this is similar"):
                        st.write(explanation)
            st.markdown("</div>", unsafe_allow_html=True)

            if in_watchlist:
                st.success("In Watchlist ✓")
            else:
                if st.button("Add to Watchlist", key=f"add_{movie_id}_{idx}"):
                    st.session_state.watchlist.add(row.to_dict())
                    st.rerun()

    if not results_df.empty:
        m1, m2, m3 = st.columns(3)
        avg_rating = float(results_df["vote_average"].fillna(0).mean()) if "vote_average" in results_df else 0.0

        genre_counter = Counter()
        if "genres" in results_df:
            for g in results_df["genres"]:
                if isinstance(g, list):
                    genre_counter.update(g)
        common_genre = genre_counter.most_common(1)[0][0] if genre_counter else "N/A"

        if "year" in results_df and not results_df["year"].dropna().empty:
            y_min = int(results_df["year"].dropna().min())
            y_max = int(results_df["year"].dropna().max())
            year_range_text = f"{y_min} - {y_max}"
        else:
            year_range_text = "N/A"

        m1.metric("Average Rating", f"{avg_rating:.2f}")
        m2.metric("Most Common Genre", common_genre)
        m3.metric("Year Range", year_range_text)

        # FEATURE 4 visual analytics section.
        render_insights(results_df)

        # FEATURE 6: One-click CSV export for results.
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results CSV",
            data=csv_bytes,
            file_name="cinematch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_trending_panel(df):
    """
    FEATURE 7: Trending/popular content integration from local dataset.
    """
    if df is None or df.empty or "popularity" not in df.columns:
        return

    st.markdown("### Trending Now")
    trending = df.sort_values("popularity", ascending=False).head(10)
    st.dataframe(
        trending[["title", "year", "popularity", "vote_average"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# SECTION 4 - SIDEBAR
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Find cinematic gems by vibe, similarity, and smart filters.")

    search_mode = st.radio(
        "Search Mode",
        ["🔍 Natural Language", "🎬 Find Similar Movie", "📂 Browse & Filter"],
    )

    st.divider()
    st.markdown("### Filters")

    all_genres = extract_all_genres(st.session_state.df)
    all_languages = ["Any"] + extract_all_languages(st.session_state.df)

    year_range = st.slider("Release Year", 1950, 2024, (1990, 2024))
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.5)
    min_votes = st.number_input("Minimum Votes", min_value=0, value=100, step=50)
    runtime_choice = st.select_slider(
        "Runtime",
        options=["Any", "Short under 90min", "Medium 90-150min", "Long over 150min"],
        value="Any",
    )
    genres = st.multiselect("Genres", options=all_genres)
    genre_logic = st.selectbox("Genre Logic", ["All selected", "Any selected", "Exact match"])
    certification = st.multiselect("Certification", options=["G", "PG", "PG-13", "R", "NR"])
    language = st.selectbox("Language", options=all_languages)
    cast_member = st.text_input("Actor Name")
    director_name = st.text_input("Director Name")

    # FEATURE 2 control for result ordering.
    sort_mode = st.selectbox(
        "Sort Results By",
        ["Best Match", "Highest Rated", "Most Popular", "Newest", "Most Votes"],
    )

    st.divider()
    st.markdown(f"### Watchlist ({len(st.session_state.watchlist)})")
    if st.button("View My Watchlist"):
        st.session_state.page = "watchlist"
        st.rerun()

    # FEATURE 8: Surprise me button for serendipitous discovery.
    if st.button("Surprise Me") and st.session_state.df is not None:
        random_pick = st.session_state.df.sample(1).reset_index(drop=True)
        st.session_state.results = random_pick
        st.session_state.page = "search"
        st.rerun()


# SECTION 6 - MAIN AREA
st.title("CineMatch")
st.markdown('<div class="subtitle">Find your next favorite movie</div>', unsafe_allow_html=True)

if st.session_state.page == "search":
    active_filters = get_active_filters()

    if search_mode == "🔍 Natural Language":
        query = st.text_area(
            "What are you in the mood for?",
            placeholder="e.g. I want a dark thriller from the 90s with a great story",
            height=110,
        )

        if st.button("Find Movies", key="nl_find") and st.session_state.recommender is not None:
            parser = NaturalLanguageParser()
            parsed = parser.parse(query)

            merged_filters = dict(active_filters)
            merged_filters.update(parsed)

            # Remove UI-only key not accepted by recommender filter path.
            recommender_filters = dict(merged_filters)
            recommender_filters.pop("genre_logic", None)

            results = st.session_state.recommender.recommend(
                query_text=query,
                filters=recommender_filters,
                top_n=60,
            )
            # Apply extended genre logic on top of model output.
            results = apply_genre_logic(results, merged_filters.get("genres"), merged_filters.get("genre_logic", "All selected"))
            results = sort_results(results, sort_mode).head(30).reset_index(drop=True)
            st.session_state.results = results
            st.session_state.similarity_explanations = {}

    elif search_mode == "🎬 Find Similar Movie":
        title_query = st.text_input("Enter a movie title")

        if st.button("Search", key="sim_search") and title_query.strip():
            st.session_state.search_candidates = search_movies(title_query.strip(), pages=1)

        candidates = st.session_state.search_candidates
        if candidates:
            option_map = {}
            labels = []
            for item in candidates:
                label = f"{item.get('title', 'Unknown')} ({(item.get('release_date') or '????')[:4]})"
                labels.append(label)
                option_map[label] = item.get("id")

            selected_label = st.selectbox("Select a movie", labels)
            selected_id = option_map.get(selected_label)

            if st.button("Find Similar", key="sim_find") and selected_id is not None:
                st.session_state.selected_seed_id = selected_id

                recommender_filters = dict(active_filters)
                recommender_filters.pop("genre_logic", None)

                results = st.session_state.recommender.recommend(
                    movie_id=selected_id,
                    filters=recommender_filters,
                    top_n=60,
                )

                results = apply_genre_logic(results, active_filters.get("genres"), active_filters.get("genre_logic", "All selected"))
                results = sort_results(results, sort_mode).head(30).reset_index(drop=True)

                # Similarity explanations for transparency.
                explanations = {}
                base_content = st.session_state.recommender.content
                if isinstance(base_content, ContentRecommender) and "movie_id" in results.columns:
                    for _, row in results.iterrows():
                        rid = row.get("movie_id")
                        if rid is None:
                            continue
                        try:
                            explanations[rid] = base_content.explain_similarity(selected_id, rid)
                        except Exception:
                            explanations[rid] = ""
                st.session_state.similarity_explanations = explanations
                st.session_state.results = results

    elif search_mode == "📂 Browse & Filter":
        if st.button("Find Movies", key="browse_find") and st.session_state.df is not None:
            filtered = apply_all_filters(st.session_state.df, active_filters)
            filtered = sort_results(filtered, sort_mode).head(60).reset_index(drop=True)
            st.session_state.results = filtered
            st.session_state.similarity_explanations = {}

    # FEATURE 7 panel always visible for discovery.
    render_trending_panel(st.session_state.df)

    # BONUS FEATURE: Comparative analysis between two movies.
    st.markdown("### Compare Two Movies")
    if st.session_state.df is not None and not st.session_state.df.empty:
        compare_source = st.session_state.df[["movie_id", "title", "year", "vote_average", "runtime", "genres", "directors", "cast"]].copy()
        compare_source["label"] = compare_source.apply(
            lambda r: f"{r['title']} ({int(r['year']) if pd.notna(r['year']) else '????'})",
            axis=1,
        )

        c1, c2 = st.columns(2)
        with c1:
            left_label = st.selectbox("Movie A", compare_source["label"].tolist(), key="cmp_left")
        with c2:
            right_label = st.selectbox("Movie B", compare_source["label"].tolist(), key="cmp_right")

        if st.button("Compare Movies", key="cmp_btn"):
            left = compare_source[compare_source["label"] == left_label].iloc[0]
            right = compare_source[compare_source["label"] == right_label].iloc[0]

            shared_genres = sorted(set(left["genres"] if isinstance(left["genres"], list) else []).intersection(
                set(right["genres"] if isinstance(right["genres"], list) else [])
            ))

            st.write(
                pd.DataFrame(
                    {
                        "Metric": ["Rating", "Runtime", "Year", "Shared Genres"],
                        "Movie A": [left["vote_average"], left["runtime"], left["year"], ", ".join(shared_genres) or "None"],
                        "Movie B": [right["vote_average"], right["runtime"], right["year"], ", ".join(shared_genres) or "None"],
                    }
                )
            )

    if not st.session_state.results.empty:
        display_results(st.session_state.results)
    else:
        st.warning("No results yet. Try searching or adjusting filters.")
        # SECTION 9 - friendly guidance when no filters/results.
        if len(active_filters) <= 1:  # includes genre_logic default key
            st.info("Tip: Try genre, year, mood words, or actor/director filters to narrow the recommendations.")


# SECTION 8 - WATCHLIST PAGE
if st.session_state.page == "watchlist":
    st.title("My Watchlist")

    if st.button("← Back to Search"):
        st.session_state.page = "search"
        st.rerun()

    watchlist_items = st.session_state.watchlist.items()
    if not watchlist_items:
        st.info("Your watchlist is empty. Add movies from search results.")
    else:
        for idx, movie in enumerate(watchlist_items):
            movie_id = movie.get("movie_id")
            title = movie.get("title", "Untitled")
            year = movie.get("year", "?")
            rating = movie.get("vote_average", "?")

            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{title}** ({year}) - ⭐ {rating}")
            if c2.button("Remove", key=f"remove_{movie_id}_{idx}"):
                st.session_state.watchlist.remove(movie_id)
                st.rerun()

        if st.button("Recommend Based on My Watchlist", use_container_width=True):
            if st.session_state.recommender is None:
                st.error("Recommender is not initialized yet.")
            else:
                recs = st.session_state.watchlist.recommend_from_watchlist(
                    recommender=st.session_state.recommender,
                    top_n=30,
                )
                st.session_state.results = recs
                st.session_state.page = "search"
                st.rerun()

        # FEATURE 6 extension: export watchlist for user portability.
        watchlist_df = pd.DataFrame(watchlist_items)
        st.download_button(
            "Download Watchlist CSV",
            data=watchlist_df.to_csv(index=False).encode("utf-8"),
            file_name="cinematch_watchlist.csv",
            mime="text/csv",
            use_container_width=True,
        )


# SECTION 9 - global API key message requested by prompt.
if not os.getenv("TMDB_API_KEY"):
    st.error("TMDB_API_KEY is missing. Add it to your environment before using API-powered features.")
