## Submission Checklist

- [ ] Deployed app URL added under "Live URL Template"
- [ ] `TMDB_API_KEY` configured in local environment and deployment secrets
- [ ] `movies.csv` successfully generated or app verified to build it automatically
- [ ] Natural language search tested
- [ ] Similar movie search tested
- [ ] Browse and filter mode tested
- [ ] Watchlist add/remove and watchlist-based recommendations tested
- [ ] Screenshots or demo video captured if required by instructor
- [ ] Evaluation examples filled in with actual observed results
- [ ] Known limitations reviewed and ready to discuss in submission/demo

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API key and run:

```bash
streamlit run app.py
```

# CineMatch

CineMatch is a movie recommendation application built with Python, TMDB data, NLP, and Streamlit. It combines content-based relevance, sentiment-aware matching, and hybrid scoring to help users discover films through natural language prompts, similarity search, and advanced filters.

## Project Scope

- Data source: TMDB API (`/movie`, `/credits`, `/keywords`, `/release_dates`, `/popular`, `/top_rated`, `/search/movie`)
- Pipeline: Cached API retrieval, enrichment, cleaning, and CSV persistence
- Recommenders: Content-based, sentiment-based, and hybrid ranking
- UI: Streamlit interface with watchlist, filters, explanations, visual insights, and export tools

## Business Value

### Problem
Streaming users often struggle with content overload. Basic browse pages optimize for popularity, not personal intent (for example, "dark 90s crime thriller with strong acting").

### Solution
CineMatch converts user intent into actionable recommendation constraints by combining:

- semantic content similarity (TF-IDF + cosine)
- mood-aware sentiment labels (VADER)
- quality weighting (rating + vote confidence)

This produces recommendations that are both relevant and credible, improving discovery speed and user confidence.

### Expected Outcomes

- Faster time-to-first-good-option for users
- Higher engagement via watchlist and follow-up recommendations
- Better personalization than popularity-only ranking

## Target Users

- Casual viewers who want quick guidance without deep filtering
- Film enthusiasts who need precise filter controls (genre logic, certification, runtime, language)
- Users who prefer natural-language discovery over form-based search
- Users who want explainable recommendations rather than black-box results

## Market Positioning

CineMatch positions as an explainable recommendation companion between:

- broad catalog apps that provide weak personalization
- niche cinephile tools that are powerful but not user-friendly

Differentiators:

- Multi-modal discovery in one interface (NL query, similar movie lookup, browse/filter)
- Recommendation transparency with similarity explanations and confidence badges
- User-controlled ranking and filter logic (ALL/ANY/EXACT genre semantics)

## Technical Summary

### Core Modules

- `tmdb_pipeline.py`: API integration, caching, dataset build/load/save/filter
- `recommender.py`: ContentRecommender, SentimentRecommender, HybridRecommender, parser, watchlist logic
- `app.py`: Streamlit UX, filtering workflows, recommendation views, watchlist page

### Recommendation Approaches

- Content-based filtering using enriched text + metadata features
- Sentiment-based matching using overview polarity profiles
- Hybrid ranking combining similarity with quality score

## Deployment Checklist (Streamlit Cloud)

1. Push project to GitHub repository.
2. Confirm files exist in repo root:
	- `app.py`
	- `requirements.txt`
	- `tmdb_pipeline.py`
	- `recommender.py`
3. In Streamlit Cloud, create a new app from the GitHub repo.
4. Set main file path to `app.py`.
5. Add secret/environment variable:
	- `TMDB_API_KEY = "your_api_key_here"`
6. Deploy and verify startup logs show successful dependency install.
7. Run smoke tests in deployed app:
	- Natural language search
	- Similar movie search
	- Browse/filter mode
	- Watchlist add/remove
8. Confirm dataset behavior:
	- If `movies.csv` exists, app loads cached dataset
	- If not, app builds dataset from TMDB
9. Validate UI and functionality on desktop and mobile widths.

## Live URL Template

- Deployed App URL: `<PASTE_STREAMLIT_CLOUD_URL_HERE>`
- Backup / Demo Video URL (optional): `<PASTE_OPTIONAL_DEMO_URL_HERE>`

## Evaluation

### Recommendation Quality Examples

Use this section to document final test evidence before submission.

1. Query: "dark thriller from the 90s with a great story"
	- Expected pattern: mostly Crime/Thriller titles, 1990s-heavy distribution, above-average ratings
2. Similar to seed: select a known crime thriller
	- Expected pattern: overlapping keywords/directors/cast, meaningful explanation terms
3. Browse with strict filters (runtime + certification + votes)
	- Expected pattern: reduced but higher-confidence result set

### Suggested Evaluation Metrics

- Relevance@10: manual judgment of top-10 result relevance to query intent
- Diversity@10: number of distinct genres/languages in top-10
- Explanation quality: whether top terms plausibly justify similarity
- Stability: repeated runs with same input produce consistent rankings

### Known Limitations

- Cold-start for unseen user preferences (no persistent user profile history yet)
- TMDB metadata variability can affect keyword/certification completeness
- Sentiment model uses lightweight VADER, which may miss nuanced tone in complex plots
- No true collaborative filtering from user-user interactions yet
- Data freshness depends on rebuild cadence and cache policy

## Future Improvements

- Add collaborative filtering from explicit ratings/watch behavior
- Add A/B testing for ranking weight calibration
- Add user accounts for persistent preferences across sessions
- Add multilingual query understanding and translation layer


