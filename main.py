"""
FastAPI Backend Application for Movie Recommendation System.

This module exposes asynchronous REST API endpoints for:
1. Health checks and server status.
2. TMDB movie catalog discovery (Popular, Trending, Top Rated, Upcoming, Now Playing).
3. Search functionality (autocomplete, keyword search, details lookup).
4. Content-based movie recommendations using TF-IDF NLP sparse matrix and Cosine Similarity.
5. Combined search recommendation bundle (TMDB details + TF-IDF similarity + Genre matches).
"""

import os
import sys
import pickle
import asyncio
import requests
from typing import Optional, List, Dict, Any, Tuple

import numpy
import numpy as np
import pandas as pd

# NumPy compatibility mapping for models pickled across numpy 2.x and 1.x versions
try:
    if not hasattr(numpy, '_core'):
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.numeric'] = getattr(numpy.core, 'numeric', numpy.core)
        sys.modules['numpy._core.multiarray'] = getattr(numpy.core, 'multiarray', numpy.core)
        sys.modules['numpy._core.umath'] = getattr(numpy.core, 'umath', numpy.core)
        sys.modules['numpy._core.fromnumeric'] = getattr(numpy.core, 'fromnumeric', numpy.core)
        sys.modules['numpy._core.defchararray'] = getattr(numpy.core, 'defchararray', numpy.core)
        sys.modules['numpy._core.strings'] = getattr(numpy.core, 'defchararray', numpy.core)
except Exception:
    pass

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ==============================================================================
# ENVIRONMENT & API CONFIGURATION
# Load environment variables from .env file or set default fallback values
# ==============================================================================
load_dotenv()
TMDB_API_KEY = (
    os.getenv("TMDB_API_KEY")
    or os.getenv("TMBDI_API_KEY")
    or "f050e150197472ead1ad59802b5933ff"
)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# ==============================================================================
# FASTAPI INSTANCE INITIALIZATION
# Configure application metadata and enable Cross-Origin Resource Sharing (CORS)
# ==============================================================================
app = FastAPI(
    title="Movie Recommender API",
    description="High-performance asynchronous API for movie recommendations, TF-IDF NLP similarity, and TMDB integration.",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from local Streamlit UI and remote web hosts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


#image url


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Create robust HTTP session with retry adapter
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

def _tmdb_get_sync(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous TMDB GET using robust requests session with automatic retries.
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        r = _session.get(f"{TMDB_BASE}{path}", params=q, timeout=10)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"TMDB error {r.status_code}: {r.text}")

    return r.json()


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET wrapped in asyncio.to_thread for non-blocking execution.
    """
    return await asyncio.to_thread(_tmdb_get_sync, path, params)


async def tmdb_cards_from_results(results: List[dict],
                                  limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            ))
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX safely.
    """
    title_to_idx: Dict[str, int] = {}

    if hasattr(indices, 'items'):
        for k, v in indices.items():
            if k is not None and isinstance(k, str) and k.strip():
                title_to_idx[_norm_title(k)] = int(v)

    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500,
                            detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    # Fallback 1: Try toggling "the " prefix
    if key.startswith("the "):
        key_no_the = key[4:].strip()
        if key_no_the in TITLE_TO_IDX:
            return int(TITLE_TO_IDX[key_no_the])
    else:
        key_with_the = f"the {key}"
        if key_with_the in TITLE_TO_IDX:
            return int(TITLE_TO_IDX[key_with_the])

    # Fallback 2: Substring matching
    for t_key, t_idx in TITLE_TO_IDX.items():
        if isinstance(t_key, str) and (key in t_key or t_key in key):
            return int(t_idx)

    raise HTTPException(status_code=404,
                        detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str,
                           top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500,
                            detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix (usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sanity
    if df is None or "title" not in df.columns:
        raise RuntimeError(
            "df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
        category: str = Query("popular"),
        limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []),
                                                 limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {
            "language": "en-US",
            "page": 1
        })
        return await tmdb_cards_from_results(data.get("results", []),
                                             limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
        query: str = Query(..., min_length=1),
        page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
        tmdb_id: int = Query(...),
        limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []),
                                          limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
        title: str = Query(..., min_length=1),
        top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
        query: str = Query(..., min_length=1),
        tfidf_top_n: int = Query(12, ge=1, le=30),
        genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404,
                            detail=f"No TMDB movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    cards = await asyncio.gather(*[attach_tmdb_card_by_title(t) for t, _ in recs], return_exceptions=True)
    for (title, score), card in zip(recs, cards):
        valid_card = card if isinstance(card, TMDBMovieCard) else None
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=valid_card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(discover.get("results", []),
                                              limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
