# 🎬 Movie Recommendation System & API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**An end-to-end, high-performance Movie Recommendation Application powered by TF-IDF Content-Based NLP, Cosine Similarity, FastAPI, and Streamlit with TMDB API integration.**

🌐 **Live Web Application**: [https://movie-recommendation-system-nimu.onrender.com/](https://movie-recommendation-system-nimu.onrender.com/)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [How It Works — System Architecture](#-how-it-works--system-architecture)
- [NLP & Machine Learning Pipeline](#-nlp--machine-learning-pipeline)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Setup](#️-prerequisites--setup)
- [Running the Application](#-running-the-application)
- [API Endpoints Summary](#-api-endpoints-summary)
- [Dataset Information](#-dataset-information)
- [Technology Stack](#️-technology-stack)
- [Performance & Scalability](#-performance--scalability)
- [Screenshots & UI Components](#-screenshots--ui-components)
- [Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

The **Movie Recommendation System** is a production-grade, full-stack machine learning application that provides intelligent content-based movie recommendations. It processes over **16,800+ movies** from a rich CSV dataset, builds a precomputed TF-IDF sparse matrix from movie metadata (plot overviews, genres, taglines, keywords), and uses **cosine similarity** to find movies that are semantically closest to any given title.

The system is architected into two layers:
1. **FastAPI Backend (`main.py`)** — A high-performance async REST API serving movie data, recommendations, and TMDB metadata.
2. **Streamlit Frontend (`app.py`)** — An interactive, modern UI for browsing movies, searching by title, and exploring recommendations.

This project demonstrates a practical end-to-end machine learning deployment pipeline: from raw data ingestion and feature engineering, all the way to a deployed, publicly accessible web application on **Render**.

---

## 🌟 Key Features

### 🧠 Intelligent Recommendation Engine
- **TF-IDF Content Vectorization**: Precomputed TF-IDF sparse matrix built from movie `overview`, `genres`, `tagline`, and `keywords` fields across 16,800+ movies.
- **Cosine Similarity Matching**: Lightning-fast similarity scoring between all movie pairs in the corpus — returns top-N most similar movies in milliseconds.
- **Hybrid Metadata Fusion**: Combines content signals (plot NLP) with structured signals (genre, cast) for higher-quality recommendations.

### ⚡ High-Performance FastAPI Backend
- **Asynchronous Architecture**: Fully async request handling using `asyncio` and `httpx` for non-blocking I/O.
- **TMDB Integration**: Automatically fetches high-resolution movie posters, backdrops, trailers, and live metadata from The Movie Database (TMDB) API.
- **Auto-Detect Backend URL**: The Streamlit frontend dynamically detects whether the local FastAPI server or the production Render deployment is running.
- **Swagger Docs**: Full interactive OpenAPI documentation available at `/docs`.

### 🎨 Streamlit Web UI
- **Poster Grid Layout**: Responsive movie card grid with hover effects and live TMDB posters.
- **Smart Search Dropdown**: Autocomplete-style title search dropdown powered by the movie index.
- **Genre Discovery**: Browse and filter movies by genre categories.
- **Movie Detail Pages**: Expanded view with full plot overview, cast info, release year, ratings, and trailer links.
- **Home Feed**: Curated home page showing Popular, Trending, Top Rated, Upcoming, and Now Playing sections.

---

## 🏗 How It Works — System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT BROWSER                               │
│                    (Streamlit Frontend - app.py)                      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ HTTP REST Calls
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (main.py)                        │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │  /home      │   │  /recommend/tfidf│   │  /movie/search       │  │
│  │  /health    │   │  Cosine Similarity│   │  Recommendation      │  │
│  │  /tmdb/search│  │  Matrix Lookup   │   │  Bundle Endpoint     │  │
│  └─────────────┘   └──────────────────┘   └──────────────────────┘  │
│                              │                                       │
│           ┌──────────────────┴──────────────────┐                   │
│           ▼                                     ▼                   │
│  ┌─────────────────────┐             ┌──────────────────────────┐   │
│  │  Pickled ML Artifacts│             │   TMDB External API      │   │
│  │  - df.pkl            │             │   (Posters, Metadata,    │   │
│  │  - indices.pkl       │             │    Trailers, Cast)       │   │
│  │  - tfidf.pkl         │             └──────────────────────────┘   │
│  │  - tfidf_matrix.pkl  │                                            │
│  └─────────────────────┘                                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Request Flow for a Recommendation
1. User types a movie title in the Streamlit UI.
2. Streamlit sends a `GET /movie/search?query=<title>` request to FastAPI.
3. FastAPI looks up the title in the preloaded `indices` dictionary.
4. The cosine similarity matrix is queried to retrieve the top-N closest movie vectors.
5. Simultaneously, TMDB API is called asynchronously to fetch poster images and metadata.
6. A combined recommendation bundle (movie details + similarity list + genre matches) is returned as JSON.
7. Streamlit renders the poster grid with all recommendations.

---

## 🤖 NLP & Machine Learning Pipeline

### Step 1 — Data Ingestion
The raw dataset (`movies_metadata.csv`) contains 16,800+ movie records with fields like:
`title`, `overview`, `genres`, `tagline`, `keywords`, `cast`, `crew`, `vote_average`, `release_date`, etc.

### Step 2 — Feature Engineering
Multiple text fields are combined into a single **"soup"** feature for each movie:
```python
df['soup'] = df['overview'] + ' ' + df['genres'] + ' ' + df['tagline'] + ' ' + df['keywords']
```

### Step 3 — TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['soup'])
# Shape: (16800, 10000) sparse matrix
```
Each movie is represented as a sparse, high-dimensional TF-IDF vector in a 10,000-dimensional word space.

### Step 4 — Cosine Similarity
```python
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Shape: (16800, 16800) similarity matrix
```
`linear_kernel` is used instead of `cosine_similarity` for speed since the TF-IDF vectors are already L2-normalized.

### Step 5 — Artifact Serialization
All artifacts are serialized with `pickle` for fast startup:
```python
import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
```

### Step 6 — FastAPI Serving
On startup, FastAPI loads all `.pkl` artifacts into memory and serves recommendations with sub-10ms latency.

---

## 📁 Repository Structure

```
Movie-Recommendation/
├── app.py                 # Streamlit Web UI Frontend
│                          #   - Home feed with category tabs
│                          #   - Movie search & autocomplete
│                          #   - Recommendation display grid
│                          #   - Movie detail expanded view
│
├── main.py                # FastAPI REST API Backend
│                          #   - Startup: loads all .pkl artifacts
│                          #   - TMDB API integration (async httpx)
│                          #   - TF-IDF recommendation logic
│                          #   - Full OpenAPI/Swagger docs at /docs
│
├── movies_metadata.csv    # Source movie dataset (16,800+ movies)
│                          #   - Fields: title, overview, genres,
│                          #     tagline, keywords, vote_average,
│                          #     release_date, cast, crew, etc.
│
├── df.pkl                 # Pickled Pandas DataFrame artifact
├── indices.pkl            # Pickled title-to-index lookup dict
├── tfidf.pkl              # Pickled TfidfVectorizer object
├── tfidf_matrix.pkl       # Pickled sparse TF-IDF feature matrix
│
├── requirements.txt       # Python dependency requirements
├── .env                   # Environment variables & API keys (Git ignored)
└── .gitignore             # Git ignore configuration
```

---

## ⚙️ Prerequisites & Setup

### System Requirements
- Python **3.9** or higher
- `pip` package manager
- Active **TMDB API Key** (free at [themoviedb.org](https://www.themoviedb.org/))
- At least **2GB RAM** (for loading TF-IDF matrix into memory)
- At least **500MB disk space** (for `.pkl` artifacts)

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/your-username/Movie-Recommendation.git
cd Movie-Recommendation
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
fastapi
uvicorn[standard]
streamlit
scikit-learn
pandas
numpy
httpx
requests
python-dotenv
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory with your credentials:
```env
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p/w500
```

> **Note**: The `.env` file is listed in `.gitignore` and will **never** be committed to version control. Keep your API keys safe.

---

## 🚀 Running the Application

### 1️⃣ Start the FastAPI Backend

Open **Terminal Tab 1** and run:
```bash
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- The FastAPI backend will start on: `http://127.0.0.1:8000`
- Interactive **Swagger UI** docs: `http://127.0.0.1:8000/docs`
- **ReDoc** alternative docs: `http://127.0.0.1:8000/redoc`

### 2️⃣ Start the Streamlit Frontend

Open **Terminal Tab 2** and run:
```bash
source venv/bin/activate
streamlit run app.py
```

- The Streamlit app will automatically open at: `http://localhost:8501`

### Running in Production
The application is deployed on **Render** and available at:  
🌐 [https://movie-recommendation-system-nimu.onrender.com/](https://movie-recommendation-system-nimu.onrender.com/)

The frontend uses **Auto-Detect Backend**: it first tries connecting to the local FastAPI instance (`http://127.0.0.1:8000`), and falls back to the Render production URL if the local server is unavailable.

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check — returns `{"status": "ok"}` |
| `GET` | `/home?category=popular&limit=24` | Retrieves home feed movie cards by category (`popular`, `trending`, `top_rated`, `upcoming`, `now_playing`) |
| `GET` | `/tmdb/search?query=...` | Keyword search against the full TMDB catalog |
| `GET` | `/movie/id/{tmdb_id}` | Detailed metadata for a specific TMDB movie ID (poster, overview, cast, trailer) |
| `GET` | `/recommend/tfidf?title=...` | Pure TF-IDF cosine similarity vector matching — returns top-N similar movie titles |
| `GET` | `/movie/search?query=...` | Full recommendation bundle: movie details + TF-IDF similarity list + genre-based discoveries |

### Example API Usage

**Get TF-IDF Recommendations:**
```bash
curl "http://127.0.0.1:8000/recommend/tfidf?title=The%20Dark%20Knight&limit=10"
```

**Response:**
```json
{
  "query": "The Dark Knight",
  "recommendations": [
    {"title": "Batman Begins", "similarity_score": 0.87},
    {"title": "The Dark Knight Rises", "similarity_score": 0.85},
    {"title": "Batman", "similarity_score": 0.72}
  ]
}
```

---

## 📊 Dataset Information

The `movies_metadata.csv` dataset contains information about **16,800+ movies** sourced from TMDB and other public movie databases.

### Key Fields Used

| Field | Type | Description |
| :--- | :--- | :--- |
| `title` | `str` | Movie title (used as the primary lookup key) |
| `overview` | `str` | Full plot summary (primary NLP feature) |
| `genres` | `str` | Pipe-separated genre labels (e.g., `Action\|Adventure\|Sci-Fi`) |
| `tagline` | `str` | Official movie tagline |
| `keywords` | `str` | Associated keywords/tags |
| `vote_average` | `float` | TMDB community rating (0–10 scale) |
| `vote_count` | `int` | Number of votes on TMDB |
| `release_date` | `str` | Release date in `YYYY-MM-DD` format |

---

## 🛠️ Technology Stack

### Core ML / Data Science
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **Scikit-Learn** | Latest | TF-IDF vectorization, cosine similarity |
| **Pandas** | Latest | Data loading, cleaning, feature engineering |
| **NumPy** | Latest | Numerical array operations |

### Backend & Frontend
| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | FastAPI + Uvicorn | Async REST API & OpenAPI docs |
| **Frontend** | Streamlit | Responsive, interactive Web UI |
| **HTTP Client** | HTTPX & Requests | External TMDB API integration |

---

## ❓ Frequently Asked Questions (FAQ)

**Q: How do I get a TMDB API key?**  
A: Register for a free account at [themoviedb.org](https://www.themoviedb.org/), go to Settings → API, and request a free developer API key. Add it to your `.env` file as `TMDB_API_KEY`.

**Q: Why does the Render deployment take a long time to respond?**  
A: The free tier of Render spins down idle services after inactivity. The first request may take 30–60 seconds to "wake up" the service. Subsequent requests will be fast.

**Q: Can I add more movies to the dataset?**  
A: Yes! Add new rows to `movies_metadata.csv` and re-run the feature engineering script to rebuild the `.pkl` artifacts.

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Here's how to contribute:

1. **Fork** the repository on GitHub.
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, descriptive commits.
4. **Push** to your fork and open a Pull Request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you found this project useful, please give it a star on GitHub! ⭐**

Made with ❤️ using Python, FastAPI, Streamlit, and the TMDB API.

</div>
