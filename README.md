# 🎬 Movie Recommendation System & API

An end-to-end, high-performance Movie Recommendation Application powered by **TF-IDF Content-Based Natural Language Processing (NLP)**, **Cosine Similarity**, **FastAPI Backend Services**, and a **Streamlit Web UI** with **TMDB API integration**.

🌐 **Live Web Application**: [https://movie-recommendation-system-nimu.onrender.com/](https://movie-recommendation-system-nimu.onrender.com/)

---

## 🌟 Key Features

- **TF-IDF Content Vectorization**: Precomputed TF-IDF sparse matrix built from plot overviews, genres, taglines, and keywords across 16,800+ movies.
- **Instant Similarity Engine**: Lightning-fast cosine similarity matrix matching.
- **FastAPI REST API (`main.py`)**: High-performance asynchronous backend providing home feeds, search autocomplete, poster resolution, and recommendation bundles.
- **Streamlit Web UI (`app.py`)**: Modern, interactive single-page web interface with poster grids, search dropdowns, genre discovery, and movie details.
- **TMDB API Integration**: Automatically fetches high-resolution movie posters, backdrops, and live metadata from TMDB.
- **Auto-Detect Backend**: Frontend dynamically detects whether local FastAPI server (`http://127.0.0.1:8000`) or production Render deployment is running.

---

## 📁 Repository Structure

```
Movie-Recommendation/
├── app.py                 # Streamlit Web UI Frontend
├── main.py                # FastAPI REST API Backend
├── movies_metadata.csv    # Source movie dataset
├── df.pkl                 # Pickled Pandas DataFrame artifact
├── indices.pkl            # Pickled title-to-index lookup artifact
├── tfidf.pkl              # Pickled TfidfVectorizer artifact
├── tfidf_matrix.pkl       # Pickled sparse TF-IDF feature matrix
├── requirements.txt       # Python dependency requirements
├── .env                   # Environment variables & API keys (Git ignored)
└── .gitignore             # Git ignore configuration
```

---

## ⚙️ Prerequisites & Setup

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/your-username/Movie-Recommendation.git
cd Movie-Recommendation
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p/w500
```

---

## 🚀 Running the Application

### 1️⃣ Start the FastAPI Backend
In Terminal Tab 1:
```bash
source venv/bin/activate
uvicorn main:app --reload
```
The FastAPI backend server will start on `http://127.0.0.1:8000`. You can inspect interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 2️⃣ Start the Streamlit Frontend
In Terminal Tab 2:
```bash
source venv/bin/activate
streamlit run app.py
```
The Streamlit application will launch automatically at `http://localhost:8501`.

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check endpoint |
| `GET` | `/home?category=popular&limit=24` | Retrieves home feed movie cards by category (`popular`, `trending`, `top_rated`, `upcoming`, `now_playing`) |
| `GET` | `/tmdb/search?query=...` | Keyword search against TMDB catalog |
| `GET` | `/movie/id/{tmdb_id}` | Detailed metadata for a specific TMDB movie ID |
| `GET` | `/recommend/tfidf?title=...` | Pure TF-IDF similarity vector matching |
| `GET` | `/movie/search?query=...` | Complete recommendation bundle (Movie details + TF-IDF similarity + Genre discovery) |

---

## 🛠️ Built With

- **Python 3.9+**
- **FastAPI** & **Uvicorn**
- **Streamlit**
- **Scikit-Learn** & **NumPy** & **Pandas**
- **TMDB REST API** & **HTTPX** & **Requests**

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
