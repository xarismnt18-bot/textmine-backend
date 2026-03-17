from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, os
import requests

import PyPDF2, openpyxl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI(title="TextMine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text(file: UploadFile) -> str:
    content = file.file.read()
    name = file.filename.lower()
    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".xlsx"):
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        texts = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    texts.append(row_text)
        return "\n".join(texts)
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

def clean_tokens(text, remove_stopwords=True, lemmatize=False):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    if remove_stopwords:
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop]
    if lemmatize:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return tokens

def _run_local_bertopic(docs, num_topics: int, min_topic_size: int, language: str, reduce_outliers: bool):
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="BERTopic is not installed. Add bertopic + sentence-transformers dependencies.",
        ) from exc

    topic_model = BERTopic(
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=num_topics if num_topics > 0 else "auto",
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs)
    if reduce_outliers:
        topics = topic_model.reduce_outliers(docs, topics)
        topic_model.update_topics(docs, topics)

    topic_info = topic_model.get_topic_info()
    results = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        words_payload = [
            {"word": term, "score": round(float(score), 4)}
            for term, score in words[:10]
        ]
        results.append(
            {
                "topic_id": topic_id,
                "count": int(row["Count"]),
                "words": words_payload,
                "label": row.get("Name") or ", ".join(w["word"] for w in words_payload[:3]),
            }
        )

    return {"method": "bertopic", "engine": "local", "num_docs": len(docs), "num_topics": len(results), "topics": results}


def _run_colab_bertopic(docs, num_topics: int, min_topic_size: int, language: str, reduce_outliers: bool):
    colab_url = os.environ.get("COLAB_BERTOPIC_URL")
    if not colab_url:
        raise HTTPException(
            status_code=400,
            detail="COLAB_BERTOPIC_URL is not configured. Set it to your Colab webhook URL.",
        )

    try:
        response = requests.post(
            colab_url,
            json={
                "docs": docs,
                "num_topics": num_topics,
                "min_topic_size": min_topic_size,
                "language": language,
                "reduce_outliers": reduce_outliers,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Colab BERTopic service: {exc}") from exc

    payload = response.json()
    payload.setdefault("engine", "colab")
    payload.setdefault("method", "bertopic")
    return payload

@app.get("/")
def root():
    return {"status": "ok", "message": "TextMine API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze/bertopic")
async def analyze_bertopic(
    file: UploadFile = File(...),
    num_topics: int = Form(10),
    min_topic_size: int = Form(15),
    language: str = Form("english"),
    reduce_outliers: bool = Form(True),
    engine: str = Form("local"),
):
    text = extract_text(file)
  docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    if len(docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text.")
      if engine == "colab":
        return _run_colab_bertopic(docs, num_topics, min_topic_size, language, reduce_outliers)
    return _run_local_bertopic(docs, num_topics, min_topic_size, language, reduce_outliers)

@app.post("/analyze/tfidf")
async def analyze_tfidf(
    file: UploadFile = File(...),
    max_features: int = Form(50),
    ngram_min: int = Form(1),
    ngram_max: int = Form(1),
    remove_stopwords: bool = Form(True),
):
    text = extract_text(file)
    docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
    if len(docs) < 2:
        raise HTTPException(status_code=400, detail="Not enough text.")
    stop = "english" if remove_stopwords else None
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(ngram_min, ngram_max), stop_words=stop)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.max(axis=0).toarray().flatten()
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return {"method": "tfidf", "num_docs": len(docs), "num_terms": len(ranked), "terms": [{"word": w, "score": round(float(s), 4)} for w, s in ranked]}

@app.post("/analyze/wordfreq")
async def analyze_wordfreq(
    file: UploadFile = File(...),
    max_words: int = Form(100),
