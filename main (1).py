from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os

import PyPDF2
import openpyxl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic

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

@app.get("/")
def root():
    return {"status": "ok", "message": "TextMine API is running 🚀"}

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
):
    text = extract_text(file)
    docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    if len(docs) < 10:
        raise HTTPException(status_code=400, detail="Not enough text. Please upload a larger document.")
    topic_model = BERTopic(language=language, nr_topics=num_topics, min_topic_size=min_topic_size, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    results = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        results.append({
            "topic_id": topic_id,
            "count": int(row["Count"]),
            "words": [{"word": w, "score": round(float(s), 4)} for w, s in words[:10]],
            "label": ", ".join(w for w, _ in words[:3]),
        })
    return {"method": "bertopic", "num_docs": len(docs), "num_topics": len(results), "topics": results}

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
        raise HTTPException(status_code=400, detail="Not enough sentences for TF-IDF.")
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
    remove_stopwords: bool = Form(True),
    lemmatize: bool = Form(False),
):
    text = extract_text(file)
    tokens = clean_tokens(text, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    if not tokens:
        raise HTTPException(status_code=400, detail="No usable words found in document.")
    counter = Counter(tokens)
    top = counter.most_common(max_words)
    max_count = top[0][1] if top else 1
    return {"method": "wordfreq", "total_tokens": len(tokens), "unique_tokens": len(counter), "words": [{"word": w, "count": c, "relative": round(c / max_count, 4)} for w, c in top]}

@app.post("/analyze/sentiment")
async def analyze_sentiment(
    file: UploadFile = File(...),
    per_sentence: bool = Form(True),
    model: str = Form("vader"),
):
    text = extract_text(file)
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text) if per_sentence else [text]
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    results = []
    for sent in sentences:
        scores = analyzer.polarity_scores(sent)
        label = "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral"
        results.append({"sentence": sent[:200], "compound": round(scores["compound"], 4), "positive": round(scores["pos"], 4), "neutral": round(scores["neu"], 4), "negative": round(scores["neg"], 4), "label": label})
    n = len(sentences)
    summary = {"total_sentences": n, "positive_pct": round(sum(1 for r in results if r["label"] == "positive") / n * 100, 1), "neutral_pct": round(sum(1 for r in results if r["label"] == "neutral") / n * 100, 1), "negative_pct": round(sum(1 for r in results if r["label"] == "negative") / n * 100, 1), "avg_compound": round(sum(r["compound"] for r in results) / n, 4)}
    return {"method": "sentiment", "summary": summary, "sentences": results[:100]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
