from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, os
import requests

import PyPDF2, openpyxl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Optional
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI(title="TextMine API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ──────────────────────────────────────────────────────────────────

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


def preprocess_text(
    text: str,
    remove_sw: bool = True,
    lemmatize: bool = False,
    stemming: bool = False,
    min_word_len: int = 3,
    custom_stopwords: str = "",
    remove_numbers: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """
    Full preprocessing pipeline based on LDA research best practices.
    Steps: lowercase → remove numbers/punctuation → tokenize →
           remove stopwords → custom stopwords → min length filter →
           lemmatize or stem
    """
    if lowercase:
        text = text.lower()

    # Remove numbers and punctuation
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    if remove_sw:
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop]

    # Custom stopwords
    if custom_stopwords.strip():
        custom = set(w.strip().lower() for w in custom_stopwords.split(",") if w.strip())
        tokens = [t for t in tokens if t not in custom]

    # Min word length filter
    tokens = [t for t in tokens if len(t) >= min_word_len]

    # Lemmatization (preferred over stemming per research)
    if lemmatize:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    elif stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


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


def _run_local_bertopic(docs, num_topics, min_topic_size, language, reduce_outliers):
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="BERTopic is not installed on this server. Use engine=colab instead.",
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
        words_payload = [{"word": w, "score": round(float(s), 4)} for w, s in words[:10]]
        results.append({
            "topic_id": topic_id,
            "count": int(row["Count"]),
            "words": words_payload,
            "label": ", ".join(w["word"] for w in words_payload[:3]),
        })
    return {"method": "bertopic", "engine": "local", "num_docs": len(docs), "num_topics": len(results), "topics": results}


def _run_colab_bertopic(docs, num_topics, min_topic_size, language, reduce_outliers):
    colab_url = os.environ.get("COLAB_BERTOPIC_URL")
    if not colab_url:
        raise HTTPException(status_code=400, detail="COLAB_BERTOPIC_URL is not set.")
    try:
        response = requests.post(
            colab_url,
            json={"docs": docs, "num_topics": num_topics, "min_topic_size": min_topic_size, "language": language, "reduce_outliers": reduce_outliers},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Colab: {exc}") from exc
    payload = response.json()
    payload.setdefault("engine", "colab")
    payload.setdefault("method", "bertopic")
    return payload


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "TextMine API is running! 🚀"}


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


@app.post("/analyze/lda")
async def analyze_lda(
    file: UploadFile = File(...),
    num_topics: int = Form(10),
    max_features: int = Form(1000),
    max_iter: int = Form(20),
    # ── Preprocessing options (research-backed) ──
    lemmatize: bool = Form(False),
    stemming: bool = Form(False),
    remove_stopwords: bool = Form(True),
    remove_numbers: bool = Form(True),
    min_word_len: int = Form(3),
    custom_stopwords: str = Form(""),
    # ── Vectorizer options ──
    min_df: int = Form(2),
    max_df: float = Form(0.95),
    ngram_min: int = Form(1),
    ngram_max: int = Form(1),
    # ── LDA hyperparameters ──
    alpha: str = Form("auto"),       # 'auto', 'symmetric', or float
    beta: str = Form("auto"),        # 'auto' or float (learning_offset proxy)
    learning_method: str = Form("online"),  # 'online' or 'batch'
):
    text = extract_text(file)

    # Split into documents (paragraphs are better than sentences for LDA)
    raw_docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(raw_docs) < 5:
        raw_docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    if len(raw_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text for LDA.")

    # ── Full preprocessing pipeline ──
    processed_docs = []
    for doc in raw_docs:
        tokens = preprocess_text(
            doc,
            remove_sw=remove_stopwords,
            lemmatize=lemmatize,
            stemming=stemming,
            min_word_len=min_word_len,
            custom_stopwords=custom_stopwords,
            remove_numbers=remove_numbers,
            lowercase=True,
        )
        if tokens:
            processed_docs.append(" ".join(tokens))

    if len(processed_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    # ── Vectorization ──
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
    )
    dtm = vectorizer.fit_transform(processed_docs)

    if dtm.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Vocabulary is empty after preprocessing. Try relaxing the filters.")

    # ── LDA model ──
    n_topics = min(num_topics, len(processed_docs) - 1, dtm.shape[1])

    # Handle alpha parameter
    doc_topic_prior = None if alpha == "auto" else (
        "symmetric" if alpha == "symmetric" else float(alpha)
    )
    topic_word_prior = None if beta == "auto" else float(beta)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=max_iter,
        learning_method=learning_method,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    results = []
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        words = [{"word": feature_names[i], "score": round(float(topic[i] / topic.sum()), 4)} for i in top_indices]
        results.append({
            "topic_id": idx,
            "count": int(len(processed_docs) / n_topics),
            "words": words,
            "label": ", ".join(w["word"] for w in words[:3]),
        })

    return {
        "method": "lda",
        "num_docs": len(processed_docs),
        "num_topics": len(results),
        "vocab_size": dtm.shape[1],
        "preprocessing": {
            "lemmatize": lemmatize,
            "stemming": stemming,
            "remove_stopwords": remove_stopwords,
            "remove_numbers": remove_numbers,
            "min_word_len": min_word_len,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": f"({ngram_min},{ngram_max})",
            "alpha": alpha,
            "beta": beta,
        },
        "topics": results,
    }


@app.post("/analyze/tfidf")
async def analyze_tfidf(
    file: UploadFile = File(...),
    max_features: int = Form(50),
    ngram_min: int = Form(1),
    ngram_max: int = Form(1),
    remove_stopwords: bool = Form(True),
    min_df: int = Form(2),
    max_df: float = Form(0.95),
    lemmatize: bool = Form(False),
    remove_numbers: bool = Form(True),
    custom_stopwords: str = Form(""),
):
    import numpy as np
    text = extract_text(file)

    # Use paragraphs as documents for better TF-IDF differentiation
    raw_docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(raw_docs) < 5:
        raw_docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    if len(raw_docs) < 2:
        raise HTTPException(status_code=400, detail="Not enough text.")

    # Preprocess each document
    processed_docs = []
    for doc in raw_docs:
        tokens = preprocess_text(
            doc,
            remove_sw=remove_stopwords,
            lemmatize=lemmatize,
            min_word_len=3,
            custom_stopwords=custom_stopwords,
            remove_numbers=remove_numbers,
            lowercase=True,
        )
        if tokens:
            processed_docs.append(" ".join(tokens))

    if len(processed_docs) < 2:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    # Build combined stopwords list: English + custom
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stop = set(ENGLISH_STOP_WORDS)
    if remove_stopwords:
        combined_stop.update(stopwords.words("english"))
    if custom_stopwords.strip():
        custom_list = [w.strip().lower() for w in custom_stopwords.split(",") if w.strip()]
        combined_stop.update(custom_list)

    # TF-IDF with proper min/max_df for score differentiation
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        stop_words=list(combined_stop),  # Pass ALL stopwords directly to vectorizer
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    # Use MEAN score across documents (not max) for better differentiation
    scores_mean = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    scores_max = np.asarray(tfidf_matrix.max(axis=0).todense()).flatten()

    # Combine: weighted average of mean and max
    scores = 0.6 * scores_mean + 0.4 * scores_max

    # Normalize to 0-1
    if scores.max() > 0:
        scores = scores / scores.max()

    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    return {
        "method": "tfidf",
        "num_docs": len(processed_docs),
        "num_terms": len(ranked),
        "scoring": "sublinear_tf + mean+max weighted",
        "terms": [{"word": w, "score": round(float(s), 4)} for w, s in ranked],
    }


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
        raise HTTPException(status_code=400, detail="No usable words found.")
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
    if n == 0:
        raise HTTPException(status_code=400, detail="No sentences found.")
    summary = {
        "total_sentences": n,
        "positive_pct": round(sum(1 for r in results if r["label"] == "positive") / n * 100, 1),
        "neutral_pct": round(sum(1 for r in results if r["label"] == "neutral") / n * 100, 1),
        "negative_pct": round(sum(1 for r in results if r["label"] == "negative") / n * 100, 1),
        "avg_compound": round(sum(r["compound"] for r in results) / n, 4),
        "model_used": model,
    }
    return {"method": "sentiment", "summary": summary, "sentences": results[:100]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ── Coherence Score Calculator ────────────────────────────────────────────────
@app.post("/analyze/coherence")
async def analyze_coherence(
    file: UploadFile = File(...),
    min_topics: int = Form(2),
    max_topics: int = Form(20),
    step: int = Form(1),
    max_features: int = Form(1000),
    max_iter: int = Form(20),
    lemmatize: bool = Form(False),
    stemming: bool = Form(False),
    remove_stopwords: bool = Form(True),
    remove_numbers: bool = Form(True),
    min_word_len: int = Form(3),
    custom_stopwords: str = Form(""),
    min_df: int = Form(2),
    max_df: float = Form(0.95),
    ngram_min: int = Form(1),
    ngram_max: int = Form(1),
):
    import numpy as np
    from sklearn.model_selection import train_test_split

    text = extract_text(file)

    # Split into documents
    raw_docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(raw_docs) < 5:
        raw_docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
    if len(raw_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text.")

    # Preprocessing
    processed_docs = []
    for doc in raw_docs:
        tokens = preprocess_text(
            doc,
            remove_sw=remove_stopwords,
            lemmatize=lemmatize,
            stemming=stemming,
            min_word_len=min_word_len,
            custom_stopwords=custom_stopwords,
            remove_numbers=remove_numbers,
            lowercase=True,
        )
        if tokens:
            processed_docs.append(" ".join(tokens))

    if len(processed_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    # Vectorize
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
    )
    dtm = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    if dtm.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Vocabulary empty after preprocessing.")

    # Calculate coherence for each number of topics
    # We use UMass coherence approximation (fast, no extra library needed)
    # C_v approximation: average top-word co-occurrence across topics
    results = []
    topic_range = list(range(min_topics, min(max_topics + 1, len(processed_docs)), step))

    # Get word co-occurrence matrix for coherence calculation
    dtm_array = dtm.toarray()
    doc_count = dtm_array.shape[0]

    def umass_coherence(top_words_idx, dtm_arr):
        """Calculate UMass coherence score for a topic."""
        scores = []
        for i in range(1, len(top_words_idx)):
            for j in range(i):
                wi = top_words_idx[i]
                wj = top_words_idx[j]
                # Co-document frequency
                co_doc = np.sum((dtm_arr[:, wi] > 0) & (dtm_arr[:, wj] > 0))
                doc_freq_wj = np.sum(dtm_arr[:, wj] > 0)
                if doc_freq_wj > 0:
                    scores.append(np.log((co_doc + 1) / doc_freq_wj))
        return np.mean(scores) if scores else 0

    perplexity_scores = []
    coherence_scores = []

    for n_topics in topic_range:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=max_iter,
            learning_method="online",
        )
        lda.fit(dtm)

        # Perplexity (lower = better)
        perp = lda.perplexity(dtm)
        perplexity_scores.append(round(float(perp), 2))

        # Coherence (higher = better)
        topic_coherences = []
        for topic in lda.components_:
            top_idx = topic.argsort()[-10:][::-1]
            coh = umass_coherence(top_idx, dtm_array)
            topic_coherences.append(coh)
        avg_coherence = round(float(np.mean(topic_coherences)), 4)
        coherence_scores.append(avg_coherence)

    # Find optimal: best coherence score
    best_idx = coherence_scores.index(max(coherence_scores))
    optimal_topics = topic_range[best_idx]

    # Normalize coherence for display (0-100 scale)
    min_c = min(coherence_scores)
    max_c = max(coherence_scores)
    range_c = max_c - min_c if max_c != min_c else 1
    normalized = [round((c - min_c) / range_c * 100, 1) for c in coherence_scores]

    return {
        "method": "coherence",
        "num_docs": len(processed_docs),
        "vocab_size": dtm.shape[1],
        "topic_range": topic_range,
        "coherence_scores": coherence_scores,
        "coherence_normalized": normalized,
        "perplexity_scores": perplexity_scores,
        "optimal_topics": optimal_topics,
        "optimal_coherence": coherence_scores[best_idx],
        "recommendation": f"Based on coherence analysis, {optimal_topics} topics is optimal for your corpus.",
    }
