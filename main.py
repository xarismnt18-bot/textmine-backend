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

def extract_texts_as_docs(files_list: list, single_file=None) -> list:
    """
    Given a list of UploadFile objects, return one text string per file.
    If only one file, split it into paragraphs/sentences instead.
    Handles FastAPI quirks: deduplicates by filename, ignores empty slots.
    """
    # Collect all files, deduplicate by filename to avoid double-processing
    seen = set()
    all_files = []
    candidates = list(files_list or [])
    if single_file and getattr(single_file, 'filename', None):
        candidates.append(single_file)
    for f in candidates:
        if not f or not getattr(f, 'filename', None):
            continue
        if f.filename not in seen:
            seen.add(f.filename)
            all_files.append(f)

    if len(all_files) > 1:
        docs = []
        for f in all_files:
            try:
                # Reset file pointer in case it was read already
                if hasattr(f.file, 'seek'):
                    f.file.seek(0)
                t = extract_text(f).strip()
                if len(t) > 20:
                    docs.append(t)
            except Exception:
                pass
        return docs
    elif len(all_files) == 1:
        if hasattr(all_files[0].file, 'seek'):
            all_files[0].file.seek(0)
        text = extract_text(all_files[0])
        docs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
        if len(docs) < 3:
            docs = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
        return docs
    return []


def adaptive_vectorizer_params(n_docs: int, user_min_df: int = 2, user_max_df: float = 0.95):
    """
    Continuously adaptive min_df and max_df — scales smoothly with ANY corpus size.

    min_df = sqrt(n_docs): word must appear in sqrt(n) docs to be statistically meaningful
      n=3→1, n=5→2, n=10→3, n=30→5, n=100→10, n=154→12, n=500→22

    max_df: logarithmic decay from 0.99 (tiny) to 0.85 (large)
      n=3→0.99, n=10→0.96, n=50→0.91, n=154→0.88, n=500→0.85
    """
    import math
    if n_docs < 3:
        return 1, 0.99
    effective_min_df = max(1, int(math.sqrt(n_docs)))
    scale = (math.log(n_docs) - math.log(3)) / (math.log(500) - math.log(3))
    effective_max_df = round(max(0.85, min(0.99, 0.99 - 0.14 * scale)), 2)
    return effective_min_df, effective_max_df


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

    # Min word length filter
    tokens = [t for t in tokens if len(t) >= min_word_len]

    # Lemmatization (preferred over stemming per research)
    if lemmatize:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    elif stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    # Custom stopwords AFTER lemmatization so lemmatized forms are caught
    if custom_stopwords.strip():
        custom = set(w.strip().lower() for w in custom_stopwords.split(",") if w.strip())
        tokens = [t for t in tokens if t not in custom]

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
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
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
    # Support both single file (legacy) and multiple files
    all_files = []
    if files:
        all_files = [f for f in files if f and f.filename]
    if file and file.filename:
        all_files.append(file)
    if not all_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # If multiple files, each file = one document (best for LDA)
    # If single file, split by paragraphs/sentences
    if len(all_files) > 1:
        raw_docs = []
        for f in all_files:
            try:
                t = extract_text(f).strip()
                if len(t) > 20:
                    raw_docs.append(t)
            except Exception:
                pass
        if len(raw_docs) < 3:
            raise HTTPException(status_code=400, detail="Not enough text across files.")
    else:
        text = extract_text(all_files[0])
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

    # Build combined stopword list for vectorizer (double-safety net)
    vectorizer_stop = None
    if remove_stopwords or custom_stopwords.strip():
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        combined = set(ENGLISH_STOP_WORDS)
        if remove_stopwords:
            combined.update(stopwords.words("english"))
        if custom_stopwords.strip():
            combined.update(w.strip().lower() for w in custom_stopwords.split(",") if w.strip())
        vectorizer_stop = list(combined)

    # ── Vectorization — fully adaptive to corpus size ──
    n_pdocs = len(processed_docs)
    effective_min_df, effective_max_df = adaptive_vectorizer_params(n_pdocs)

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=effective_min_df,
        max_df=effective_max_df,
        ngram_range=(ngram_min, ngram_max),
        stop_words=vectorizer_stop,
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
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
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
    raw_docs = extract_texts_as_docs(files, file)
    if len(raw_docs) < 2:
        raise HTTPException(status_code=400, detail="Not enough text.")
    # Override user min_df/max_df with adaptive values
    min_df, max_df = adaptive_vectorizer_params(len(raw_docs))

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


@app.post("/analyze/tfidf_perdoc")
async def analyze_tfidf_perdoc(
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
    max_features: int = Form(20),
    ngram_min: int = Form(1),
    ngram_max: int = Form(2),
    remove_stopwords: bool = Form(True),
    lemmatize: bool = Form(False),
    remove_numbers: bool = Form(True),
    custom_stopwords: str = Form(""),
):
    import numpy as np
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    paragraphs = extract_texts_as_docs(files, file)
    if len(paragraphs) < 3:
        raise HTTPException(status_code=400, detail="Not enough text.")

    # Build stopwords
    combined_stop = set(ENGLISH_STOP_WORDS)
    if remove_stopwords:
        combined_stop.update(stopwords.words("english"))
    if custom_stopwords.strip():
        combined_stop.update([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])

    # Preprocess
    processed = []
    for p in paragraphs:
        tokens = preprocess_text(p, remove_sw=remove_stopwords, lemmatize=lemmatize,
                                  min_word_len=3, custom_stopwords=custom_stopwords,
                                  remove_numbers=remove_numbers, lowercase=True)
        if tokens:
            processed.append(" ".join(tokens))

    if len(processed) < 3:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    # Fully adaptive to corpus size
    n_docs = len(processed)
    auto_min_df, auto_max_df = adaptive_vectorizer_params(n_docs, 1, 0.95)

    # Fit TF-IDF on ALL documents (IDF from full corpus)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(ngram_min, ngram_max),
        stop_words=list(combined_stop),
        sublinear_tf=True,
        min_df=auto_min_df,
        max_df=auto_max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(processed)
    feature_names = vectorizer.get_feature_names_out()

    # Get top terms per paragraph, then aggregate uniqueness
    # A term is "distinctive" if it scores HIGH in few paragraphs (truly specific)
    scores_array = tfidf_matrix.toarray()

    # For each term: score = max_score * (1 - frequency_ratio)
    # This rewards terms that are very high in some docs but not all
    max_scores = scores_array.max(axis=0)
    doc_freq = (scores_array > 0).sum(axis=0) / len(processed)
    distinctiveness = max_scores * (1 - doc_freq * 0.5)

    # Get top N
    top_idx = distinctiveness.argsort()[-max_features:][::-1]

    # Normalize
    top_scores = distinctiveness[top_idx]
    if top_scores.max() > 0:
        top_scores = top_scores / top_scores.max()

    terms = [{"word": feature_names[i], "score": round(float(top_scores[j]), 4)}
             for j, i in enumerate(top_idx)]
    terms.sort(key=lambda x: x["score"], reverse=True)

    return {
        "method": "tfidf",
        "num_docs": len(processed),
        "num_terms": len(terms),
        "scoring": "distinctiveness (max_score × inverse_frequency)",
        "terms": terms,
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
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
    per_sentence: bool = Form(True),
    model: str = Form("vader"),
):
    # Collect all uploaded files, deduplicate by filename
    seen = set()
    all_files = []
    candidates = list(files or [])
    if file and getattr(file, 'filename', None):
        candidates.append(file)
    for f in candidates:
        if not f or not getattr(f, 'filename', None):
            continue
        if f.filename not in seen:
            seen.add(f.filename)
            all_files.append(f)

    if not all_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Concatenate text from ALL files
    full_text = ""
    for f in all_files:
        try:
            if hasattr(f.file, 'seek'):
                f.file.seek(0)
            t = extract_text(f).strip()
            if t:
                full_text += t + "\n"
        except Exception:
            pass

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded files.")

    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(full_text) if per_sentence else [full_text]
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
    return {"method": "sentiment", "summary": summary, "sentences": results[:500]}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ── Coherence Score Calculator ────────────────────────────────────────────────
@app.post("/analyze/coherence")
async def analyze_coherence(
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
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

    # ── 1. Split into documents ──────────────────────────────────────────────
    raw_docs = extract_texts_as_docs(files, file)
    if len(raw_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text.")

    # ── 2. Preprocessing — keep token lists (needed for C_V / NPMI) ──────────
    tokenized_docs = []
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
            tokenized_docs.append(tokens)

    if len(tokenized_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    # ── 3. Build stopword list for sklearn vectorizer ────────────────────────
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    coh_vectorizer_stop = None
    if remove_stopwords or custom_stopwords.strip():
        coh_combined = set(ENGLISH_STOP_WORDS)
        if remove_stopwords:
            coh_combined.update(stopwords.words("english"))
        if custom_stopwords.strip():
            coh_combined.update(w.strip().lower() for w in custom_stopwords.split(",") if w.strip())
        coh_vectorizer_stop = list(coh_combined)

    # ── 4. sklearn vectorizer (for LDA + perplexity) ─────────────────────────
    processed_docs = [" ".join(t) for t in tokenized_docs]
    n_docs = len(processed_docs)

    # Fully adaptive to corpus size
    effective_min_df, effective_max_df = adaptive_vectorizer_params(n_docs)

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=effective_min_df,
        max_df=effective_max_df,
        ngram_range=(ngram_min, ngram_max),
        stop_words=coh_vectorizer_stop,
    )
    dtm = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    if dtm.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Vocabulary empty after preprocessing.")

    # ── 5. Attempt Gensim C_V coherence (gold standard) ─────────────────────
    use_cv = False
    gensim_dictionary = None
    gensim_corpus = None
    try:
        import gensim
        import gensim.corpora as corpora
        from gensim.models import LdaModel
        from gensim.models.coherencemodel import CoherenceModel

        gensim_dictionary = corpora.Dictionary(tokenized_docs)
        # Filter extremes to match sklearn settings
        gensim_dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)
        gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in tokenized_docs]

        if len(gensim_dictionary) > 0:
            use_cv = True
    except ImportError:
        use_cv = False

    # ── 6. Loop over topic range ─────────────────────────────────────────────
    topic_range = list(range(min_topics, min(max_topics + 1, len(tokenized_docs)), step))
    perplexity_scores = []
    coherence_scores = []
    coherence_metric = "c_v" if use_cv else "npmi"

    for n_topics in topic_range:
        # sklearn LDA for perplexity
        lda_sk = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=max_iter,
            learning_method="online",
        )
        lda_sk.fit(dtm)
        perp = lda_sk.perplexity(dtm)
        perplexity_scores.append(round(float(perp), 2))

        if use_cv:
            # ── Gensim LDA + C_V coherence ────────────────────────────────
            try:
                lda_gensim = LdaModel(
                    corpus=gensim_corpus,
                    num_topics=n_topics,
                    id2word=gensim_dictionary,
                    passes=5,
                    random_state=42,
                    alpha="auto",
                )
                cm = CoherenceModel(
                    model=lda_gensim,
                    texts=tokenized_docs,
                    dictionary=gensim_dictionary,
                    coherence="c_v",
                    topn=10,
                )
                score = round(float(cm.get_coherence()), 4)
            except Exception:
                # fallback to NPMI if C_V fails
                try:
                    cm = CoherenceModel(
                        model=lda_gensim,
                        texts=tokenized_docs,
                        dictionary=gensim_dictionary,
                        coherence="c_npmi",
                        topn=10,
                    )
                    score = round(float(cm.get_coherence()), 4)
                    coherence_metric = "npmi"
                except Exception:
                    score = 0.0
        else:
            # ── NPMI fallback (no gensim) — better than UMass ─────────────
            # NPMI = log(p(wi,wj) / p(wi)*p(wj)) / -log(p(wi,wj))
            # computed over sliding windows of size 10
            window_size = 10
            all_tokens_flat = [t for doc in tokenized_docs for t in doc]
            vocab_list = list(feature_names)
            word2idx = {w: i for i, w in enumerate(vocab_list)}

            # Build co-occurrence counts using sliding windows
            n_vocab = len(vocab_list)
            word_count = np.zeros(n_vocab)
            pair_count = {}

            for doc_tokens in tokenized_docs:
                idxs = [word2idx[t] for t in doc_tokens if t in word2idx]
                for k in range(len(idxs)):
                    word_count[idxs[k]] += 1
                    window = idxs[max(0, k - window_size): k + window_size + 1]
                    for m in window:
                        if m != idxs[k]:
                            pair = (min(idxs[k], m), max(idxs[k], m))
                            pair_count[pair] = pair_count.get(pair, 0) + 1

            total = max(sum(word_count), 1)

            def npmi(wi, wj):
                pair = (min(wi, wj), max(wi, wj))
                co = pair_count.get(pair, 0)
                if co == 0:
                    return -1.0
                p_wi = word_count[wi] / total
                p_wj = word_count[wj] / total
                p_co = co / total
                pmi = np.log(p_co / (p_wi * p_wj + 1e-12))
                npmi_val = pmi / (-np.log(p_co + 1e-12))
                return float(npmi_val)

            topic_scores = []
            for topic in lda_sk.components_:
                top_idx = topic.argsort()[-10:][::-1]
                pairs = [(top_idx[i], top_idx[j])
                         for i in range(len(top_idx))
                         for j in range(i + 1, len(top_idx))]
                s = np.mean([npmi(wi, wj) for wi, wj in pairs]) if pairs else 0.0
                topic_scores.append(s)
            score = round(float(np.mean(topic_scores)), 4)

        coherence_scores.append(score)

    # ── 7. Find optimal — peak of C_V/NPMI curve ────────────────────────────
    best_idx = coherence_scores.index(max(coherence_scores))
    optimal_topics = topic_range[best_idx]

    # Normalize for display (0–100)
    min_c = min(coherence_scores)
    max_c = max(coherence_scores)
    range_c = max_c - min_c if max_c != min_c else 1
    normalized = [round((c - min_c) / range_c * 100, 1) for c in coherence_scores]

    metric_label = "C_V (Gensim)" if coherence_metric == "c_v" else "NPMI"
    recommendation = (
        f"Based on {metric_label} coherence analysis, {optimal_topics} topics is optimal for your corpus. "
        f"(Score: {coherence_scores[best_idx]:.4f}, Docs: {len(tokenized_docs)}, Vocab: {dtm.shape[1]} words)"
    )

    return {
        "method": "coherence",
        "coherence_metric": coherence_metric,
        "num_docs": len(tokenized_docs),
        "vocab_size": dtm.shape[1],
        "topic_range": topic_range,
        "coherence_scores": coherence_scores,
        "coherence_normalized": normalized,
        "perplexity_scores": perplexity_scores,
        "optimal_topics": optimal_topics,
        "optimal_coherence": coherence_scores[best_idx],
        "recommendation": recommendation,
    }
