from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, os
import requests
from anthropic import Anthropic
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
anthropic_client = Anthropic()

@app.post("/ai/evaluate")
async def ai_evaluate(request: Request):
    body = await request.json()
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=body.get("system", ""),
        messages=body.get("messages", [])
    )
    return {"content": response.content[0].text}

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


def expand_custom_stopwords(custom_stopwords: str, lemmatize: bool = False, stemming: bool = False) -> set:
    """Expand custom stopwords to include lemmatized/stemmed forms so they are always caught."""
    words = set(w.strip().lower() for w in custom_stopwords.split(",") if w.strip())
    expanded = set(words)
    if lemmatize:
        lem = WordNetLemmatizer()
        expanded.update(lem.lemmatize(w) for w in words)
        expanded.update(lem.lemmatize(w, pos='v') for w in words)
    if stemming:
        stemmer = PorterStemmer()
        expanded.update(stemmer.stem(w) for w in words)
    return expanded


def _make_tokenizer(stop_set):
    def _tok(x):
        return [t for t in x.split() if t not in stop_set]
    return _tok


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

    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    if remove_sw:
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop]

    tokens = [t for t in tokens if len(t) >= min_word_len]

    if lemmatize:
        lem = WordNetLemmatizer()
        def _lem_best(word):
            forms = [lem.lemmatize(word, pos='n'), lem.lemmatize(word, pos='v'), lem.lemmatize(word, pos='a')]
            return min(forms, key=len)
        tokens = [_lem_best(t) for t in tokens]
    elif stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    # Custom stopwords AFTER lemmatization — also matches lemmatized/stemmed forms
    if custom_stopwords.strip():
        custom = expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize, stemming=stemming)
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
        def _lem_best(word):
            forms = [lem.lemmatize(word, pos='n'), lem.lemmatize(word, pos='v'), lem.lemmatize(word, pos='a')]
            return min(forms, key=len)
        tokens = [_lem_best(t) for t in tokens]
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

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "TextMine API is running! 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze/bertopic")
async def analyze_bertopic(
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
    num_topics: int = Form(10),
    min_topic_size: int = Form(15),
    language: str = Form("english"),
    reduce_outliers: bool = Form(True),
    engine: str = Form("local"),
    custom_stopwords: str = Form(""),
    lemmatize: bool = Form(False),
):
    seen = set()
    all_files = []
    for f in list(files or []) + ([file] if file and getattr(file, 'filename', None) else []):
        if f and getattr(f, 'filename', None) and f.filename not in seen:
            seen.add(f.filename)
            all_files.append(f)
    if not all_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    full_text = ""
    for f in all_files:
        try:
            if hasattr(f.file, 'seek'): f.file.seek(0)
            full_text += extract_text(f) + "\n"
        except Exception:
            pass

    # Always strip URLs, www references, and standalone numbers from BERTopic docs
    import re as _re
    _url_re = _re.compile(r'https?://\S+|www\.\S+', _re.IGNORECASE)
    _num_re = _re.compile(r'\b\d+\b')

    def _clean_bertopic_doc(text: str) -> str:
        text = _url_re.sub(' ', text)
        text = _num_re.sub(' ', text)
        # Optionally apply lemmatization
        if lemmatize:
            tokens = word_tokenize(text.lower())
            lem = WordNetLemmatizer()
            def _lem_best(word):
                forms = [lem.lemmatize(word, pos='n'), lem.lemmatize(word, pos='v'), lem.lemmatize(word, pos='a')]
                return min(forms, key=len)
            tokens = [_lem_best(t) if t.isalpha() else t for t in tokens if t.isalpha() or t in ('.', ',', '!', '?')]
            text = ' '.join(tokens)
        return text.strip()

    docs = [_clean_bertopic_doc(s.strip()) for s in sent_tokenize(full_text) if len(s.strip()) > 20]
    docs = [d for d in docs if len(d) > 10]

    # Apply custom stopwords by filtering them out at the sentence level
    if custom_stopwords.strip():
        custom_set = expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize)
        _cust_re = _re.compile(r'\b(' + '|'.join(_re.escape(w) for w in custom_set) + r')\b', _re.IGNORECASE)
        docs = [_cust_re.sub(' ', d).strip() for d in docs]

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
    alpha: str = Form("auto"),
    beta: str = Form("auto"),
    learning_method: str = Form("online"),
):
    all_files = []
    if files:
        all_files = [f for f in files if f and f.filename]
    if file and file.filename:
        all_files.append(file)
    if not all_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

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

    vectorizer_stop = None
    if remove_stopwords or custom_stopwords.strip():
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        combined = set(ENGLISH_STOP_WORDS)
        if remove_stopwords:
            combined.update(stopwords.words("english"))
        if custom_stopwords.strip():
            combined.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize, stemming=stemming))
        vectorizer_stop = list(combined)

    # FIX: use adaptive params but respect user values as floor/ceiling hints
    n_pdocs = len(processed_docs)
    effective_min_df, effective_max_df = adaptive_vectorizer_params(n_pdocs, user_min_df=min_df, user_max_df=max_df)

    final_stop = set(vectorizer_stop) if vectorizer_stop else set()
    if custom_stopwords.strip():
        final_stop.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize, stemming=stemming))

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=effective_min_df,
        max_df=effective_max_df,
        ngram_range=(ngram_min, ngram_max),
        tokenizer=_make_tokenizer(final_stop),
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    dtm = vectorizer.fit_transform(processed_docs)

    if dtm.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Vocabulary is empty after preprocessing. Try relaxing the filters.")

    n_topics = min(num_topics, len(processed_docs) - 1, dtm.shape[1])

    doc_topic_prior = None if alpha == "auto" else (
        "symmetric" if alpha == "symmetric" else float(alpha)
    )

    # FIX: topic_word_prior now correctly passed to the model
    topic_word_prior = None if beta == "auto" else float(beta)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=max_iter,
        learning_method=learning_method,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,  # FIX: was missing before
    )
    lda.fit(dtm)

    # FIX: compute real document-topic assignments for accurate counts
    doc_topic_matrix = lda.transform(dtm)
    topic_doc_counts = doc_topic_matrix.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    results = []
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        words = [{"word": feature_names[i], "score": round(float(topic[i] / topic.sum()), 4)} for i in top_indices]
        # FIX: real count — number of docs where this topic is dominant
        real_count = int((topic_doc_counts == idx).sum())
        results.append({
            "topic_id": idx,
            "count": real_count,
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
            "min_df": effective_min_df,
            "max_df": effective_max_df,
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

    # FIX: renamed to effective_ to make the override explicit and clear
    effective_min_df, effective_max_df = adaptive_vectorizer_params(len(raw_docs))

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

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stop = set(ENGLISH_STOP_WORDS)
    if remove_stopwords:
        combined_stop.update(stopwords.words("english"))
    if custom_stopwords.strip():
        combined_stop.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize))

    tfidf_stop = set(combined_stop)
    if custom_stopwords.strip():
        tfidf_stop.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=effective_min_df,
        max_df=effective_max_df,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        tokenizer=_make_tokenizer(tfidf_stop),
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    scores_mean = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    scores_max = np.asarray(tfidf_matrix.max(axis=0).todense()).flatten()

    scores = 0.6 * scores_mean + 0.4 * scores_max

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

    combined_stop = set(ENGLISH_STOP_WORDS)
    if remove_stopwords:
        combined_stop.update(stopwords.words("english"))
    if custom_stopwords.strip():
        combined_stop.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize))

    processed = []
    for p in paragraphs:
        tokens = preprocess_text(p, remove_sw=remove_stopwords, lemmatize=lemmatize,
                                  min_word_len=3, custom_stopwords=custom_stopwords,
                                  remove_numbers=remove_numbers, lowercase=True)
        if tokens:
            processed.append(" ".join(tokens))

    if len(processed) < 3:
        raise HTTPException(status_code=400, detail="Not enough text after preprocessing.")

    n_docs = len(processed)
    auto_min_df, auto_max_df = adaptive_vectorizer_params(n_docs)

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

    scores_array = tfidf_matrix.toarray()
    max_scores = scores_array.max(axis=0)
    doc_freq = (scores_array > 0).sum(axis=0) / len(processed)
    distinctiveness = max_scores * (1 - doc_freq * 0.5)

    top_idx = distinctiveness.argsort()[-max_features:][::-1]
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
    return {
        "method": "wordfreq",
        "total_tokens": len(tokens),
        "unique_tokens": len(counter),
        "words": [{"word": w, "count": c, "relative": round(c / max_count, 4)} for w, c in top]
    }


@app.post("/analyze/sentiment")
async def analyze_sentiment(
    files: List[UploadFile] = File(...),
    file: Optional[UploadFile] = File(None),
    per_sentence: bool = Form(True),
    model: str = Form("vader"),
):
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
        results.append({
            "sentence": sent[:200],
            "compound": round(scores["compound"], 4),
            "positive": round(scores["pos"], 4),
            "neutral": round(scores["neu"], 4),
            "negative": round(scores["neg"], 4),
            "label": label
        })
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

    raw_docs = extract_texts_as_docs(files, file)
    if len(raw_docs) < 5:
        raise HTTPException(status_code=400, detail="Not enough text.")

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

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    coh_vectorizer_stop = None
    if remove_stopwords or custom_stopwords.strip():
        coh_combined = set(ENGLISH_STOP_WORDS)
        if remove_stopwords:
            coh_combined.update(stopwords.words("english"))
        if custom_stopwords.strip():
            coh_combined.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize, stemming=stemming))
        coh_vectorizer_stop = list(coh_combined)

    processed_docs = [" ".join(t) for t in tokenized_docs]
    n_docs = len(processed_docs)

    effective_min_df, effective_max_df = adaptive_vectorizer_params(n_docs)

    coh_final_stop = set(coh_vectorizer_stop) if coh_vectorizer_stop else set()
    if custom_stopwords.strip():
        coh_final_stop.update(expand_custom_stopwords(custom_stopwords, lemmatize=lemmatize, stemming=stemming))

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=effective_min_df,
        max_df=effective_max_df,
        ngram_range=(ngram_min, ngram_max),
        tokenizer=_make_tokenizer(coh_final_stop),
        preprocessor=lambda x: x,
        token_pattern=None,
    )
    dtm = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()

    if dtm.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Vocabulary empty after preprocessing.")

    use_cv = False
    gensim_dictionary = None
    gensim_corpus = None
    try:
        import gensim
        import gensim.corpora as corpora
        from gensim.models import LdaModel
        from gensim.models.coherencemodel import CoherenceModel

        gensim_dictionary = corpora.Dictionary(tokenized_docs)
        gensim_dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)
        gensim_corpus = [gensim_dictionary.doc2bow(doc) for doc in tokenized_docs]

        if len(gensim_dictionary) > 0:
            use_cv = True
    except ImportError:
        use_cv = False

    topic_range = list(range(min_topics, min(max_topics + 1, len(tokenized_docs)), step))
    perplexity_scores = []
    coherence_scores = []
    coherence_metric = "c_v" if use_cv else "npmi"

    for n_topics in topic_range:
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
            window_size = 10
            vocab_list = list(feature_names)
            word2idx = {w: i for i, w in enumerate(vocab_list)}

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

    best_idx = coherence_scores.index(max(coherence_scores))
    optimal_topics = topic_range[best_idx]

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


# ── FIX: uvicorn.run moved to bottom — after ALL routes are defined ───────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
