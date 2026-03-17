# textmine-backend

FastAPI backend for text analysis (TF-IDF, word frequency, sentiment, and BERTopic).

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

Server starts on `http://localhost:10000` by default.

## BERTopic endpoint

`POST /analyze/bertopic`

Form fields:
- `file`: `.txt`, `.pdf`, or `.xlsx`
- `num_topics` (default: `10`)
- `min_topic_size` (default: `15`)
- `language` (default: `english`)
- `reduce_outliers` (default: `true`)
- `engine` (`local` or `colab`, default: `local`)

### Local BERTopic mode

Use `engine=local` to run BERTopic directly in this backend.

### Google Colab mode

Use `engine=colab` to delegate BERTopic processing to a Colab-hosted webhook.

Set this environment variable in your backend before starting it:

```bash
export COLAB_BERTOPIC_URL="https://<your-colab-endpoint>"
```

Your Colab service should accept JSON like:

```json
{
  "docs": ["sentence one", "sentence two"],
  "num_topics": 10,
  "min_topic_size": 15,
  "language": "english",
  "reduce_outliers": true
}
```

And return a BERTopic-style payload (topics, counts, top words).


## Deploy on Render

Use these Render settings (or keep `render.yaml` in this repo):
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Fix for your current error

If Render logs show:

`Invalid requirement: 'diff --git a/requirements.txt b/requirements.txt'`

then your **GitHub `requirements.txt` file contains git diff text** instead of only package lines.
Replace the file with clean dependency lines only (the exact file in this repo), commit, and redeploy.

Quick check before pushing:

```bash
python -m pip install -r requirements.txt
```

If this command fails locally with the same error, your `requirements.txt` still has extra diff/comment text.
