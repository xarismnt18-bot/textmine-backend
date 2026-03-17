diff --git a/README.md b/README.md
index e556d720706bd07dab451f3600df9626191f73d4..b5586aaad286b21c79b36734403815a75c836d28 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,52 @@
-# textmine-backend
\ No newline at end of file
+# textmine-backend
+
+FastAPI backend for text analysis (TF-IDF, word frequency, sentiment, and BERTopic).
+
+## Run locally
+
+```bash
+pip install -r requirements.txt
+python main.py
+```
+
+Server starts on `http://localhost:10000` by default.
+
+## BERTopic endpoint
+
+`POST /analyze/bertopic`
+
+Form fields:
+- `file`: `.txt`, `.pdf`, or `.xlsx`
+- `num_topics` (default: `10`)
+- `min_topic_size` (default: `15`)
+- `language` (default: `english`)
+- `reduce_outliers` (default: `true`)
+- `engine` (`local` or `colab`, default: `local`)
+
+### Local BERTopic mode
+
+Use `engine=local` to run BERTopic directly in this backend.
+
+### Google Colab mode
+
+Use `engine=colab` to delegate BERTopic processing to a Colab-hosted webhook.
+
+Set this environment variable in your backend before starting it:
+
+```bash
+export COLAB_BERTOPIC_URL="https://<your-colab-endpoint>"
+```
+
+Your Colab service should accept JSON like:
+
+```json
+{
+  "docs": ["sentence one", "sentence two"],
+  "num_topics": 10,
+  "min_topic_size": 15,
+  "language": "english",
+  "reduce_outliers": true
+}
+```
+
+And return a BERTopic-style payload (topics, counts, top words).
