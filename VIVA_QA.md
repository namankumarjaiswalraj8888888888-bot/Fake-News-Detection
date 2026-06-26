# 🎓 Viva Questions & Answers
## Fake News Detection System — AI & ML Internship

---

### Q1. What is Fake News Detection?
**A:** Fake news detection is the task of automatically classifying news articles as
**real** (factual, credible) or **fake** (fabricated, misleading, or satirical) using
Natural Language Processing (NLP) and Machine Learning techniques.

---

### Q2. Which machine learning algorithm did you use and why?
**A:** I used **Logistic Regression** because:
- It's a proven, interpretable binary classifier
- Works extremely well with high-dimensional TF-IDF features
- Provides calibrated probability estimates (useful for confidence %)
- Trains fast even on large text datasets
- Avoids overfitting with L2 regularization built in

---

### Q3. What is TF-IDF and how does it work?
**A:** TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts raw
text into numerical vectors by:
- **TF** (Term Frequency): How often a word appears in a document — frequent words in that document get higher weight
- **IDF** (Inverse Document Frequency): Words common across ALL documents (like "the", "is") get penalized (lower weight)
- Together, TF-IDF highlights words that are **unique and important** to a specific document

---

### Q4. What preprocessing did you apply to the text?
**A:** The `clean_text()` function in `utils.py` performs:
1. **Lowercase conversion** — "The" and "the" become the same word
2. **URL removal** — removes http/https links
3. **HTML tag removal** — strips `<b>`, `<p>` etc.
4. **Punctuation removal** — removes !@#$% etc.
5. **Whitespace normalization** — collapses multiple spaces into one

---

### Q5. What is the difference between TF-IDF and Bag of Words?
**A:**
| Feature | Bag of Words | TF-IDF |
|---|---|---|
| Representation | Raw word counts | Weighted counts |
| Common words | Over-weighted | Penalized by IDF |
| Unique words | Under-valued | Boosted |
| Information | Less discriminative | More discriminative |

TF-IDF is generally better for text classification.

---

### Q6. What is Logistic Regression? Is it a classification or regression algorithm?
**A:** Despite the name, Logistic Regression is a **classification** algorithm. It uses the
**sigmoid function** to map any real-valued input to a probability between 0 and 1.
For binary classification (Real/Fake), if the probability > 0.5, it predicts class 1 (Real),
otherwise class 0 (Fake).

---

### Q7. What is a confusion matrix?
**A:** A confusion matrix is a table that shows:
- **True Positives (TP)**: Correctly predicted Real news
- **True Negatives (TN)**: Correctly predicted Fake news
- **False Positives (FP)**: Fake predicted as Real (dangerous!)
- **False Negatives (FN)**: Real predicted as Fake

From these, we compute Precision, Recall, and F1-Score.

---

### Q8. What is precision, recall, and F1-score?
**A:**
- **Precision** = TP / (TP + FP) — Of all predicted Real, how many were actually Real?
- **Recall** = TP / (TP + FN) — Of all actual Real, how many did we catch?
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — Harmonic mean, balances both

---

### Q9. What dataset did you use?
**A:** The project supports the **WELFake Dataset** (publicly available on Kaggle) which
contains ~72,000 news articles labeled as Real or Fake. It combines four public datasets:
Kaggle, McIntire, Reuters, and BuzzFeed Political. For demonstration, a synthetic dataset
is auto-generated if the real dataset isn't available.

---

### Q10. What is train-test split and why do we need it?
**A:** We split the dataset into:
- **Training set (80%)**: The model learns patterns from this data
- **Test set (20%)**: Never seen by the model during training — used to measure real-world performance

Without a test set, we can't know if the model has truly learned or just memorized the training data (**overfitting**).

---

### Q11. What is overfitting and how did you prevent it?
**A:** Overfitting occurs when the model memorizes training data and fails on new data.
Prevention methods used:
- **Regularization** (C parameter in Logistic Regression) — penalizes complex models
- **Train-test split** — evaluate on unseen data
- **min_df=2** in TF-IDF — ignore words appearing in only one document
- **max_features=50000** — cap the vocabulary size

---

### Q12. What is Gradio and why did you use it?
**A:** Gradio is a Python library that lets you build interactive web UIs for ML models
in just a few lines of code. Benefits:
- No HTML/CSS/JS knowledge needed
- Automatic mobile-responsive layout
- One-click deployment to Hugging Face Spaces
- Built-in support for loading indicators, file uploads, etc.

---

### Q13. What is Hugging Face Spaces?
**A:** Hugging Face Spaces is a free hosting platform for ML applications. It:
- Provides a public HTTPS URL for your app
- Automatically installs requirements.txt
- Supports Gradio, Streamlit, and Docker apps
- Is free for public projects

---

### Q14. What are ngrams and why did you use ngram_range=(1,2)?
**A:** An n-gram is a sequence of N consecutive words.
- **Unigram (1-gram)**: "fake", "news", "breaking"
- **Bigram (2-gram)**: "fake news", "breaking news", "hidden truth"

Using `ngram_range=(1,2)` captures both individual words AND two-word phrases, which improves detection of characteristic fake news patterns like "share before", "mainstream media", "they don't".

---

### Q15. How does the confidence score work?
**A:** Logistic Regression's `predict_proba()` returns probabilities for each class:
- e.g., [0.08, 0.92] means 8% chance Fake, 92% chance Real
- Confidence = `max(probabilities) × 100`
- This is displayed as a percentage with a color-coded progress bar

---

### Q16. What would you do to improve accuracy further?
**A:** Possible improvements:
1. Use **BERT** or **RoBERTa** (deep learning transformer models) for context-aware embeddings
2. Use a larger, cleaner dataset (e.g., ISOT with 44k articles)
3. Add **metadata features**: author, source URL, publication date
4. Ensemble methods: combine multiple classifiers (Random Forest + SVM + Logistic Regression)
5. Apply **SMOTE** if dataset is imbalanced

---

### Q17. What is the pickle module used for?
**A:** Python's `pickle` module serializes (saves) Python objects to binary files.
We use it to save the trained `model` and `vectorizer` to `.pkl` files so we don't
have to re-train every time the app starts. On startup, `app.py` loads these files.

---

### Q18. What is the role of utils.py?
**A:** `utils.py` is a helper module containing reusable functions:
- `clean_text()` — text preprocessing (used by both training and inference)
- `get_confidence_label()` — converts % to human-readable label
- `get_sample_news()` — returns demo news snippets for the UI

Keeping utilities separate follows the **Single Responsibility Principle** of good software design.

---

### Q19. What challenges did you face and how did you solve them?
**A:** Key challenges:
1. **Text variability** — solved with thorough preprocessing
2. **Large vocabulary** — solved with `max_features=50000` cap
3. **Imbalanced data** — used `stratify=y` in train_test_split
4. **Slow prediction** — solved by using Logistic Regression (millisecond inference)
5. **Mobile UI** — Gradio handles this automatically

---

### Q20. How would you explain this project during a job interview?
**A:** "I built an end-to-end AI system that detects fake news articles. The system converts
raw text into numerical features using TF-IDF vectorization, then classifies them using
Logistic Regression with 96%+ accuracy. I deployed the model as a live web application
using Gradio on Hugging Face Spaces, where anyone can paste a news article and get an
instant prediction with a confidence score. The entire project is production-ready with
proper error handling, code documentation, and a professional UI."

---

### Q21. What is the difference between supervised and unsupervised learning?
**A:**
- **Supervised Learning**: Training data has labels (Real=1, Fake=0). Model learns mapping from features → labels. *This project uses supervised learning.*
- **Unsupervised Learning**: No labels. Model finds hidden patterns (e.g., clustering similar articles).

---

### Q22. Can this model detect misinformation in other languages?
**A:** No — the current model is trained on English text only. For multilingual support:
1. Train separate models per language
2. Use multilingual embeddings (e.g., `multilingual-BERT`)
3. Apply translation as a preprocessing step

---

*Good luck with your viva! Remember: understand each concept, don't just memorize.*
