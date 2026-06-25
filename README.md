# AI-Powered Resume Analyzer

An NLP-based system that classifies resumes into job categories and ranks candidates against a job description, built with scikit-learn and Streamlit.

## What it does

This project has two parts:

1. **Resume Category Classifier** — a Random Forest model trained on TF-IDF text features that predicts which of 25 job categories (e.g. Data Science, HR, DevOps Engineer, Java Developer) a resume belongs to.
2. **Candidate Ranking App** — a Streamlit web app where a recruiter pastes a job description and uploads multiple candidate resumes (`.txt`). Each resume is labeled with its predicted category, and all resumes are ranked by **cosine similarity** between their TF-IDF vector and the job description's TF-IDF vector.

**Important distinction:** the Random Forest classifier and the ranking mechanism are independent. The classifier only assigns a category label. Ranking is driven entirely by cosine similarity on TF-IDF vectors, not by the classifier.

## How it works

### Preprocessing
Resume text is lowercased and stripped of non-word characters using a simple regex (`re.sub(r'\W+', ' ', text.lower())`).

### Feature extraction — TF-IDF
Text is converted into numerical features using `TfidfVectorizer(max_features=1000, stop_words='english')`. This scores words by how frequent they are in a given resume *and* how rare they are across all resumes — so common filler words are downweighted and category-distinguishing terms (e.g. "Tensorflow", "litigation") are emphasized.

### Classification — Random Forest
A `RandomForestClassifier(n_estimators=100, random_state=42)` is trained on the TF-IDF features to predict job category. Random Forest combines 100 decision trees trained on random subsets of the data, reducing overfitting compared to a single tree.

### Ranking — Cosine Similarity
For the deployed app, the job description and each resume are both converted to TF-IDF vectors. Cosine similarity is computed between the job description vector and each resume vector, and resumes are sorted by this score — independent of the classifier's category prediction.

## Model Performance

Trained and evaluated on the public Kaggle "Resume Dataset" (`UpdatedResumeDataSet.csv`, ~962 resumes across 25 categories), using an 80/20 train-test split (`random_state=42`):

- **Test accuracy: 99.48%** (193 test samples)
- Per-category precision/recall is 1.00 for most categories; two exceptions: `DevOps Engineer` (precision 1.00, recall 0.93, n=14) and `PMO` (precision 0.88, recall 1.00, n=7)

**Caveat on these numbers:** the test set is small (193 samples) and unevenly distributed across 25 categories — several categories have only 3-4 test examples, meaning a single misclassification swings that category's score by 25-33%. Near-perfect accuracy on a small test split like this should be treated cautiously rather than as a finished benchmark. Two likely explanations: (a) these job categories use sufficiently distinct vocabulary that the task is genuinely easier than typical text classification, or (b) there may be near-duplicate resumes split across train/test (a known characteristic of this public dataset), inflating the score. No deduplication or leakage check has been run yet — that would be the first step before trusting this number for a production use case.

**A more rigorous evaluation** would use k-fold cross-validation (averaging accuracy across multiple train/test splits) rather than relying on a single 80/20 split, plus an explicit near-duplicate check between train and test resumes.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Feature extraction | scikit-learn `TfidfVectorizer` |
| Classifier | scikit-learn `RandomForestClassifier` |
| Ranking | `sklearn.metrics.pairwise.cosine_similarity` |
| Web app | Streamlit |
| Model persistence | `joblib` |

## Files

- `train_final_model.py` — trains the TF-IDF vectorizer + Random Forest classifier, saves both as `.pkl` files
- `app.py` — Streamlit app: upload resumes + job description, get ranked + classified results
- `final_model.pkl`, `vectorizer.pkl` — saved trained artifacts
- `Resume_Analyser.ipynb` — exploratory notebook with the original training/evaluation runs

## Known Limitations

- **Small, imbalanced test set** for accuracy reporting (see Model Performance above)
- **No deduplication check** between train/test splits on the source dataset
- **TF-IDF + cosine similarity for ranking** only captures vocabulary overlap, not semantic meaning — a resume describing "led a team" won't strongly match a job description asking for "management experience" despite the conceptual overlap. Sentence embeddings would likely improve ranking quality.
- **Classifier and ranking are independent** — the predicted category is informational only and does not influence the similarity-based ranking order.

## Possible Improvements

- Replace single train/test split evaluation with k-fold cross-validation for a more reliable accuracy estimate
- Add a near-duplicate detection step on the source dataset before splitting
- Swap TF-IDF + cosine similarity for sentence embeddings (e.g. `sentence-transformers`) to capture semantic similarity in ranking, not just keyword overlap
- Expand training data beyond the ~962-resume public dataset for more robust per-category performance, especially for categories with very few examples (Advocate, Network Security Engineer, Business Analyst — all under 10 test samples)
