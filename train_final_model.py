import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Preprocess the resume text
def clean_text(text):
    return re.sub(r'\W+', ' ', str(text).lower())

X = df['Resume'].apply(clean_text)
y = df['Category']

# TF-IDF vectorizer (this should be the same as before)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# --- The Key Change: Use RandomForestClassifier ---
# This is a more powerful model than Logistic Regression
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the final model
y_pred = model.predict(X_test)
print("\nClassification Report for the Final Model:")
print(classification_report(y_test, y_pred))

# Save the final model and vectorizer
joblib.dump(model, "final_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n Final model and vectorizer saved successfully!")
