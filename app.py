import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Function to clean text ---
def clean_text(text):
    return re.sub(r'\W+', ' ', str(text).lower())

# --- Load Model and Vectorizer ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("final_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Streamlit App Interface ---
st.title("NLP-Based Resume Evaluation System")
st.write("Upload resumes and paste a job description to rank candidates.")

# --- Input from User ---
job_description = st.text_area("Enter the Job Description Here", height=200)
uploaded_files = st.file_uploader("Upload Candidate Resumes (TXT files)", type=["txt"], accept_multiple_files=True)

if st.button("Evaluate and Rank Resumes"):
    if not job_description:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # --- Processing and Ranking Logic ---
        results = []
        
        # Clean and vectorize the job description
        cleaned_jd = clean_text(job_description)
        jd_vec = vectorizer.transform([cleaned_jd])

        for file in uploaded_files:
            # Read and decode the uploaded file
            resume_text = file.read().decode('utf-8', errors='ignore')
            cleaned_resume = clean_text(resume_text)
            
            # 1. Classify the resume's category
            resume_vec = vectorizer.transform([cleaned_resume])
            predicted_category = model.predict(resume_vec)[0]
            
            # 2. Calculate Similarity Score
            similarity_score = cosine_similarity(jd_vec, resume_vec)[0][0]
            
            results.append({
                "Filename": file.name,
                "Predicted Category": predicted_category,
                "Similarity Score": f"{similarity_score * 100:.2f}%"
            })
            
        # --- Display Results ---
        st.subheader("Evaluation Results")
        
        # Sort results by similarity score (descending)
        ranked_results = sorted(results, key=lambda x: float(x['Similarity Score'][:-1]), reverse=True)
        
        # Display as a DataFrame
        df_results = pd.DataFrame(ranked_results)
        st.dataframe(df_results)
