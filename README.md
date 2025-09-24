NLP-Based Resume Evaluation System
🚀 Overview
This project is an intelligent system designed to automate the initial stages of the recruitment process. It uses machine learning and natural language processing to classify resumes into job categories and rank them against a given job description.

The goal is to help recruiters and hiring managers quickly identify the most qualified candidates from a large pool of applicants, saving significant time and effort. The project is deployed as an interactive web application using Streamlit.

✨ Features
Automated Resume Classification: Automatically categorizes resumes into predefined roles (e.g., Data Scientist, Web Developer) using a highly accurate Random Forest model.

JD-Resume Matching: Ranks candidates based on the contextual similarity between their resume and the job description.

Similarity Scoring: Utilizes TF-IDF and Cosine Similarity to provide a quantitative score for each candidate's relevance.

Interactive Web App: A user-friendly interface built with Streamlit that allows for easy uploading of resumes and job descriptions.

🛠️ Tech Stack
Backend: Python

Machine Learning: Scikit-learn (RandomForest, TfidfVectorizer)

NLP & Data Processing: Pandas, NLTK (for potential future enhancements)

Web Framework: Streamlit

Serialization: Joblib

⚙️ How It Works
The system follows a robust NLP pipeline to process and evaluate resumes:

Text Preprocessing: Raw text from resumes is cleaned by removing special characters, converting to lowercase, and handling noise.

TF-IDF Vectorization: The cleaned text is converted into numerical feature vectors using TF-IDF, which captures the importance of different keywords.

Resume Classification: The pre-trained Random Forest model predicts the job category for each resume.

Cosine Similarity Ranking: For resumes that match the relevant category, the system calculates the cosine similarity between the job description's TF-IDF vector and each resume's vector.

Ranked Output: The Streamlit application displays a ranked list of candidates, sorted from most to least relevant based on their similarity score.

📈 Model Performance
The core classification model was trained and evaluated rigorously to ensure high performance and reliability.

Model: Random Forest Classifier

Validation Method: K-Fold Cross-Validation

Classification Accuracy: 99.48%

🔧 Setup and Installation
To run this project locally, please follow these steps:

Clone the repository:

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:

Bash

pip install -r requirements.txt
Run the Streamlit application:

Bash

streamlit run app.py
The application will now be running and accessible in your web browser.
