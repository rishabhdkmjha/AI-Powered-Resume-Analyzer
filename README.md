# Resume Category Classification using NLP

## 1. Project Overview
This project is a machine learning pipeline that classifies resumes into different job categories. It uses Natural Language Processing (NLP) techniques to process resume text and a Logistic Regression model to predict the most relevant category. The system is designed to automate the initial screening process for recruiters.

---

## 2. Dataset
The model is trained on the **Updated Resume Dataset** from Kaggle, which contains resumes sorted into categories like 'Data Science', 'HR', 'Advocate', and more.

- **Source**: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauthambv/resume-dataset)
- **To run this project**, please download the dataset from the link above and place `UpdatedResumeDataSet.csv` in the root directory.

---

## 3. Workflow
1.  **Data Loading**: Resumes are loaded from `UpdatedResumeDataSet.csv` using pandas.
2.  **Text Preprocessing**: Each resume's text is cleaned by converting it to lowercase and removing special characters.
3.  **Feature Extraction**: The cleaned text is converted into numerical features using `TfidfVectorizer`.
4.  **Model Training**: A Logistic Regression classifier is trained on the vectorized text data.
5.  **Evaluation**: The model's performance is evaluated on a held-out test set using metrics like accuracy and precision.
6.  **Persistence**: The trained model and vectorizer are saved as `.pkl` files using `joblib` for future use.

---

## 4. Technologies Used
- **Python**
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning (TF-IDF, Logistic Regression, and evaluation metrics).
- **Joblib**: For saving and loading the trained model.
- **Matplotlib**: For data visualization.

---

## 5. How to Run the Project

### Prerequisites
- Python 3.x
- Install the required libraries:
  ```bash
  pip install pandas scikit-learn joblib matplotlib
