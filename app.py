import streamlit as st
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def cleanText(text):
    # Remove URLs, mentions, hashtags, and non-letter words
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\S*[^a-zA-Z\s]\S*', '', text)  # Removes words containing any non-letters

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize and remove stopwords
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in set(stopwords.words('english'))]

    clean_sentence = ' '.join(filtered_words)
    return clean_sentence


# Web App
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])
    if uploaded_file is not None:
        resume_bytes = uploaded_file.read()
        try:
            resume_txt = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt = resume_bytes.decode('latin-1')  # fallback decoding

        # Clean text (pass string, not list)
        cleaned_resume_txt = cleanText(resume_txt)

        # Transform text using TF-IDF
        cleaned_resume_tfidf = tfidf.transform([cleaned_resume_txt])

        # Predict category / ID
        prediction_id = clf.predict(cleaned_resume_tfidf)[0]
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category: ", category_name)



if __name__ == "__main__":
    main()
