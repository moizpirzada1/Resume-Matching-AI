import nltk
nltk.download('stopwords')

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


# PDF TEXT EXTRACTION
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


# PREPROCESSING
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [w for w in words if w.isalnum() and w not in stop_words]
    return " ".join(words)


# SIMILARITY
def get_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return score * 100


# KEYWORDS
def get_keywords(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    return list(resume_words.intersection(jd_words))[:15]


# INTERPRETATION
def interpret(score):
    if score > 80:
        return "Strong Match"
    elif score > 50:
        return "Moderate Match"
    else:
        return "Weak Match"
