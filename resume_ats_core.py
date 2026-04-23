{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "af6f98f0-2782-4853-bdf7-8f24591c6a68",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to C:\\Users\\R Y Z E\n",
      "[nltk_data]     N\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "import PyPDF2\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from nltk.corpus import stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ed044588-9296-4018-8fcc-9442c60c24e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# TEXT EXTRACTION\n",
    "def extract_text_from_pdf(file_path):\n",
    "    text = \"\"\n",
    "    with open(file_path, 'rb') as f:\n",
    "        reader = PyPDF2.PdfReader(f)\n",
    "        for page in reader.pages:\n",
    "            if page.extract_text():\n",
    "                text += page.extract_text()\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e1187dd4-b6c5-4bbd-a1b4-2b625e5ca3e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# PREPROCESSING\n",
    "def preprocess(text):\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    words = text.lower().split()\n",
    "    words = [w for w in words if w.isalnum() and w not in stop_words]\n",
    "    return \" \".join(words)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "99b9e3a3-8710-4c92-ba48-e4199b18f9e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SIMILARITY MODEL\n",
    "def get_similarity(resume_text, jd_text):\n",
    "    vectorizer = TfidfVectorizer()\n",
    "    vectors = vectorizer.fit_transform([resume_text, jd_text])\n",
    "    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]\n",
    "    return score * 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "224eeecb-4e08-4f33-a82d-c66d85519182",
   "metadata": {},
   "outputs": [],
   "source": [
    "# MATCHING KEYWORDS\n",
    "def get_keywords(resume_text, jd_text):\n",
    "    resume_words = set(resume_text.split())\n",
    "    jd_words = set(jd_text.split())\n",
    "    return list(resume_words.intersection(jd_words))[:15]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b157afa6-c0df-4d7c-b38e-3a5daff97ab9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# INTERPRETATION\n",
    "def interpret(score):\n",
    "    if score > 80:\n",
    "        return \"Strong Match\"\n",
    "    elif score > 50:\n",
    "        return \"Moderate Match\"\n",
    "    else:\n",
    "        return \"Weak Match\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (shopgenie)",
   "language": "python",
   "name": "shopgenie"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
