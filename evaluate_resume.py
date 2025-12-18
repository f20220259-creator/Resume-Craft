import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# --- Cleaning Function ---
def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip().lower()

# --- Metrics ---
def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def jaccard_sim(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

def fuzzy_score(text1, text2):
    return fuzz.token_set_ratio(text1, text2) / 100

def tfidf_score(text1, text2):
    return cosine_sim(text1, text2)  # TF-IDF + cosine

# --- Example Run ---
jd = open("jd.txt").read()
tailored_resume = open("tailored_resume.txt").read()

jd, tailored_resume = clean_text(jd), clean_text(tailored_resume)

# Scores after tailoring
tail_cosine = cosine_sim(tailored_resume, jd)
tail_jaccard = jaccard_sim(tailored_resume, jd)
tail_fuzzy = fuzzy_score(tailored_resume, jd)
tail_tfidf = tfidf_score(tailored_resume, jd)

print("📊 Tailored Resume vs JD")
print(f"Cosine Similarity: {tail_cosine:.4f}")
print(f"Jaccard Similarity: {tail_jaccard:.4f}")
print(f"Fuzzy Score: {tail_fuzzy:.4f}")
print(f"TF-IDF Score: {tail_tfidf:.4f}")
