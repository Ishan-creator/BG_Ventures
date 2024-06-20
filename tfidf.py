import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

descriptions_to_match = [
    "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
    "Providing, mixing and placing 75mm thick PCC (1:3:6) using PPC Cement equivalent in foundation trenches and plinths.",
    "Earthback filling in foundation trenches including watering, consolidating drawing, specification & instructions.",
    "Moulding or Nosing works in Granite edges in skirting and tops all complete"
]


def get_norms(path):
    df1 = pd.read_csv(path)
    df = df1["Norms"].dropna()
    return df


target_descriptions = get_norms("norms_final.csv")
target_descriptions = target_descriptions.tolist()


def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = text.split()  
    return ' '.join(tokens)


preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]


all_descriptions = preprocessed_descriptions_to_match + preprocessed_target_descriptions


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_descriptions)


cosine_similarities = cosine_similarity(tfidf_matrix[:len(descriptions_to_match)], tfidf_matrix[len(descriptions_to_match):])

best_matches = np.argmax(cosine_similarities, axis=1)


for i, match_index in enumerate(best_matches):
    print(f"Description to match: {descriptions_to_match[i]}")
    print(f"Best matched description: {target_descriptions[match_index]}")
    print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
    print("-" * 80)
