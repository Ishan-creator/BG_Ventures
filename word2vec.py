import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Sample descriptions to be matched
descriptions_to_match = [
    "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
    "Providing, mixing and placing 75mm thick PCC (1:3:6) using PPC Cement equivalent in foundation trenches and plinths.",
    "Earthback filling in foundation trenches including watering, consolidating drawing, specification & instructions."
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
    tokens = text.split()  # Tokenize
    return tokens

preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]

all_descriptions = preprocessed_descriptions_to_match + preprocessed_target_descriptions
word2vec_model = Word2Vec(all_descriptions, vector_size=300, window=5, min_count=1, sg=1)

def get_average_word_vector(tokens, model, vector_size):
    vector_sum = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vector_sum += model.wv[word]
            count += 1
    if count != 0:
        return vector_sum / count
    else:
        return vector_sum

vector_size = 300  
w2v_descriptions_to_match = [get_average_word_vector(tokens, word2vec_model, vector_size) for tokens in preprocessed_descriptions_to_match]
w2v_target_descriptions = [get_average_word_vector(tokens, word2vec_model, vector_size) for tokens in preprocessed_target_descriptions]

w2v_descriptions_to_match = np.array(w2v_descriptions_to_match)
w2v_target_descriptions = np.array(w2v_target_descriptions)

cosine_similarities = cosine_similarity(w2v_descriptions_to_match, w2v_target_descriptions)

best_matches = np.argmax(cosine_similarities, axis=1)

for i, match_index in enumerate(best_matches):
    print(f"Description to match: {descriptions_to_match[i]}")
    print(f"Best matched description: {target_descriptions[match_index]}")
    print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
    print("-" * 80)
