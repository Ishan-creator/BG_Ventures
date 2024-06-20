import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import gensim.downloader as api


fasttext_model = api.load("fasttext-wiki-news-subwords-300")

# Sample descriptions to be matched
descriptions_to_match = [
    "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
    "Providing, mixing and placing 75mm thick PCC (1:3:6) using PPC Cement equivalent in foundation trenches and plinths.",
    "Earthback filling in foundation trenches including watering, consolidating drawing, specification & instructions."
]

# Function to fetch target descriptions from a CSV file
def get_norms(path):
    df1 = pd.read_csv(path)
    df = df1["Norms"].dropna()
    return df

# Fetch the target descriptions from the CSV file
target_descriptions = get_norms("norms_final.csv")
target_descriptions = target_descriptions.tolist()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize
    return tokens

# Preprocess both lists of descriptions
preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]

# Function to get sentence vector using FastText embeddings
def get_sentence_vector(tokens, model, vector_size):
    vector = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model:
            vector += model[word]
            count += 1
    if count != 0:
        vector /= count
    return vector

# Get sentence vectors for descriptions to match and target descriptions
fasttext_vector_size = fasttext_model.vector_size  # Get vector size from the model
fasttext_descriptions_to_match = [get_sentence_vector(tokens, fasttext_model, fasttext_vector_size) for tokens in preprocessed_descriptions_to_match]
fasttext_target_descriptions = [get_sentence_vector(tokens, fasttext_model, fasttext_vector_size) for tokens in preprocessed_target_descriptions]

# Convert lists to numpy arrays for cosine similarity calculation
fasttext_descriptions_to_match = np.array(fasttext_descriptions_to_match)
fasttext_target_descriptions = np.array(fasttext_target_descriptions)

# Compute cosine similarities between descriptions to match and target descriptions
cosine_similarities = cosine_similarity(fasttext_descriptions_to_match, fasttext_target_descriptions)

# Find the best match for each description
best_matches = np.argmax(cosine_similarities, axis=1)

# Display the matches
for i, match_index in enumerate(best_matches):
    print(f"Description to match: {descriptions_to_match[i]}")
    print(f"Best matched description: {target_descriptions[match_index]}")
    print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
    print("-" * 80)
