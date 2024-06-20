import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model_path = 'roberta-base-nli-stsb-mean-tokens'  
model = SentenceTransformer(model_path)

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
    tokens = text.split()  
    return ' '.join(tokens)

preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]

sentence_embeddings_to_match = model.encode(preprocessed_descriptions_to_match, convert_to_tensor=True)
sentence_embeddings_target = model.encode(preprocessed_target_descriptions, convert_to_tensor=True)

embeddings_to_match = sentence_embeddings_to_match.numpy()
embeddings_target = sentence_embeddings_target.numpy()

cosine_similarities = cosine_similarity(embeddings_to_match, embeddings_target)

best_matches = np.argmax(cosine_similarities, axis=1)

for i, match_index in enumerate(best_matches):
    print(f"Description to match: {descriptions_to_match[i]}")
    print(f"Best matched description: {target_descriptions[match_index]}")
    print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
    print("-" * 80)
