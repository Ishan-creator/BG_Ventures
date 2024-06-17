# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer, util

# # Sample documents from two corpora
# corpus1 = [
# "Brick Work. Supplying preparing and construction of chimney made brick work of approved quality with 1:3 cement sand mortar ratio with 30 m lead"
# ]

# corpus2 = [
# "Construction of Temporary shelter for material storage and workers room.Clearing & grubbing of site along with layout of  buidling for construction as per drawing specification &  instructions.",
# "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
# "Earthback filling in foundation trenches including watering, consolidating drawing , specification & instructions.",
# "Sand filling in Plinth under floors of 50mm thick including watering, consolidating and dressing complete as per drawing, specification & instruction"
# ]

# model = SentenceTransformer('all-MiniLM-L6-v2')


# embeddings1 = model.encode(corpus1, convert_to_tensor=True)
# embeddings2 = model.encode(corpus2, convert_to_tensor=True)

# similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)

# mapping = similarity_matrix.argmax(dim=1)
# similarity_scores = similarity_matrix.max(dim=1).values

# for i, doc in enumerate(corpus1):
#     matched_doc_index = mapping[i].item()
#     print(f"Document in corpus1: '{doc}'")
#     print(f"-> Most similar document in corpus2: '{corpus2[matched_doc_index]}' with similarity score: {similarity_scores[i]:.4f}\n")


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(texts):
    processed_texts = []
    
    for text in texts:
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]
        
        tokens = [[handle_special_terms(word) for word in sentence] for sentence in tokens]
        
        tokens = [[word for word in sentence if word.isalnum()] for sentence in tokens]
        
        tokens = [[word for word in sentence if not word.isdigit()] for sentence in tokens]
        
        processed_sentences = [" ".join(sentence) for sentence in tokens]
        
        processed_texts.append(processed_sentences)
    
    return processed_texts

def handle_special_terms(word):
    if re.match(r'^\d+:\d+$', word):
        return 'ratio_' + word.replace(':', '_')
    if re.match(r'^\d+\s*m$', word):
        return word.replace(' ', '_') + '_meter'
    return word



# if __name__ == "__main__":
#     # Example with multiple texts
#     sample_texts = [
#         "Brick Work. Supplying preparing and construction of chimney made brick work of approved quality with 1:3 cement sand mortar ratio with 30 m lead.",
#         "The quick brown fox jumps over the lazy dog."
#     ]
    
#     processed_texts = preprocess_text(sample_texts)
    
#     for processed_sentences in processed_texts:
#         for sentence in processed_sentences:
#             print(sentence)
#         print()  # Separate each text's output with a blank line
