# import spacy
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# import re
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Load the SpaCy medium-sized model
# nlp = spacy.load('en_core_web_md')

# def preprocess_text(texts):
#     processed_texts = []
#     lemmatizer = WordNetLemmatizer()
#     for text in texts:
#         text = text.lower()  # Convert to lowercase
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#         sentences = sent_tokenize(text)  # Tokenize into sentences
#         tokens = [word_tokenize(sentence) for sentence in sentences]  # Tokenize each sentence into words
#         tokens = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]  # Lemmatize each word
#         flat_tokens = [word for sentence in tokens for word in sentence]  # Flatten tokens into a single list
#         processed_texts.append(" ".join(flat_tokens))  # Join tokens back into a single string
#     return processed_texts

# sample_texts = [
#     "Brick Work Supplying preparing and construction of chimney made brick work of approved quality with 1:2 chuna ( white cement) : Surkhi mortar ratio with 30 m lead",
#     "1:2:4 PCC Works"
# ]
# sample_sentences_processed = preprocess_text(sample_texts)

# def document_to_vector_spacy(doc_text, nlp_model):
#     doc = nlp_model(doc_text)
#     return doc.vector

# def compute_spacy_vectors(documents, nlp_model):
#     vectors = [document_to_vector_spacy(doc, nlp_model) for doc in documents]
#     return np.array(vectors)

# def calculate_cosine_similarity(vector1, vector2):
#     similarity = cosine_similarity([vector1], [vector2])
#     return similarity[0][0]

# # Print word embeddings for each word in each sample sentence
# for sentence in sample_sentences_processed:
#     doc = nlp(sentence)  # Process the sentence with SpaCy
#     print(f"Sentence: '{sentence}'")
#     for token in doc:
#         print(f"Word: '{token.text}' -> Vector: {token.vector[:5]}...")  # Print the first 5 elements of the vector for brevity
#     print("-" * 100)

# # Example: Check cosine similarity between vectors of the sample sentences
# sample_vectors = compute_spacy_vectors(sample_sentences_processed, nlp)
# for i, vector1 in enumerate(sample_vectors):
#     for j, vector2 in enumerate(sample_vectors):
#         if i != j:
#             similarity = calculate_cosine_similarity(vector1, vector2)
#             print(f"Cosine similarity between sentence {i+1} and sentence {j+1}: {similarity:.4f}")
            
            
            
# print(sample_vectors.shape)


import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

nlp = spacy.load('en_core_web_md')

# Sample sentences
sample_sentences = [
    "Brick Work Supplying preparing and construction of chimney made brick work of approved quality with 1:2 chuna (white cement) : Surkhi mortar ratio with 30 m lead",
    "1:2:4 PCC Works"
]

# Preprocess the text
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        flat_tokens = [word for sentence in tokens for word in sentence]
        processed_texts.append(" ".join(flat_tokens))
    return processed_texts

sample_sentences_processed = preprocess_text(sample_sentences)

# Print word embeddings for each word in each sample sentence
for sentence in sample_sentences_processed:
    doc = nlp(sentence)
    print(f"Sentence: '{sentence}'")
    for token in doc:
        print(f"Word: '{token.text}' -> Vector shape: {token.vector.shape} -> Vector (first 5 values): {token.vector[:50]}")
    print(f"Document vector shape: {doc.vector.shape} -> Document vector (first 5 values): {doc.vector[:5]}")
    print("-" * 100)
