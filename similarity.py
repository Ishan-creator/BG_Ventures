# # from sklearn.metrics.pairwise import cosine_similarity
# # from norms import get_norms
# # from word_embedding import compute_tfidf_matrices
# # from preprocessing import preprocess_text

# # def calculate_cosine_similarity(tfidf_vector1, tfidf_vector2):
# #     similarity = cosine_similarity(tfidf_vector1, tfidf_vector2)
# #     return similarity[0][0]

# # sample_texts = [
# #     "Brick Work. Supplying preparing and construction of chimney made brick work of approved quality with 1:3 cement sand mortar ratio with 30 m lead.",
# # ]

# # # Compute TF-IDF matrix for the sample text
# # sample_tfidf_matrix = compute_tfidf_matrices(sample_texts)[0]

# # get = get_norms("/home/ishan-pc/Desktop/BG_Ventures/Residence Building RCC Project Template New-3.xlsx", sheet_name="Norms", header=1)
# # get_list = get.tolist()

# # # Compute TF-IDF matrices for the documents in get_list
# # final_matrices = compute_tfidf_matrices(get_list)

# # max_similarity = -1
# # best_match_index = -1

# # # Iterate over each document's TF-IDF matrix in final_matrices
# # for index, document_matrix in enumerate(final_matrices):
# #     similarity = calculate_cosine_similarity(sample_tfidf_matrix, document_matrix)
# #     if similarity > max_similarity:
# #         max_similarity = similarity
# #         best_match_index = index

# # print(f"Best match index: {best_match_index}")
# # print(f"Highest similarity score: {max_similarity}")



# from sklearn.metrics.pairwise import cosine_similarity
# from norms import get_norms
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import WordNetLemmatizer

# def preprocess_text(texts):
#     processed_texts = []
#     for text in texts:
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         sentences = sent_tokenize(text)
#         tokens = [word_tokenize(sentence) for sentence in sentences]
#         lemmatizer = WordNetLemmatizer()
#         tokens = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]
#         tokens = [[word for word in sentence if word.isalnum()] for sentence in tokens]
#         tokens = [[word for word in sentence if not word.isdigit()] for sentence in tokens]
#         processed_sentences = [" ".join(sentence) for sentence in tokens]
#         processed_texts.extend(processed_sentences)
#     return processed_texts

# def compute_tfidf_matrices(documents):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(documents)
#     return tfidf_matrix, vectorizer.get_feature_names_out()

# def calculate_cosine_similarity(tfidf_vector1, tfidf_vector2):
#     similarity = cosine_similarity(tfidf_vector1, tfidf_vector2)
#     return similarity[0][0]

# sample_texts = [
#     "Construction of Temporary shelter for material storage and workers room.Clearing & grubbing of site along with layout of  buidling for construction as per drawing specification &  instructions.",
# ]

# # Preprocess texts
# sample_texts_processed = preprocess_text(sample_texts)

# # Load and preprocess documents from Excel
# get = get_norms("/home/ishan-pc/Desktop/BG_Ventures/Residence Building RCC Project Template New-3.xlsx", sheet_name="Norms", header=1)
# get_list = get.tolist()
# get_list_processed = preprocess_text(get_list)

# # Combine sample text and documents for TF-IDF vectorization
# all_texts = sample_texts_processed + get_list_processed

# # Compute the combined TF-IDF matrix
# combined_tfidf_matrix, feature_names = compute_tfidf_matrices(all_texts)

# # Split the combined TF-IDF matrix back into sample and document matrices
# sample_tfidf_matrix = combined_tfidf_matrix[0:len(sample_texts_processed)]
# document_tfidf_matrices = combined_tfidf_matrix[len(sample_texts_processed):]

# print(sample_tfidf_matrix)
# print("5"*50)
# print(document_tfidf_matrices)

# max_similarity = -1
# best_match_index = -1

# # Iterate over each document's TF-IDF matrix to find the most similar document
# for index in range(document_tfidf_matrices.shape[0]):
#     document_matrix = document_tfidf_matrices[index:index+1]  # Keep it as a 2D matrix for compatibility
#     similarity = calculate_cosine_similarity(sample_tfidf_matrix, document_matrix)
#     if similarity > max_similarity:
#         max_similarity = similarity
#         best_match_index = index

# print(f"Best match index: {best_match_index}")
# print(f"Highest similarity score: {max_similarity}")

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy's pre-trained model
nlp = spacy.load('en_core_web_md')

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        lemmatizer = WordNetLemmatizer()
        tokens = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]
        tokens = [[word for word in sentence if word.isalnum()] for sentence in tokens]
        # tokens = [[word for word in sentence if not word.isdigit()] for sentence in tokens]
        processed_texts.append(tokens)
    return processed_texts

def document_to_vector_spacy(doc_tokens, nlp_model):
    flattened_tokens = [word for sentence in doc_tokens for word in sentence]
    doc_text = " ".join(flattened_tokens)
    doc = nlp_model(doc_text)
    return doc.vector

def compute_spacy_vectors(documents, nlp_model):
    vectors = []
    for doc in documents:
        doc_vector = document_to_vector_spacy(doc, nlp_model)
        vectors.append(doc_vector)
    return np.array(vectors)

def calculate_cosine_similarity(vector1, vector2):
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

sample_texts = [
    "hello world",
]

# Preprocess the sample text
sample_texts_processed = preprocess_text(sample_texts)

# Load documents from Excel
def get_norms(url, sheet_name, header): 
    df1 = pd.read_excel(url, sheet_name=sheet_name, header=header)
    df = df1["Norms"].dropna()
    return df

get = get_norms("/home/ishan-pc/Desktop/BG_Ventures/Residence Building RCC Project Template New-3.xlsx", sheet_name="Norms", header=1)
get_list = get.tolist()
get_list_processed = preprocess_text(get_list)

print(get_list_processed[6])

# Compute spaCy vectors
sample_vectors = compute_spacy_vectors(sample_texts_processed, nlp)
document_vectors = compute_spacy_vectors(get_list_processed, nlp)

max_similarity = -1
best_match_index = -1

# Find the most similar document
for index, document_vector in enumerate(document_vectors):
    similarity = calculate_cosine_similarity(sample_vectors[0], document_vector)
    if similarity > max_similarity:
        max_similarity = similarity
        best_match_index = index

print(f"Best match index: {best_match_index}")
print(f"Highest similarity score: {max_similarity}")
