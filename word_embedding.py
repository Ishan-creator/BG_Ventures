from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text  

def compute_tfidf_matrices(texts):
    processed_texts = preprocess_text(texts)
    
    all_words = [word for processed_sentences in processed_texts for sentence in processed_sentences for word in sentence.split()]

    vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, token_pattern=r'\b\w+\b')
    vectorizer.fit([' '.join(all_words)])

    tfidf_matrices = []
    for text in processed_texts:
        tfidf_matrix = vectorizer.transform(text)
        tfidf_matrices.append(tfidf_matrix)

    return tfidf_matrices

# Example usage:
sample_texts = [
    "Brick Work. Supplying preparing and construction of chimney made brick work of approved quality with 1:3 cement sand mortar ratio with 30 m lead.",
    "The quick brown fox jumps over the lazy dog."
]

tfidf_matrices, feature_names = compute_tfidf_matrices(sample_texts)

# # Print TF-IDF matrices for each text
# for i, tfidf_matrix in enumerate(tfidf_matrices):
#     print(f"\nTF-IDF Matrix for Text {i+1}:")
#     print(tfidf_matrix.toarray())

