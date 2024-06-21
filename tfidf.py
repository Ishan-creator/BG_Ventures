# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pandas as pd

# descriptions_to_match = [
#     "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
#     "Providing, mixing and placing 75mm thick PCC (1:3:6) using PPC Cement equivalent in foundation trenches and plinths.",
#     "Earthback filling in foundation trenches including watering, consolidating drawing, specification & instructions.",
#     "Moulding or Nosing works in Granite edges in skirting and tops all complete"
# ]


# def get_norms(path):
#     df1 = pd.read_csv(path)
#     df = df1["Norms"].dropna()
#     return df


# target_descriptions = get_norms("norms_final.csv")
# target_descriptions = target_descriptions.tolist()


# def preprocess_text(text):
#     text = text.lower() 
#     text = re.sub(r'[^\w\s]', '', text)  
#     tokens = text.split()  
#     return ' '.join(tokens)


# preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
# preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]


# all_descriptions = preprocessed_descriptions_to_match + preprocessed_target_descriptions


# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(all_descriptions)


# cosine_similarities = cosine_similarity(tfidf_matrix[:len(descriptions_to_match)], tfidf_matrix[len(descriptions_to_match):])

# best_matches = np.argmax(cosine_similarities, axis=1)


# for i, match_index in enumerate(best_matches):
#     print(f"Description to match: {descriptions_to_match[i]}")
#     print(f"Best matched description: {target_descriptions[match_index]}")
#     print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
#     print("-" * 80)



import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from itertools import product

# Sample descriptions to be matched
descriptions_to_match = [
    "Earthwork in excavation.",
    "Providing and laying glazed porcelain wall tiles of (Italian or Equivalent) approved color, size & make in proper line & level laid in 12mm thick base plaster of cement mortar MM5.",
    "Granite in Window Sill and Parapet Wall"
]

def get_norms(path, sheet_name, header):
    df1 = pd.read_excel(path, sheet_name=sheet_name, header=header)
    df = df1[["Norm Name", "BOQ Description"]].dropna()
    return df

target_descriptions1 = get_norms("data For training.xlsx", sheet_name="Project 1", header=0)
target_descriptions2 = get_norms("data For training.xlsx", sheet_name="data ", header=0)
target_descriptions3 = get_norms("data For training.xlsx", sheet_name="Project 2", header=0)
target_descriptions4 = get_norms("data For training.xlsx", sheet_name="Project 3", header=0)
target_descriptions5 = get_norms("data For training.xlsx", sheet_name="Project 4", header=0)

combined_df = pd.concat([target_descriptions1, target_descriptions2, target_descriptions3, target_descriptions4, target_descriptions5], ignore_index=True)

print(combined_df)
# target_descriptions = target_descriptions1 +target_descriptions2 + target_descriptions3 + target_descriptions4 + target_descriptions5
# print(target_descriptions[:50])
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     tokens = text.split()
#     return ' '.join(tokens)

# preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
# preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]

# all_descriptions = preprocessed_descriptions_to_match + preprocessed_target_descriptions

# def cosine_similarity_scorer(estimator, X, y):
#     tfidf_matrix = estimator.transform(X)
#     cosine_similarities = cosine_similarity(tfidf_matrix[:len(preprocessed_descriptions_to_match)], tfidf_matrix[len(preprocessed_descriptions_to_match):])
#     best_matches = np.argmax(cosine_similarities, axis=1)
#     scores = [cosine_similarities[i][best_matches[i]] for i in range(len(best_matches))]
#     return np.mean(scores)

# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer())
# ])

# param_grid = {
#     'tfidf__max_df': [0.85, 0.90, 0.95, 1.0],
#     'tfidf__min_df': [1, 2, 5, 10],
#     'tfidf__ngram_range': [(1, 1), (1, 2)],
#     'tfidf__use_idf': [True, False],
#     'tfidf__smooth_idf': [True, False],
#     'tfidf__norm': ['l1', 'l2', None]
# }

# grid_search = GridSearchCV(pipeline, param_grid, scoring=make_scorer(cosine_similarity_scorer, greater_is_better=True), cv=3)

# grid_search.fit(all_descriptions, np.zeros(len(all_descriptions)))

# best_params = grid_search.best_params_



# vectorizer = TfidfVectorizer(
#     max_df=best_params['tfidf__max_df'],
#     min_df=best_params['tfidf__min_df'],
#     ngram_range=best_params['tfidf__ngram_range'],
#     use_idf=best_params['tfidf__use_idf'],
#     smooth_idf=best_params['tfidf__smooth_idf'],
#     norm=best_params['tfidf__norm']
# )

# tfidf_matrix = vectorizer.fit_transform(all_descriptions)

# cosine_similarities = cosine_similarity(tfidf_matrix[:len(preprocessed_descriptions_to_match)], tfidf_matrix[len(preprocessed_descriptions_to_match):])

# best_matches = np.argmax(cosine_similarities, axis=1)

# print("Best Parameters:")
# print(best_params)


# for i, match_index in enumerate(best_matches):
#     print(f"Description to match: {descriptions_to_match[i]}")
#     print(f"Best matched description: {target_descriptions[match_index]}")
#     print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
#     print("-" * 80)
    

