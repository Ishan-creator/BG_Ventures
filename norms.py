import pandas as pd
from preprocessing import preprocess_text
from word_embedding import compute_tfidf_matrices

def get_norms(url, sheet_name, header): 
    df1 = pd.read_excel(url, sheet_name=sheet_name, header=header)
    df = df1["Norms"].dropna()
    return df

get = get_norms("/home/ishan-pc/Desktop/BG_Ventures/Residence Building RCC Project Template New-3.xlsx", sheet_name="Norms", header=1)

get_list = get.tolist()


# checkk = preprocess_text(get)


# finall = compute_tfidf_matrices(checkk)






