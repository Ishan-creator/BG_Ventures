# import re
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# model_path = 'roberta-base-nli-stsb-mean-tokens'  
# model = SentenceTransformer(model_path)

# descriptions_to_match = [
#     "Earthwork in excavation in trenches, found. In medium clayey soil including dressing of sides, ramming bottom, lift upto 2.0m, disposing of surplus soil.",
#     "Providing, mixing and placing 75mm thick PCC (1:3:6) using PPC Cement equivalent in foundation trenches and plinths.",
#     "Earthback filling in foundation trenches including watering, consolidating drawing, specification & instructions."
# ]

# def get_norms(path):
#     df1 = pd.read_csv(path)
#     df = df1["Norms"].dropna()
#     return df

# target_descriptions = get_norms("norms_final.csv")
# target_descriptions = target_descriptions.tolist()

# def preprocess_text(text):
#     text = text.lower() 
#     # text = re.sub(r'[^\w\s]', '', text)  
#     # tokens = text.split()  
#     return text

# preprocessed_descriptions_to_match = [preprocess_text(desc) for desc in descriptions_to_match]
# preprocessed_target_descriptions = [preprocess_text(desc) for desc in target_descriptions]

# sentence_embeddings_to_match = model.encode(preprocessed_descriptions_to_match, convert_to_tensor=True)
# sentence_embeddings_target = model.encode(preprocessed_target_descriptions, convert_to_tensor=True)

# embeddings_to_match = sentence_embeddings_to_match.numpy()
# embeddings_target = sentence_embeddings_target.numpy()

# cosine_similarities = cosine_similarity(embeddings_to_match, embeddings_target)

# best_matches = np.argmax(cosine_similarities, axis=1)

# for i, match_index in enumerate(best_matches):
#     print(f"Description to match: {descriptions_to_match[i]}")
#     print(f"Best matched description: {target_descriptions[match_index]}")
#     print(f"Similarity score: {cosine_similarities[i][match_index]:.4f}")
#     print("-" * 80)



import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

import pandas as pd

# Function to load the data and create a new BOQ Description
def get_norms(path, sheet_name, header):
    # Load the Excel sheet into a DataFrame
    df1 = pd.read_excel(path, sheet_name=sheet_name, header=header)
    
    # Filter the relevant columns and drop rows with missing values in important columns
    df = df1[["Norm Name", "BOQ Description", "Item Name", "Specification", "SubSpec"]].dropna(subset=["Norm Name", "BOQ Description", "Item Name", "Specification", "SubSpec"])
    
    # Create a new 'BOQ Description' by concatenating the relevant columns
    df['Combined BOQ Description'] = df.apply(lambda row: ' '.join([str(row['BOQ Description']), str(row['Item Name']), str(row['Specification']), str(row['SubSpec'])]), axis=1)
    
    # Select only the necessary columns for the final DataFrame
    final_df = df[["Norm Name", "Combined BOQ Description"]]
    
    return final_df

# Load data from each sheet into DataFrames
target_descriptions1 = get_norms("data For training.xlsx", sheet_name="Project 1", header=0)
target_descriptions2 = get_norms("data For training.xlsx", sheet_name="data ", header=0)
target_descriptions3 = get_norms("data For training.xlsx", sheet_name="Project 2", header=0)
target_descriptions4 = get_norms("data For training.xlsx", sheet_name="Project 3", header=0)
target_descriptions5 = get_norms("data For training.xlsx", sheet_name="Project 4", header=0)

# Combine all data into a single DataFrame
combined_df = pd.concat([target_descriptions1, target_descriptions2, target_descriptions3, target_descriptions4, target_descriptions5], ignore_index=True)

# Display the combined DataFrame
print(combined_df)


# Create training data with sentence pairs and labels
train_examples = [InputExample(texts=[row["Norm Name"], row["Combined BOQ Description"]], label=1.0) for index, row in combined_df.iterrows()]



# Optional: Create negative examples by pairing mismatched Norms and BOQ Descriptions
negative_examples = []
boq_descriptions = combined_df["Combined BOQ Description"].tolist()
for index, row in combined_df.iterrows():
    random_boq = random.choice(boq_descriptions)
    if random_boq != row["Combined BOQ Description"]:  # Ensure we don't accidentally pick the matching pair
        negative_examples.append(InputExample(texts=[row["Norm Name"], random_boq], label=0.0))

# Combine positive and negative examples
train_examples.extend(negative_examples)




# print(negative_examples)

# # Load a pre-trained sentence transformer model
# model_name = 'distilbert-base-nli-stsb-mean-tokens'
# model = SentenceTransformer(model_name)

# # Create a DataLoader
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# # Define the loss function (CosineSimilarityLoss is good for similarity tasks)
# train_loss = losses.CosineSimilarityLoss(model)

# # Train the model on CPU
# num_epochs = 3
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=num_epochs,
#     warmup_steps=100,  # Usually a good practice to avoid high initial learning rates
#     output_path='./trained_model',  # Where to save the model
#     use_amp=False  # Automatic mixed precision, not used here since we're on CPU
# )

# # Save the trained model
# model.save('./trained_model')

# print("Model training complete and saved at './trained_model'.")
