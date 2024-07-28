import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from difflib import SequenceMatcher
import pickle

# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer # type: ignore
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
def preprocess_data(data_path, sample_size):
    # Read the data from specific path
    data = pd.read_csv(data_path, low_memory=False)
    # Drop articles without Abstract
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    # Get "sample_size" random articles
    data = data.sample(sample_size)[['abstract', 'id']]
    return data

new_data_path = "/kaggle/input/research-papers-dataset/dblp-v10.csv"
new_data = preprocess_data(new_data_path, sample_size=30000)
model_name = 'all-MiniLM-L6-v2'  # A good general-purpose SBERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name).to(device)
def create_vector_from_text(model, text, max_length=128):
    # Encode the text
    with torch.no_grad():
        vector = model.encode(text, show_progress_bar=False, normalize_embeddings=True, device=device)
    return vector
def create_vector_database(data):
    vectors = []
    source_data = data.abstract.values
    
    for text in tqdm(source_data):
        vector = create_vector_from_text(model, text)
        vectors.append(vector)
    
    data["vectors"] = vectors
    return data
# vector_database = create_vector_database(source_data)
# import pickle
# import os

# # Path to save the new pickle file
# save_path = '/kaggle/working/Main_combined.pkl'

# try:
#     # Save using pickle
#     with open(save_path, 'wb') as f:
#         pickle.dump(vector_database, f)
    
#     # Verify the file was created and has content
#     if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
#         print(f"Combined vector database saved as pickle file at {save_path}")
#         print(f"File size: {os.path.getsize(save_path)} bytes")
        
#         # Attempt to load the file to verify it's readable
#         with open(save_path, 'rb') as f:
#             test_load = pickle.load(f)
#         print("Successfully verified the saved file by loading it.")
        
#     else:
#         print("Error: File not created or is empty.")

# except Exception as e:
#     print(f"An error occurred while saving or verifying the file: {e}")

# # # # Optionally, you can print a sample of the data to further verify
# # # # print(test_load.sample(5))  # Uncomment this if you want to see a sample
def process_document(text):
  
    text_vect = create_vector_from_text(model, text)
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)

    return text_vect

    
def is_plagiarism(similarity_score, plagiarism_threshold):

  is_plagiarism = False

  if(similarity_score >= plagiarism_threshold):
    is_plagiarism = True

  return is_plagiarism
def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.85, string_similarity_threshold=0.3):
    top_N = 3
    # Step 1: Check if the text is long enough for reliable plagiarism detection
    if len(query_text.split()) < 20:  # Adjust this threshold as needed
        return {'is_plagiarism': False, 'reason': 'Text too short for reliable check'}

    # Step 2: Process the query text
    query_vect = process_document(query_text)
    query_vect = normalize(query_vect.reshape(1, -1))

    # Run similarity Search
    data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, normalize(x.reshape(1, -1))))
    data["similarity"] = data["similarity"].apply(lambda x: x[0][0])
    
    similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
    formated_result = similar_articles[["abstract", "id", "similarity"]].reset_index(drop=True)
    similarity_score = formated_result.iloc[0]["similarity"] 
    most_similar_article = formated_result.iloc[0]["abstract"] 
    
    # Additional check for text similarity
    if similarity_score > plagiarism_threshold:
        text_similarity = string_similarity(query_text, most_similar_article)
#         if text_similarity < string_similarity_threshold:
#             is_plagiarism_bool = 'Suspicious'  # New category
#         else:
        is_plagiarism_bool = True
    else:
        is_plagiarism_bool = False

    plagiarism_decision = {
        'similarity_score': similarity_score, 
        'is_plagiarism': is_plagiarism_bool,
        'most_similar_article': most_similar_article, 
        'article_submitted': query_text
    }
    return plagiarism_decision

# Updated process_document function
def process_document(text):
    """
    Create a vector for given text and adjust it for cosine similarity search
    """
    text_vect = create_vector_from_text(model, text)
    return text_vect.flatten()  # Ensure it's a 1D array
def get_vectordatabase():
    with open('Main_combined.pkl', 'rb') as f:
        vector_database = pickle.load(f)
    return vector_database
vector_database=get_vectordatabase()
mytext="""Ingredient-based Culinary Engineering (IBCE) is concerned with the development of recipes from reusable elements (ingredients), and the development and preservation of these elements. This approach addresses the issue of taste testing in an IBCE environment, and specifically automatically measuring the palatability of different ingredients in a single dish. The proposed taste measure is derived from the flavor interactions between ingredients recorded in a taste profile. The measure was validated in an experimental evaluation. Four different prototypes of a signature dish were subjected to taste tests, in which 40 food critics participated. Results show that the palatability of the individual ingredients can be measured, and that they can be prioritized based on their potential for flavor enhancement."""
res=run_plagiarism_analysis(mytext,vector_database)
print(res)