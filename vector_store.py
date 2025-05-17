
import pandas as pd
# Importing pickle for saving metadata and vectorizer
import pickle
# Importing faiss for similarity search
import faiss
# Importing tfidfvectorizer for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

import os

# Loading the  data 
df = pd.read_csv(r"C:\Users\akmet\OneDrive\Desktop\recommender\amz_ca_total_products_data_processed.csv")

# Dropping rows with nulls in product column
df = df.dropna(subset=["product"])

# Creating a  vectorizer with removing stop words 
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)

# Fitting the vectorizer to the product column and transforming  data into vectors
X = vectorizer.fit_transform(df["product"])

# Converting the  matrix to a dense float array 
X_f32 = X.toarray().astype("float32")

# Creating a faiss index that uses L2 distance metric for  search
index = faiss.IndexFlatL2(X_f32.shape[1])

# Adding the vectorized data to the faiss index
index.add(X_f32)

# Creating a directory to store the faiss index and metadata and vectorizer 
os.makedirs("light_vector_store", exist_ok=True)

# Saving the faiss index to  file
faiss.write_index(index, "light_vector_store/index.faiss")

# Saving the DataFrame as metadata in pickle format
with open("light_vector_store/metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

# Saving the  TF-IDF vectorizer to a file
with open("light_vector_store/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

