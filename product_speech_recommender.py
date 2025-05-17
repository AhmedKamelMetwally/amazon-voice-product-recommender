
import streamlit as st
# Importing faiss for similarity search
import faiss

# Importing pickle to load metadata and vectorizer
import pickle

import numpy as np

# Imported to transform text into vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing speech recognition library 
import speech_recognition as sr

# Caching the loading of FAISS index, metadata,  vectorizer 
@st.cache_resource
def load_vector_store():
    # Loading faiss index from file
    index = faiss.read_index("light_vector_store/index.faiss")
    
    # Loading product info from pickle file
    with open("light_vector_store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Loading the  vectorizer from pickle file
    with open("light_vector_store/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    
    return index, metadata, vectorizer

#streamlit
st.set_page_config(page_title="Product Recommender", layout="centered")


st.title(" Amazon Product Recommender by Ahmed Kamel")


st.caption("Tell me the description of the  product")


query = ""


if st.button("Speak"):
    
    recognizer = sr.Recognizer()
    
    
    with sr.Microphone() as source:
        
        st.info("Listening ....")
        
        
        audio = recognizer.listen(source)
    
    
    try:
        
        query = recognizer.recognize_google(audio)
        
        
        st.success(f"your describtion: {query}")
    
    
    except sr.UnknownValueError:
        st.error("Sorry I could not understand that Please try again")
    
    
    except sr.RequestError as e:
        st.error(f"Error with the speech recognition service: {e}")


if query:
    try:
        # Loading faiss index, metadata, and vectorizer
        index, metadata_list, vectorizer = load_vector_store()

        # Vectorizing  user query and converting it to float
        query_vector = vectorizer.transform([query]).toarray().astype("float32")
        
        # Searching the faiss index for the top 5 most similar product vectors
        D, I = index.search(query_vector, k=5)

         
        st.subheader("Recommendations")
        
        # Looping through  the top results
        for idx in I[0]:
            # getting metadata of each recommended product
            meta = metadata_list[idx]
            
            # Extracting  name ,price and category
            product_name = meta.get('product', 'Unnamed Product')
            price = meta.get('price', 'N/A')
            category = meta.get('categoryName', 'N/A')
            
           
            st.markdown(f"### {product_name}")
            
            
            st.write(f"**Price**: ${price}")
            
            
            st.write(f"**Category**: {category}")
            
            
            st.markdown("---")
    
    
    except Exception as e:
        st.error(f"An error occurred while processing the query: {e}")
