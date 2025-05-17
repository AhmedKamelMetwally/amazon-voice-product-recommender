# 🛍️ Amazon Product Recommender (Voice-Powered)  
_By Ahmed Kamel_

A voice-enabled product recommendation system that helps users discover similar products from Amazon listings using text similarity and speech input.

---

## 🔍 What This Project Does

This project allows users to **speak** a product description (e.g., “a heavy-duty extension cord”), and get back a list of **similar Amazon products** from a dataset using vector-based similarity search. It's built using **TF-IDF**, **FAISS**, and **Streamlit**, with integrated **speech recognition**.

---

## 🎯 Features

- 🎙️ **Voice Input**: Users can describe products using their voice
- 🔡 **TF-IDF Vectorization**: Converts product descriptions into vectors
- ⚡ **FAISS Similarity Search**: Quickly finds the most similar products
- 🖥️ **Streamlit Web App**: Clean and simple UI
- 📦 **Product Metadata Display**: Name, price, and category shown for recommendations

---

## 🧠 How It Works

1. **Preprocessing**: Product descriptions are vectorized using `TfidfVectorizer` (max 1000 features).
2. **Indexing**: A `FlatL2` FAISS index is built using the dense vector matrix.
3. **Saving**: The index, vectorizer, and metadata are saved in a directory (`light_vector_store`).
4. **Querying**: The user’s voice is transcribed and vectorized.
5. **Searching**: The FAISS index retrieves the top 5 similar products.
6. **Results**: Results are displayed in the Streamlit UI with product name, price, and category.

---

## 🧰 Tech Stack

- **Python**
- **FAISS** – Facebook AI Similarity Search
- **scikit-learn** – TF-IDF Vectorization
- **Streamlit** – Web App UI
- **SpeechRecognition** – Google Speech API integration
- **Pandas, NumPy, Pickle** – Data manipulation and model persistence

---

## 📂 Project Structure

