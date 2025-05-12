# flipkart_project1

# 💻 AI-Powered Laptop Recommendation Assistant

This project is an intelligent laptop search assistant built with Streamlit, FAISS, Sentence Transformers, and Groq API (LLaMA3). It helps users find the best laptops based on natural language queries like _"best gaming laptop under ₹50000"_.

---

## 🚀 Features

- 🔍 **Semantic Search**: Finds relevant laptops using FAISS + sentence embeddings.
- 🤖 **LLM Recommendations**: Uses Groq's LLaMA3 model to generate user-friendly responses.
- 🧠 **Follow-up Support**: Lets users ask questions about the results (multi-turn).
- 📊 **Product Ranking**: Displays top matching laptops with detailed specs.
- 🛒 **Streamlit UI**: Clean, interactive web interface.

---

## 📦 Tech Stack

| Tool         | Purpose                             |
|--------------|--------------------------------------|
| Streamlit    | Frontend web UI                     |
| FAISS        | Vector similarity search            |
| SentenceTransformer (`paraphrase-albert-small-v2`) | Embedding text data |
| Groq API     | LLM response generation (LLaMA3)    |
| Pandas       | Data handling and filtering         |
| Regex        | Price parsing from natural language |

---

## 🏗️ Project Structure

├── app.py # Main Streamlit app
├── clean_data.csv # Laptop dataset
├── README.md # Project documentation
├── .gitignore # Ignore large/unwanted files


🧠 How It Works
Data Embedding: Each laptop's info is converted into a sentence and embedded using SentenceTransformer.

FAISS Search: User queries are encoded and matched against product embeddings.

LLM Generation: The Groq LLM (LLaMA3) generates a natural language recommendation from the top results.

Chat Loop: User can follow up with questions that continue from the current context.
