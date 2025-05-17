# flipkart_project #

# 💻 AI-Powered Laptop Recommendation System 🤖

Welcome to the **AI-Powered Laptop Recommendation System**! This project uses **LangChain Agents**, **FAISS**, **Sentence Transformers**, and the **Groq API (LLaMA3)** to recommend the best laptops based on user queries like:


-
![ChatGPT Image May 10, 2025, 07_08_59 PM](https://github.com/user-attachments/assets/c612318f-9c10-4048-98ae-2f9c5587c93b)


## 🧠 What It Does

* Understands user queries like “Suggest laptops under ₹60000 for programming”
* Retrieves relevant laptops from a local dataset
* Uses a local LLM to generate personalized responses
* Remembers previous interactions for follow-up queries
  

## 🌐  Web Scraping

* Laptop listings are scraped from Flipkart using automation tools like *Selenium* or *BeautifulSoup*.
* Fields like product name, price, and descriptions are extracted.
* The result is saved as a raw CSV file containing mixed and inconsistent data.

> *Note:* Web scraping is performed before running the main app. This step is not automated in the app and needs to be done manually or via an external script.



## 🧼  Data Cleaning

* The raw scraped CSV file is processed using the data_cleaning.py script.
* Key tasks include:

  * Cleaning the price field and removing unwanted characters
  * Standardizing product names
  * Extracting specs such as Processor, RAM, Storage, OS, Display, and Warranty using regular expressions
* The cleaned data is saved into a new CSV file that is used by the chatbot app

  
## 🗂 Project Workflow Overview

1. *Web Scraping* (external process)
2. *Data Cleaning* (data_cleaning.py)
3. *RAG-based Chatbot App* (laptop_recommendation_app.py)
4. *User Interaction via Streamlit*
   
## 📌 Features

- 🔍 **Semantic Search** using Sentence Transformers + FAISS
- 🧠 **LangChain Agent** with memory for intelligent multi-turn chat
- 💬 Ask follow-up questions like a real conversation
- ⚙️ Dynamic filters for **RAM** and **Operating System**
- 🖼️ **Streamlit** front-end for a clean and interactive UI
- 🤖 Powered by **LLaMA3-8B (Groq API)** for smart, fast replies

---

## 🧩 Tech Stack

| Tool/Library | Purpose |
|-------------|---------|
| **Streamlit** | UI for interaction |
| **Pandas** | Data handling |
| **FAISS** | Fast semantic search |
| **SentenceTransformers** | Embedding laptop specs & queries |
| **LangChain** | Conversational agent framework |
| **Groq + LLaMA3** | LLM-based responses |
| **Regex (`re`)** | Extract price filters from queries |
| **dotenv** | Manage API keys securely |

---

## 🗂️ Project Structure

📁 laptop-recommender/
│
├── EDA/ # (Optional) Data cleaning, preprocessing notebooks
├── app.py # Main Streamlit application
├── clean_data.csv # Cleaned laptop dataset
├── .env # API key environment file
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## ✅ Final Notes

* All responses are based on the uploaded dataset (not live data)
* The app is fully local and doesn’t require internet after setup
* Suitable for offline, private, and personalized recommendation use cases
* Memory allows the chatbot to behave more naturally in a conversation




