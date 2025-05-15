# import streamlit as st
# import pandas as pd
# import faiss
# import numpy as np
# import re
# from sentence_transformers import SentenceTransformer
# import requests
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")

# # Load and preprocess data
# df = pd.read_csv("E:/pandas/clean_data.csv")

# def create_text(row):
#     return (
#         f"Product: {row['Product Name']}, Rating: {row['Rating']}, Price: ₹{row['Price']}, "
#         f"Processor: {row['Processor']}, RAM: {row['RAM']}, Storage: {row['Storage']}, "
#         f"Display: {row['Display']}, OS: {row['OS']}, Warranty: {row['Warranty']}"
#     )

# df["text"] = df.apply(create_text, axis=1)

# # Embed products
# model = SentenceTransformer('paraphrase-albert-small-v2')
# embedding_matrix = model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")

# # FAISS index
# dimension = embedding_matrix.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embedding_matrix)

# # --- Custom Prompt Template ---
# template = """
# You are an expert assistant for laptop recommendations, specializing in matching user queries to laptops based on price, specifications, and use cases.

# Your task is to analyze the provided context, containing laptop details in the format: Product Name | Processor | RAM | Storage | Price | Brand | Weight | Rating. Each laptop has metadata including price, RAM, storage, processor, brand, weight, rating, display_size, os, and warranty. Recommend laptops that match the user's query, considering price range, specific features, and use case requirements.

# Guidelines:
# - For price range queries (e.g., "₹50000 to ₹70000"), filter using metadata['price'].
# - For feature queries (e.g., "16GB RAM", "512GB SSD", "Intel Core i5"), filter using metadata['ram'], metadata['storage'], metadata['processor'].
# - For use case queries (e.g., "laptop for a software engineer"), map to requirements:
#   - Software engineer: ≥16GB RAM, ≥512GB SSD, Intel Core i5/i7 or AMD Ryzen 5/7 (if processor unknown, prioritize RAM/storage), weight <2kg.
#   - Gaming: ≥16GB RAM, ≥512GB SSD, Intel Core i7 or AMD Ryzen 7, weight <2.5kg.
#   - Student: ≥8GB RAM, ≥256GB SSD, Intel Core i3 or AMD Ryzen 3, price <₹40000.
# - Use chat history to refine queries (e.g., "same price range" refers to prior query).
# - Define "best" as highest rating, then strongest specs (higher RAM, SSD, newer processor) within constraints.
# - List up to 3 laptops, sorted by price (cheapest first), including product name, processor, RAM, storage, price, rating.
# - If no laptops match, respond: "Sorry, no laptops found matching your criteria."
# - Do not invent information or recommend laptops outside the context.
# - Keep responses concise and user-friendly.

# Context:
# {context}

# Chat History:
# {history}

# Question:
# {question}

# Helpful Answer:
# """

# # --- Helper Functions ---
# def extract_price_limit(query):
#     match = re.search(r"under\s*₹?\s*(\d+)", query)
#     if match:
#         return int(match.group(1))
#     return None

# def search_products(query, top_k=5):
#     price_limit = extract_price_limit(query)
    
#     filtered_df = df
#     if price_limit:
#         filtered_df = df[df['Price'].astype(str).str.replace(",", "").astype(float) <= price_limit]

#     if filtered_df.empty:
#         return []

#     query_vector = model.encode([query]).astype("float32")
#     filtered_embeddings = model.encode(filtered_df["text"].tolist()).astype("float32")

#     temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
#     temp_index.add(filtered_embeddings)

#     distances, indices = temp_index.search(query_vector, min(top_k, len(filtered_df)))
#     filtered_data = filtered_df.to_dict(orient="records")
#     return [filtered_data[i] for i in indices[0]]

# # --- LLM Response ---
# def generate_response(query, retrieved_products, model_name="llama3-8b-8192", groq_api_key=api_key):
#     if not retrieved_products:
#         return "Sorry, no matching laptops found for your query."

#     product_lines = []
#     for p in retrieved_products:
#         line = f"{p['Product Name']} | {p['Processor']} | {p['RAM']} | {p['Storage']} | ₹{p['Price']} | {p.get('Brand', 'Unknown')} | {p.get('Weight', 'Unknown')} | {p['Rating']}"
#         product_lines.append(line)

#     context = "\n".join(product_lines)

#     history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

#     prompt = template.format(context=context, history=history, question=query)

#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {groq_api_key}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model_name,
#         "messages": [
#             {"role": "system", "content": "You are a helpful shopping assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.7
#     }

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         return response.json()["choices"][0]["message"]["content"]
#     else:
#         return f"API Error {response.status_code}: {response.text}"

# # --- Streamlit UI ---
# st.set_page_config(page_title="Laptop Recommender", layout="wide")
# st.title("💻 AI-Powered Laptop Recommendation Assistant")

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# query = st.text_input("Enter your laptop requirements 👇", placeholder="e.g. best laptop under 50000 for gaming")

# if st.button("🔍 Find Laptops"):
#     if not query.strip():
#         st.warning("Please enter a query.")
#     else:
#         with st.spinner("Searching and generating recommendations..."):
#             top_k_results = search_products(query)

#             if not top_k_results:
#                 st.error("No laptops found matching your query.")
#             else:
#                 st.session_state.chat_history.append({"role": "user", "content": query})

#                 st.subheader("🔍 Top Matching Laptops")
#                 for i, p in enumerate(top_k_results, 1):
#                     st.markdown(f"**{i}. {p['Product Name']}**")
#                     st.markdown(f"💰 Price: ₹{p['Price']}")  
#                     st.markdown(f"⭐ Rating: {p['Rating']}")
#                     st.markdown(f"🧐 Processor: {p['Processor']}, 📀 RAM: {p['RAM']}, 📍 Storage: {p['Storage']}")
#                     st.markdown(f"💡 Display: {p['Display']}")
#                     st.markdown(f"🛡️ Warranty: {p['Warranty']}")
#                     st.markdown(f"🛡️ OS: {p['OS']}")
#                     st.markdown("---")

#                 st.subheader("🤖 Chatbot Recommendation")
#                 response = generate_response(query, top_k_results)
#                 st.session_state.chat_history.append({"role": "assistant", "content": response})
#                 st.success(response)

# # --- Follow-up Questions ---
# if st.session_state.chat_history:
#     follow_up = st.text_input("Ask a follow-up question 🧐", key="follow_up_q")

#     if st.button("💬 Ask"):
#         if not follow_up.strip():
#             st.warning("Please enter your follow-up question.")
#         else:
#             st.session_state.chat_history.append({"role": "user", "content": follow_up})
#             with st.spinner("Generating follow-up answer..."):
#                 response = generate_response(follow_up, top_k_results)
#                 st.session_state.chat_history.append({"role": "assistant", "content": response})
#                 st.info(response)

# # --- Chat History Display ---
# if st.session_state.chat_history:
#     st.subheader("💂️ Chat History")
#     for msg in st.session_state.chat_history:
#         role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 Bot"
#         st.markdown(f"**{role}:** {msg['content']}")


# This is the updated version of your modularized laptop recommender with:
# ✅ Streamlit filter dropdowns
# 🔁 LangChain Agents
# 🧠 LangChain memory

# import streamlit as st
# import pandas as pd
# import faiss
# import numpy as np
# import re
# import os
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from langchain.llms import OpenAI
# from langchain.agents import initialize_agent, Tool
# from langchain.memory import ConversationBufferMemory
# # from langchain.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage
# import requests
# from langchain_groq import ChatGroq


# # Load API key
# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")

# # Load data
# @st.cache_data

# def load_data():
#     df = pd.read_csv("E:/pandas/clean_data.csv")
#     return df

# df = load_data()

# # Create text from row
# def create_text(row):
#     return (
#         f"Product: {row['Product Name']}, Rating: {row['Rating']}, Price: ₹{row['Price']}, "
#         f"Processor: {row['Processor']}, RAM: {row['RAM']}, Storage: {row['Storage']}, "
#         f"Display: {row['Display']}, OS: {row['OS']}, Warranty: {row['Warranty']}"
#     )

# df["text"] = df.apply(create_text, axis=1)

# # Embed
# model = SentenceTransformer("paraphrase-albert-small-v2")
# embedding_matrix = model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")
# index = faiss.IndexFlatL2(embedding_matrix.shape[1])
# index.add(embedding_matrix)

# # UI
# st.title("💻 AI-Powered Laptop Recommendation")

# Product Name = df['Product Name'].dropna().unique().tolist()
# rams = df['RAM'].dropna().unique().tolist()
# os_options = df['OS'].dropna().unique().tolist()

# selected_brand = st.selectbox("Choose Product Name", ["Any"] + Product Name)
# selected_ram = st.selectbox("Choose RAM", ["Any"] + rams)
# selected_os = st.selectbox("Choose OS", ["Any"] + os_options)

# query = st.text_input("Enter your laptop requirements")

# # Helper: extract price

# def extract_price_limit(query):
#     match = re.search(r"under\s*₹?\s*(\d+)", query)
#     return int(match.group(1)) if match else None

# # Filter and search

# def search_products(query, brand, ram, os_name, top_k=5):
#     filtered_df = df.copy()

#     if Product Name != "Any":
#         filtered_df = filtered_df[filtered_df['Product Name'] == Product Name]
#     if ram != "Any":
#         filtered_df = filtered_df[filtered_df['RAM'] == ram]
#     if os_name != "Any":
#         filtered_df = filtered_df[filtered_df['OS'] == os_name]

#     price_limit = extract_price_limit(query)
#     if price_limit:
#         filtered_df = filtered_df[filtered_df['Price'].astype(str).str.replace(",", "").astype(float) <= price_limit]

#     if filtered_df.empty:
#         return []

#     query_vector = model.encode([query]).astype("float32")
#     subset_embeddings = model.encode(filtered_df["text"].tolist()).astype("float32")

#     temp_index = faiss.IndexFlatL2(subset_embeddings.shape[1])
#     temp_index.add(subset_embeddings)

#     distances, indices = temp_index.search(query_vector, min(top_k, len(filtered_df)))
#     return [filtered_df.iloc[i].to_dict() for i in indices[0]]

# # LangChain Agent Tool

# def laptop_tool_func(query):
#     results = search_products(query, selected_brand, selected_ram, selected_os)
#     if not results:
#         return "No laptops matched your query."
#     return "\n".join([f"{r['Product Name']} - ₹{r['Price']}" for r in results])

# laptop_tool = Tool(
#     name="LaptopSearchTool",
#     func=laptop_tool_func,
#     description="Search laptops from the dataset based on user query."
# )

# # LangChain Memory & Agent
# llm = ChatGroq(api_key=api_key, model_name="llama3-8b-8192")
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent = initialize_agent(tools=[laptop_tool], llm=llm, memory=memory, agent="chat-conversational-react-description", verbose=True)

# # Search & Agent Response
# if st.button("🔍 Find Laptops"):
#     if not query.strip():
#         st.warning("Please enter a query.")
#     else:
#         st.subheader("🔍 Matching Laptops")
#         results = search_products(query, selected_brand, selected_ram, selected_os)

#         if not results:
#             st.error("No matching laptops found.")
#         else:
#             for i, r in enumerate(results, 1):
#                 st.markdown(f"**{i}. {r['Product Name']}**")
#                 st.markdown(f"💰 Price: ₹{r['Price']}, ⭐ Rating: {r['Rating']}")
#                 st.markdown(f"🔧 {r['Processor']}, 💾 {r['RAM']}, 💽 {r['Storage']} | 🖥️ {r['Display']}")
#                 st.markdown("---")

#         st.subheader("🤖 Agent Response")
#         with st.spinner("Thinking..."):
#             reply = agent.run(query)
#             st.success(reply)

# # Follow-up chat
# follow_up = st.text_input("Ask a follow-up question")
# if st.button("💬 Ask"):
#     if not follow_up.strip():
#         st.warning("Please enter your follow-up question.")
#     else:
#         with st.spinner("Responding..."):
#             reply = agent.run(follow_up)
#             st.info(reply)

# # Show chat memory
# if memory.chat_memory.messages:
#     st.subheader("🗂️ Chat History")
#     for m in memory.chat_memory.messages:
#         st.markdown(f"**{m.type.capitalize()}:** {m.content}")


import streamlit as st
import pandas as pd
import faiss
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("E:/pandas/clean_data.csv")
    return df

df = load_data()

# Create text from row
def create_text(row):
    return (
        f"Product: {row['Product Name']}, Rating: {row['Rating']}, Price: ₹{row['Price']}, "
        f"Processor: {row['Processor']}, RAM: {row['RAM']}, Storage: {row['Storage']}, "
        f"Display: {row['Display']}, OS: {row['OS']}, Warranty: {row['Warranty']}"
    )

df["text"] = df.apply(create_text, axis=1)

# Embed

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Use "cuda" if GPU is available


# model = SentenceTransformer("paraphrase-albert-small-v2")
embedding_matrix = model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# UI
st.title("💻 AI-Powered Laptop Recommendation")

rams = df['RAM'].dropna().unique().tolist()
os_options = df['OS'].dropna().unique().tolist()

selected_ram = st.selectbox("Choose RAM", ["Any"] + rams)
selected_os = st.selectbox("Choose OS", ["Any"] + os_options)

query = st.text_input("Enter your laptop requirements")

# Helper: extract price

def extract_price_limit(query):
    match = re.search(r"under\s*₹?\s*(\d+)", query)
    return int(match.group(1)) if match else None

# Filter and search

def search_products(query, ram, os_name, top_k=5):
    filtered_df = df.copy()

    if ram != "Any":
        filtered_df = filtered_df[filtered_df['RAM'] == ram]
    if os_name != "Any":
        filtered_df = filtered_df[filtered_df['OS'] == os_name]

    price_limit = extract_price_limit(query)
    if price_limit:
        filtered_df = filtered_df[filtered_df['Price'].astype(str).str.replace(",", "").astype(float) <= price_limit]

    if filtered_df.empty:
        return []

    query_vector = model.encode([query]).astype("float32")
    subset_embeddings = model.encode(filtered_df["text"].tolist()).astype("float32")

    temp_index = faiss.IndexFlatL2(subset_embeddings.shape[1])
    temp_index.add(subset_embeddings)

    distances, indices = temp_index.search(query_vector, min(top_k, len(filtered_df)))
    return [filtered_df.iloc[i].to_dict() for i in indices[0]]

# LangChain Agent Tool

def laptop_tool_func(query):
    results = search_products(query, selected_ram, selected_os)
    if not results:
        return "No laptops matched your query."
    return "\n".join([f"{r['Product Name']} - ₹{r['Price']}" for r in results])

laptop_tool = Tool(
    name="LaptopSearchTool",
    func=laptop_tool_func,
    description="Search laptops from the dataset based on user query."
)

# LangChain Memory & Agent
llm = ChatGroq(api_key=api_key, model_name="llama3-8b-8192")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools=[laptop_tool], llm=llm, memory=memory, agent="chat-conversational-react-description", verbose=True)

# Search & Agent Response
if st.button("🔍 Find Laptops"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.subheader("🔍 Matching Laptops")
        results = search_products(query, selected_ram, selected_os)

        if not results:
            st.error("No matching laptops found.")
        else:
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['Product Name']}**")
                st.markdown(f"💰 Price: ₹{r['Price']}, ⭐ Rating: {r['Rating']}")
                st.markdown(f"🔧 {r['Processor']}, 📂 {r['RAM']}, 💽 {r['Storage']} | 💥 {r['Display']}")
                st.markdown("---")

        st.subheader("🤖 Agent Response")
        with st.spinner("Thinking..."):
            reply = agent.run(query)
            st.success(reply)

# Follow-up chat
follow_up = st.text_input("Ask a follow-up question")
if st.button("💬 Ask"):
    if not follow_up.strip():
        st.warning("Please enter your follow-up question.")
    else:
        with st.spinner("Responding..."):
            reply = agent.run(follow_up)
            st.info(reply)

# Show chat memory
if memory.chat_memory.messages:
    st.subheader("🗂️ Chat History")
    for m in memory.chat_memory.messages:
        st.markdown(f"**{m.type.capitalize()}:** {m.content}")
