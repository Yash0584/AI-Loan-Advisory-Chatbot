import os
import re
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st

# 1. SETUP
PROJECT_DIR = "/content/drive/MyDrive/AI_Loan_Advisory_Chatbot"
EMBEDDINGS_PATH = os.path.join(PROJECT_DIR, "processed/embeddings/embeddings.pkl")
METADATA_PATH = os.path.join(PROJECT_DIR, "processed/embeddings/metadata.pkl")
FAISS_INDEX_PATH = os.path.join(PROJECT_DIR, "vector_db/faiss_index.bin")
MODEL_PATH = os.path.join(PROJECT_DIR, "models/gru_intent_model.pth")
VOCAB_PATH = os.path.join(PROJECT_DIR, "models/vocabulary.pkl")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "models/tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_DIR, "models/label_encoder.pkl")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# 2. MODEL
class GRUIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.fc(self.dropout(hidden[-1]))

# 3. RESOURCES
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f: metadata = pickle.load(f)
    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    with open(TOKENIZER_PATH, 'rb') as f: tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f: le = pickle.load(f)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    net = GRUIntentClassifier(ckpt['vocab_size'], ckpt['embedding_dim'], ckpt['hidden_dim'], ckpt['num_layers'], ckpt['num_classes'], 0.2)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return embedder, index, metadata, net, vocab, tokenizer, le

# 4. PIPELINE
def rag_pipeline(query, res):
    embedder, index, meta, net, vocab, tok, le = res
    tokens = [vocab.get(w, 1) for w in re.sub(r'[^a-z0-9\s]', '', query.lower()).split()]
    tokens = torch.tensor([tokens + [0]*(tok['max_sequence_length']-len(tokens))]).long()
    with torch.no_grad():
        intent_idx = net(tokens).argmax().item()
    intent = le.inverse_transform([intent_idx])[0]
    
    vec = embedder.encode([query]).astype('float32')
    dists, idxs = index.search(vec, 3)
    context = "\n".join([meta[i]['chunk_text'] for i in idxs[0] if i != -1])
    prompt = f"Answer using: {context}\nQuestion: {query}"
    
    for _ in range(3):
        try:
            return model.generate_content(prompt).text, intent, context
        except ResourceExhausted:
            time.sleep(60)
    return "Rate limit hit. Try again in 60s.", intent, context

# 5. UI
st.set_page_config(page_title="AI Loan Advisor", layout="wide")
res = load_resources()

# Sidebar
with st.sidebar:
    st.header("Project Info")
    st.markdown("### Supported Banks\n• SBI\n• HDFC")
    st.markdown("### Supported Loans\n• Home, Personal, Car, Edu, Gold, LAP")
    st.success("Built with RAG & Gemini 2.5 Flash")

# Main Page
st.title("🏦 AI Loan Advisory Chatbot")
st.markdown("Ask any questions regarding loan policies, eligibility, or documentation.")

# Keep hint questions at the top
with st.expander("💡 Click for Example Questions"):
    st.markdown("- What documents are needed for a Car Loan?\n- How do I apply for an SBI Home Loan?")

# Persistent Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            st.caption(f"**Intent:** {message['metadata']['intent']}")
            with st.expander("View Context"): st.write(message["metadata"]["docs"])

if prompt := st.chat_input("Ask a loan-related question..."):
    st.chat_message("user").markdown(prompt)
    with st.spinner("Retrieving information..."):
        ans, intent, docs = rag_pipeline(prompt, res)
        with st.chat_message("assistant"):
            st.markdown(ans)
            st.caption(f"**Intent Detected:** {intent}")
            with st.expander("View Retrieved Context"): st.write(docs)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ans,
                "metadata": {"intent": intent, "docs": docs}
            })