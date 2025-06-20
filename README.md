# AskSuraj 🤖 — Portfolio RAG Assistant

**AskSuraj** is a Retrieval-Augmented Generation (RAG)-powered assistant designed to answer questions about my portfolio, skills, projects, and experiences using natural language.

It uses vector-based semantic search to retrieve relevant information and generates responses using an LLM. This project showcases my ability to integrate AI, NLP, and fullstack development into a smart, job-ready portfolio tool.

---

## 🚀 Features

- 🔎 Semantic Search over portfolio documents
- 🧠 LLM-powered Q&A system
- 🗂️ Supports resume, project write-ups, blogs, and more
- 🪄 Built with FastAPI, LangServe, and OpenAI embeddings
- 💡 Great for recruiters, collaborators, or curious devs!

---

## 🛠️ Tech Stack

- **FastAPI** – for serving the API
- **LangChain + LangServe** – for RAG pipeline
- **FAISS / Chroma** – for vector indexing
- **OpenAI / Local LLM** – for response generation


---

## 📚 Use Case

> Curious about my projects? Just ask:  
> _"Tell me about the tech stack used in your Tic Tac Toe game."_  
> or  
> _"What is your current DSA plan?"_  

And boom — the AI pulls the exact info from my actual docs!

---

## 📦 How to Run Locally

```bash
git clone https://github.com/smartcraze/SurajGPT.git
cd asksuraj
pip install -r requirements.txt
python rag_chain.py   # index your portfolio docs
uvicorn app:app --reload
