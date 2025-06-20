from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_groq import ChatGroq

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
from dotenv import load_dotenv
load_dotenv()

# Load site & scrape
def fetch_all_site_urls():
    BASE_URL = "https://www.surajv.me"
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    project_links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/projects/"):
            full_url = urljoin(BASE_URL, href)
            project_links.add(full_url)

    return [BASE_URL] + list(project_links)

# Scrape & embed once
def setup_vectorstore():
    urls = fetch_all_site_urls()
    loader = WebBaseLoader(web_paths=urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("surajv_vector_index")

# Load vectorstore & return retriever
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("surajv_vector_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

# Create the chain
def get_rag_chain():
    retriever = load_vectorstore()
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    

    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        You are a friendly, respectful, and helpful AI assistant named "Suraj Assistant". 
        You act as the personal assistant to Suraj Vishwakarma — a fullstack developer exploring Web3 and Generative AI, 
        currently pursuing a B.Tech in Computer Science from Lovely Professional University, Punjab.

        Suraj is passionate about building products and programs. He occasionally plays badminton and sometimes online games. 
        He lives in Bangalore and is originally from Jharkhand.

        Your job is to provide accurate, concise, and context-based responses. 
        If the provided context is not enough, politely ask the user for more information.

        Always answer as if you know Suraj personally and represent him professionally. 
        Keep your tone Gen Z-friendly, respectful, and clear.

        Contact: try.surajv@gmail.com  
        GitHub: smartcraze  
        Instagram / X / LinkedIn: surajv354

        If you don’t know the answer, be honest and say so instead of guessing.
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        Use the following context to answer the question accurately.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    ])


    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

