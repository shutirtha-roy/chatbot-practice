from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urls_list import urls

import os

os.environ["USER_AGENT"] = os.getenv('USER_AGENT')

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_documents(loader.load())

def storeVector(pdfs):
    embedder = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))

    for index, pdf in enumerate(pdfs):
        split_docs = load_and_split_pdf(pdf)
        if index == 0:
            vector_store = FAISS.from_documents(split_docs, embedder)
        else:
            vector_store = FAISS.from_documents(split_docs, embedder)
            vector_store.merge_from(vector_store)

        vector_store.save_local("Swinburne_Chat_Bot_FROM_PDF")

#splitter = RecursiveCharacterTextSplitter()
#embedder = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))

#splitted_docs = splitter.split_documents(WebBaseLoader(urls).load())

#vector_store = FAISS.from_documents(splitted_docs, embedder)
#vector_store.save_local("Swinburne_Chat_Bot")


