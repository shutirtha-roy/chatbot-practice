from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urls_list import urls

import os

os.environ["USER_AGENT"] = os.getenv('USER_AGENT')

splitter = RecursiveCharacterTextSplitter()
embedder = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))

splitted_docs = splitter.split_documents(WebBaseLoader(urls).load())

vector_store = FAISS.from_documents(splitted_docs, embedder)
vector_store.save_local("Swinburne_Chat_Bot")


